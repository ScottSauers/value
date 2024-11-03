import requests
import xml.etree.ElementTree as ET
import pandas as pd
import finagg
import logging
import time
from pathlib import Path
from tqdm import tqdm
from threading import Lock
from collections import deque

# Configuration
INPUT_FILE = 'small_cap_stocks_latest.csv'
TICKER_COLUMN = 'ticker'
USER_AGENT = 'useragent@gmail.com'  # Replace with a valid email
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_PATH = OUTPUT_DIR / "company_xbrl_tags_summary.csv"
NO_10K_CSV_PATH = OUTPUT_DIR / "no_10k_tickers.csv"
FILING_TYPE = '10-K'
MAX_REQUESTS_PER_SECOND = 8  # Strict rate limit

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load existing data if available
if CSV_PATH.exists():
    existing_data = pd.read_csv(CSV_PATH, index_col=0)
else:
    existing_data = pd.DataFrame()

if NO_10K_CSV_PATH.exists():
    no_10k_data = pd.read_csv(NO_10K_CSV_PATH)
    no_10k_tickers = set(no_10k_data[TICKER_COLUMN].unique())
else:
    no_10k_tickers = set()

processed_tickers = set(existing_data.columns) | no_10k_tickers

# Rate limiting mechanism
lock = Lock()
request_times = deque()

def rate_limited_request(url, headers):
    with lock:
        current_time = time.time()
        # Remove timestamps older than 1 second
        while request_times and current_time - request_times[0] >= 1:
            request_times.popleft()
        # If we've reached the max requests per second, wait
        if len(request_times) >= MAX_REQUESTS_PER_SECOND:
            sleep_time = 1 - (current_time - request_times[0])
            time.sleep(sleep_time)
            current_time = time.time()
            # Remove the timestamp we're now past
            request_times.popleft()
        request_times.append(current_time)
    response = requests.get(url, headers=headers)
    return response

# Retrieve XBRL tags for a company's latest 10-K filing
def get_xbrl_tags(cik: str) -> dict:
    try:
        search_url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        headers = {'User-Agent': USER_AGENT}
        response = rate_limited_request(search_url, headers)
        response.raise_for_status()
        data = response.json()

        # Locate the most recent 10-K filing
        accession_number = None
        for form, accession in zip(data['filings']['recent']['form'], data['filings']['recent']['accessionNumber']):
            if form == FILING_TYPE:
                accession_number = accession.replace('-', '')
                break

        if not accession_number:
            logger.warning(f"No 10-K found for CIK {cik}")
            return None

        # Access the filing directory via `index.json`
        base_url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}'
        index_url = f'{base_url}/index.json'
        response = rate_limited_request(index_url, headers)
        response.raise_for_status()
        index_data = response.json()

        # Find primary XML instance document
        xbrl_url = None
        for file in index_data['directory']['item']:
            if file['name'].endswith('.xml') and all(x not in file['name'] for x in ['cal', 'def', 'lab', 'pre']):
                xbrl_url = f"{base_url}/{file['name']}"
                break

        if not xbrl_url:
            logger.warning(f"No primary XBRL document found for CIK {cik}")
            return None

        # Download and parse the XBRL instance document
        response = rate_limited_request(xbrl_url, headers)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # Collect and format tags by namespace
        tags = {}
        for element in root.iter():
            namespace = element.tag.split('}')[0].strip('{')
            tag_name = element.tag.split('}')[1] if '}' in element.tag else element.tag
            if namespace not in tags:
                tags[namespace] = []
            tags[namespace].append(tag_name)

        # Deduplicate tags within each namespace
        for ns in tags:
            tags[ns] = sorted(set(tags[ns]))

        return tags

    except Exception as e:
        logger.error(f"Failed to retrieve XBRL tags for CIK {cik}: {e}")
        return None

# Function to process a single ticker and fetch its XBRL tags
def process_ticker(ticker):
    try:
        cik = finagg.sec.api.get_cik(ticker, user_agent=USER_AGENT)
    except Exception as e:
        logger.error(f"Error retrieving CIK for {ticker}: {e}")
        return ticker, None, None

    if not cik:
        logger.warning(f"No CIK found for ticker {ticker}")
        return ticker, None, None

    tags = get_xbrl_tags(cik)
    if tags is None:
        return ticker, None, None

    return ticker, cik, tags

# Main processing function
def main():
    # Load tickers from input file
    try:
        tickers_df = pd.read_csv(INPUT_FILE)
        all_tickers = set(tickers_df[TICKER_COLUMN].unique())
        tickers_to_process = all_tickers - processed_tickers
        cached_count = len(processed_tickers)
        to_process_count = len(tickers_to_process)

        logger.info(f"Total tickers: {len(all_tickers)}")
        logger.info(f"Already cached tickers: {cached_count} ({(cached_count / len(all_tickers)) * 100:.2f}%)")
        logger.info(f"Tickers to process: {to_process_count} ({(to_process_count / len(all_tickers)) * 100:.2f}%)")
    except Exception as e:
        logger.error(f"Error loading tickers from {INPUT_FILE}: {e}")
        return

    all_data = {}
    no_10k_count = 0

    with tqdm(total=to_process_count, desc="Processing Tickers", unit="ticker") as pbar:
        for ticker in tickers_to_process:
            ticker, cik, tags = process_ticker(ticker)
            if ticker is None:
                continue

            if tags is None:
                no_10k_count += 1
                no_10k_tickers.add(ticker)
            else:
                all_data[ticker] = {}
                for namespace, tag_list in tags.items():
                    for tag in tag_list:
                        all_data[ticker][f"{namespace}:{tag}"] = 1

            pbar.update(1)

            # Save incremental results to prevent data loss
            if all_data:
                pd.DataFrame.from_dict(all_data, orient='index').fillna(0).T.to_csv(CSV_PATH)
            pd.DataFrame({TICKER_COLUMN: list(no_10k_tickers)}).to_csv(NO_10K_CSV_PATH, index=False)

    # Final summary statistics
    percent_no_10k = (no_10k_count / to_process_count) * 100 if to_process_count > 0 else 0
    logger.info(f"\nSummary Statistics:\nTotal tickers processed: {to_process_count}\n"
                f"No 10-K found for {no_10k_count} companies ({percent_no_10k:.2f}% of total)\n"
                f"Results saved to {CSV_PATH}")

    if all_data:
        final_data = pd.DataFrame.from_dict(all_data, orient='index').fillna(0).T
        print("\nNamespace Tags Summary:")
        for ticker in final_data.columns:
            concept_count = final_data[ticker].sum()
            logger.info(f"Ticker: {ticker}, Total Concepts Extracted: {int(concept_count)}")

if __name__ == "__main__":
    main()
