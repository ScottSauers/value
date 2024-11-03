import requests
import xml.etree.ElementTree as ET
import pandas as pd
import finagg
import logging
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
INPUT_FILE = 'small_cap_stocks_latest.csv'
TICKER_COLUMN = 'ticker'
USER_AGENT = 'useragent@email.com'  # Replace with a valid email
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_PATH = OUTPUT_DIR / "company_xbrl_tags_summary.csv"
NO_10K_CSV_PATH = OUTPUT_DIR / "no_10k_tickers.csv"
FILING_TYPE = '10-K'
MAX_REQUESTS_PER_SECOND = 10
BATCH_SIZE = MAX_REQUESTS_PER_SECOND

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

# Retrieve XBRL tags for a company's latest 10-K filing
def get_xbrl_tags(cik: str) -> dict:
    try:
        search_url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(search_url, headers=headers)
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
        response = requests.get(index_url, headers=headers)
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
        response = requests.get(xbrl_url, headers=headers)
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
        tickers = [ticker for ticker in tickers_df[TICKER_COLUMN].unique() if ticker not in processed_tickers]
        logger.info(f"Loaded {len(tickers)} tickers to process from {INPUT_FILE}")
    except Exception as e:
        logger.error(f"Error loading tickers from {INPUT_FILE}: {e}")
        return

    all_data = {ticker: {} for ticker in tickers}
    no_10k_count = 0

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = []
        with tqdm(total=len(tickers), desc="Processing ", unit="tickers") as pbar:
            for i in range(0, len(tickers), BATCH_SIZE):
                batch = tickers[i:i + BATCH_SIZE]
                for ticker in batch:
                    futures.append(executor.submit(process_ticker, ticker))

                # Wait for the batch to complete and apply the rate limit
                for future in as_completed(futures):
                    ticker, cik, tags = future.result()
                    if ticker is None:
                        continue

                    if tags is None:
                        no_10k_count += 1
                        no_10k_tickers.add(ticker)
                    else:
                        for namespace, tag_list in tags.items():
                            for tag in tag_list:
                                all_data[ticker][f"{namespace}:{tag}"] = 1

                    pbar.update(1)  # Update progress bar after each completed future

                # Save incremental results to prevent data loss
                pd.DataFrame.from_dict(all_data, orient='index').fillna(0).T.to_csv(CSV_PATH)
                pd.DataFrame({TICKER_COLUMN: list(no_10k_tickers)}).to_csv(NO_10K_CSV_PATH, index=False)

                # Pause to respect the rate limit
                time.sleep(1)  # Limit to max 10 requests per second by pausing 1 sec after each batch

    # Final summary statistics
    total_tickers = len(tickers)
    percent_no_10k = (no_10k_count / total_tickers) * 100 if total_tickers > 0 else 0
    logger.info(f"\nSummary Statistics:\nTotal companies processed: {total_tickers}\n"
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
