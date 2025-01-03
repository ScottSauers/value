import requests
import xml.etree.ElementTree as ET
import pandas as pd
import finagg
import logging
import time
from pathlib import Path
from tqdm import tqdm
from threading import Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import re

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
MAX_WORKERS = 10  # Number of parallel threads

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
semaphore = Semaphore(MAX_REQUESTS_PER_SECOND)
request_times = deque()

def rate_limited_request(url, headers):
    with semaphore:
        current_time = time.time()
        # Remove timestamps older than 1 second
        with lock:
            while request_times and current_time - request_times[0] >= 1:
                request_times.popleft()
            request_times.append(current_time)
        response = requests.get(url, headers=headers)
    return response

# Retrieve XBRL tags for a company's latest 10-K filing
def get_xbrl_tags(cik: str) -> set:
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

        # Collect and format tags by concept
        concepts = set()
        for element in root.iter():
            tag_name = element.tag.split('}')[-1]  # Extract tag name after namespace
            concept_name = re.sub(r'\d{4}', '', tag_name)  # Remove any years from the tag name
            concepts.add(concept_name)

        return concepts

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

    def process_and_cache_ticker(ticker):
        nonlocal no_10k_count
        ticker, cik, tags = process_ticker(ticker)
        if ticker is None:
            return

        if tags is None:
            no_10k_count += 1
            no_10k_tickers.add(ticker)
        else:
            all_data[ticker] = {tag: 1 for tag in tags}

        # Save incremental results to prevent data loss
        with lock:
            if all_data:
                updated_data = existing_data.combine_first(pd.DataFrame.from_dict(all_data, orient='index').fillna(0).T)
                updated_data.to_csv(CSV_PATH)
            pd.DataFrame({TICKER_COLUMN: list(no_10k_tickers)}).to_csv(NO_10K_CSV_PATH, index=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_and_cache_ticker, tickers_to_process), total=to_process_count, desc="Processing Tickers", unit="ticker"))

    # Final summary statistics
    if to_process_count == 0 and not all_data:
        # Use cached data for summary statistics if everything was already processed
        final_data = existing_data
    else:
        final_data = pd.DataFrame.from_dict(all_data, orient='index').fillna(0).T if all_data else existing_data

    percent_no_10k = (no_10k_count / to_process_count) * 100 if to_process_count > 0 else 0
    logger.info(f"\nSummary Statistics:\nTotal tickers processed: {to_process_count}\n"
                f"No 10-K found for {no_10k_count} companies ({percent_no_10k:.2f}% of total)\n"
                f"Results saved to {CSV_PATH}")

    if not final_data.empty:
        # Calculate concept presence percentages
        concept_presence = (final_data.sum(axis=1) / len(final_data.columns)) * 100
        final_data['Percent of Companies'] = concept_presence
        final_data.to_csv(CSV_PATH)

        print("\nConcept Tags Summary:")
        for ticker in final_data.columns[:-1]:
            concept_count = final_data[ticker].sum()
            logger.info(f"Ticker: {ticker}, Total Concepts Extracted: {int(concept_count)}")

        # Additional statistics
        concepts_more_than_one = (final_data.sum(axis=1) > 1).sum()
        concepts_over_50_percent = (concept_presence > 50).sum()
        total_concepts = len(final_data.index)
        max_concept_presence = concept_presence.max()
        concepts_at_least_half_max = (concept_presence >= (max_concept_presence / 2)).sum()

        logger.info(f"\nAdditional Summary Statistics:\n"
                    f"Total Concepts: {total_concepts}\n"
                    f"Concepts present in more than one company: {concepts_more_than_one}\n"
                    f"Concepts present in over 50% of companies: {concepts_over_50_percent}\n"
                    f"Concepts present in at least half of the number of companies that the most common concept is present in: {concepts_at_least_half_max}")

if __name__ == "__main__":
    main()
