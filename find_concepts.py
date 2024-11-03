import requests
import xml.etree.ElementTree as ET
import pandas as pd
import finagg
import logging
import time
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_FILE = 'small_cap_stocks_latest.csv'
TICKER_COLUMN = 'ticker'
USER_AGENT = 'useragent@email.com'  # Use a valid email
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_PATH = OUTPUT_DIR / "company_xbrl_tags_summary.csv"
NO_10K_CSV_PATH = OUTPUT_DIR / "no_10k_tickers.csv"
FILING_TYPE = '10-K'

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

    # Placeholder for data collection
    data_dict = {ticker: {} for ticker in tickers}
    no_10k_count = 0

    for ticker in tqdm(tickers, desc="Processing tickers"):
        logger.info(f"Processing {ticker}")
        try:
            cik = finagg.sec.api.get_cik(ticker, user_agent=USER_AGENT)
        except Exception as e:
            logger.error(f"Error retrieving CIK for {ticker}: {e}")
            continue

        if not cik:
            logger.warning(f"No CIK found for ticker {ticker}")
            continue

        tags = get_xbrl_tags(cik)
        if tags:
            # Flatten and populate the data_dict with tags
            for namespace, tag_list in tags.items():
                for tag in tag_list:
                    data_dict[ticker][f"{namespace}:{tag}"] = 1  # Mark presence of tag for this ticker
        else:
            no_10k_count += 1
            no_10k_tickers.add(ticker)

            # Save to no_10k CSV after each ticker without 10-K found
            pd.DataFrame({TICKER_COLUMN: list(no_10k_tickers)}).to_csv(NO_10K_CSV_PATH, index=False)
            continue

        # Save incremental progress to CSV for each ticker processed
        pd.DataFrame.from_dict(data_dict, orient='index').fillna(0).T.to_csv(CSV_PATH)

        # Pause to respect SEC rate limits
        time.sleep(0.1)

    # Final summary statistics
    total_tickers = len(tickers)
    percent_no_10k = (no_10k_count / total_tickers) * 100 if total_tickers > 0 else 0
    logger.info(f"\nSummary Statistics:\nTotal companies processed: {total_tickers}\n"
                f"No 10-K found for {no_10k_count} companies ({percent_no_10k:.2f}% of total)\n"
                f"Results saved to {CSV_PATH}")

if __name__ == "__main__":
    main()
