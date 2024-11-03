import requests
import xml.etree.ElementTree as ET
import pandas as pd
import finagg
import logging
import time
from pathlib import Path

# Configuration
INPUT_FILE = 'small_cap_stocks_latest.csv'
TICKER_COLUMN = 'ticker'
USER_AGENT = 'useragent@email.com'  # Replace with a valid email
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_PATH = OUTPUT_DIR / "company_xbrl_tags_summary.csv"
FILING_TYPE = '10-K'

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
            namespace = element.tag.split('}')[0].strip('{')  # Extract namespace
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
        tickers = tickers_df[TICKER_COLUMN].unique().tolist()
        logger.info(f"Loaded {len(tickers)} tickers from {INPUT_FILE}")
    except Exception as e:
        logger.error(f"Error loading tickers from {INPUT_FILE}: {e}")
        return

    all_data = []
    for ticker in tickers:
        logger.info(f"Processing {ticker}")
        try:
            # Get CIK for ticker
            cik = finagg.sec.api.get_cik(ticker, user_agent=USER_AGENT)
        except Exception as e:
            logger.error(f"Error retrieving CIK for {ticker}: {e}")
            continue

        if not cik:
            logger.warning(f"No CIK found for ticker {ticker}")
            continue

        tags = get_xbrl_tags(cik)
        if tags:
            # Flatten the dictionary to save in a DataFrame-friendly format
            for namespace, tag_list in tags.items():
                for tag in tag_list:
                    all_data.append({
                        'ticker': ticker,
                        'cik': cik,
                        'namespace': namespace,
                        'tag': tag
                    })
            # Respect rate limits
            time.sleep(0.11)

    # Convert all_data to DataFrame and save
    if all_data:
        full_df = pd.DataFrame(all_data)
        full_df.to_csv(CSV_PATH, index=False)
        logger.info(f"Saved all data to {CSV_PATH}")
    else:
        logger.warning("No data was retrieved for any ticker.")

    # Print summary statistics if data is available
    if all_data:
        print("\nSummary Statistics:")
        print(f"Total companies processed: {len(tickers)}")
        print(f"Total tags extracted: {len(full_df)}")
        print("Tags count per namespace:")
        print(full_df['namespace'].value_counts())

if __name__ == "__main__":
    main()
