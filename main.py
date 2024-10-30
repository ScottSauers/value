# main.py

import sys
import logging
from url_finder import SECFieldExtractor
import parser

# Configure logging (ensure it does not duplicate if already configured in modules)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <CIK/Ticker> <your_company_name> <your_email>")
        print("Examples:")
        print("  python main.py AAPL MyCompany your.email@example.com")
        print("  python main.py 320193 MyCompany your.email@example.com")
        sys.exit(1)

    identifier = sys.argv[1]
    company_name = sys.argv[2]
    email = sys.argv[3]

    extractor = SECFieldExtractor(company_name, email)
    fetched_data = extractor.get_latest_10k_fields(identifier)

    if fetched_data:
        company_info = fetched_data.get('company_info')
        html_content = fetched_data.get('html_content')

        # Extract financial data
        financial_data = parser.extract_financial_data(html_content)

        # Add company_info to financial_data
        financial_data['company_info'] = {
            'name': company_info.name,
            'cik': company_info.cik,
            'filing_date': company_info.filing_date,
            'report_date': company_info.report_date
        }

        # Save to TSV
        parser.save_fields_to_tsv(financial_data)

        logger.info("Data extraction complete. Results saved to sec_fields.tsv")
    else:
        logger.error("Failed to extract fields from 10-K")

if __name__ == "__main__":
    main()
