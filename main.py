# main.py

import sys
import logging
from finder import SECFieldExtractor
import parser

# Configure logging (ensure it does not duplicate if already configured in modules)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) != 5:
        print("Usage: python main.py <CIK/Ticker> <your_company_name> <your_email> <number_of_quarters>")
        print("Examples:")
        print("  python main.py AAPL MyCompany your.email@example.com 4")
        print("  python main.py 320193 MyCompany your.email@example.com 2")
        sys.exit(1)

    identifier = sys.argv[1]
    company_name = sys.argv[2]
    email = sys.argv[3]

    try:
        number_of_quarters = int(sys.argv[4])
        if number_of_quarters < 1:
            raise ValueError
    except ValueError:
        print("Error: <number_of_quarters> must be a positive integer.")
        sys.exit(1)

    extractor = SECFieldExtractor(company_name, email)

    # Define form types to fetch: 10-K (annual) and 10-Q (quarterly)
    form_types = ["10-K", "10-Q"]

    # Since 10-K is annual, and 10-Q is quarterly, adjust the limit accordingly
    # For example, for 4 quarters, you might expect 1 10-K and 3 10-Q filings
    # To ensure enough filings, set the limit to number_of_quarters + 1 per form type
    limit_per_form = number_of_quarters + 1

    fetched_data = extractor.get_latest_fields(identifier, form_types, limit_per_form)

    if fetched_data:
        # Initialize a list to hold all financial data
        all_financial_data = []

        for filing in fetched_data:
            company_info = filing.get('company_info')
            html_content = filing.get('html_content')
            form_type = filing.get('form_type')
            filing_date = filing.get('filing_date')

            if not html_content:
                logger.warning(f"Skipping filing dated {filing_date} due to empty content.")
                continue

            # Extract financial data
            financial_data = parser.extract_financial_data(html_content)

            # Add company_info and filing details to financial_data
            financial_data['company_info'] = {
                'name': company_info.name,
                'cik': company_info.cik,
                'filing_date': company_info.filing_date,
                'report_date': company_info.report_date,
                'form_type': form_type,
                'filing_date_actual': filing_date
            }

            # Append to the master list
            all_financial_data.append(financial_data)

        if not all_financial_data:
            logger.error("No financial data extracted from the filings.")
            sys.exit(1)

        # Save all financial data to TSV
        # Initialize the TSV file with headers
        tsv_filename = "sec_fields.tsv"
        try:
            with open(tsv_filename, 'w') as f:
                f.write("Tag\tValue\tColumn Name\tSection\tForm Type\tFiling Date\n")

            for financial_data in all_financial_data:
                parser.save_fields_to_tsv(financial_data, filename=tsv_filename)

            logger.info(f"Data extraction complete. Results saved to {tsv_filename}")
        except Exception as e:
            logger.error(f"Error writing to TSV file: {e}")
            sys.exit(1)

    else:
        logger.error("Failed to extract fields from filings")

if __name__ == "__main__":
    main()
