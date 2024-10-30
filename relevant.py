import sys
from typing import Optional, Dict, Any, List
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
from bs4 import BeautifulSoup
from dataclasses import dataclass
import logging
import re
import pandas as pd
from io import StringIO
import warnings
from bs4 import XMLParsedAsHTMLWarning

# Suppress XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CompanyInfo:
    name: str
    cik: str
    filing_date: str
    report_date: str

def clean_text(text: Any) -> str:
    """Clean text by removing extra whitespace and special characters."""
    # Convert non-string types to string, then clean
    text = str(text) if not isinstance(text, str) else text
    return ' '.join(text.split()).strip()

def extract_years_from_string(s: str) -> List[str]:
    """Extract four-digit years from a string."""
    return re.findall(r'(?:19|20)\d{2}', s)

def preprocess_table(df):
    """Preprocess table to handle complex headers and clean data."""
    try:
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Ensure column names are strings
        df.columns = [str(col) for col in df.columns]

        # Define header keywords to identify potential header rows
        header_keywords = ['year', 'month', 'period', 'date', 'fiscal', 'quarter', 'type', 'item', 'line', 'description']

        # Initialize headers list
        header_rows = []

        # Iterate through the first 6 rows to find header rows
        for i in range(min(6, len(df))):
            row = df.iloc[i].astype(str).str.lower()
            # Check if the row contains any header keywords
            if row.str.contains('|'.join(header_keywords), regex=True).any():
                header_rows.append(row)

        if header_rows:
            # Combine multiple header rows into a single header by concatenation
            combined_header = header_rows[0].fillna('')
            for additional_header in header_rows[1:]:
                combined_header = combined_header + ' ' + additional_header.fillna('')

            # Clean and format the combined header
            combined_header = combined_header.apply(clean_text).str.replace(r'\s+', ' ', regex=True)

            # Assign the combined header to the DataFrame
            df.columns = combined_header

            # Drop the header rows from the DataFrame
            df = df.iloc[len(header_rows):].reset_index(drop=True)

            logger.debug("Combined multi-level headers and set as column names.")
        else:
            # If no headers detected, attempt to use the first row as header
            first_row = df.iloc[0].astype(str).str.lower()
            if first_row.str.contains('|'.join(header_keywords), regex=True).any():
                combined_header = first_row.apply(clean_text).str.replace(r'\s+', ' ', regex=True)
                df = df[1:].reset_index(drop=True)
                df.columns = combined_header
                logger.debug("Assigned first row as header.")
            else:
                # Assign default column names if no headers detected
                df.columns = [f"column_{i+1}" for i in range(len(df.columns))]
                logger.debug("No header rows detected. Assigned generic column names.")

        # Further clean column names
        df.columns = [re.sub(r'\s+', ' ', col).strip().lower() for col in df.columns]

        # Reset index after dropping rows
        df = df.reset_index(drop=True)

        # Log the final column names for verification
        logger.debug(f"Processed columns: {df.columns.tolist()}")

        return df
    except Exception as e:
        logger.error(f"Exception in preprocess_table: {e}")
        raise

def parse_table_row(row: List[str], headers: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a table row with improved header handling."""
    try:
        if not headers or not row or len(row) < 2:
            return None

        # Get the label from the first column
        label = clean_text(str(row[0])).lower()
        if not label or label in ['nan', 'none', '']:
            return None

        # Initialize value and column name
        value = None
        column_name = None

        # Iterate through cells to find the first valid numeric value
        for i, cell in enumerate(row[1:], start=1):
            cell_str = str(cell)
            # Clean the cell value
            cleaned_cell = clean_text(cell_str)

            # Skip empty cells
            if not cleaned_cell or cleaned_cell.lower() in ['nan', 'none']:
                continue

            # Check if the cell contains a numeric value or parenthesized number
            numeric_pattern = r'^-?\$?\s*\(?\d[\d,\.]*\)?%?$'
            if re.match(numeric_pattern, cleaned_cell.replace(' ', '')):
                value = cleaned_cell
                # Get corresponding header if available
                if i < len(headers):
                    column_name = clean_text(str(headers[i]))
                break

        if not value:
            return None

        return {
            'label': label,
            'value': value,
            'column_name': column_name if column_name else 'N/A'
        }
    except Exception as e:
        logger.error(f"Exception in parse_table_row: {e} | Row: {row} | Headers: {headers}")
        return None

def save_fields_to_tsv(data: Dict[str, Any], filename: str = "sec_fields.tsv"):
    """Save fields to a TSV file."""
    logger.debug(f"Saving data to {filename}")
    try:
        with open(filename, 'w') as f:
            f.write("Tag\tValue\tColumn Name\tSection\n")

            company_info = data.get('company_info', {})
            f.write(f"Company Name\t{company_info.get('name', 'N/A')}\tN/A\tCompany Information\n")
            f.write(f"CIK\t{company_info.get('cik', 'N/A')}\tN/A\tCompany Information\n")
            f.write(f"Filing Date\t{company_info.get('filing_date', 'N/A')}\tN/A\tCompany Information\n")
            f.write(f"Report Date\t{company_info.get('report_date', 'N/A')}\tN/A\tCompany Information\n")

            sections = {
                'Income Statement': data.get('income_statement', []),
                'Balance Sheet': data.get('balance_sheet', []),
                'Cash Flow Statement': data.get('cash_flow', []),
                'Key Financial Ratios': data.get('key_ratios', [])
            }

            for section_name, items in sections.items():
                for item in items:
                    column_name = item.get('column_name', 'N/A')
                    # Capitalize the first letter of the label for consistency
                    label = item['label'].capitalize()
                    f.write(f"{label}\t{item['value']}\t{column_name}\t{section_name}\n")
        logger.debug("Data successfully saved to TSV.")
    except Exception as e:
        logger.error(f"Error saving TSV: {e}")

class SECFieldExtractor:
    def __init__(self, company_name: str, email: str):
        self.downloader = Downloader(company_name, email)

    def _get_company_info(self, identifier: str, filing_metadata) -> Optional[CompanyInfo]:
        try:
            if not filing_metadata:
                logger.error("Filing metadata is empty.")
                return None

            return CompanyInfo(
                name=filing_metadata.company_name,
                cik=filing_metadata.cik,
                filing_date=filing_metadata.filing_date,
                report_date=filing_metadata.report_date
            )
        except Exception as e:
            logger.error(f"Error getting company info: {e}")
            return None

    def get_latest_10k_fields(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            logger.debug(f"Requesting latest 10-K filings for identifier: {identifier}")
            request = RequestedFilings(ticker_or_cik=identifier, form_type="10-K", limit=1)
            metadatas = self.downloader.get_filing_metadatas(request)

            if not metadatas:
                logger.error(f"No 10-K filings found for: {identifier}")
                return None

            metadata = metadatas[0]
            company_info = self._get_company_info(identifier, metadata)

            if not company_info:
                logger.error("Company information could not be retrieved.")
                return None

            logger.info(f"Processing 10-K for {company_info.name} (CIK: {company_info.cik})")

            # Download and decode the HTML content
            html_content = self.downloader.download_filing(url=metadata.primary_doc_url).decode()

            if not html_content:
                logger.error("Downloaded HTML content is empty.")
                return None

            # Parse the HTML content using 'lxml' parser
            self.soup = BeautifulSoup(html_content, 'lxml')  # or 'xml' if appropriate
            financial_data = self.extract_financial_data()
            financial_data['company_info'] = {
                'name': company_info.name,
                'cik': company_info.cik,
                'filing_date': company_info.filing_date,
                'report_date': company_info.report_date
            }
            return financial_data

        except Exception as e:
            logger.error(f"Error processing 10-K: {e}")
            return None

    def extract_financial_data(self) -> Dict[str, Any]:
        """Extract key financial data from the document with table parsing."""
        data = {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow': [],
            'key_ratios': []
        }

        try:
            # Use more flexible table parsing options
            tables = pd.read_html(StringIO(str(self.soup)),
                                   flavor='bs4',
                                   thousands=',',  # Handle number formatting
                                   decimal='.',
                                   na_values=['', 'N/A', 'None'],
                                   keep_default_na=True)

            logger.info(f"Found {len(tables)} tables in the document.")

            # Optionally, print the first few tables for debugging
            for idx, df in enumerate(tables[:10], start=1):
                print(f"\n--- Raw Table {idx} ---")
                for row_num in range(min(5, len(df))):
                    row = df.iloc[row_num].tolist()
                    print(f"Row {row_num + 1}: {row}")
                if len(df) > 5:
                    last_row = df.iloc[-1].tolist()
                    print(f"Last Row: {last_row}")
                print("----------------------\n")

            for idx, df in enumerate(tables, start=1):
                # Apply preprocessing to clean and structure the table
                df = preprocess_table(df)

                # Log the DataFrame shape and columns for debugging
                logger.debug(f"Table {idx} shape: {df.shape} | Columns: {df.columns.tolist()}")

                # Identify the table type based on content
                table_text = df.to_string().lower()

                income_keywords = ['income', 'revenue', 'net income', 'operating income', 'gross margin']
                balance_keywords = ['assets', 'liabilities', 'equity', 'total assets', 'total liabilities',
                                    "shareholders’ equity", "shareholders' equity", 'balance sheet']
                cash_flow_keywords = ['cash flow', 'operating activities', 'investing activities',
                                      'financing activities', 'net cash', 'cash generated']

                if any(keyword in table_text for keyword in income_keywords):
                    section = 'Income Statement'
                elif any(keyword in table_text for keyword in balance_keywords):
                    section = 'Balance Sheet'
                elif any(keyword in table_text for keyword in cash_flow_keywords):
                    section = 'Cash Flow Statement'
                else:
                    section = 'Key Financial Ratios'

                logger.info(f"Table {idx}: Classified as '{section}'")

                # Improved row parsing with header context
                for row_idx, row in df.iterrows():
                    # Skip rows that are likely sub-headers or notes
                    if row.astype(str).str.contains('year|month|period|date|item|line|description', case=False).any():
                        continue

                    row_data = parse_table_row(row.tolist(), df.columns.tolist())
                    if row_data:
                        # Store data in appropriate section
                        section_key = section.lower().replace(' ', '_')
                        if section_key in data:
                            data[section_key].append(row_data)

        except Exception as e:
            logger.error(f"Error processing tables: {e}")

        return data

def main():
    if len(sys.argv) != 4:
        print("Usage: python relevant.py <CIK/Ticker> <your_company_name> <your_email>")
        print("Examples:")
        print("  python relevant.py AAPL MyCompany your.email@example.com")
        print("  python relevant.py 320193 MyCompany your.email@example.com")
        sys.exit(1)

    identifier = sys.argv[1]
    company_name = sys.argv[2]
    email = sys.argv[3]

    extractor = SECFieldExtractor(company_name, email)
    data = extractor.get_latest_10k_fields(identifier)

    if data:
        save_fields_to_tsv(data)
        logger.info("Data extraction complete. Results saved to sec_fields.tsv")
    else:
        logger.error("Failed to extract fields from 10-K")

if __name__ == "__main__":
    main()
