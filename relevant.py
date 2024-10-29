import sys
from typing import Optional, Dict, Any, List
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
from bs4 import BeautifulSoup
from dataclasses import dataclass
import logging
import re
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class CompanyInfo:
    name: str
    cik: str
    filing_date: str
    report_date: str

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    return ' '.join(text.split()).strip()

def extract_years_from_string(s: str) -> List[str]:
    """Extract four-digit years from a string."""
    return re.findall(r'(?:19|20)\d{2}', s)

def parse_table_row(row: List[str], headers: List[str]) -> Optional[Dict[str, Any]]:
    """
    Parse a table row into a key-value pair if possible.
    Args:
        row: The data row
        headers: The header row cells
    """
    if not headers or not row or len(row) < 2:
        logger.debug("Row skipped: Insufficient data or headers missing.")
        return None

    label = clean_text(row[0]).lower()
    if not label:
        logger.debug("Row skipped: Label is empty.")
        return None

    # Initialize default values
    value = None
    column_name = 'N/A'

    # Iterate through the cells to find the first numeric value
    for idx, cell in enumerate(row[1:], start=1):  # Start from index 1
        cleaned_cell = clean_text(cell)
        if not cleaned_cell:
            continue
        # Check if the cell contains a numeric value
        if re.search(r'\d', cleaned_cell.replace(',', '').replace('.', '').replace('(', '').replace(')', '')):
            value = cleaned_cell
            # Assign the column name based on headers
            if headers and len(headers) > idx:
                column_name = clean_text(headers[idx])
                # Extract year if present
                years = extract_years_from_string(column_name)
                if years:
                    column_name = ','.join(years)
                else:
                    column_name = clean_text(headers[idx])
            break

    if not value:
        logger.debug(f"No numeric value found in row: {row}")
        return None

    logger.debug(f"Parsed Row - Label: {label}, Value: {value}, Column Name: {column_name}")
    return {
        'label': label,
        'value': value,
        'column_name': column_name if column_name else 'N/A'
    }

def save_fields_to_tsv(data: Dict[str, Any], filename: str = "sec_fields.tsv"):
    """Save fields to a TSV file."""
    logger.debug(f"Saving data to {filename}")
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
                f.write(f"{item['label']}\t{item['value']}\t{column_name}\t{section_name}\n")
    logger.debug("Data successfully saved to TSV.")

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
            
            html_content = self.downloader.download_filing(url=metadata.primary_doc_url).decode()
            
            if not html_content:
                logger.error("Downloaded HTML content is empty.")
                return None
                
            self.soup = BeautifulSoup(html_content, 'html.parser')
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
        """Extract key financial data from the document."""
        data = {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow': [],
            'key_ratios': []
        }
        
        # Use pandas to read all tables
        try:
            tables = pd.read_html(str(self.soup), header=0, flavor='bs4')
            logger.info(f"Found {len(tables)} tables in the document.")
        except Exception as e:
            logger.error(f"Error reading tables with pandas: {e}")
            return data
        
        # Define financial keywords for classification
        income_keywords = ['income', 'revenue', 'net income', 'operating income']
        balance_keywords = ['assets', 'liabilities', 'equity', 'total assets', 'total liabilities', 'shareholders’ equity']
        cash_flow_keywords = ['cash flow', 'operating activities', 'investing activities', 'financing activities', 'net cash']
        
        # Target keywords for special debugging
        target_keywords = ['net income', 'total assets']
        
        for idx, df in enumerate(tables, start=1):
            # Clean column headers
            df.columns = [clean_text(str(col)).lower() for col in df.columns]
            
            # Check if the table contains financial keywords
            table_text = df.to_string().lower()
            if any(keyword in table_text for keyword in income_keywords):
                section = 'Income Statement'
            elif any(keyword in table_text for keyword in balance_keywords):
                section = 'Balance Sheet'
            elif any(keyword in table_text for keyword in cash_flow_keywords):
                section = 'Cash Flow Statement'
            else:
                section = 'Key Financial Ratios'
            
            logger.info(f"Table {idx}: Classified as '{section}'")
            
            # Check if table contains target keywords for detailed debugging
            if any(keyword in table_text for keyword in target_keywords):
                logger.debug(f"Table {idx} contains target keywords '{target_keywords}'. Printing entire table for debugging.")
                print(f"\n=== Table {idx}: {section} ===")
                print(df.to_string(index=False))
                print("=== End of Table ===\n")
                logger.debug(f"Table {idx} printed for detailed debugging.")
            
            # Iterate over rows and parse data
            for _, row in df.iterrows():
                row_data = parse_table_row(row.tolist(), df.columns.tolist())
                if row_data:
                    # Depending on section, append to appropriate list
                    if section in ['Income Statement', 'Balance Sheet', 'Cash Flow Statement']:
                        data[section.lower().replace(' ', '_')].append(row_data)
                    else:
                        data['key_ratios'].append(row_data)
        
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
