import sys
from typing import Optional, Dict, Any, List
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
from bs4 import BeautifulSoup
from dataclasses import dataclass
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
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

def parse_table_row(row: List[str], headers: List[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a table row into a list of dictionaries mapping headers to cell values.
    Args:
        row: The data row
        headers: The header row cells
    """
    if not headers or not row or len(row) != len(headers):
        return None

    row_data = []
    for header, cell in zip(headers, row):
        header_clean = clean_text(header).lower()
        cell_clean = clean_text(cell)
        if header_clean and cell_clean:
            row_data.append({
                'label': header_clean,
                'value': cell_clean
            })
    return row_data if row_data else None

def save_fields_to_tsv(data: Dict[str, Any], filename: str = "sec_fields.tsv"):
    """Save fields to a TSV file."""
    with open(filename, 'w') as f:
        f.write("Tag\tValue\tColumn Name\tSection\n")
        
        company_info = data.get('company_info', {})
        f.write(f"Company Name\t{company_info.get('name', 'N/A')}\tN/A\tCompany Information\n")
        f.write(f"CIK\t{company_info.get('cik', 'N/A')}\tN/A\tCompany Information\n")
        f.write(f"Filing Date\t{company_info.get('filing_date', 'N/A')}\tN/A\tCompany Information\n")
        f.write(f"Report Date\t{company_info.get('report_date', 'N/A')}\tN/A\tCompany Information\n")
        
        sections = ['income_statement', 'balance_sheet', 'cash_flow']
        for section in sections:
            items = data.get(section, [])
            for item in items:
                f.write(f"{item['label']}\t{item['value']}\t{item.get('column_name', 'N/A')}\t{section.replace('_', ' ').title()}\n")

class SECFieldExtractor:
    def __init__(self, company_name: str, email: str):
        self.downloader = Downloader(company_name, email)
        
    def _get_company_info(self, identifier: str, filing_metadata) -> Optional[CompanyInfo]:
        try:
            if not filing_metadata:
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
            request = RequestedFilings(ticker_or_cik=identifier, form_type="10-K", limit=1)
            metadatas = self.downloader.get_filing_metadatas(request)
            
            if not metadatas:
                logger.error(f"No 10-K filings found for: {identifier}")
                return None
                
            metadata = metadatas[0]
            company_info = self._get_company_info(identifier, metadata)
            
            if not company_info:
                return None
                
            logger.info(f"Processing 10-K for {company_info.name} (CIK: {company_info.cik})")
            
            html = self.downloader.download_filing(url=metadata.primary_doc_url).decode()
            
            if not html:
                logger.error("Downloaded HTML content is empty.")
                return None
                
            self.soup = BeautifulSoup(html, 'html.parser')
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
            'cash_flow': []
        }
        
        tables = self.soup.find_all('table')
        logger.info(f"Found {len(tables)} tables in the document.")

        for idx, table in enumerate(tables, start=1):
            table_text = clean_text(table.get_text()).lower()

            # Skip tables that are unlikely to contain financial data
            if any(keyword in table_text for keyword in ['index', 'table of contents', 'signature', 'exhibit']):
                logger.debug(f"Table {idx}: Skipped due to irrelevant content.")
                continue

            rows = table.find_all('tr')
            if not rows:
                logger.debug(f"Table {idx}: No rows found.")
                continue

            # Attempt to identify headers
            headers = []
            header_row_index = None
            for i, row in enumerate(rows[:5]):  # Check first 5 rows for headers
                cells = row.find_all(['th', 'td'])
                cell_texts = [clean_text(cell.get_text()) for cell in cells]
                if any('year' in cell.lower() or 'ended' in cell.lower() for cell in cell_texts):
                    headers = cell_texts
                    header_row_index = i
                    break

            if not headers:
                logger.debug(f"Table {idx}: No headers found, skipping table.")
                continue

            # Check if headers contain financial keywords
            financial_keywords = ['net income', 'revenue', 'assets', 'liabilities', 'equity', 'cash', 'operating activities', 'investing activities', 'financing activities']
            if not any(any(keyword in header.lower() for keyword in financial_keywords) for header in headers):
                logger.debug(f"Table {idx}: Headers do not contain financial keywords, skipping table.")
                continue

            logger.info(f"Table {idx}: Processing with headers: {headers}")

            # Extract data rows
            data_rows = rows[header_row_index + 1:]
            for row in data_rows:
                cells = row.find_all(['td', 'th'])
                cell_texts = [clean_text(cell.get_text()) for cell in cells]

                # Skip empty rows
                if not any(cell_texts):
                    continue

                # Adjust row if number of cells doesn't match headers
                if len(cell_texts) < len(headers):
                    # Pad the row with empty strings
                    cell_texts.extend([''] * (len(headers) - len(cell_texts)))
                elif len(cell_texts) > len(headers):
                    # Trim the row to match the headers
                    cell_texts = cell_texts[:len(headers)]

                parsed_row = parse_table_row(cell_texts, headers)
                if not parsed_row:
                    continue

                # Classify and store the data
                if any('income' in header.lower() for header in headers):
                    data['income_statement'].extend(parsed_row)
                elif any('asset' in header.lower() or 'liabilit' in header.lower() or 'equity' in header.lower() for header in headers):
                    data['balance_sheet'].extend(parsed_row)
                elif any('cash' in header.lower() for header in headers):
                    data['cash_flow'].extend(parsed_row)

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
        print(f"Data extraction complete. Results saved to sec_fields.tsv")
    else:
        print("Failed to extract fields from 10-K")

if __name__ == "__main__":
    main()
