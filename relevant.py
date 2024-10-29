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

def extract_dates_from_headers(headers: List[str]) -> Optional[str]:
    """Extract date from headers if present."""
    date_pattern = re.compile(r'([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})')
    for header in headers:
        match = date_pattern.search(header)
        if match:
            return match.group(1)
    return None

def parse_table_row(row: List[str], headers: List[str] = None) -> Optional[Dict[str, Any]]:
    """
    Parse a table row into a meaningful key-value pair if possible.
    Args:
        row: The data row
        headers: The header row cells
    """
    if len(row) < 2 or not headers:
        return None

    label = clean_text(row[0]).lower()
    if not label:
        return None

    value = None
    column_name = 'N/A'

    # Iterate through the cells to find the first numeric value
    for idx, cell in enumerate(row[1:], start=1):  # Start from index 1
        cleaned_cell = clean_text(cell)
        # Skip if cell is empty
        if not cleaned_cell:
            continue
        # Check if the cell contains a numeric value
        if re.search(r'\d', cleaned_cell):
            # Assign the value
            value = cleaned_cell
            # Assign the column name based on headers
            if headers and len(headers) > idx:
                column_name = clean_text(headers[idx])
            break

    if not value:
        return None

    return {
        'label': label,
        'value': value,
        'column_name': column_name if column_name else 'N/A'
    }

def save_fields_to_tsv(data: Dict[str, Any], filename: str = "sec_fields.tsv"):
    """Save fields to a TSV file."""
    with open(filename, 'w') as f:
        f.write("Tag\tValue\tColumn Name\tSection\n")
        
        company_info = data.get('company_info', {})
        f.write(f"Company Name\t{company_info.get('name', 'N/A')}\tN/A\tCompany Information\n")
        f.write(f"CIK\t{company_info.get('cik', 'N/A')}\tN/A\tCompany Information\n")
        f.write(f"Filing Date\t{company_info.get('filing_date', 'N/A')}\tN/A\tCompany Information\n")
        f.write(f"Report Date\t{company_info.get('report_date', 'N/A')}\tN/A\tCompany Information\n")
        
        for metric, value in data['summary_metrics'].items():
            f.write(f"{metric}\t{value}\tN/A\tSummary Metrics\n")
        
        sections = {
            'Income Statement': data['income_statement'],
            'Balance Sheet': data['balance_sheet'],
            'Cash Flow Statement': data['cash_flow'],
            'Key Financial Ratios': data['key_ratios']
        }
        
        for section_name, items in sections.items():
            for item in items:
                column_name = item.get('column_name', 'N/A')
                f.write(f"{item['label']}\t{item['value']}\t{column_name}\t{section_name}\n")

class SECFieldExtractor:
    def __init__(self, company_name: str, email: str):
        self.downloader = Downloader(company_name, email)
        
    def _normalize_cik(self, cik: str) -> str:
        return cik.zfill(10)
        
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
            'summary_metrics': {},
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow': [],
            'key_ratios': []
        }
        
        tables = self.soup.find_all('table')
        logger.info(f"Found {len(tables)} tables in the document.")
        
        # Limit detailed debugging to first 3 tables
        debug_table_limit = 3
        
        for idx, table in enumerate(tables):
            rows = table.find_all('tr')
            if not rows:
                logger.debug(f"Table {idx+1}: No rows found.")
                continue

            header_cells = []
            header_found = False

            # Define a list of common financial header keywords
            financial_header_keywords = ['year', 'amount', 'total', 'value', 'usd', 'description', 'item', 'date', 'balance', 'shares']

            # Attempt to identify header row based on keywords
            max_header_rows = 5  # Adjust as needed based on table structure
            for i in range(min(max_header_rows, len(rows))):
                potential_headers = [clean_text(cell.get_text(strip=True)) for cell in rows[i].find_all(['td', 'th'])]
                
                # Check if any of the header keywords are present in the potential headers
                if any(any(keyword in cell.lower() for keyword in financial_header_keywords) for cell in potential_headers):
                    header_cells = potential_headers
                    rows = rows[i+1:]  # Exclude the header row from data rows
                    header_found = True
                    logger.info(f"Table {idx+1}: Header identified in row {i+1}: {header_cells}")
                    break
            
            if not header_found:
                # Fallback: Assume the first row is the header if no header row was identified
                header_cells = [clean_text(cell.get_text(strip=True)) for cell in rows[0].find_all(['td', 'th'])]
                rows = rows[1:]  # Exclude the first row from data rows
                logger.warning(f"Table {idx+1}: No header row identified based on keywords. Using first row as header: {header_cells}")
            
            logger.info(f"Table {idx+1}: Final Headers found: {header_cells}")
            
            # Debugging: Print sample tables
            if idx < debug_table_limit:
                logger.debug(f"--- Table {idx+1} Sample ---")
                logger.debug(f"Headers: {header_cells}")
                sample_rows = [clean_text(cell.get_text(strip=True)) for cell in rows[:3].find_all(['td', 'th'])]
                logger.debug(f"Sample Rows: {sample_rows}")
                logger.debug(f"--------------------------")
            
            parsed_rows = []
            
            for row_num, row in enumerate(rows, start=1):
                cells = [clean_text(cell.get_text(strip=True)) for cell in row.find_all(['td', 'th'])]
                
                # Skip entirely empty rows
                if not any(cells):
                    logger.debug(f"Table {idx+1}, Row {row_num}: Empty row skipped.")
                    continue
                
                row_data = parse_table_row(cells, header_cells)
                if row_data:
                    parsed_rows.append(row_data)
                else:
                    logger.debug(f"Table {idx+1}, Row {row_num}: No valid data extracted.")
            
            if parsed_rows:
                table_text = clean_text(table.get_text()).lower()
                if any(keyword in table_text for keyword in ['income statement', 'statement of operations', 'comprehensive income']):
                    data['income_statement'].extend(parsed_rows)
                    logger.info(f"Table {idx+1}: Classified as 'Income Statement'")
                elif any(keyword in table_text for keyword in ['balance sheet', 'balance sheets']):
                    data['balance_sheet'].extend(parsed_rows)
                    logger.info(f"Table {idx+1}: Classified as 'Balance Sheet'")
                elif any(keyword in table_text for keyword in ['cash flow', 'cash flows']):
                    data['cash_flow'].extend(parsed_rows)
                    logger.info(f"Table {idx+1}: Classified as 'Cash Flow Statement'")
                else:
                    data['key_ratios'].extend(parsed_rows)
                    logger.info(f"Table {idx+1}: Classified as 'Key Financial Ratios'")
        
        all_text = clean_text(self.soup.get_text()).lower()
        
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
        print(data)
        save_fields_to_tsv(data)
    else:
        print("Failed to extract fields from 10-K")

if __name__ == "__main__":
    main()
