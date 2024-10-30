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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class CompanyInfo:
    name: str
    cik: str
    filing_date: str
    report_date: str

def clean_text(text: Any) -> str:
    """Clean text by removing extra whitespace and special characters."""
    return ' '.join(str(text).split()).strip()

def extract_years_from_string(s: str) -> List[str]:
    """Extract four-digit years from a string."""
    return re.findall(r'(?:19|20)\d{2}', s)

def preprocess_table(df):
    """Preprocess table to handle complex headers and clean data."""
    # Drop completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Handle multi-row headers by combining them
    if df.iloc[0:3].apply(lambda x: x.astype(str).str.contains('Year|Month|Period|Date', case=False)).any().any():
        # Combine first few rows if they contain header information
        headers = []
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            if row.astype(str).str.contains('Year|Month|Period|Date', case=False).any():
                headers.append(row)
        
        if headers:
            # Combine headers vertically
            header = pd.concat(headers).fillna('')
            # Create meaningful column names
            columns = [' '.join(filter(None, col.split())) for col in header]
            # Set new headers and drop header rows
            df.columns = columns
            df = df.iloc[len(headers):]
    
    # Clean column names
    df.columns = [clean_text(str(col)).lower() for col in df.columns]
    
    # Reset index after dropping rows
    df = df.reset_index(drop=True)
    
    return df

def parse_table_row(row: List[str], headers: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a table row with improved header handling."""
    if not headers or not row or len(row) < 2:
        return None

    # Get the label from the first column
    label = clean_text(str(row[0])).lower()
    if not label or label in ['nan', 'none', '']:
        return None

    # Look for value and corresponding header
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
                    f.write(f"{item['label']}\t{item['value']}\t{column_name}\t{section_name}\n")
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
            
            # Wrap the HTML content in StringIO to fix the FutureWarning
            html_content = self.downloader.download_filing(url=metadata.primary_doc_url).decode()
            
            if not html_content:
                logger.error("Downloaded HTML content is empty.")
                return None
                
            # Use 'lxml' parser for better handling and suppress XMLParsedAsHTMLWarning
            self.soup = BeautifulSoup(html_content, 'lxml')
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
            
            for idx, df in enumerate(tables, start=1):
                # Apply preprocessing to clean and structure the table
                df = preprocess_table(df)
                
                # Identify the table type based on content
                table_text = df.to_string().lower()
                
                # Classification logic remains the same...
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
                for idx, row in df.iterrows():
                    # Skip rows that are likely sub-headers
                    if row.astype(str).str.contains('Year|Month|Period|Date', case=False).any():
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
