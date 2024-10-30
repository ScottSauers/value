# parser.py

import logging
from typing import Any, Dict, List
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

def clean_text(text: Any) -> str:
    """Clean text by removing extra whitespace and special characters."""
    # Convert non-string types to string, then clean
    text = str(text) if not isinstance(text, str) else text
    return ' '.join(text.split()).strip()

def is_date(string: str) -> bool:
    """Check if the string matches common date patterns."""
    date_pattern1 = r'^[A-Za-z]+\s+\d{1,2},\s+\d{4}$'  # e.g., September 30, 2023
    date_pattern2 = r'^\d{4}$'  # e.g., 2023
    return bool(re.match(date_pattern1, string)) or bool(re.match(date_pattern2, string))

def preprocess_table(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess table to handle complex headers and clean data."""
    try:
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Ensure column names are strings
        df.columns = [str(col) for col in df.columns]

        # Identify header rows based on date patterns
        date_header_rows = []
        for i in range(min(10, len(df))):
            row = df.iloc[i].astype(str)
            date_matches = row.apply(is_date)
            if date_matches.sum() >= 2:  # At least two date-like entries
                date_header_rows.append(row)
                break  # Assume only one date header row

        if date_header_rows:
            # Use the identified date header row as column headers
            date_header = date_header_rows[0].apply(clean_text)
            # Assign to column names
            df.columns = date_header

            # Drop the header row from the DataFrame
            df = df.iloc[len(date_header_rows):].reset_index(drop=True)

            logger.debug("Set date header row as column names.")
        else:
            # Fallback to existing logic
            # Define header keywords to identify potential header rows
            header_keywords = ['year', 'month', 'period', 'date', 'fiscal', 'quarter', 'type', 'item', 'line', 'description']

            header_rows = []
            for i in range(min(10, len(df))):
                row = df.iloc[i].astype(str).str.lower()
                keyword_matches = row.str.contains('|'.join(header_keywords), regex=True)
                match_ratio = keyword_matches.sum() / len(row)
                if match_ratio > 0.5:
                    header_rows.append(row)

            # Define label row keywords to exclude
            label_row_keywords = ['years ended', 'total', 'percentage', 'change', 'subtotal']
            filtered_header_rows = []
            for row in header_rows:
                if any(row.str.contains(keyword, regex=False).any() for keyword in label_row_keywords):
                    logger.debug("Skipping label row in headers.")
                    continue
                filtered_header_rows.append(row)

            if filtered_header_rows:
                # Combine multiple header rows into a single header by concatenation
                combined_header = filtered_header_rows[0].fillna('')
                for additional_header in filtered_header_rows[1:]:
                    combined_header = combined_header + ' ' + additional_header.fillna('')

                # Clean and format the combined header
                combined_header = combined_header.apply(clean_text).str.replace(r'\s+', ' ', regex=True)

                # Assign the combined header to the DataFrame
                df.columns = combined_header

                # Drop the header rows from the DataFrame
                df = df.iloc[len(header_rows):].reset_index(drop=True)

                logger.debug("Combined multi-level headers and set as column names.")
            else:
                # Attempt to use the first row as header if it contains header keywords
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
        df.columns = [re.sub(r'\s+', ' ', col).strip() for col in df.columns]

        # Reset index after dropping rows
        df = df.reset_index(drop=True)

        # Log the final column names for verification
        logger.debug(f"Processed columns: {df.columns.tolist()}")

        return df
    except Exception as e:
        logger.error(f"Exception in preprocess_table: {e}")
        raise

def classify_table(df: pd.DataFrame) -> str:
    """Classify the table into sections based on content."""
    try:
        table_text = df.to_string().lower()

        # Define keywords for classification
        income_keywords = ['income', 'revenue', 'net income', 'operating income', 'gross margin']
        balance_keywords = ['assets', 'liabilities', 'equity', 'total assets', 'total liabilities',
                            "shareholders’ equity", "shareholders' equity", 'balance sheet']
        cash_flow_keywords = ['cash flow', 'operating activities', 'investing activities',
                              'financing activities', 'net cash', 'cash generated']

        # Priority-based classification
        if any(keyword in table_text for keyword in income_keywords):
            return 'Income Statement'
        elif any(keyword in table_text for keyword in balance_keywords):
            return 'Balance Sheet'
        elif any(keyword in table_text for keyword in cash_flow_keywords):
            return 'Cash Flow Statement'
        else:
            return 'Key Financial Ratios'
    except Exception as e:
        logger.error(f"Exception in classify_table: {e}")
        return 'Key Financial Ratios'

def parse_table_row(row: List[str], headers: List[str]) -> List[Dict[str, Any]]:
    """
    Parse a table row to extract all label-value pairs.

    Returns a list of dictionaries, each containing:
    - label: The financial metric label.
    - value: The numeric value.
    - column_name: The corresponding fiscal year.
    """
    try:
        if not headers or not row or len(row) < 2:
            return []

        # Get the label from the first column
        label = clean_text(str(row[0])).lower()
        if not label or label in ['nan', 'none', '']:
            return []

        # Initialize list to hold multiple entries
        entries = []

        # Iterate through all cells to extract numeric values
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
                # Remove any non-numeric characters for value extraction
                numeric_value = re.sub(r'[^\d\.-]', '', cleaned_cell)
                # Get corresponding header if available
                if i < len(headers):
                    column_name = clean_text(str(headers[i]))
                else:
                    column_name = 'N/A'

                entries.append({
                    'label': label.capitalize(),
                    'value': numeric_value,
                    'column_name': column_name if column_name else 'N/A'
                })

        return entries
    except Exception as e:
        logger.error(f"Exception in parse_table_row: {e} | Row: {row} | Headers: {headers}")
        return []

def extract_financial_data(html_content: str) -> Dict[str, Any]:
    """Extract key financial data from the document with table parsing."""
    from bs4 import BeautifulSoup
    data = {
        'income_statement': [],
        'balance_sheet': [],
        'cash_flow': [],
        'key_ratios': []
    }

    try:
        # Parse the HTML content using 'lxml' parser
        soup = BeautifulSoup(html_content, 'lxml')  # or 'xml' if appropriate

        # Use more flexible table parsing options
        tables = pd.read_html(StringIO(str(soup)),
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
            section = classify_table(df)
            logger.info(f"Table {idx}: Classified as '{section}'")

            # Improved row parsing with header context
            for row_idx, row in df.iterrows():
                # Skip rows that are likely sub-headers or notes
                if row.astype(str).str.contains('year|month|period|date|item|line|description', case=False).any():
                    continue

                row_entries = parse_table_row(row.tolist(), df.columns.tolist())
                if row_entries:
                    # Store each entry in the appropriate section
                    for entry in row_entries:
                        section_key = section.lower().replace(' ', '_')
                        if section_key in data:
                            data[section_key].append(entry)

    except Exception as e:
        logger.error(f"Error processing tables: {e}")

    return data

def save_fields_to_tsv(data: Dict[str, Any], filename: str = "sec_fields.tsv"):
    """Save fields to a TSV file."""
    logger.debug(f"Saving data to {filename}")
    try:
        with open(filename, 'a') as f:  # Changed to append mode
            company_info = data.get('company_info', {})

            # Write Company Information only once per filing
            if 'Company Information' not in f.read():
                # Write headers if file is empty
                f.seek(0, 2)  # Move to the end of the file
                if f.tell() == 0:
                    f.write("Tag\tValue\tColumn Name\tSection\tForm Type\tFiling Date\n")

            # Write Company Information
            f.write(f"Company Name\t{company_info.get('name', 'N/A')}\tN/A\tCompany Information\t{company_info.get('form_type', 'N/A')}\t{company_info.get('filing_date_actual', 'N/A')}\n")
            f.write(f"CIK\t{company_info.get('cik', 'N/A')}\tN/A\tCompany Information\t{company_info.get('form_type', 'N/A')}\t{company_info.get('filing_date_actual', 'N/A')}\n")
            f.write(f"Filing Date\t{company_info.get('filing_date', 'N/A')}\tN/A\tCompany Information\t{company_info.get('form_type', 'N/A')}\t{company_info.get('filing_date_actual', 'N/A')}\n")
            f.write(f"Report Date\t{company_info.get('report_date', 'N/A')}\tN/A\tCompany Information\t{company_info.get('form_type', 'N/A')}\t{company_info.get('filing_date_actual', 'N/A')}\n")

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
                    label = item['label']
                    value = item['value']
                    # Retrieve form type and filing date from company_info
                    form_type = company_info.get('form_type', 'N/A')
                    filing_date = company_info.get('filing_date_actual', 'N/A')
                    f.write(f"{label}\t{value}\t{column_name}\t{section_name}\t{form_type}\t{filing_date}\n")
        logger.debug("Data successfully saved to TSV.")
    except Exception as e:
        logger.error(f"Error saving TSV: {e}")
