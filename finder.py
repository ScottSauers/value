# finder.py

import logging
from typing import Optional, Dict, Any, List
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
from bs4 import BeautifulSoup
from dataclasses import dataclass
import re
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
    form_type: str
    fiscal_year_end: str
    period_of_report: str
    accepted_date: str
    sec_url: str
    document_url: str

class SECFieldExtractor:
    def __init__(self, company_name: str, email: str):
        self.downloader = Downloader(company_name, email)
    
    def _get_company_info(self, identifier: str, filing_metadata) -> Optional[CompanyInfo]:
        if not filing_metadata:
            logger.error("Filing metadata is missing.")
            return None
    
        # Access attributes directly
        return CompanyInfo(
            name=filing_metadata.company_name if hasattr(filing_metadata, 'company_name') else 'N/A',
            cik=filing_metadata.cik if hasattr(filing_metadata, 'cik') else 'N/A',
            filing_date=filing_metadata.filing_date if hasattr(filing_metadata, 'filing_date') else 'N/A',
            report_date=filing_metadata.report_date if hasattr(filing_metadata, 'report_date') else 'N/A',
            form_type=filing_metadata.form_type if hasattr(filing_metadata, 'form_type') else 'N/A',
            fiscal_year_end=filing_metadata.fiscal_year_end if hasattr(filing_metadata, 'fiscal_year_end') else 'N/A',
            period_of_report=filing_metadata.period_of_report if hasattr(filing_metadata, 'period_of_report') else 'N/A',
            accepted_date=filing_metadata.accepted_date if hasattr(filing_metadata, 'accepted_date') else 'N/A',
            sec_url=filing_metadata.sec_url if hasattr(filing_metadata, 'sec_url') else 'N/A',
            document_url=filing_metadata.primary_doc_url if hasattr(filing_metadata, 'primary_doc_url') else 'N/A'
        )
    
    def get_latest_fields(
        self, 
        identifier: str, 
        form_types: List[str], 
        limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch the latest filings based on form types and limit.

        Args:
            identifier (str): CIK or ticker symbol.
            form_types (List[str]): List of form types to fetch (e.g., ["10-K", "10-Q"]).
            limit (int): Total number of filings to fetch.

        Returns:
            Optional[List[Dict[str, Any]]]: List of dictionaries containing company_info and html_content.
        """
        try:
            logger.debug(f"Requesting latest filings for identifier: {identifier} with form types: {form_types} and limit: {limit}")
            # Create a combined request for all specified form types
            request = RequestedFilings(
                ticker_or_cik=identifier, 
                form_type=form_types, 
                limit=limit
            )
            metadatas = self.downloader.get_filing_metadatas(request)
    
            if not metadatas:
                logger.error(f"No filings found for: {identifier} with form types: {form_types}")
                return None
    
            results = []
            for metadata in metadatas:
                company_info = self._get_company_info(identifier, metadata)
    
                if not company_info:
                    logger.error("Company information could not be retrieved for a filing.")
                    continue
    
                logger.info(f"Processing {metadata.form_type} for {company_info.name} (CIK: {company_info.cik})")
    
                # Download and decode the HTML content
                html_content = self.downloader.download_filing(url=metadata.primary_doc_url).decode()
    
                if not html_content:
                    logger.error("Downloaded HTML content is empty.")
                    continue
    
                # Append the result
                results.append({
                    'company_info': company_info,
                    'html_content': html_content,
                    'form_type': metadata.form_type,
                    'filing_date': metadata.filing_date
                })
    
            if not results:
                logger.error("No valid filings were processed.")
                return None
    
            return results
    
        except Exception as e:
            logger.error(f"Error processing filings: {e}")
            return None
