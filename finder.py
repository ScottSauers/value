# finder.py

import logging
from typing import Optional, Dict, Any
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
from bs4 import BeautifulSoup
from dataclasses import dataclass
import sys
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
    
            # Return company_info and html_content
            return {
                'company_info': company_info,
                'html_content': html_content
            }
    
        except Exception as e:
            logger.error(f"Error processing 10-K: {e}")
            return None
