# extractor.py
from typing import Optional, Dict, Any
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
from dataclasses import dataclass
import logging

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
        return CompanyInfo(
            name=filing_metadata.company_name or 'N/A',
            cik=filing_metadata.cik or 'N/A',
            filing_date=filing_metadata.filing_date or 'N/A',
            report_date=filing_metadata.report_date or 'N/A',
            form_type=filing_metadata.form_type or 'N/A',
            fiscal_year_end=filing_metadata.fiscal_year_end or 'N/A',
            period_of_report=filing_metadata.period_of_report or 'N/A',
            accepted_date=filing_metadata.accepted_date or 'N/A',
            sec_url=filing_metadata.sec_url or 'N/A',
            document_url=filing_metadata.primary_doc_url or 'N/A'
        )

    def get_latest_10k_url(self, identifier: str) -> Optional[Dict[str, Any]]:
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
        
        return {"company_info": company_info, "html_content": html_content}
