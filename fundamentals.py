import os
import finagg
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv
import threading
import time

@dataclass
class SECConcept:
    """Represents an SEC XBRL concept with its taxonomy and units."""
    tag: str
    taxonomy: str = "us-gaap"
    units: str = "USD"
    description: str = ""

class SECDataExtractor:
    """Extracts raw SEC fundamental data without derived calculations."""
    
    # Comprehensive list of SEC concepts to collect
    SEC_CONCEPTS = [
        # Balance Sheet - Assets
        SECConcept("Assets"),
        SECConcept("AssetsCurrent"),
        SECConcept("AssetsNoncurrent"),
        SECConcept("CashAndCashEquivalentsAtCarryingValue"),
        SECConcept("MarketableSecurities"),
        SECConcept("AvailableForSaleSecurities"),
        SECConcept("AccountsReceivableNetCurrent"),
        SECConcept("InventoryNet"),
        SECConcept("PrepaidExpenseAndOtherAssetsCurrent"),
        SECConcept("PropertyPlantAndEquipmentNet"),
        SECConcept("OperatingLeaseRightOfUseAsset"),
        SECConcept("IntangibleAssetsNetExcludingGoodwill"),
        SECConcept("Goodwill"),
        SECConcept("OtherAssets"),
        SECConcept("InvestmentsAndAdvances"),
        
        # Balance Sheet - Liabilities
        SECConcept("Liabilities"),
        SECConcept("LiabilitiesCurrent"),
        SECConcept("AccountsPayableCurrent"),
        SECConcept("AccruedLiabilitiesCurrent"),
        SECConcept("ContractWithCustomerLiabilityCurrent"),
        SECConcept("OperatingLeaseLiabilityCurrent"),
        SECConcept("ShortTermBorrowings"),
        SECConcept("LongTermDebtCurrent"),
        SECConcept("LongTermDebtNoncurrent"),
        SECConcept("OperatingLeaseLiabilityNoncurrent"),
        SECConcept("DeferredTaxLiabilitiesNoncurrent"),
        SECConcept("OtherLiabilitiesNoncurrent"),
        SECConcept("PensionAndOtherPostretirementBenefitPlans"),
        
        # Stockholders' Equity
        SECConcept("StockholdersEquity"),
        SECConcept("CommonStockParOrStatedValuePerShare"),
        SECConcept("CommonStockSharesAuthorized"),
        SECConcept("CommonStockSharesIssued"),
        SECConcept("CommonStockSharesOutstanding"),
        SECConcept("RetainedEarningsAccumulatedDeficit"),
        SECConcept("AccumulatedOtherComprehensiveIncomeLossNetOfTax"),
        
        # Income Statement
        SECConcept("Revenues"),
        SECConcept("CostOfRevenue"),
        SECConcept("CostOfGoodsAndServicesSold"),
        SECConcept("GrossProfit"),
        SECConcept("ResearchAndDevelopmentExpense"),
        SECConcept("SellingGeneralAndAdministrativeExpense"),
        SECConcept("OperatingExpenses"),
        SECConcept("OperatingIncomeLoss"),
        SECConcept("NonoperatingIncomeExpense"),
        SECConcept("InterestExpense"),
        SECConcept("InterestIncome"),
        SECConcept("OtherNonoperatingIncomeExpense"),
        SECConcept("IncomeTaxExpenseBenefit"),
        SECConcept("NetIncomeLoss"),
        SECConcept("ComprehensiveIncomeNetOfTax"),
        
        # Per Share Data
        SECConcept("EarningsPerShareBasic", units="USD/shares"),
        SECConcept("EarningsPerShareDiluted", units="USD/shares"),
        SECConcept("WeightedAverageNumberOfSharesOutstandingBasic", units="shares"),
        SECConcept("WeightedAverageNumberOfDilutedSharesOutstanding", units="shares"),
        SECConcept("DividendsPerShareDeclared", units="USD/shares"),
        
        # Cash Flow
        SECConcept("NetCashProvidedByUsedInOperatingActivities"),
        SECConcept("NetCashProvidedByUsedInInvestingActivities"),
        SECConcept("NetCashProvidedByUsedInFinancingActivities"),
        SECConcept("DepreciationDepletionAndAmortization"),
        SECConcept("ShareBasedCompensation"),
        SECConcept("CapitalExpendituresIncurredButNotYetPaid"),
        SECConcept("PaymentsToAcquirePropertyPlantAndEquipment"),
        SECConcept("PaymentsToAcquireBusinessesNetOfCashAcquired"),
        SECConcept("PaymentsToAcquireOtherInvestments"),
        SECConcept("ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities"),
        SECConcept("ProceedsFromIssuanceOfLongTermDebt"),
        SECConcept("RepaymentsOfLongTermDebt"),
        SECConcept("PaymentsOfDividends"),
        SECConcept("PaymentsForRepurchaseOfCommonStock"),
        
        # Other Important Metrics
        SECConcept("CommitmentsAndContingencies"),
        SECConcept("GuaranteeObligations"),
        SECConcept("SegmentReportingInformation"),
        SECConcept("LeaseCost"),
        SECConcept("LeasePayments"),
        SECConcept("StockRepurchaseProgramAuthorizedAmount"),
        SECConcept("DerivativeInstrumentsAndHedgingActivitiesDisclosure"),
        SECConcept("FairValueMeasurementsRecurring"),
        SECConcept("RevenueFromContractWithCustomerExcludingAssessedTax"),
        SECConcept("RevenueRemainingPerformanceObligation")
    ]

    # Class-level variables for rate limiting
    _rate_limit_lock = threading.Lock()
    _request_timestamps: List[float] = []
    _MAX_CALLS = 5
    _PERIOD = 1.0  # seconds

    def __init__(self, output_dir: str = "./data"):
        """Initialize the extractor with output directory."""
        load_dotenv()
        # Set SEC API user agent
        sec_user_agent = os.getenv('SEC_API_USER_AGENT')
        if not sec_user_agent:
            raise ValueError("SEC_API_USER_AGENT environment variable not set")
        
        # Set the user agent for finagg
        finagg.sec.api.USER_AGENT = sec_user_agent
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'sec_data.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def rate_limited_get(self, tag: str, ticker: str, taxonomy: str, units: str) -> pd.DataFrame:
        """
        Wrapper for the finagg.sec.api.company_concept.get method with rate limiting.
        
        Args:
            tag: SEC concept tag
            ticker: Stock ticker symbol
            taxonomy: Taxonomy to use
            units: Units of the concept
            
        Returns:
            DataFrame containing the concept data
        """
        while True:
            with SECDataExtractor._rate_limit_lock:
                current_time = time.time()
                # Remove timestamps older than PERIOD
                SECDataExtractor._request_timestamps = [
                    timestamp for timestamp in SECDataExtractor._request_timestamps
                    if current_time - timestamp < SECDataExtractor._PERIOD
                ]
                if len(SECDataExtractor._request_timestamps) < SECDataExtractor._MAX_CALLS:
                    # Record the current timestamp and proceed
                    SECDataExtractor._request_timestamps.append(current_time)
                    break
                else:
                    # Calculate the time to wait until the earliest timestamp is older than PERIOD
                    earliest_timestamp = SECDataExtractor._request_timestamps[0]
                    sleep_time = SECDataExtractor._PERIOD - (current_time - earliest_timestamp)
                    if sleep_time > 0:
                        self.logger.debug(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                    else:
                        sleep_time = SECDataExtractor._PERIOD / SECDataExtractor._MAX_CALLS
                        self.logger.debug(f"Rate limit edge case. Sleeping for {sleep_time:.2f} seconds.")
            # Sleep outside the lock to allow other threads to proceed
            time.sleep(sleep_time)

        try:
            response = finagg.sec.api.company_concept.get(
                tag,
                ticker=ticker,
                taxonomy=taxonomy,
                units=units
            )
            return response
        except Exception as e:
            self.logger.error(f"Error during API call for {tag} and {ticker}: {e}")
            raise

    def get_sec_data(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve comprehensive SEC fundamental data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame containing all fundamental data
        """
        all_data = []
        
        for concept in self.SEC_CONCEPTS:
            try:
                df = self.rate_limited_get(
                    tag=concept.tag,
                    ticker=ticker,
                    taxonomy=concept.taxonomy,
                    units=concept.units
                )
                
                if not df.empty:
                    # Keep only original filings
                    df = finagg.sec.api.filter_original_filings(df)
                    
                    # Rename value column to concept tag
                    df = df.rename(columns={'value': concept.tag})
                    
                    # Keep only essential columns
                    df = df[['end', concept.tag]]
                    df = df.rename(columns={'end': 'filing_date'})
                    
                    all_data.append(df)
                    self.logger.info(f"Successfully retrieved {concept.tag} data for {ticker}")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve {concept.tag} data for {ticker}: {str(e)}")
                
        if not all_data:
            self.logger.warning(f"No SEC data found for {ticker}")
            return pd.DataFrame()
            
        # Merge all data on filing date
        merged_df = all_data[0]
        for df in all_data[1:]:
            merged_df = pd.merge(merged_df, df, on='filing_date', how='outer')
            
        # Sort by filing date
        merged_df = merged_df.sort_values('filing_date', ascending=False)
        
        return merged_df

    def save_data(self, df: pd.DataFrame, ticker: str, metadata: Dict) -> Tuple[str, str]:
        """Save the SEC data and metadata to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        data_filename = f"{ticker}_sec_data_{timestamp}.tsv"
        data_filepath = self.output_dir / data_filename
        df.to_csv(data_filepath, sep='\t', index=False)
        self.logger.info(f"Data saved to {data_filepath}")
        
        metadata_filename = f"{ticker}_sec_data_{timestamp}.json"
        metadata_filepath = self.output_dir / metadata_filename
        pd.Series(metadata).to_json(metadata_filepath)
        self.logger.info(f"Metadata saved to {metadata_filepath}")
        
        return str(data_filepath), str(metadata_filepath)

    def process_ticker(self, ticker: str) -> Tuple[str, str]:
        """Process a single ticker to extract and save SEC data."""
        metadata = {
            'ticker': ticker,
            'extraction_date': datetime.now().isoformat(),
            'source': 'SEC EDGAR API',
            'concepts_requested': [c.tag for c in self.SEC_CONCEPTS]
        }
        
        try:
            df = self.get_sec_data(ticker)
            if df.empty:
                raise ValueError(f"No SEC data found for {ticker}")
            
            return self.save_data(df, ticker, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to process {ticker}: {str(e)}")
            raise

def main():
    """Main execution function."""
    extractor = SECDataExtractor()
    
    # List of tickers to process
    tickers = ["AAPL"]
    
    for ticker in tickers:
        try:
            data_file, metadata_file = extractor.process_ticker(ticker)
            print(f"Successfully processed {ticker}")
            print(f"Data file: {data_file}")
            print(f"Metadata file: {metadata_file}")
        except Exception as e:
            print(f"Failed to process {ticker}: {str(e)}")

if __name__ == "__main__":
    main()
