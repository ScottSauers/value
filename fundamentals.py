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
import sqlite3
import time
import hashlib
import gc

@dataclass
class SECConcept:
    """Represents an SEC XBRL concept with its taxonomy and units."""
    tag: str
    taxonomy: str = "us-gaap"
    units: str = "USD"
    description: str = ""

class SECDataExtractor:
    """Extracts raw SEC fundamental data with granular caching."""
    
    def _init_granular_cache(self):
        """Initialize SQLite database with granular caching tables and improved schema."""
        with sqlite3.connect(str(self.cache_dir / 'granular_cache.db')) as conn:
            try:
                conn.execute('BEGIN IMMEDIATE')
                
                # Table for storing individual concept data points with fetch status
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS concept_cache (
                        ticker TEXT,
                        concept_tag TEXT,
                        filing_date TEXT,
                        concept_value REAL,
                        taxonomy TEXT,
                        units TEXT,
                        fetch_status TEXT DEFAULT 'unknown',
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (ticker, concept_tag, filing_date)
                    )
                ''')
                
                # Table for tracking concept fetch status with more detail
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS concept_status (
                        ticker TEXT,
                        concept_tag TEXT,
                        last_attempt TIMESTAMP,
                        status TEXT,
                        error TEXT,
                        retry_count INTEGER DEFAULT 0,
                        PRIMARY KEY (ticker, concept_tag)
                    )
                ''')
                
                # Create indices for better performance
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_concept_cache_ticker 
                    ON concept_cache(ticker)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_concept_cache_last_updated 
                    ON concept_cache(last_updated)
                ''')
                
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
    
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
        SECConcept("OtherAssets"),
        
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
        SECConcept("OtherNonoperatingIncomeExpense"),
        SECConcept("IncomeTaxExpenseBenefit"),
        SECConcept("NetIncomeLoss"),
        SECConcept("ComprehensiveIncomeNetOfTax"),
        
        # Per Share Data
        SECConcept("EarningsPerShareBasic", units="USD/shares"),
        SECConcept("EarningsPerShareDiluted", units="USD/shares"),
        SECConcept("WeightedAverageNumberOfSharesOutstandingBasic", units="shares"),
        SECConcept("WeightedAverageNumberOfDilutedSharesOutstanding", units="shares"),
        
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

        self.cache_dir = Path(output_dir) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_granular_cache()

    def get_cached_concept(self, ticker: str, concept: SECConcept) -> Optional[pd.DataFrame]:
        """Retrieve cached data for a specific concept if available."""
        with sqlite3.connect(str(self.cache_dir / 'granular_cache.db')) as conn:
            query = '''
                SELECT filing_date, concept_value as value
                FROM concept_cache 
                WHERE ticker = ? AND concept_tag = ?
                AND last_updated > datetime('now', '-7 days')
                ORDER BY filing_date DESC
            '''
            try:
                df = pd.read_sql_query(query, conn, params=(ticker, concept.tag))
                if not df.empty:
                    # Rename columns to match expected format
                    df = df.rename(columns={'value': concept.tag})
                    return df
            except Exception as e:
                self.logger.debug(f"Cache miss for {ticker} {concept.tag}: {str(e)}")
        return None

    
    def cache_concept_data(self, ticker: str, concept: SECConcept, data: pd.DataFrame):
        """Cache the data for a specific concept with explicit missing data handling."""
        with sqlite3.connect(str(self.cache_dir / 'granular_cache.db')) as conn:
            try:
                conn.execute('BEGIN IMMEDIATE')
                
                # Handle empty or missing data case
                if data is None or data.empty:
                    conn.execute('''
                        INSERT OR REPLACE INTO concept_cache
                        (ticker, concept_tag, filing_date, concept_value, taxonomy, units, fetch_status, last_updated)
                        VALUES (?, ?, datetime('now'), NULL, ?, ?, 'N/A', datetime('now'))
                    ''', (ticker, concept.tag, concept.taxonomy, concept.units))
                    conn.commit()
                    self.logger.info(f"Cached 'N/A' for {ticker} {concept.tag}")
                    return
                
                # Prepare data for caching
                cache_df = data.copy()
                if concept.tag not in cache_df.columns:
                    # If the expected tag is missing, treat it as 'N/A'
                    conn.execute('''
                        INSERT OR REPLACE INTO concept_cache
                        (ticker, concept_tag, filing_date, concept_value, taxonomy, units, fetch_status, last_updated)
                        VALUES (?, ?, datetime('now'), NULL, ?, ?, 'N/A', datetime('now'))
                    ''', (ticker, concept.tag, concept.taxonomy, concept.units))
                    conn.commit()
                    self.logger.info(f"Cached 'N/A' for {ticker} {concept.tag} due to missing tag in data")
                    return

                # Rename the concept column to 'concept_value'
                cache_df = cache_df.rename(columns={concept.tag: 'concept_value'})
                cache_df['ticker'] = ticker
                cache_df['concept_tag'] = concept.tag
                cache_df['taxonomy'] = concept.taxonomy
                cache_df['units'] = concept.units
                
                # Determine fetch_status based on concept_value
                if cache_df['concept_value'].iloc[0] == 'N/A':
                    cache_df['fetch_status'] = 'N/A'
                else:
                    cache_df['fetch_status'] = 'success'
                
                cache_df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Delete existing data for the ticker and concept
                conn.execute('''
                    DELETE FROM concept_cache
                    WHERE ticker = ? AND concept_tag = ?
                ''', (ticker, concept.tag))

                # Insert new data
                cache_df.to_sql('concept_cache', conn, if_exists='append',
                               index=False, method='multi')
                
                conn.commit()
                
                status = cache_df['fetch_status'].iloc[0]
                if status == 'success':
                    self.logger.info(f"Successfully cached data for {ticker} {concept.tag}")
                elif status == 'N/A':
                    self.logger.info(f"Cached 'N/A' for {ticker} {concept.tag}")
                    
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error caching data for {ticker} {concept.tag}: {e}")
                raise

    def cache_concept_error(self, ticker: str, concept: SECConcept, error: str):
        """Cache error status for a concept."""
        with sqlite3.connect(str(self.cache_dir / 'granular_cache.db')) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO concept_status 
                (ticker, concept_tag, last_attempt, status, error)
                VALUES (?, ?, CURRENT_TIMESTAMP, 'error', ?)
            ''', (ticker, concept.tag, str(error)))

    def rate_limited_get(self, tag: str, ticker: str, taxonomy: str, units: str) -> pd.DataFrame:
        """
        Wrapper for the finagg.sec.api.company_concept.get method with rate limiting.
        Implements backoff for rate limits and handles 404s by returning 'N/A'.
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

        try:
            response = finagg.sec.api.company_concept.get(
                tag,
                ticker=ticker,
                taxonomy=taxonomy,
                units=units
            )
            return response
        except Exception as e:
            error_str = str(e)
            if "404" in error_str:
                self.logger.info(f"No data found for {tag} and {ticker} (404)")
                return pd.DataFrame({
                    'end': [datetime.now().strftime('%Y-%m-%d')],
                    'value': ['N/A'],
                    'filed': [datetime.now().strftime('%Y-%m-%d')]
                })
    
    def get_sec_data(self, ticker: str) -> pd.DataFrame:
        """Retrieve SEC fundamental data with efficient memory management."""
        CHUNK_SIZE = 10  # Process concepts in batches of 10
            
        def optimize_df(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            return df
    
        # Get all unique dates first
        all_dates = set()
        valid_concepts = []  # Track concepts with valid data
        for concept in self.SEC_CONCEPTS:
            try:
                cached_data = self.get_cached_concept(ticker, concept)
                if cached_data is not None and not cached_data.empty:
                    all_dates.update(cached_data['filing_date'])
                    valid_concepts.append(concept)  # Regardless of N/A since we already checked empty
            except Exception:
                continue
        
        if not all_dates or not valid_concepts:
            self.logger.warning(f"No SEC data found for {ticker}")
            return pd.DataFrame()
        
        # Create base dataframe
        base_df = pd.DataFrame({'filing_date': sorted(list(all_dates))})
        base_df = optimize_df(base_df)
        
        concept_batches = [
            valid_concepts[i:i + CHUNK_SIZE] 
            for i in range(0, len(valid_concepts), CHUNK_SIZE)
        ]
        
        result_df = base_df
        for batch_idx, concept_batch in enumerate(concept_batches, 1):
            self.logger.info(f"Processing concept batch {batch_idx}/{len(concept_batches)}")
            result_df = self.process_concept_batch(concept_batch, result_df)  # Add self.
            gc.collect()
        
        result_df = result_df.sort_values('filing_date', ascending=False)
        result_df = result_df.drop_duplicates(subset='filing_date')
        
        return result_df

    def process_concept_batch(self, concepts: List[SECConcept], base_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Process a batch of concepts for a given ticker.
        
        Args:
            concepts: List of SECConcept objects to process
            base_df: Base DataFrame with filing dates
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame with processed concept data merged
        """
        result_df = base_df.copy()
        
        # Log initial size
        memory_usage = result_df.memory_usage(deep=True).sum()
        self.logger.info(f"Initial batch size: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB ({memory_usage/1024/1024/1024:.3f}GB)")
        
        for concept in concepts:
            try:
                cached_data = self.get_cached_concept(ticker, concept)
                if cached_data is not None and not cached_data.empty and concept.tag in cached_data.columns:
                    self.logger.info(f"Using cached data for {ticker} {concept.tag}")
                    self.logger.info(f"Retrieved shape for {concept.tag}: {cached_data.shape}")
                    
                    # Log incoming cached data size
                    cached_memory = cached_data.memory_usage(deep=True).sum()
                    self.logger.info(f"Cached data size: {cached_data.shape}, Memory: {cached_memory/1024/1024:.2f}MB ({cached_memory/1024/1024/1024:.3f}GB)")
                    
                    cached_data.reset_index(drop=True, inplace=True)
                    result_df = pd.merge(result_df, cached_data, on='filing_date', how='left')
                    
                    # Log after merge
                    memory_usage = result_df.memory_usage(deep=True).sum()
                    self.logger.info(f"Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB ({memory_usage/1024/1024/1024:.3f}GB)")
                    continue
     
                df = self.rate_limited_get(
                    tag=concept.tag,
                    ticker=ticker,
                    taxonomy=concept.taxonomy,
                    units=concept.units
                )
                
                if df is not None and not df.empty and 'value' in df.columns:
                    if df['value'].iloc[0] != 'N/A':
                        df = finagg.sec.api.filter_original_filings(df)
                        df = df.rename(columns={'value': concept.tag, 'end': 'filing_date'})
                        df = df[['filing_date', concept.tag]]
                        
                        # Log incoming API data size
                        api_memory = df.memory_usage(deep=True).sum()
                        self.logger.info(f"API data size: {df.shape}, Memory: {api_memory/1024/1024:.2f}MB ({api_memory/1024/1024/1024:.3f}GB)")
                        
                        result_df = pd.merge(result_df, df, on='filing_date', how='left')
                        self.cache_concept_data(ticker, concept, df)
                        
                        # Log after merge
                        memory_usage = result_df.memory_usage(deep=True).sum()
                        self.logger.info(f"Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB ({memory_usage/1024/1024/1024:.3f}GB)")
                    else:
                        self.logger.info(f"No data available for {concept.tag} and {ticker}")
                        self.cache_concept_data(ticker, concept, df)
                    
            except Exception as e:
                self.logger.debug(f"Failed to retrieve {concept.tag} data for {ticker}: {str(e)}")
                self.cache_concept_error(ticker, concept, str(e))
            
            gc.collect()
        
        # Log final batch size
        memory_usage = result_df.memory_usage(deep=True).sum()
        self.logger.info(f"Final batch size: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB ({memory_usage/1024/1024/1024:.3f}GB)")
        
        return result_df

    def save_data(self, df: pd.DataFrame, ticker: str, metadata: Dict) -> Tuple[str, str]:
        """Save the SEC data and metadata to files with deduplication."""
        # First check if identical data already exists
        existing_files = list(self.output_dir.glob(f"{ticker}_sec_data_*.tsv"))
        
        # Generate hash of current dataframe content
        current_content = df.to_csv(sep='\t', index=False).encode()
        current_hash = hashlib.md5(current_content).hexdigest()
        
        # Check for duplicates
        for existing_file in existing_files:
            with open(existing_file, 'rb') as f:
                existing_hash = hashlib.md5(f.read()).hexdigest()
                if existing_hash == current_hash:
                    # Found identical file, use its timestamp for metadata
                    timestamp = existing_file.stem.split('_')[-1]
                    metadata_file = self.output_dir / f"{ticker}_sec_data_{timestamp}.json"
                    
                    # Update metadata timestamp if needed
                    if metadata_file.exists():
                        return str(existing_file), str(metadata_file)
                    else:
                        # Just create new metadata if missing
                        pd.Series(metadata).to_json(metadata_file)
                        return str(existing_file), str(metadata_file)
        
        # If no duplicate found, create new files
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
        """Process a single ticker to extract and save SEC data with deduplication."""
        metadata = {
            'ticker': ticker,
            'extraction_date': datetime.now().isoformat(),
            'source': 'SEC EDGAR API',
            'concepts_requested': [c.tag for c in self.SEC_CONCEPTS]
        }
        
        try:
            df = self.get_sec_data(ticker)
            
            # Check if all concepts are 'N/A'
            if df.empty or all(df[col].iloc[0] == 'N/A' for col in df.columns if col != 'filing_date'):
                self.logger.warning(f"All concepts returned 'N/A' for {ticker}. Skipping save.")
                return ("N/A", "N/A")
            
            # Sort DataFrame consistently to ensure same content produces same hash
            df = df.sort_values(['filing_date'] + list(df.columns.drop('filing_date')))
            df = df.reset_index(drop=True)
            
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
