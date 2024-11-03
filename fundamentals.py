# fundamentals.py

import os
import finagg
import pandas as pd
from datetime import datetime, timedelta
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

class CacheManager:
    """Manages caching of SEC fundamental data using SQLite."""
    
    MIN_CONCEPT_THRESHOLD = 70  # Minimum concepts required for a ticker to be considered fully processed
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / 'granular_cache.db'
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._init_cache_db()
        self._check_and_create_tables()
        self.resync_ticker_cache()

    def _init_cache_db(self):
        """Initializes the cache database connection."""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.logger.info(f"Connected to cache database at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to cache database: {e}")
            raise

    def _check_and_create_tables(self):
        """Ensures that required tables exist in the cache database."""
        with self.conn:
            try:
                # Create concept_cache table if it doesn't exist
                self.conn.execute('''
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
                
                # Create concept_status table if it doesn't exist
                self.conn.execute('''
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
                self.conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_concept_cache_ticker 
                    ON concept_cache(ticker)
                ''')
                self.conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_concept_cache_last_updated 
                    ON concept_cache(last_updated)
                ''')
                
                self.logger.info("Verified existence of concept_cache and concept_status tables.")
            except Exception as e:
                self.logger.error(f"Error creating tables: {e}")
                raise

    def resync_ticker_cache(self):
        """Rebuild ticker_cache.db by identifying tickers with sufficient cached data."""
        ticker_stats_dict = {}

        # Scan existing cached tickers
        try:
            cursor = self.conn.execute('''
                SELECT ticker, COUNT(DISTINCT concept_tag) as concept_count
                FROM concept_cache 
                WHERE last_updated > datetime('now', '-7 days')
                GROUP BY ticker
                HAVING concept_count >= ?
            ''', (self.MIN_CONCEPT_THRESHOLD,))
            ticker_stats = cursor.fetchall()
            for ticker, concept_count in ticker_stats:
                ticker_stats_dict[ticker] = concept_count
            self.logger.info(f"Resynced ticker cache with {len(ticker_stats_dict)} tickers.")
        except Exception as e:
            self.logger.error(f"Error during resync_ticker_cache: {e}")
            raise

    def get_cached_concept(self, ticker: str, concept: SECConcept) -> Optional[pd.DataFrame]:
        """Retrieve cached data for a specific concept if available."""
        query = '''
            SELECT filing_date, concept_value as value
            FROM concept_cache 
            WHERE ticker = ? AND concept_tag = ?
            AND last_updated > datetime('now', '-7 days')
            ORDER BY filing_date DESC
        '''
        try:
            df = pd.read_sql_query(query, self.conn, params=(ticker, concept.tag))
            if not df.empty:
                # Rename columns to match expected format
                df = df.rename(columns={'value': concept.tag})
                return df
        except Exception as e:
            self.logger.debug(f"Cache miss for {ticker} {concept.tag}: {str(e)}")
        return None

    def cache_concept_data(self, ticker: str, concept: SECConcept, data: pd.DataFrame):
        """Cache the data for a specific concept with explicit missing data handling."""
        try:
            with self._lock:
                # Handle empty or missing data case
                if data is None or data.empty:
                    self.conn.execute('''
                        INSERT OR REPLACE INTO concept_cache
                        (ticker, concept_tag, filing_date, concept_value, taxonomy, units, fetch_status, last_updated)
                        VALUES (?, ?, datetime('now'), NULL, ?, ?, 'N/A', datetime('now'))
                    ''', (ticker, concept.tag, concept.taxonomy, concept.units))
                    self.conn.commit()
                    self.logger.info(f"Cached 'N/A' for {ticker} {concept.tag}")
                    return
                
                # Prepare data for caching
                cache_df = data.copy()
                if concept.tag not in cache_df.columns:
                    # If the expected tag is missing, treat it as 'N/A'
                    self.conn.execute('''
                        INSERT OR REPLACE INTO concept_cache
                        (ticker, concept_tag, filing_date, concept_value, taxonomy, units, fetch_status, last_updated)
                        VALUES (?, ?, datetime('now'), NULL, ?, ?, 'N/A', datetime('now'))
                    ''', (ticker, concept.tag, concept.taxonomy, concept.units))
                    self.conn.commit()
                    self.logger.info(f"Cached 'N/A' for {ticker} {concept.tag} due to missing tag in data")
                    return

                # Rename the concept column to 'concept_value'
                cache_df = cache_df.rename(columns={concept.tag: 'concept_value'})
                cache_df['taxonomy'] = concept.taxonomy
                cache_df['units'] = concept.units
                
                # Determine fetch_status based on concept_value
                if cache_df['concept_value'].iloc[0] == 'N/A':
                    fetch_status = 'N/A'
                else:
                    fetch_status = 'success'
                
                cache_df['fetch_status'] = fetch_status
                cache_df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Insert or replace records
                cache_df.to_sql('concept_cache', self.conn, if_exists='append', index=False, method='multi')
                self.conn.commit()
                
                self.logger.info(f"Successfully cached data for {ticker} {concept.tag}")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error caching data for {ticker} {concept.tag}: {e}")
            raise

    def cache_concept_error(self, ticker: str, concept: SECConcept, error: str):
        """Cache error status for a concept."""
        try:
            with self._lock:
                self.conn.execute('''
                    INSERT OR REPLACE INTO concept_status 
                    (ticker, concept_tag, last_attempt, status, error)
                    VALUES (?, ?, CURRENT_TIMESTAMP, 'error', ?)
                ''', (ticker, concept.tag, str(error)))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error caching concept error for {ticker} {concept.tag}: {e}")
            raise

    def get_processed_tickers_count(self) -> int:
        """Get count of successfully processed tickers within the last 7 days that meet the concept threshold."""
        try:
            cursor = self.conn.execute('''
                SELECT COUNT(DISTINCT ticker) FROM concept_cache
                WHERE last_updated > datetime('now', '-7 days')
                GROUP BY ticker
                HAVING COUNT(DISTINCT concept_tag) >= ?
            ''', (self.MIN_CONCEPT_THRESHOLD,))
            count = len(cursor.fetchall())
            return count
        except Exception as e:
            self.logger.error(f"Error counting processed tickers: {e}")
            return 0

    def close_connection(self):
        """Closes the cache database connection."""
        try:
            self.conn.close()
            self.logger.info("Closed cache database connection.")
        except Exception as e:
            self.logger.error(f"Error closing cache database: {e}")

class SECDataExtractor:
    """Extracts raw SEC fundamental data with granular caching."""
    
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

    # Rate limiting parameters
    _rate_limit_lock = threading.Lock()
    _request_timestamps: List[float] = []
    _MAX_CALLS = 5
    _PERIOD = 1.0  # seconds

    def __init__(self, cache_manager: CacheManager, output_dir: str = "./data"):
        """Initialize the extractor with cache manager and output directory."""
        self.cache_manager = cache_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

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
            time.sleep(0.05)  # Sleep briefly to prevent tight loop

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
            else:
                self.logger.error(f"Error fetching data for {tag} and {ticker}: {str(e)}")
                raise

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

        # Get all unique filing dates from cached data
        all_dates = set()
        valid_concepts = []  # Track concepts with valid data
        for concept in self.SEC_CONCEPTS:
            try:
                cached_data = self.cache_manager.get_cached_concept(ticker, concept)
                if cached_data is not None and not cached_data.empty:
                    all_dates.update(cached_data['filing_date'])
                    valid_concepts.append(concept)
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
            result_df = self.process_concept_batch(concepts=concept_batch, base_df=result_df, ticker=ticker)
            gc.collect()
        
        result_df = result_df.sort_values('filing_date', ascending=False)
        result_df = result_df.drop_duplicates(subset='filing_date')
        
        return result_df

    def process_concept_batch(self, concepts: List[SECConcept], base_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Process a batch of concepts for a given ticker."""
        result_df = base_df.copy()
        
        # Log initial size
        memory_usage = result_df.memory_usage(deep=True).sum()
        self.logger.info(f"Initial batch size: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
        
        for concept in concepts:
            try:
                cached_data = self.cache_manager.get_cached_concept(ticker, concept)
                if cached_data is not None and not cached_data.empty and concept.tag in cached_data.columns:
                    self.logger.info(f"Using cached data for {ticker} {concept.tag}")
                    
                    # Deduplicate cached data by filing date
                    cached_data = cached_data.sort_values('filing_date').drop_duplicates('filing_date', keep='last')
                    
                    # Merge with base dataframe
                    result_df = pd.merge(result_df, cached_data[['filing_date', concept.tag]], on='filing_date', how='left')
                    
                    # Validate no row explosion occurred
                    if len(result_df) > len(base_df) * 1.1:  # Allow 10% tolerance
                        raise ValueError(f"Unexpected row multiplication for {concept.tag} in cached data")
                    
                    # Log after merge
                    memory_usage = result_df.memory_usage(deep=True).sum()
                    self.logger.info(f"Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
                    continue
     
                # Fetch data from API with rate limiting
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
                        
                        # Deduplicate API data by filing date
                        df = df.sort_values('filing_date').drop_duplicates('filing_date', keep='last')
                        
                        # Merge with base dataframe
                        result_df = pd.merge(result_df, df, on='filing_date', how='left')
                        
                        # Validate no row explosion occurred
                        if len(result_df) > len(base_df) * 1.1:  # Allow 10% tolerance
                            raise ValueError(f"Unexpected row multiplication for {concept.tag} in API data")
                        self.cache_manager.cache_concept_data(ticker, concept, df)
                        
                        # Log after merge
                        memory_usage = result_df.memory_usage(deep=True).sum()
                        self.logger.info(f"Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
                    else:
                        self.logger.info(f"No data available for {concept.tag} and {ticker}")
                        self.cache_manager.cache_concept_data(ticker, concept, df)
                    
            except Exception as e:
                self.logger.debug(f"Failed to retrieve {concept.tag} data for {ticker}: {str(e)}")
                self.cache_manager.cache_concept_error(ticker, concept, str(e))
            
            gc.collect()
        
        # Log final batch size
        memory_usage = result_df.memory_usage(deep=True).sum()
        self.logger.info(f"Final batch size: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
        
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

    def __del__(self):
        """Destructor to ensure the cache database connection is closed."""
        self.close_connection()
