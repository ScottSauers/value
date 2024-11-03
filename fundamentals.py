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
                self.logger.info("Initialized granular_cache.db with concept_cache and concept_status tables.")
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error initializing granular_cache.db: {e}")
                raise

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
            time.sleep(0.01)  # Sleep briefly to prevent tight loop

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
            result_df = self.process_concept_batch(concepts=concept_batch, base_df=result_df, ticker=ticker)
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
                    
                    # Deduplicate cached data by most recent filing
                    cached_data = cached_data.sort_values('filing_date').groupby('filing_date').last().reset_index()
                    
                    # Set up indexes for efficient joining
                    if 'filing_date' not in result_df.index.names:
                        result_df = result_df.set_index('filing_date')
                    cached_data = cached_data.set_index('filing_date')
                    
                    # Join using index (more efficient than merge)
                    result_df = result_df.join(cached_data[[concept.tag]], how='left')
                    
                    # Reset index for next iteration
                    result_df = result_df.reset_index()
                    
                    # Validate no row explosion occurred
                    if len(result_df) > len(base_df) * 1.1:  # Allow 10% tolerance
                        raise ValueError(f"Unexpected row multiplication for {concept.tag} in cached data")
                    
                    # Log after merge
                    memory_usage = result_df.memory_usage(deep=True).sum()
                    self.logger.info(f"Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB ({memory_usage/1024/1024/1024:.3f}GB)")
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
                        
                        # Log incoming API data size
                        api_memory = df.memory_usage(deep=True).sum()
                        self.logger.info(f"API data size: {df.shape}, Memory: {api_memory/1024/1024:.2f}MB ({api_memory/1024/1024/1024:.3f}GB)")
                        
                        # Deduplicate API data by most recent filing
                        df = df.sort_values('filing_date').groupby('filing_date').last().reset_index()
                        
                        # Set up indexes for efficient joining
                        if 'filing_date' not in result_df.index.names:
                            result_df = result_df.set_index('filing_date')
                        df = df.set_index('filing_date')
                        
                        # Join using index (more efficient than merge)
                        result_df = result_df.join(df[[concept.tag]], how='left')
                        
                        # Reset index for next iteration 
                        result_df = result_df.reset_index()
                        
                        # Validate no row explosion occurred
                        if len(result_df) > len(base_df) * 1.1:  # Allow 10% tolerance
                            raise ValueError(f"Unexpected row multiplication for {concept.tag} in API data")
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

class CacheManager:
    MIN_CONCEPT_THRESHOLD = 70  # Minimum concepts required for a ticker to be considered fully processed
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / 'ticker_cache.db'
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._init_db()
        self._check_schema()
        self.resync_ticker_cache()


    def _init_db(self):
        """Initialize SQLite database with necessary tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processed_tickers (
                    ticker TEXT PRIMARY KEY,
                    last_processed TIMESTAMP,
                    status TEXT,
                    data_file TEXT,
                    metadata_file TEXT,
                    error TEXT,
                    data_hash TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_stats (
                    batch_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_tickers INTEGER,
                    successful INTEGER,
                    failed INTEGER
                )
            ''')

    def resync_ticker_cache(self):
        """Rebuild ticker_cache.db by identifying tickers with .tsv files modified in the last 7 days."""
        ticker_stats_dict = {}

        # First, get existing processed tickers from ticker_cache.db
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute('''
                SELECT ticker, last_processed, status
                FROM processed_tickers
                WHERE last_processed > datetime('now', '-7 days')
                AND status = 'success'
            ''')
            existing_tickers = {row[0]: row[1] for row in cursor.fetchall()}
            ticker_stats_dict.update(existing_tickers)

        # Then get tickers from granular_cache.db
        granular_cache_path = self.cache_dir / 'granular_cache.db'
        if granular_cache_path.exists():
            with sqlite3.connect(str(granular_cache_path)) as gconn:
                cursor = gconn.execute('''
                    SELECT ticker, COUNT(DISTINCT concept_tag) as concept_count,
                           MAX(last_updated) as last_update
                    FROM concept_cache 
                    WHERE concept_value IS NOT NULL
                    AND last_updated > datetime('now', '-7 days')
                    GROUP BY ticker
                    HAVING COUNT(DISTINCT concept_tag) >= ?
                ''', (self.MIN_CONCEPT_THRESHOLD,))
                ticker_stats = cursor.fetchall()
                for ticker, _, last_update in ticker_stats:
                    if ticker not in ticker_stats_dict:
                        ticker_stats_dict[ticker] = last_update

        # Now scan for recent .tsv files
        data_dir = self.cache_dir.parent.parent  # Adjusted to match the cache directory structure
        recent_files = list(data_dir.glob('**/*sec_data_*.tsv'))
        
        for file_path in recent_files:
            try:
                modification_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if modification_time >= (datetime.now() - timedelta(days=7)):
                    ticker = file_path.stem.split('_')[0]
                    mod_time_str = modification_time.strftime('%Y-%m-%d %H:%M:%S')
                    ticker_stats_dict[ticker] = mod_time_str
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {e}")

        # Write all to ticker_cache.db
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                # Clear existing processed tickers
                conn.execute('DELETE FROM processed_tickers')

                # Insert updated processed tickers
                conn.executemany('''
                    INSERT INTO processed_tickers 
                    (ticker, last_processed, status, data_file, metadata_file, error, data_hash)
                    VALUES (?, ?, 'success', NULL, NULL, NULL, NULL)
                ''', [(t, v) for t, v in ticker_stats_dict.items()])

                conn.commit()
                self.logger.info(f"Resynced ticker cache with {len(ticker_stats_dict)} tickers.")
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error during resync_ticker_cache: {e}")
                raise

    def cleanup_old_entries(self):
        """Remove entries older than 7 days."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                DELETE FROM processed_tickers 
                WHERE last_processed < datetime('now', '-7 days')
            ''')

    def get_cached_result(self, ticker: str) -> Optional[Dict]:
        """Retrieve cached result for a ticker if it exists and is recent, along with concept count."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('''
                    SELECT * FROM processed_tickers 
                    WHERE ticker = ? AND last_processed > datetime("now", "-7 days")
                ''', (ticker,))
                result = cursor.fetchone()

                if result:
                    # Count the number of concepts present for this ticker
                    concept_count_cursor = conn.execute('''
                        SELECT COUNT(DISTINCT concept_tag) 
                        FROM concept_cache 
                        WHERE ticker = ? 
                          AND last_updated > datetime("now", "-7 days")
                    ''', (ticker,))
                    concept_count = concept_count_cursor.fetchone()[0] if concept_count_cursor else 0

                    return {
                        'ticker': result[0],
                        'last_processed': result[1],
                        'status': result[2],
                        'data_file': result[3],
                        'metadata_file': result[4],
                        'error': result[5],
                        'data_hash': result[6],
                        'concept_count': concept_count
                    }

                # If not found, mark as processing
                conn.execute('''
                    INSERT OR REPLACE INTO processed_tickers 
                    (ticker, last_processed, status)
                    VALUES (?, datetime('now'), 'processing')
                ''', (ticker,))
                conn.commit()
                return None

    def get_processed_tickers_count(self) -> int:
        """Get count of successfully processed tickers within the last 7 days that meet the concept threshold."""
        # Get count from processed_tickers in ticker_cache.db
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute('''
                SELECT COUNT(DISTINCT ticker) FROM processed_tickers
                WHERE status = 'success'
                AND last_processed > datetime("now", "-7 days")
            ''')
            ticker_count = cursor.fetchone()[0]

        # Get count from concept_cache in granular_cache.db
        try:
            granular_db_path = self.cache_dir / 'granular_cache.db'
            if granular_db_path.exists():
                with sqlite3.connect(str(granular_db_path)) as conn:
                    # Count tickers with sufficient concepts
                    cursor = conn.execute('''
                        SELECT COUNT(DISTINCT ticker) 
                        FROM (
                            SELECT ticker
                            FROM concept_cache 
                            WHERE last_updated > datetime("now", "-7 days")
                            GROUP BY ticker 
                            HAVING COUNT(DISTINCT concept_tag) >= ?
                        )
                    ''', (self.MIN_CONCEPT_THRESHOLD,))
                    concept_count = cursor.fetchone()[0]
            else:
                concept_count = 0
        except Exception as e:
            self.logger.warning(f"Error checking concept cache: {e}")
            concept_count = 0

        return max(ticker_count, concept_count)


    def _check_schema(self):
        """Check that the database schema is up to date, including the existence of the concept_cache table."""
        granular_db_path = self.cache_dir / 'granular_cache.db'
        if granular_db_path.exists():
            with sqlite3.connect(str(granular_db_path)) as conn:
                # Check if concept_cache table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='concept_cache';
                """)
                table_exists = cursor.fetchone() is not None

                if not table_exists:
                    # Create the concept_cache table without deleting existing data
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
                    conn.execute('''
                        CREATE INDEX IF NOT EXISTS idx_concept_cache_ticker 
                        ON concept_cache(ticker)
                    ''')
                    conn.execute('''
                        CREATE INDEX IF NOT EXISTS idx_concept_cache_last_updated 
                        ON concept_cache(last_updated)
                    ''')
                    conn.commit()
                    self.logger.info("Created missing 'concept_cache' table in granular_cache.db.")
                else:
                    # Existing concept_cache table; check for required columns
                    cursor = conn.execute('''
                        SELECT COUNT(*) FROM pragma_table_info('concept_cache')
                        WHERE name = 'last_updated'
                    ''')
                    has_timestamp = cursor.fetchone()[0] > 0

                    if not has_timestamp:
                        conn.execute('''
                            ALTER TABLE concept_cache
                            ADD COLUMN last_updated TIMESTAMP
                        ''')
                        conn.execute('''
                            UPDATE concept_cache 
                            SET last_updated = datetime('now')
                            WHERE last_updated IS NULL
                        ''')

                    # Add fetch_status if it doesn't exist
                    cursor = conn.execute('''
                        SELECT COUNT(*) FROM pragma_table_info('concept_cache')
                        WHERE name = 'fetch_status'
                    ''')
                    has_fetch_status = cursor.fetchone()[0] > 0

                    if not has_fetch_status:
                        conn.execute('''
                            ALTER TABLE concept_cache
                            ADD COLUMN fetch_status TEXT DEFAULT 'unknown'
                        ''')

                    conn.commit()
        else:
            self.logger.error(f"Granular cache database not found at {granular_db_path}.")

    def get_processed_tickers_in_batch(self, batch_tickers: List[str]) -> int:
        """Get count of successfully processed tickers from a specific batch."""
        placeholders = ','.join(['?' for _ in batch_tickers])
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(f'''
                SELECT COUNT(*) FROM processed_tickers 
                WHERE ticker IN ({placeholders})
                AND status = 'success' 
                AND last_processed > datetime("now", "-7 days")
            ''', batch_tickers)
            return cursor.fetchone()[0]

    def cache_result(self, ticker: str, result: Dict):
        """Cache the processing result for a ticker with transaction safety."""
        with self._lock:  # Use the existing lock
            with sqlite3.connect(str(self.db_path)) as conn:
                try:
                    conn.execute('BEGIN IMMEDIATE')  # Ensure atomic operation
                    conn.execute('''
                        INSERT OR REPLACE INTO processed_tickers 
                        (ticker, last_processed, status, data_file, metadata_file, error, data_hash)
                        VALUES (?, datetime('now'), ?, ?, ?, ?, ?)
                        ''',
                        (
                            ticker,
                            result['status'],
                            result.get('data_file'),
                            result.get('metadata_file'),
                            result.get('error'),
                            result.get('data_hash')
                        )
                    )
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    self.logger.error(f"Error caching result for {ticker}: {e}")
                    raise

    def save_batch_stats(self, batch_id: str, stats: Dict):
        """Save statistics for a processing batch."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                '''
                INSERT OR REPLACE INTO processing_stats 
                (batch_id, start_time, end_time, total_tickers, successful, failed)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    batch_id,
                    stats['start_time'],
                    stats['end_time'],
                    stats['total_tickers'],
                    stats['successful'],
                    stats['failed']
                )
            )

class ProgressTracker:
    """Tracks and displays processing progress using rich console."""
    
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "•",
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            console=self.console
        )
        self.total_task = None
        self.current_batch_task = None

    def init_progress(self, total_tickers: int):
        """Initialize progress bars."""
        self.progress.start()
        self.total_task = self.progress.add_task(
            "[blue]Overall Progress", total=total_tickers
        )
        return self.progress

    def update_progress(self, completed: Optional[int], batch_completed: Optional[int] = None):
        """Update progress bars."""
        if completed is not None and self.total_task:
            self.progress.update(self.total_task, completed=completed)
        if batch_completed is not None and self.current_batch_task:
            self.progress.update(self.current_batch_task, completed=batch_completed)

    def new_batch(self, batch_size: int):
        """Create a new progress bar for the current batch."""
        if self.current_batch_task:
            self.progress.remove_task(self.current_batch_task)
        self.current_batch_task = self.progress.add_task(
            "[green]Current Batch", total=batch_size
        )

    def finish(self):
        """Complete progress tracking."""
        try:
            if self.progress:
                self.progress.stop()
                self.progress = None
        except:
            pass  # Don't hang if progress bar is already stopped
        finally:
            # Aggressively clear all references
            self.total_task = None
            self.current_batch_task = None
            self.console = None

def setup_environment():
    """Set up environment variables and configurations."""
    load_dotenv()
    
    sec_user_agent = os.getenv('SEC_API_USER_AGENT')
    if not sec_user_agent or sec_user_agent == "Your Name your-email@example.com":
        raise EnvironmentError(
            "SEC_API_USER_AGENT is not properly configured. "
            "Please update your .env file with your actual name and email"
        )
    
    finagg.sec.api.company_concept.SEC_USER_AGENT = sec_user_agent

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up rich logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(output_dir / 'parallel_fundamentals.log')
        ]
    )
    return logging.getLogger(__name__)

def calculate_data_hash(data_file: Path) -> str:
    """Calculate SHA-256 hash of data file for cache validation."""
    if not data_file.exists():
        return ""
    
    sha256_hash = hashlib.sha256()
    with open(data_file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

MAX_RETRIES = 3
BASE_RETRY_DELAY = 1

def process_ticker_batch(tickers: List[str], output_dir: Path, cache_manager: CacheManager, progress_tracker: ProgressTracker) -> List[dict]:
    """Process a batch of tickers."""
    results = []
    completed_in_batch = 0
    extractor = SECDataExtractor(str(output_dir))

    for ticker in tickers:
        try:
            # Retrieve cached result
            cached_result = cache_manager.get_cached_result(ticker)
            if cached_result and cached_result['status'] == 'success':
                concept_count = cached_result.get('concept_count', 0)
                if concept_count >= cache_manager.MIN_CONCEPT_THRESHOLD:
                    # Sufficient concepts present, skip processing
                    results.append(cached_result)
                    completed_in_batch += 1
                    progress_tracker.update_progress(completed=None, batch_completed=completed_in_batch)
                    continue
                else:
                    # Insufficient concepts, proceed to fetch missing data
                    cache_manager.logger.info(f"Ticker {ticker} has {concept_count} concepts. Proceeding to fetch missing data.")
                    pass  # Continue to processing below

            # Process ticker with retry only for rate limits
            for attempt in range(MAX_RETRIES):
                try:
                    data_file, metadata_file = extractor.process_ticker(ticker)
                    
                    if data_file == "N/A" and metadata_file == "N/A":
                        result = {
                            'ticker': ticker,
                            'status': 'N/A',
                            'data_file': "N/A",
                            'metadata_file': "N/A"
                        }
                    else:
                        result = {
                            'ticker': ticker,
                            'status': 'success',
                            'data_file': str(data_file),
                            'metadata_file': str(metadata_file)
                        }
                    
                    cache_manager.cache_result(ticker, result)
                    results.append(result)
                    break

                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        if attempt < MAX_RETRIES - 1:
                            delay = BASE_RETRY_DELAY * (2 ** attempt)
                            cache_manager.logger.warning(
                                f"Rate limit hit for {ticker}, attempt {attempt + 1}/{MAX_RETRIES}. "
                                f"Waiting {delay}s"
                            )
                            time.sleep(delay)
                            continue
                    raise

        except Exception as e:
            error_result = {
                'ticker': ticker,
                'status': 'failed', 
                'error': str(e)
            }
            cache_manager.cache_result(ticker, error_result)
            results.append(error_result)
            cache_manager.logger.error(f"Failed to process {ticker}: {str(e)}")

        completed_in_batch += 1
        progress_tracker.update_progress(completed=None, batch_completed=completed_in_batch)

    return results

def parallel_process_tickers(
    input_file: str,
    output_dir: str,
    max_workers: Optional[int] = None,
    ticker_column: str = 'ticker',
    batch_size: int = 50
) -> pd.DataFrame:
    """Process multiple tickers in parallel with enhanced progress tracking and caching."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup environment, logging, and managers
    setup_environment()
    logger = setup_logging(output_path)
    cache_manager = CacheManager(output_path / 'cache')
    cache_manager.cleanup_old_entries()
    progress_tracker = ProgressTracker()
    
    # Generate batch ID
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Read ticker list
    try:
        df = pd.read_csv(input_file)
        tickers = df[ticker_column].unique().tolist()
        logger.info(f"📋 Found {len(tickers)} unique tickers to process")
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        raise
    
    # Determine optimal batch size based on available memory
    available_memory = psutil.virtual_memory().available
    estimated_memory_per_ticker = 1024 * 1024  # 1MB estimate
    optimal_batch_size = min(
        batch_size,
        max(10, int(available_memory / (estimated_memory_per_ticker * 2)))
    )
    logger.info(f"Optimal batch size set to {optimal_batch_size} based on available memory.")
    
    # Split tickers into batches
    ticker_batches = [
        tickers[i:i + optimal_batch_size] 
        for i in range(0, len(tickers), optimal_batch_size)
    ]
    
    # Get count of already processed tickers
    already_processed_count = cache_manager.get_processed_tickers_count()
    
    # Get specific tickers from cache for comparison
    cached_tickers = set()
    with sqlite3.connect(str(cache_manager.db_path)) as conn:
        cursor = conn.execute('''
            SELECT ticker FROM processed_tickers
            WHERE status = 'success'
            AND last_processed > datetime('now', '-7 days')
        ''')
        cached_tickers.update(row[0] for row in cursor.fetchall())
    
    # Compare input vs cache
    input_tickers = set(tickers)
    extra_cached = cached_tickers - input_tickers
    missing_from_cache = input_tickers - cached_tickers
    
    if extra_cached:
        logger.warning(f"Cache contains {len(extra_cached)} tickers not in input file: {sorted(extra_cached)}")
    if missing_from_cache:
        logger.info(f"Input file contains {len(missing_from_cache)} new tickers: {sorted(missing_from_cache)}")

    logger.info(f"📋 Found {len(tickers)} unique tickers to process "
                f"({already_processed_count} already processed, "
                f"{len(tickers) - already_processed_count} remaining)")
    
    # Initialize progress tracking with already processed count
    progress_tracker.init_progress(len(tickers))
    if already_processed_count > 0:
        progress_tracker.update_progress(completed=None, batch_completed=already_processed_count)
        completed_total = already_processed_count
    else:
        completed_total = 0

    # Process batches in parallel with memory management
    start_time = datetime.now()
    results = []
    MAX_CONCURRENT_BATCHES = min(4, max_workers if max_workers else 4)
    active_futures = set()
    
    def save_batch_results(batch_results, batch_num):
        batch_df = pd.DataFrame(batch_results)
        batch_file = output_path / f'batch_results_{batch_id}_part{batch_num}.csv'
        batch_df.to_csv(batch_file, index=False)
        logger.info(f"Batch {batch_num} results saved to {batch_file}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_num, batch in enumerate(ticker_batches, start=1):
            # Wait if too many active futures
            while len(active_futures) >= MAX_CONCURRENT_BATCHES:
                done, active_futures = concurrent.futures.wait(
                    active_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        completed_total += len(batch_results)
                        progress_tracker.update_progress(completed=None, batch_completed=completed_total)
                        
                        # Save and clear results periodically
                        if len(results) > 1000:
                            results = save_batch_results(results, batch_num)
                    except Exception as e:
                        logger.error(f"Batch processing error: {str(e)}")
                
                # Force garbage collection
                gc.collect()
            
            # Monitor memory usage
            memory_percent = psutil.Process().memory_percent()
            if memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
                # Wait for all current batches to complete
                concurrent.futures.wait(active_futures)
                results = save_batch_results(results, batch_num)
                gc.collect()
            
            # Submit new batch
            future = executor.submit(
                process_ticker_batch,
                batch,
                output_path,
                cache_manager,
                progress_tracker
            )
            active_futures.add(future)
        
        # Wait for remaining futures with a timeout
        remaining_futures = list(active_futures)
        while remaining_futures:
            done, remaining_futures = concurrent.futures.wait(
                remaining_futures, 
                timeout=10.0,  # 10 second timeout
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                try:
                    batch_results = future.result(timeout=5.0)  # 5 second timeout
                    results.extend(batch_results)
                    completed_total += len(batch_results)
                    progress_tracker.update_progress(completed=None, batch_completed=completed_total)
                except concurrent.futures.TimeoutError:
                    logger.error("Timeout waiting for batch results")
                    # Cancel the future if possible
                    future.cancel()
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
            
            # Force cleanup
            gc.collect()
    
    # Complete progress tracking
    progress_tracker.finish()
    progress_tracker.progress = None
    gc.collect()
    
    # Combine all results
    final_results_df = pd.DataFrame(results)
    
    # Save summary
    summary_file = output_path / f'processing_summary_{batch_id}.csv'
    final_results_df.to_csv(summary_file, index=False)
    
    # Calculate final statistics
    end_time = datetime.now()
    success_count = len(final_results_df[final_results_df['status'] == 'success'])
    fail_count = len(final_results_df[final_results_df['status'] == 'failed'])
    
    # Save batch statistics
    batch_stats = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_tickers': len(final_results_df),
        'successful': success_count,
        'failed': fail_count
    }
    cache_manager.save_batch_stats(batch_id, batch_stats)
    
    # Print final summary
    logger.info("\n📋 Processing Summary:")
    logger.info(f"Total tickers processed: {len(final_results_df)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed: {fail_count}")
    logger.info(f"⏱️ Total time: {end_time - start_time}")
    logger.info(f"📁 Summary saved to: {summary_file}")
    
    return final_results_df

def main():
    """Main execution function with configuration."""
    # Configuration
    INPUT_FILE = 'small_cap_stocks_latest.csv'
    OUTPUT_DIR = './data/fundamentals'
    MAX_WORKERS = 8
    TICKER_COLUMN = 'ticker'
    BATCH_SIZE = 16
    
    try:
        console = Console()
        console.print("\n[bold blue]🚀 Starting SEC Data Processing[/bold blue]\n")
        
        results = parallel_process_tickers(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            max_workers=MAX_WORKERS,
            ticker_column=TICKER_COLUMN,
            batch_size=BATCH_SIZE
        )
        
        # Calculate success rate
        success_rate = (len(results[results['status'] == 'success']) / len(results)) * 100 if len(results) > 0 else 0
        
        # Display final summary with rich formatting
        console.print("\n[bold green]✨ Processing Complete![/bold green]")
        console.print("\n[bold]Final Statistics:[/bold]")
        console.print(f"📊 Success Rate: [green]{success_rate:.1f}%[/green]")
        console.print(f"📁 Output Directory: [blue]{OUTPUT_DIR}[/blue]")
        
        # Display error summary if there were failures
        failed_results = results[results['status'] == 'failed']
        if not failed_results.empty:
            console.print("\n[bold red]⚠️ Failed Tickers:[/bold red]")
            for _, row in failed_results.iterrows():
                console.print(f"❌ {row['ticker']}: {row['error']}")
                
        console.print("\n[bold green]🎉 Process completed successfully![/bold green]\n")
        
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Process interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Fatal error: {str(e)}[/bold red]")
        console.print_exception()
        sys.exit(1)

if __name__ == "__main__":
    main()
