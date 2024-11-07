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
    _init_lock = threading.Lock()

    def _init_granular_cache(self):
        with SECDataExtractor._init_lock:
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
                    print("✅ Successfully initialized granular_cache.db with required tables.")
                except Exception as e:
                    conn.rollback()
                    self.logger.error(f"Error initializing granular_cache.db: {e}")
                    print(f"❌ Error initializing granular_cache.db: {e}")
                    raise

    # Comprehensive list of SEC concepts to collect
    SEC_CONCEPTS = [
        # Assets
        SECConcept("CashAndCashEquivalentsAtCarryingValue"),
        SECConcept("PropertyPlantAndEquipmentNet"),
        SECConcept("InventoryNet"),
        SECConcept("AccountsReceivableNetCurrent"),
        SECConcept("AvailableForSaleSecurities"),
        SECConcept("OperatingLeaseRightOfUseAsset"),
        SECConcept("IntangibleAssetsNetExcludingGoodwill"),
        SECConcept("PropertyPlantAndEquipmentGross"),
        SECConcept("AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"),

        # Liabilities
        SECConcept("Liabilities"),
        SECConcept("LiabilitiesCurrent"),
        SECConcept("LongTermDebtNoncurrent"),
        SECConcept("OperatingLeaseLiabilityNoncurrent"),
        SECConcept("DeferredTaxLiabilitiesNoncurrent"),
        SECConcept("OtherLiabilitiesNoncurrent"),

        # Equity & Share Data
        SECConcept("CommonStockSharesOutstanding"),
        SECConcept("StockholdersEquity"),

        # Income Statement
        SECConcept("Revenues"),
        SECConcept("CostOfGoodsAndServicesSold"),
        SECConcept("CostOfRevenue"),
        SECConcept("OperatingIncomeLoss"),
        SECConcept("DepreciationDepletionAndAmortization"),

        # Cash Flow & Related
        SECConcept("NetCashProvidedByUsedInOperatingActivities"),
        SECConcept("PaymentsToAcquirePropertyPlantAndEquipment"),
        SECConcept("CapitalExpendituresIncurredButNotYetPaid"),
        SECConcept("PaymentsToAcquireBusinessesNetOfCashAcquired"),

        # Other
        SECConcept("RevenueFromContractWithCustomerExcludingAssessedTax"),
        SECConcept("IncreaseDecreaseInAccountsReceivable"),
        SECConcept("IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets")
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
        print(f"🔍 SEC_API_USER_AGENT set to: {sec_user_agent}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory set to: {self.output_dir.resolve()}")

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'sec_data.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Logger initialized with DEBUG level.")
        print("📝 Logging initialized. All DEBUG level logs will be recorded.")

        self.cache_dir = Path(output_dir) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"🗄️ Cache directory set to: {self.cache_dir.resolve()}")
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
                self.logger.debug(f"Attempting to retrieve cached data for {ticker}, concept {concept.tag}")
                print(f"🔎 Retrieving cached data for {ticker}, concept {concept.tag}")
                df = pd.read_sql_query(query, conn, params=(ticker, concept.tag))
                if not df.empty:
                    # Rename columns to match expected format
                    df = df.rename(columns={'value': concept.tag})
                    self.logger.debug(f"Cached data found for {ticker}, concept {concept.tag}. DataFrame shape: {df.shape}")
                    print(f"✅ Cached data found for {ticker}, concept {concept.tag}.")
                    return df
                else:
                    self.logger.debug(f"No cached data available for {ticker}, concept {concept.tag}.")
                    print(f"⚠️ No cached data available for {ticker}, concept {concept.tag}.")
            except Exception as e:
                self.logger.debug(f"Cache retrieval failed for {ticker} {concept.tag}: {str(e)}")
                print(f"❌ Cache retrieval failed for {ticker}, concept {concept.tag}: {e}")
        return None

    def cache_concept_data(self, ticker: str, concept: SECConcept, data: pd.DataFrame):
        """Cache the data for a specific concept with explicit missing data handling."""
        with sqlite3.connect(str(self.cache_dir / 'granular_cache.db')) as conn:
            try:
                self.logger.debug(f"Attempting to cache data for {ticker}, concept {concept.tag}")
                print(f"💾 Caching data for {ticker}, concept {concept.tag}")
                conn.execute('BEGIN IMMEDIATE')

                # Handle missing or empty data
                if data is None or data.empty:
                    self.logger.warning(f"No data to cache for {ticker}, concept {concept.tag}. Inserting 'N/A'.")
                    print(f"⚠️ No data to cache for {ticker}, concept {concept.tag}. Inserting 'N/A'.")
                    conn.execute('''
                        INSERT OR REPLACE INTO concept_cache
                        (ticker, concept_tag, filing_date, concept_value, taxonomy, units, fetch_status, last_updated)
                        VALUES (?, ?, datetime('now'), NULL, ?, ?, 'N/A', datetime('now'))
                    ''', (ticker, concept.tag, concept.taxonomy, concept.units))
                    conn.commit()
                    self.logger.info(f"Cached 'N/A' for {ticker} {concept.tag}")
                    print(f"✅ Cached 'N/A' for {ticker}, concept {concept.tag}")
                    return

                # Check if the expected concept tag is in the data
                if concept.tag not in data.columns:
                    self.logger.warning(f"Data for {ticker}, concept {concept.tag} does not contain the expected tag. Inserting 'N/A'.")
                    print(f"⚠️ Data for {ticker}, concept {concept.tag} does not contain the expected tag. Inserting 'N/A'.")
                    conn.execute('''
                        INSERT OR REPLACE INTO concept_cache
                        (ticker, concept_tag, filing_date, concept_value, taxonomy, units, fetch_status, last_updated)
                        VALUES (?, ?, datetime('now'), NULL, ?, ?, 'N/A', datetime('now'))
                    ''', (ticker, concept.tag, concept.taxonomy, concept.units))
                    conn.commit()
                    self.logger.info(f"Cached 'N/A' for {ticker} {concept.tag} due to missing tag in data")
                    print(f"✅ Cached 'N/A' for {ticker}, concept {concept.tag} due to missing tag in data")
                    return

                # Rename and prepare your dataframe
                cache_df = data.copy()
                cache_df = cache_df.rename(columns={concept.tag: 'concept_value'})
                cache_df['ticker'] = ticker
                cache_df['concept_tag'] = concept.tag
                cache_df['taxonomy'] = concept.taxonomy
                cache_df['units'] = concept.units
                cache_df['fetch_status'] = 'N/A' if cache_df['concept_value'].iloc[0] == 'N/A' else 'success'
                cache_df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Delete existing data for this ticker/concept combination
                conn.execute('''
                    DELETE FROM concept_cache
                    WHERE ticker = ? AND concept_tag = ?
                ''', (ticker, concept.tag))

                # Insert new data using explicit INSERT
                for _, row in cache_df.iterrows():
                    conn.execute('''
                        INSERT INTO concept_cache
                        (ticker, concept_tag, filing_date, concept_value, taxonomy, units, fetch_status, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['ticker'],
                        row['concept_tag'],
                        row['filing_date'],
                        row['concept_value'],
                        row['taxonomy'],
                        row['units'],
                        row['fetch_status'],
                        row['last_updated']
                    ))

                conn.commit()

                status = cache_df['fetch_status'].iloc[0]
                if status == 'success':
                    self.logger.info(f"Successfully cached data for {ticker} {concept.tag}")
                    print(f"✅ Successfully cached data for {ticker}, concept {concept.tag}")
                elif status == 'N/A':
                    self.logger.info(f"Cached 'N/A' for {ticker} {concept.tag}")
                    print(f"✅ Cached 'N/A' for {ticker}, concept {concept.tag}")

            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error caching data for {ticker} {concept.tag}: {e}")
                print(f"❌ Error caching data for {ticker}, concept {concept.tag}: {e}")
                raise

    def cache_concept_error(self, ticker: str, concept: SECConcept, error: str):
        """Cache error status for a concept."""
        with sqlite3.connect(str(self.cache_dir / 'granular_cache.db')) as conn:
            self.logger.debug(f"Caching error for {ticker}, concept {concept.tag}: {error}")
            print(f"🛑 Caching error for {ticker}, concept {concept.tag}: {error}")
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
                else:
                    # Calculate time to wait
                    earliest_request = min(SECDataExtractor._request_timestamps)
                    wait_time = SECDataExtractor._PERIOD - (current_time - earliest_request)
                    self.logger.debug(f"Rate limit reached. Sleeping for {wait_time:.2f} seconds.")
                    print(f"⏳ Rate limit reached. Sleeping for {wait_time:.2f} seconds.")
                    time.sleep(wait_time)

        try:
            self.logger.debug(f"Attempting API call for {ticker}, concept {tag}")
            print(f"📡 Making API call for {ticker}, concept {tag}")
            response = finagg.sec.api.company_concept.get(
                tag,
                ticker=ticker,
                taxonomy=taxonomy,
                units=units
            )
            if response is None or response.empty:
                self.logger.warning(f"Empty response received for {ticker}, concept {tag}")
                print(f"⚠️ Empty response received for {ticker}, concept {tag}")
            else:
                self.logger.debug(f"Received response for {ticker}, concept {tag}: {response.shape}")
                print(f"✅ Received response for {ticker}, concept {tag}: {response.shape}")
            return response
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"API call failed for {ticker}, concept {tag}: {error_str}", exc_info=True)
            print(f"❌ API call failed for {ticker}, concept {tag}: {error_str}")
            return pd.DataFrame({
                'end': [datetime.now().strftime('%Y-%m-%d')],
                'value': ['N/A'],
                'filed': [datetime.now().strftime('%Y-%m-%d')]
            })

    def get_sec_data(self, ticker: str) -> pd.DataFrame:
        """Retrieve SEC fundamental data with efficient memory management."""
        CHUNK_SIZE = 10  # Process concepts in batches of 10
        print(f"📈 Starting SEC data retrieval for ticker: {ticker}")
        self.logger.info(f"Starting SEC data retrieval for ticker: {ticker}")

        def optimize_df(df: pd.DataFrame) -> pd.DataFrame:
            """Optimize DataFrame memory usage by downcasting numeric types."""
            self.logger.debug("Optimizing DataFrame memory usage.")
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            self.logger.debug(f"DataFrame optimized. New memory usage: {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
            print(f"🔧 DataFrame optimized. New memory usage: {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
            return df

        # Collect all concepts to process
        concepts_to_process = self.SEC_CONCEPTS.copy()
        all_dates = set()

        # Try to get cached data first to collect dates
        for concept in concepts_to_process:
            try:
                cached_data = self.get_cached_concept(ticker, concept)
                if cached_data is not None and not cached_data.empty:
                    all_dates.update(cached_data['filing_date'])
            except Exception as e:
                self.logger.error(f"Error accessing cached data for {ticker}, concept {concept.tag}: {e}")

        # Initialize base DataFrame
        if not all_dates:
            # If no dates from cache, we can proceed without base_df
            result_df = pd.DataFrame()
        else:
            # Create base dataframe
            result_df = pd.DataFrame({'filing_date': sorted(all_dates)})
            result_df = optimize_df(result_df)
            self.logger.debug(f"Base DataFrame created with {result_df.shape[0]} filing dates.")
            print(f"📊 Base DataFrame created with {result_df.shape[0]} filing dates.")

        # Process concepts in batches
        concept_batches = [
            concepts_to_process[i:i + CHUNK_SIZE]
            for i in range(0, len(concepts_to_process), CHUNK_SIZE)
        ]
        self.logger.debug(f"Divided {len(concepts_to_process)} concepts into {len(concept_batches)} batches.")
        print(f"📚 Divided {len(concepts_to_process)} concepts into {len(concept_batches)} batches.")

        for batch_idx, concept_batch in enumerate(concept_batches, 1):
            self.logger.info(f"Processing concept batch {batch_idx}/{len(concept_batches)}")
            print(f"🔄 Processing concept batch {batch_idx}/{len(concept_batches)}")
            result_df = self.process_concept_batch(concepts=concept_batch, base_df=result_df, ticker=ticker)
            gc.collect()
            self.logger.debug(f"Garbage collection performed after batch {batch_idx}.")
            print(f"🗑️ Garbage collection performed after batch {batch_idx}.")

        if 'filing_date' in result_df.columns:
            result_df = result_df.sort_values('filing_date', ascending=False)
            result_df = result_df.drop_duplicates(subset='filing_date')
            result_df = result_df.reset_index(drop=True)
        else:
            self.logger.warning(f"No filing_date column in result_df for {ticker}")
            print(f"⚠️ No filing_date column in result_df for {ticker}")

        self.logger.debug(f"Final DataFrame sorted and deduplicated. Shape: {result_df.shape}")
        print(f"📉 Final DataFrame sorted and deduplicated. Shape: {result_df.shape}")

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
        batch_size = len(concepts)
        self.logger.info(f"Starting processing of {batch_size} concepts for {ticker}.")
        print(f"🛠️ Starting processing of {batch_size} concepts for {ticker}.")
        
        # Log initial size
        memory_usage = result_df.memory_usage(deep=True).sum()
        self.logger.info(f"Initial batch size: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB ({memory_usage/1024/1024/1024:.3f}GB)")
        print(f"📐 Initial batch size: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
        
        for concept in concepts:
            self.logger.debug(f"Processing concept: {concept.tag}")
            print(f"📊 Processing concept: {concept.tag}")
            try:
                cached_data = self.get_cached_concept(ticker, concept)
                if cached_data is not None and not cached_data.empty and concept.tag in cached_data.columns:
                    self.logger.info(f"Using cached data for {ticker}, concept {concept.tag}. Shape: {cached_data.shape}")
                    print(f"✅ Using cached data for {ticker}, concept {concept.tag}. Shape: {cached_data.shape}")
                    
                    # Log incoming cached data size
                    cached_memory = cached_data.memory_usage(deep=True).sum()
                    self.logger.debug(f"Cached data size: {cached_memory/1024/1024:.2f} MB")
                    print(f"💾 Cached data size: {cached_memory/1024/1024:.2f} MB")
                    
                    # Deduplicate cached data by filing_date
                    cached_data = cached_data.sort_values('filing_date').drop_duplicates(subset='filing_date', keep='last')
                    
                    # Set up indexes for efficient joining
                    if 'filing_date' not in result_df.index.names:
                        result_df = result_df.set_index('filing_date')
                        self.logger.debug("Set 'filing_date' as index for result_df.")
                        print("🔗 Set 'filing_date' as index for result_df.")
                    cached_data = cached_data.set_index('filing_date')
                    
                    # Join using index (more efficient than merge)
                    result_df = result_df.join(cached_data[[concept.tag]], how='left')
                    self.logger.debug(f"Joined cached data for {concept.tag}. Resulting DataFrame shape: {result_df.shape}")
                    print(f"🔗 Joined cached data for {concept.tag}. New shape: {result_df.shape}")
                    
                    # Reset index for next iteration
                    result_df = result_df.reset_index()
                    
                    # Validate no row explosion occurred
                    if len(result_df) > len(base_df) * 1.1:  # Allow 10% tolerance
                        error_msg = f"Unexpected row multiplication for {concept.tag} in cached data"
                        self.logger.error(error_msg)
                        print(f"❌ {error_msg}")
                        raise ValueError(error_msg)
                    
                    # Log after merge
                    memory_usage = result_df.memory_usage(deep=True).sum()
                    self.logger.debug(f"Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
                    print(f"📊 Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
                    continue

                # Fetch data from API
                self.logger.debug(f"Fetching data from API for {ticker}, concept {concept.tag}")
                print(f"📡 Fetching data from API for {ticker}, concept {concept.tag}")
                df = self.rate_limited_get(
                    tag=concept.tag,
                    ticker=ticker,
                    taxonomy=concept.taxonomy,
                    units=concept.units
                )
                
                if df is not None and not df.empty and 'value' in df.columns:
                    if df['value'].iloc[0] != 'N/A':
                        self.logger.debug(f"Data fetched successfully for {ticker}, concept {concept.tag}")
                        print(f"✅ Data fetched successfully for {ticker}, concept {concept.tag}")
                        df = finagg.sec.api.filter_original_filings(df)
                        self.logger.debug(f"Filtered original filings for {concept.tag}. Shape: {df.shape}")
                        print(f"🔍 Filtered original filings for {concept.tag}. Shape: {df.shape}")
                        df = df.rename(columns={'value': concept.tag, 'end': 'filing_date'})
                        df = df[['filing_date', concept.tag]]
                        
                        # Log incoming API data size
                        api_memory = df.memory_usage(deep=True).sum()
                        self.logger.debug(f"API data size: {df.shape}, Memory: {api_memory/1024/1024:.2f}MB")
                        print(f"💾 API data size: {df.shape}, Memory: {api_memory/1024/1024:.2f}MB")
                        
                        # Deduplicate API data by filing_date
                        df = df.sort_values('filing_date').drop_duplicates(subset='filing_date', keep='last')
                        self.logger.debug(f"Deduplicated API data for {concept.tag}. Shape: {df.shape}")
                        print(f"🧹 Deduplicated API data for {concept.tag}. Shape: {df.shape}")
                        
                        # Set up indexes for efficient joining
                        if 'filing_date' not in result_df.index.names:
                            result_df = result_df.set_index('filing_date')
                            self.logger.debug("Set 'filing_date' as index for result_df.")
                            print("🔗 Set 'filing_date' as index for result_df.")
                        df = df.set_index('filing_date')
                        
                        # Join using index (more efficient than merge)
                        result_df = result_df.join(df[[concept.tag]], how='left')
                        self.logger.debug(f"Joined API data for {concept.tag}. Resulting DataFrame shape: {result_df.shape}")
                        print(f"🔗 Joined API data for {concept.tag}. New shape: {result_df.shape}")
                        
                        # Reset index for next iteration 
                        result_df = result_df.reset_index()
                        
                        # Validate no row explosion occurred
                        if len(result_df) > len(base_df) * 1.1:  # Allow 10% tolerance
                            error_msg = f"Unexpected row multiplication for {concept.tag} in API data"
                            self.logger.error(error_msg)
                            print(f"❌ {error_msg}")
                            raise ValueError(error_msg)
                        
                        # Cache the retrieved data
                        self.cache_concept_data(ticker, concept, df)
                        
                        # Log after merge
                        memory_usage = result_df.memory_usage(deep=True).sum()
                        self.logger.debug(f"Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
                        print(f"📊 Size after merging {concept.tag}: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
                    else:
                        self.logger.warning(f"No data available for {concept.tag} and {ticker}. Inserting 'N/A'.")
                        print(f"⚠️ No data available for {concept.tag} and {ticker}. Inserting 'N/A'.")
                        self.cache_concept_data(ticker, concept, df)
                        print(f"✅ Inserted 'N/A' for {concept.tag} and {ticker}")
                
            except Exception as e:
                self.logger.error(f"Failed to retrieve {concept.tag} data for {ticker}: {e}")
                print(f"❌ Failed to retrieve {concept.tag} data for {ticker}: {e}")
                self.cache_concept_error(ticker, concept, str(e))
            
            # Garbage collect to free memory
            gc.collect()
            self.logger.debug(f"Garbage collection performed after processing {concept.tag} for {ticker}.")
            print(f"🗑️ Garbage collection performed after processing {concept.tag} for {ticker}.")
        
        # Log final batch size
        memory_usage = result_df.memory_usage(deep=True).sum()
        self.logger.info(f"Final batch size: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
        print(f"📊 Final batch size: {result_df.shape}, Memory: {memory_usage/1024/1024:.2f}MB")
        
        return result_df

    def save_data(self, df: pd.DataFrame, ticker: str, metadata: Dict) -> Tuple[str, str]:
        """Save the SEC data and metadata to files with deduplication."""
        print(f"💾 Saving data for {ticker}")
        self.logger.info(f"Saving data for {ticker}")
        
        # First check if identical data already exists
        existing_files = list(self.output_dir.glob(f"{ticker}_sec_data_*.tsv"))
        self.logger.debug(f"Existing files for {ticker}: {existing_files}")
        print(f"📂 Checking for existing files for {ticker}: {existing_files}")
        
        # Generate hash of current dataframe content
        current_content = df.to_csv(sep='\t', index=False).encode()
        current_hash = hashlib.md5(current_content).hexdigest()
        self.logger.debug(f"Generated MD5 hash for current data: {current_hash}")
        print(f"🔑 Generated MD5 hash for current data: {current_hash}")
        
        # Check for duplicates
        for existing_file in existing_files:
            with open(existing_file, 'rb') as f:
                existing_hash = hashlib.md5(f.read()).hexdigest()
                self.logger.debug(f"Comparing with existing file {existing_file}: {existing_hash}")
                print(f"🔄 Comparing with existing file {existing_file.name}: {existing_hash}")
                if existing_hash == current_hash:
                    # Found identical file, use its timestamp for metadata
                    timestamp = existing_file.stem.split('_')[-1]
                    metadata_file = self.output_dir / f"{ticker}_sec_data_{timestamp}.json"
                    
                    # Update metadata timestamp if needed
                    if metadata_file.exists():
                        self.logger.info(f"Duplicate data found for {ticker}. Using existing files.")
                        print(f"✅ Duplicate data found for {ticker}. Using existing files.")
                        return str(existing_file), str(metadata_file)
                    else:
                        # Just create new metadata if missing
                        pd.Series(metadata).to_json(metadata_file)
                        self.logger.info(f"Metadata file created for {ticker}: {metadata_file}")
                        print(f"📝 Metadata file created for {ticker}: {metadata_file.name}")
                        return str(existing_file), str(metadata_file)
        
        # If no duplicate found, create new files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_filename = f"{ticker}_sec_data_{timestamp}.tsv"
        data_filepath = self.output_dir / data_filename
        df.to_csv(data_filepath, sep='\t', index=False)
        self.logger.info(f"Data saved to {data_filepath}")
        print(f"✅ Data saved to {data_filename}")
        
        metadata_filename = f"{ticker}_sec_data_{timestamp}.json"
        metadata_filepath = self.output_dir / metadata_filename
        pd.Series(metadata).to_json(metadata_filepath)
        self.logger.info(f"Metadata saved to {metadata_filepath}")
        print(f"📝 Metadata saved to {metadata_filename}")
        
        return str(data_filepath), str(metadata_filepath)

    def process_ticker(self, ticker: str) -> Tuple[str, str]:
        """Process a single ticker to extract and save SEC data with deduplication."""
        metadata = {
            'ticker': ticker,
            'extraction_date': datetime.now().isoformat(),
            'source': 'SEC EDGAR API',
            'concepts_requested': [c.tag for c in self.SEC_CONCEPTS]
        }
        
        self.logger.info(f"Starting processing for ticker: {ticker}")
        print(f"🚀 Starting processing for ticker: {ticker}")
        
        try:
            df = self.get_sec_data(ticker)
            
            # Check if all concepts are 'N/A'
            if df.empty:
                self.logger.warning(f"No data retrieved for ticker {ticker}. Skipping save.")
                print(f"⚠️ No data retrieved for ticker {ticker}. Skipping save.")
                return ("N/A", "N/A")
            
            all_n_a = True
            for col in df.columns:
                if col != 'filing_date' and df[col].iloc[0] != 'N/A':
                    all_n_a = False
                    break
            
            if all_n_a:
                self.logger.warning(f"All concepts returned 'N/A' for {ticker}. Skipping save.")
                print(f"⚠️ All concepts returned 'N/A' for ticker {ticker}. Skipping save.")
                return ("N/A", "N/A")
            
            # Sort DataFrame consistently to have same content produce same hash
            df = df.sort_values(['filing_date'] + [col for col in df.columns if col != 'filing_date']).reset_index(drop=True)
            self.logger.debug(f"DataFrame sorted and reset index for {ticker}.")
            print(f"📑 DataFrame sorted and index reset for {ticker}.")
            
            data_file, metadata_file = self.save_data(df, ticker, metadata)
            self.logger.info(f"Finished processing ticker {ticker}. Data saved to {data_file}, metadata saved to {metadata_file}.")
            print(f"✅ Successfully processed {ticker}.")
            print(f"📂 Data file: {Path(data_file).name}")
            print(f"📝 Metadata file: {Path(metadata_file).name}")
            return (data_file, metadata_file)
            
        except Exception as e:
            self.logger.error(f"Failed to process {ticker}: {e}", exc_info=True)
            print(f"❌ Failed to process ticker {ticker}: {e}")
            raise

def main():
    """Main execution function."""
    extractor = SECDataExtractor()

    # List of tickers to process. Example.
    tickers = ["AAPL", "NVDA", "MSFT", "ME"]

    for ticker in tickers:
        try:
            extractor.logger.info(f"Initiating processing for ticker: {ticker}")
            print(f"🔄 Initiating processing for ticker: {ticker}")
            data_file, metadata_file = extractor.process_ticker(ticker)
        except Exception as e:
            extractor.logger.error(f"Exception occurred while processing ticker {ticker}: {e}", exc_info=True)
            print(f"❌ Exception occurred while processing ticker {ticker}: {e}")
            continue  # Continue with the next ticker

if __name__ == "__main__":
    main()
