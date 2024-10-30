import finagg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import time
import json
import yfinance as yf
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import os
import pickle
from collections import defaultdict

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

class MarketCapScreener:
    """Enhanced screener with progressive caching and rate limiting."""
    
    CACHE_DIR = Path("cache")
    BATCH_SIZE = 2  # Smaller batches for better reliability
    
    def __init__(self, max_workers: int = 20, cache_expiry_days: int = 1, use_cache: bool = True):
        self.use_cache = use_cache
        print("\nInitializing MarketCapScreener...")
        self.max_workers = max_workers
        self.cache_expiry_days = cache_expiry_days
        self.stats = {
            'processed': 0,
            'success': 0,
            'errors': defaultdict(int),
            'last_save': time.time()
        }
        print(f"Max workers set to: {self.max_workers}")
        print(f"Cache expiry set to: {self.cache_expiry_days} days")
        
        # Create cache directory
        self.CACHE_DIR.mkdir(exist_ok=True)
        print(f"Cache directory ensured at: {self.CACHE_DIR.resolve()}")
        
        print("Setting up CachedLimiterSession...")
        self.session = CachedLimiterSession(
            limiter=Limiter(RequestRate(2, Duration.SECOND)),
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache("yfinance.cache")
        )
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        print("CachedLimiterSession initialized.")
        
        # Setup logging
        print("Configuring logging...")
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('screener.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Logging configured with DEBUG level.")
        
        # Track processed tickers for resume capability
        self.processed_tickers = set()
        self.current_results = []
        print("Initialized processed tickers set and current results list.")
        
        self.BATCH_SIZE = 50  # Process more tickers at once
        print(f"Batch size set to: {self.BATCH_SIZE}")

    def _get_cache_path(self, cache_type: str) -> Path:
        """Get cache file path without date stamping."""
        cache_path = self.CACHE_DIR / f"{cache_type}.pkl"
        print(f"Cache path for '{cache_type}': {cache_path.resolve()}")
        return cache_path
    
    def _get_single_ticker_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get data for a single ticker with proper rate limiting."""
        try:
            # First get basic price data
            price_data = finagg.yfinance.api.get(ticker, period="5d")
            if price_data.empty:
                print(f"Price data for {ticker} is empty. Skipping.")
                return None
            latest_price = price_data['close'].iloc[-1]
            print(f"Latest price for {ticker}: {latest_price}")
    
            try:
                # Create ticker object and get fast_info first
                ticker_obj = yf.Ticker(ticker)
                fast_info = ticker_obj.fast_info
                
                # Get market cap from different methods
                market_cap = None
                shares_outstanding = None
                
                # Method 1: Try fast_info market cap
                if hasattr(fast_info, 'market_cap'):
                    market_cap = fast_info.market_cap 
                    print(f"{ticker} Market Cap from fast_info: {market_cap}")
                
                # Method 2: Try shares * price if available
                if market_cap is None and hasattr(fast_info, 'shares_outstanding'):
                    shares_outstanding = fast_info.shares_outstanding
                    market_cap = shares_outstanding * latest_price
                    print(f"{ticker} Market Cap calculated: {market_cap}")
                
                # Method 3: Try to get from regular info as last resort
                if market_cap is None:
                    try:
                        time.sleep(0.1)  # Rate limit
                        info = ticker_obj.get_info()  # Use get_info() instead of .info
                        if 'marketCap' in info:
                            market_cap = info['marketCap']
                            print(f"{ticker} Market Cap from info: {market_cap}")
                        elif 'sharesOutstanding' in info:
                            shares_outstanding = info['sharesOutstanding']
                            market_cap = shares_outstanding * latest_price
                            print(f"{ticker} Market Cap from shares: {market_cap}")
                    except Exception as e:
                        print(f"Could not get detailed info for {ticker}: {str(e)}")
    
                # Final check and data return
                if market_cap is None or market_cap <= 0:
                    print(f"No valid market cap data for {ticker}")
                    return None
    
                return {
                    'ticker': ticker,
                    'price': latest_price,
                    'shares_outstanding': shares_outstanding or (market_cap / latest_price if latest_price > 0 else None),
                    'market_cap': market_cap,
                    'high': price_data['high'].iloc[-1],
                    'low': price_data['low'].iloc[-1],
                    'open': price_data['open'].iloc[-1],
                    'volume': price_data['volume'].iloc[-1]
                }
    
            except Exception as e:
                print(f"Error getting market cap for {ticker}: {str(e)}")
                return None
    
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            return None

    def _get_resume_path(self) -> Path:
        """Get path for resume data."""
        resume_path = self.CACHE_DIR / "resume_data.pkl"
        print(f"Resume path: {resume_path.resolve()}")
        return resume_path

    def _load_cache(self, cache_type: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        print(f"\nAttempting to load cache for '{cache_type}'...")
        cache_path = self._get_cache_path(cache_type)
        if cache_path.exists():
            print(f"Cache file exists for '{cache_type}'. Checking validity...")
            try:
                cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
                print(f"Cache last modified at: {cache_time}")
                if datetime.now() - cache_time < timedelta(days=self.cache_expiry_days):
                    print(f"Cache is valid (within {self.cache_expiry_days} days). Loading cache...")
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"Cache loaded successfully for '{cache_type}'.")
                    return data
                else:
                    print(f"Cache expired for '{cache_type}'.")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {str(e)}")
                print(f"Exception occurred while loading cache for '{cache_type}': {str(e)}")
        else:
            print(f"No cache file found for '{cache_type}'.")
        return None

    def _save_cache(self, data: pd.DataFrame, cache_type: str):
        """Save data to cache with error handling."""
        print(f"\nSaving cache for '{cache_type}' with {len(data)} records...")
        try:
            cache_path = self._get_cache_path(cache_type)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.stats['last_save'] = time.time()
            print(f"Cache saved successfully for '{cache_type}' at {cache_path.resolve()}.")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {str(e)}")
            print(f"Exception occurred while saving cache for '{cache_type}': {str(e)}")

    def _save_resume_state(self):
        """Save current state for resume capability."""
        print("\nSaving resume state...")
        try:
            resume_data = {
                'processed_tickers': self.processed_tickers,
                'current_results': self.current_results,
                'stats': self.stats
            }
            with open(self._get_resume_path(), 'wb') as f:
                pickle.dump(resume_data, f)
            print("Resume state saved successfully.")
        except Exception as e:
            self.logger.warning(f"Failed to save resume state: {str(e)}")
            print(f"Exception occurred while saving resume state: {str(e)}")

    def _load_resume_state(self) -> bool:
        """Load previous state if exists."""
        print("\nAttempting to load resume state...")
        try:
            resume_path = self._get_resume_path()
            if resume_path.exists():
                print("Resume file found. Loading resume state...")
                with open(resume_path, 'rb') as f:
                    resume_data = pickle.load(f)
                self.processed_tickers = resume_data['processed_tickers']
                self.current_results = resume_data['current_results']
                self.stats = resume_data['stats']
                print(f"Resume state loaded. Processed tickers count: {len(self.processed_tickers)}")
                return True
            else:
                print("No resume file found.")
        except Exception as e:
            self.logger.warning(f"Failed to load resume state: {str(e)}")
            print(f"Exception occurred while loading resume state: {str(e)}")
        return False

    def get_exchange_listed_companies(self) -> pd.DataFrame:
        """Get companies from major exchanges with caching."""
        print("\nFetching exchange-listed companies...")
        cached_data = self._load_cache('exchanges')
        if cached_data is not None:
            print(f"Cache loaded for exchanges with {len(cached_data)} entries.")
            return cached_data
            
        try:
            print("Fetching exchange data using finagg SEC API...")
            df = finagg.sec.api.exchanges.get()
            print(f"Exchange data fetched. Total entries: {len(df)}")
            
            # Filter for major exchanges
            major_exchanges = ['NYSE', 'Nasdaq', 'NYSE Arca', 'NYSE American']
            print(f"Filtering for major exchanges: {major_exchanges}")
            df = df[df['exchange'].isin(major_exchanges)]
            print(f"Entries after filtering major exchanges: {len(df)}")
            
            df = df[df['name'].notna()]
            print(f"Entries after ensuring 'name' is not NA: {len(df)}")
            
            # Exclude tickers with special characters but keep the regex pattern valid
            print("Excluding tickers with special characters...")
            df = df[~df['ticker'].str.contains(r'[\^$]')]  # Fixed regex pattern
            print(f"Entries after excluding special characters: {len(df)}")
            
            df = df[df['ticker'].str.len() <= 5]
            print(f"Entries after limiting ticker length to <=5: {len(df)}")
            
            self._save_cache(df, 'exchanges')
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get exchange data: {str(e)}")
            print(f"Exception occurred while fetching exchange data: {str(e)}")
            raise

    def get_stock_data_batch(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get stock data for multiple tickers with true parallelization."""
        print(f"\nStarting batch data retrieval for {len(tickers)} tickers...")
        results = {}
        futures = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            print(f"Submitting {len(tickers)} tickers to ThreadPoolExecutor...")
            # Submit each ticker for parallel processing
            for ticker in tickers:
                futures.append((ticker, executor.submit(self._get_single_ticker_data, ticker)))
                print(f"Submitted ticker: {ticker}")
                
            # Collect results as they complete
            for ticker, future in futures:
                try:
                    #print(f"Waiting for result of ticker: {ticker}")
                    data = future.result()
                    if data:
                        results[ticker] = data
                        print(f"Data received for ticker: {ticker}")
                    else:
                        print(f"No data returned for ticker: {ticker}")
                except Exception as e:
                    self.logger.debug(f"Error processing {ticker}: {str(e)}")
                    print(f"Exception occurred while processing ticker {ticker}: {str(e)}")
                    self.stats['errors']['processing'] += 1
        print(f"Batch data retrieval complete. {len(results)} tickers processed successfully.")
        return results

    def screen_small_caps(self, max_market_cap: float = 50_000_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Screen for small cap stocks with progressive caching."""
        print("\nStarting small cap screening process...")
        # Try loading existing results first
        cached_data = self._load_cache('market_caps')
        print(f"\nAttempting to load cache for 'market_caps'...")
        if cached_data is not None:
            print(f"Cache found with {len(cached_data)} entries.")
            if not cached_data.empty:
                print("Using cached market cap data...")
                all_stocks_df = self.validate_market_cap(cached_data)
                print(f"After validation: {len(all_stocks_df)} entries.")
                small_caps = all_stocks_df[all_stocks_df['market_cap'] < max_market_cap].copy()
                print(f"Found {len(small_caps)} companies under ${max_market_cap:,} market cap.")
                if not small_caps.empty:
                    print("Returning cached small cap data.")
                    return small_caps, all_stocks_df
                else:
                    print("No small cap companies found in cache.")
            else:
                print("Cached data was empty.")
        else:
            print("No valid cache found for 'market_caps'.")
        
        # Check for resume data
        if self._load_resume_state():
            print("\nResuming previous run...")
        else:
            print("\nStarting fresh run. Clearing processed tickers and current results.")
            self.processed_tickers.clear()
            self.current_results.clear()
        
        # Get exchange-listed companies
        print("\nFetching exchange-listed companies...")
        exchange_df = self.get_exchange_listed_companies()
        
        # Filter out already processed tickers
        remaining_tickers = [t for t in exchange_df['ticker'] if t not in self.processed_tickers]
        total_remaining = len(remaining_tickers)
        print(f"Total remaining tickers to process: {total_remaining}")
        
        results = self.current_results.copy()  # Start with any existing results
        
        # Process in batches
        all_results = []  # Keep track of ALL results
        with tqdm(total=total_remaining, desc="Processing") as pbar:
            for i in range(0, total_remaining, self.BATCH_SIZE):
                batch = remaining_tickers[i:i + self.BATCH_SIZE]
                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)  # Save ALL results
                pbar.update(len(batch))
                
                # Progressive caching - save everything we have so far
                if len(all_results) >= 100 or i + self.BATCH_SIZE >= total_remaining:
                    df = pd.DataFrame(all_results)  # Use all_results instead of results
                    if not df.empty:
                        print("Merging with exchange data for consistency...")
                        merged_df = pd.merge(df, exchange_df, on='ticker', how='inner')
                        self._save_cache(merged_df, 'market_caps')
        
        if not all_results:  # Check all_results instead of results
            print("No results obtained from processing batches.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Final processing
        market_caps_df = pd.DataFrame(all_results)
        print(f"Total market cap records: {len(market_caps_df)}")
        print("Merging with exchange data...")
        all_stocks_df = pd.merge(market_caps_df, exchange_df, on='ticker', how='inner')
        print(f"Total merged records: {len(all_stocks_df)}")
        all_stocks_df = self.validate_market_cap(all_stocks_df)
        print(f"After final validation: {len(all_stocks_df)} records.")
        
        # Save final results
        self._save_cache(all_stocks_df, 'market_caps')
        
        # Filter and format
        print(f"\nFiltering companies with market cap under ${max_market_cap:,}...")
        small_caps = all_stocks_df[all_stocks_df['market_cap'] < max_market_cap].copy()
        print(f"Total small cap companies found: {len(small_caps)}")
        for df in [small_caps, all_stocks_df]:
            df['market_cap_millions'] = df['market_cap'] / 1_000_000
            print("Converted market cap to millions.")
        
        small_caps = small_caps.sort_values('market_cap', ascending=True)
        print("Sorted small cap companies by market cap in ascending order.")
        
        # Clean up resume file
        try:
            print("Attempting to remove resume file...")
            self._get_resume_path().unlink(missing_ok=True)
            print("Resume file removed successfully.")
        except Exception as e:
            print(f"Failed to remove resume file: {str(e)}")
        
        self._print_stats()
        print("Small cap screening process completed.")
        return small_caps, all_stocks_df

    def validate_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate market cap data with filtering."""
        print(f"\nValidating market cap data for {len(df)} records...")
        if df.empty:
            return df
                
        validated = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['market_cap', 'price', 'shares_outstanding']
        for col in numeric_cols:
            if col in validated.columns:
                validated[col] = pd.to_numeric(validated[col], errors='coerce')
    
        # Strict validation
        before_filter = len(validated)
        validated = validated[
            (validated['market_cap'].notna()) &
            (validated['market_cap'] > 0) &
            (validated['price'].notna()) &
            (validated['price'] > 0) &
            (validated['shares_outstanding'].notna()) &
            (validated['shares_outstanding'] > 0)
        ].copy()
    
        # Cross-validate market cap with price * shares
        calculated_cap = validated['price'] * validated['shares_outstanding']
        validated = validated[
            (validated['market_cap'] >= calculated_cap * 0.5) &  # Allow variance
            (validated['market_cap'] <= calculated_cap * 2)
        ]
    
        print(f"After validation: {len(validated)} of {before_filter} records passed")
        return validated
        
        # Convert numeric columns
        numeric_cols = ['market_cap', 'price', 'shares_outstanding']
        for col in numeric_cols:
            if col in validated.columns:
                print(f"Converting column '{col}' to numeric...")
                validated[col] = pd.to_numeric(validated[col], errors='coerce')
                null_count = validated[col].isna().sum()
                print(f"Column '{col}' converted. Null values introduced: {null_count}")
        
        # Basic validation filters
        print("Applying validation filters...")
        before_filter = len(validated)
        validated = validated[
            (validated['market_cap'].notna()) &
            (validated['market_cap'] > 0) &
            (validated['market_cap'] < 1e16) &  # Dollar cap
            (validated['price'].notna()) &
            (validated['price'] > 0)  # Any positive price
        ].copy()
        after_filter = len(validated)
        print(f"Filtered out {before_filter - after_filter} invalid records.")
        
        # Log validation stats
        self.logger.info(f"Validated entries: {len(validated)} out of {len(df)}")
        print(f"Validation complete: {len(validated)} valid records out of {len(df)}.")
        
        return validated

    def process_batch(self, batch_tickers: List[str]) -> List[Dict]:
        """Process batch with better error logging."""
        print(f"\nProcessing batch of {len(batch_tickers)} tickers...")
        results = []
        local_results = []  # New local list for batch saving
        print(f"Tickers in this batch: {batch_tickers}")
        batch_results = self.get_stock_data_batch(batch_tickers)
        print(f"Got results for {len(batch_results)} tickers in this batch.")
        
        for ticker, data in batch_results.items():
            if data:
                market_cap = data.get('market_cap')
                if market_cap:
                    result = {
                        'ticker': ticker,
                        **data,
                        'last_updated': datetime.now().strftime('%Y-%m-%d')
                    }
                    results.append(result)  # Add to main results
                    local_results.append(result)  # Add to local batch
                    print(f"  Including ticker {ticker} with market cap ${market_cap:,.2f}.")
                    if market_cap < 50_000_000:
                        print(f"  ✓ INCLUDED: Under $50M cap.")
                    else:
                        print(f"  FILTERED: Above $50M cap but saved to all_stocks.")
                    self.stats['success'] += 1
    
            # Save progress more frequently
            if len(local_results) >= 10:
                print(f"\nProgress threshold reached. Saving {len(local_results)} results...")
                self.current_results.extend(local_results)
                self._save_resume_state()
                local_results = []  # Only clear local batch
                print("Resume state updated.")
        
        return results  # Return ALL results for the batch

    def format_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format results for display."""
        print("\nFormatting results for display...")
        if df.empty:
            print("DataFrame is empty. Skipping formatting.")
            return df
            
        display_df = df.copy()
        print("Creating a copy of the DataFrame for formatting.")
        
        numeric_cols = [
            'market_cap_millions', 'price', 'shares_outstanding',
            'volume', 'avg_volume', 'high', 'low', 'open'
        ]
        
        for col in numeric_cols:
            if col in display_df.columns:
                print(f"Ensuring column '{col}' is numeric...")
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                null_count = display_df[col].isna().sum()
                print(f"Column '{col}' conversion complete. Null values: {null_count}")
        
        # Format numbers
        if 'market_cap_millions' in display_df:
            print("Rounding 'market_cap_millions' to 2 decimal places.")
            display_df['market_cap_millions'] = display_df['market_cap_millions'].round(2)
        if 'price' in display_df:
            print("Rounding 'price' to 2 decimal places.")
            display_df['price'] = display_df['price'].round(2)
        if 'shares_outstanding' in display_df:
            print("Rounding 'shares_outstanding' to 0 decimal places.")
            display_df['shares_outstanding'] = display_df['shares_outstanding'].round(0)
        
        # Select columns
        base_cols = [
            'ticker', 'name', 'exchange', 'market_cap_millions', 
            'price', 'shares_outstanding', 'volume', 'avg_volume',
            'high', 'low', 'open', 'last_updated'
        ]

        # Select columns
        cols = [col for col in base_cols if col in display_df.columns]
        print(f"Selecting columns for display: {cols}")
        formatted_df = display_df[cols]
        print("Formatting complete.")
        return formatted_df

    def _print_stats(self):
        """Print detailed statistics."""
        print("\nProcessing Statistics:")
        print(f"Total processed tickers: {self.stats['processed']}")
        success_percentage = (self.stats['success'] / max(1, self.stats['processed'])) * 100
        print(f"Successful: {self.stats['success']} ({success_percentage:.1f}%)")
        print("\nErrors:")
        for error_type, count in self.stats['errors'].items():
            if count > 0:
                print(f"- {error_type}: {count}")

    def clear_caches(self):
        """Clear all cache files to ensure fresh data."""
        cache_files = ['market_caps.pkl', 'exchanges.pkl', 'resume_data.pkl']
        try:
            for cache_file in cache_files:
                cache_path = self.CACHE_DIR / cache_file
                if cache_path.exists():
                    cache_path.unlink()
                    print(f"Deleted cache: {cache_file}")
            if os.path.exists("yfinance.cache"):
                os.remove("yfinance.cache")
                print("Deleted YFinance cache")
        except Exception as e:
            print(f"Error clearing caches: {e}")
    
def main():
    print("\n=== MarketCapScreener Execution Started ===")
    start_time = time.time()
    
    try:
        print("\nCreating MarketCapScreener instance...")
        screener = MarketCapScreener(max_workers=10, use_cache=True)
        
        print("\nClearing all caches for fresh data...")
        screener.clear_caches()
        
        try:
            print("\nStarting screening for small cap stocks...")
            small_caps, all_stocks = screener.screen_small_caps(max_market_cap=50_000_000)
            
            if not small_caps.empty:
                print("\nSmall cap companies found. Formatting results...")
                # Format and save results
                small_caps = small_caps.sort_values('market_cap', ascending=True)
                print("Sorted small cap companies by market cap.")
                
                # Save results with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                all_stocks_filename = f'all_stocks_{timestamp}.csv'
                small_caps_filename = f'small_cap_stocks_{timestamp}.csv'
                print(f"Saving all stocks data to {all_stocks_filename}...")
                all_stocks.to_csv(all_stocks_filename, index=False)
                print(f"Saving small cap stocks data to {small_caps_filename}...")
                small_caps.to_csv(small_caps_filename, index=False)
                
                # Create symlinks to latest files
                print("Creating symlinks to the latest results...")
                for old, new in [
                    (all_stocks_filename, 'all_stocks_latest.csv'),
                    (small_caps_filename, 'small_cap_stocks_latest.csv')
                ]:
                    if os.path.exists(new):
                        print(f"Removing existing symlink or file: {new}")
                        os.remove(new)
                    print(f"Creating symlink from {old} to {new}...")
                    os.symlink(old, new)
                    print(f"Symlink created: {new} -> {old}")
                
                # Display results
                print("\nFormatting results for display...")
                display_df = screener.format_results(small_caps)
                
                print(f"\nFound {len(small_caps)} companies under $50,000,000 market cap.")
                print("\nSmall Cap Companies Found:")
                print("=" * 100)
                
                # Configure pandas display
                pd.set_option('display.max_rows', None)
                pd.set_option('display.float_format', lambda x: '%.2f' % x)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(display_df.to_string(index=False))
                
                print(f"\nResults saved to:")
                print(f"- {small_caps_filename} (filtered results)")
                print(f"- {all_stocks_filename} (complete dataset)")
                print(f"- small_cap_stocks_latest.csv (symlink to latest results)")
                print(f"- all_stocks_latest.csv (symlink to latest results)")
                
            else:
                print("\nNo companies found matching the criteria.")
                
        except KeyboardInterrupt:
            print("\nOperation interrupted by user. Saving partial results...")
            screener._save_resume_state()
            print("Partial results saved. Run the script again to resume.")
            return
            
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f"\nScreening failed: {str(e)}")
        logging.exception("Fatal error in main execution")
        print("Fatal error logged.")

    print("\n=== MarketCapScreener Execution Finished ===")

if __name__ == "__main__":
    main()
