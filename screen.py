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
    
    def __init__(self, max_workers: int = 20, cache_expiry_days: int = 1):
        """Initialize with better parallelization settings."""
        self.max_workers = max_workers
        self.cache_expiry_days = cache_expiry_days
        self.stats = {
            'processed': 0,
            'success': 0,
            'errors': defaultdict(int),
            'last_save': time.time()
        }
        
        # Create cache directory
        self.CACHE_DIR.mkdir(exist_ok=True)
        
        # Setup better session with increased rate limits
        self.session = CachedLimiterSession(
            limiter=Limiter(RequestRate(10, Duration.SECOND)),  # 10 requests/sec
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache("yfinance.cache")
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('screener.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Track processed tickers for resume capability
        self.processed_tickers = set()
        self.current_results = []
        
        self.BATCH_SIZE = 50  # Process more tickers at once

    def _get_cache_path(self, cache_type: str) -> Path:
            """Get cache file path without date stamping."""
            return self.CACHE_DIR / f"{cache_type}.pkl"  # Think about date dependency
    
    def _get_single_ticker_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get data for a single ticker with proper rate limiting."""
        try:
            # Add small delay between requests
            time.sleep(0.1)
            
            # Use the finagg API for basic price data
            try:
                price_data = finagg.yfinance.api.get(ticker, period="5d")
                if price_data.empty:
                    return None
                latest_price = price_data['close'].iloc[-1]
                avg_volume = price_data['volume'].mean()
            except Exception as e:
                self.logger.debug(f"Failed to get finagg data for {ticker}: {str(e)}")
                return None
                
            # Use yfinance for additional info
            ticker_obj = yf.Ticker(ticker, session=self.session)
            info = ticker_obj.info
            
            if not info:
                return None
                
            # Get market cap directly or calculate it
            market_cap = info.get('marketCap')
            if market_cap is None and latest_price is not None and info.get('sharesOutstanding') is not None:
                market_cap = latest_price * info['sharesOutstanding']
            
            # Only require market cap and price for basic validity
            if market_cap is None or latest_price is None:
                print(f"  Skipped {ticker}: Missing required data - Market Cap: {market_cap}, Price: {latest_price}")
                return None
                
            result = {
                'ticker': ticker,
                'price': latest_price,
                'volume': avg_volume,  # Use averaged volume from price data
                'shares_outstanding': info.get('sharesOutstanding'),
                'market_cap': market_cap,
                'high': price_data['high'].iloc[-1] if not price_data.empty else None,
                'low': price_data['low'].iloc[-1] if not price_data.empty else None,
                'open': price_data['open'].iloc[-1] if not price_data.empty else None
            }
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Error getting data for {ticker}: {str(e)}")
            return None

    def _get_resume_path(self) -> Path:
        """Get path for resume data."""
        return self.CACHE_DIR / "resume_data.pkl"

    def _load_cache(self, cache_type: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(cache_type)
        if cache_path.exists():
            try:
                cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if datetime.now() - cache_time < timedelta(days=self.cache_expiry_days):
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {str(e)}")
        return None

    def _save_cache(self, data: pd.DataFrame, cache_type: str):
        """Save data to cache with error handling."""
        try:
            cache_path = self._get_cache_path(cache_type)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.stats['last_save'] = time.time()
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {str(e)}")

    def _save_resume_state(self):
        """Save current state for resume capability."""
        try:
            resume_data = {
                'processed_tickers': self.processed_tickers,
                'current_results': self.current_results,
                'stats': self.stats
            }
            with open(self._get_resume_path(), 'wb') as f:
                pickle.dump(resume_data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save resume state: {str(e)}")

    def _load_resume_state(self) -> bool:
        """Load previous state if exists."""
        try:
            resume_path = self._get_resume_path()
            if resume_path.exists():
                with open(resume_path, 'rb') as f:
                    resume_data = pickle.load(f)
                self.processed_tickers = resume_data['processed_tickers']
                self.current_results = resume_data['current_results']
                self.stats = resume_data['stats']
                return True
        except Exception as e:
            self.logger.warning(f"Failed to load resume state: {str(e)}")
        return False

    def get_exchange_listed_companies(self) -> pd.DataFrame:
        """Get companies from major exchanges with caching."""
        cached_data = self._load_cache('exchanges')
        if cached_data is not None:
            return cached_data
            
        try:
            df = finagg.sec.api.exchanges.get()
            
            # Filter for major exchanges
            major_exchanges = ['NYSE', 'Nasdaq', 'NYSE Arca', 'NYSE American']
            df = df[df['exchange'].isin(major_exchanges)]
            df = df[df['name'].notna()]
            # Exclude tickers with special characters but keep the regex pattern valid
            df = df[~df['ticker'].str.contains(r'[\^$]')]  # Fixed regex pattern
            df = df[df['ticker'].str.len() <= 5]
            
            self._save_cache(df, 'exchanges')
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get exchange data: {str(e)}")
            raise

    def get_stock_data_batch(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get stock data for multiple tickers with true parallelization."""
        results = {}
        futures = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit each ticker for parallel processing
            for ticker in tickers:
                futures.append((ticker, executor.submit(self._get_single_ticker_data, ticker)))
                
            # Collect results as they complete
            for ticker, future in futures:
                try:
                    data = future.result()
                    if data:
                        results[ticker] = data
                except Exception as e:
                    self.logger.debug(f"Error processing {ticker}: {str(e)}")
                    self.stats['errors']['processing'] += 1
                    
        return results

    def screen_small_caps(self, max_market_cap: float = 50_000_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Screen for small cap stocks with progressive caching."""
        # Try loading existing results first
        cached_data = self._load_cache('market_caps')
        print(f"\nAttempting to load cache...")
        if cached_data is not None:
            print(f"Cache found with {len(cached_data)} entries")
            if not cached_data.empty:
                print("Using cached market cap data...")
                all_stocks_df = self.validate_market_cap(cached_data)
                print(f"After validation: {len(all_stocks_df)} entries")
                small_caps = all_stocks_df[all_stocks_df['market_cap'] < max_market_cap].copy()
                print(f"Found {len(small_caps)} companies under {max_market_cap:,} market cap")
                if not small_caps.empty:
                    return small_caps, all_stocks_df
            else:
                print("Cached data was empty")
        else:
            print("No valid cache found")
        
        # Check for resume data
        if self._load_resume_state():
            print("\nResuming previous run...")
        else:
            self.processed_tickers.clear()
            self.current_results.clear()
        
        # Get exchange-listed companies
        print("\nFetching exchange-listed companies...")
        exchange_df = self.get_exchange_listed_companies()
        
        # Filter out already processed tickers
        remaining_tickers = [t for t in exchange_df['ticker'] if t not in self.processed_tickers]
        total_remaining = len(remaining_tickers)
        print(f"Processing {total_remaining} remaining companies...")
        
        results = self.current_results.copy()  # Start with any existing results
        
        # Process in batches
        with tqdm(total=total_remaining, desc="Processing") as pbar:
            for i in range(0, total_remaining, self.BATCH_SIZE):
                batch = remaining_tickers[i:i + self.BATCH_SIZE]
                batch_results = self.process_batch(batch)
                results.extend(batch_results)
                pbar.update(len(batch))
                
                # Progressive caching
                if len(results) >= 100 or i + self.BATCH_SIZE >= total_remaining:
                    df = pd.DataFrame(results)
                    if not df.empty:
                        merged_df = pd.merge(df, exchange_df, on='ticker', how='inner')
                        self._save_cache(merged_df, 'market_caps')
        
        if not results:
            return pd.DataFrame(), pd.DataFrame()
        
        # Final processing
        market_caps_df = pd.DataFrame(results)
        all_stocks_df = pd.merge(market_caps_df, exchange_df, on='ticker', how='inner')
        all_stocks_df = self.validate_market_cap(all_stocks_df)
        
        # Save final results
        self._save_cache(all_stocks_df, 'market_caps')
        
        # Filter and format
        small_caps = all_stocks_df[all_stocks_df['market_cap'] < max_market_cap].copy()
        for df in [small_caps, all_stocks_df]:
            df['market_cap_millions'] = df['market_cap'] / 1_000_000
            
        small_caps = small_caps.sort_values('market_cap', ascending=True)
        
        # Clean up resume file
        try:
            self._get_resume_path().unlink(missing_ok=True)
        except:
            pass
            
        self._print_stats()
        return small_caps, all_stocks_df

    def validate_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate market cap data with less strict filtering."""
        if df.empty:
            return df
            
        validated = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['market_cap', 'price', 'shares_outstanding']
        for col in numeric_cols:
            if col in validated.columns:
                validated[col] = pd.to_numeric(validated[col], errors='coerce')
        
        # Basic validation filters
        validated = validated[
            (validated['market_cap'].notna()) &
            (validated['market_cap'] > 0) &
            (validated['market_cap'] < 1e15) &  # Trillion dollar cap
            (validated['price'].notna()) &
            (validated['price'] > 0)  # Any positive price
        ].copy()
        
        # Log validation stats
        self.logger.info(f"Validated entries: {len(validated)} out of {len(df)}")
        
        return validated
    
    def process_batch(self, batch_tickers: List[str]) -> List[Dict]:
        """Process batch with better error logging."""
        results = []
        print(f"\nProcessing batch of {len(batch_tickers)} tickers...")
        batch_results = self.get_stock_data_batch(batch_tickers)
        print(f"Got results for {len(batch_results)} tickers")

        for ticker, data in batch_results.items():
            print(f"\nProcessing {ticker}:")
            if data:
                market_cap = data.get('market_cap')
                if market_cap:
                    print(f"  Market Cap: ${market_cap:,.2f}")
                    results.append({
                        'ticker': ticker,
                        **data,
                        'last_updated': datetime.now().strftime('%Y-%m-%d')
                    })
                    if market_cap < 50_000_000:
                        print(f"  ✓ INCLUDED: Under $50M cap")
                    else:
                        print(f"  FILTERED: Above $50M cap but saved to all_stocks")
                    self.stats['success'] += 1
                else:
                    print(f"  Skipped: No market cap data")
            else:
                print(f"  Skipped: No valid data")
            self.stats['processed'] += 1
            self.processed_tickers.add(ticker)

            # Save progress more frequently and include ALL processed companies
            if time.time() - self.stats['last_save'] > 30 or len(results) >= 10:
                print(f"\nSaving {len(results)} results to resume state...")
                self.current_results.extend(results)
                self._save_resume_state()
                results = []  # Clear after saving to avoid duplicates

        return results

    def format_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format results for display."""
        if df.empty:
            return df
            
        display_df = df.copy()
        
        numeric_cols = [
            'market_cap_millions', 'price', 'shares_outstanding',
            'volume', 'avg_volume', 'high', 'low', 'open'
        ]
        
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
        
        # Format numbers
        if 'market_cap_millions' in display_df:
            display_df['market_cap_millions'] = display_df['market_cap_millions'].round(2)
        if 'price' in display_df:
            display_df['price'] = display_df['price'].round(2)
        if 'shares_outstanding' in display_df:
            display_df['shares_outstanding'] = display_df['shares_outstanding'].round(0)
        
        # Select columns
        base_cols = [
            'ticker', 'name', 'exchange', 'market_cap_millions', 
            'price', 'shares_outstanding', 'volume', 'avg_volume',
            'high', 'low', 'open', 'last_updated'
        ]

        # Select columns
        cols = [col for col in base_cols if col in display_df.columns]
        return display_df[cols]

    def _print_stats(self):
        """Print detailed statistics."""
        print("\nProcessing Statistics:")
        print(f"Total processed: {self.stats['processed']}")
        print(f"Successful: {self.stats['success']} ({(self.stats['success']/max(1, self.stats['processed'])*100):.1f}%)")
        print("\nErrors:")
        for error_type, count in self.stats['errors'].items():
            if count > 0:
                print(f"- {error_type}: {count}")

def main():
    """Main execution function with improved error handling and reporting."""
    start_time = time.time()
    
    try:
        screener = MarketCapScreener(max_workers=10)  # Reduced workers for better stability
        
        # Catch keyboard interrupts to allow graceful shutdown
        try:
            small_caps, all_stocks = screener.screen_small_caps(max_market_cap=50_000_000)
            
            if not small_caps.empty:
                # Format and save results
                small_caps = small_caps.sort_values('market_cap', ascending=True)
                
                # Save results with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                all_stocks.to_csv(f'all_stocks_{timestamp}.csv', index=False)
                small_caps.to_csv(f'small_cap_stocks_{timestamp}.csv', index=False)
                
                # Create symlinks to latest files
                for old, new in [
                    (f'all_stocks_{timestamp}.csv', 'all_stocks_latest.csv'),
                    (f'small_cap_stocks_{timestamp}.csv', 'small_cap_stocks_latest.csv')
                ]:
                    if os.path.exists(new):
                        os.remove(new)
                    os.symlink(old, new)
                
                # Display results
                display_df = screener.format_results(small_caps)
                
                print(f"\nFound {len(small_caps)} companies under $50,000,000 market cap")
                print("\nSmall Cap Companies Found:")
                print("=" * 100)
                
                # Configure pandas display
                pd.set_option('display.max_rows', None)
                pd.set_option('display.float_format', lambda x: '%.2f' % x)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(display_df.to_string(index=False))
                
                print(f"\nResults saved to:")
                print(f"- small_cap_stocks_{timestamp}.csv (filtered results)")
                print(f"- all_stocks_{timestamp}.csv (complete dataset)")
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
        print(f"Screening failed: {str(e)}")
        logging.exception("Fatal error in main execution")

if __name__ == "__main__":
    main()
