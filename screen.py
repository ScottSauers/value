import finagg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import time
import json
from urllib3 import PoolManager
from urllib3.util import Retry
import requests
from requests.adapters import HTTPAdapter
import os
import pickle

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class CustomHTTPAdapter(HTTPAdapter):
    """Custom HTTP Adapter with proper retry strategy."""
    def __init__(self, max_retries=3):
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        super().__init__(max_retries=retry_strategy, pool_connections=50, pool_maxsize=50)

class MarketCapScreener:
    """Enhanced screener with caching and robust error handling."""
    
    SHARES_OUTSTANDING_TAGS = [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
        "SharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding"
    ]
    
    CACHE_DIR = Path("cache")
    
    def __init__(self, max_workers: int = 20, cache_expiry_days: int = 1):
        """Initialize the screener with caching support."""
        self.max_workers = max_workers
        self.cache_expiry_days = cache_expiry_days
        self.error_counts = {
            'price_fetch_failed': 0,
            'shares_fetch_failed': 0,
            'invalid_market_cap': 0,
            'connection_error': 0,
            'other_errors': 0
        }
        self.processed_count = 0
        self.success_count = 0
        
        # Create cache directory
        self.CACHE_DIR.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup custom session with proper connection pooling
        self.session = requests.Session()
        adapter = CustomHTTPAdapter(max_retries=3)
        self.session.mount('https://', adapter)
        
    def _get_cache_path(self, cache_type: str) -> Path:
        """Get cache file path."""
        return self.CACHE_DIR / f"{cache_type}_{datetime.now().strftime('%Y%m%d')}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < timedelta(days=self.cache_expiry_days)

    def _load_cache(self, cache_type: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(cache_type)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {str(e)}")
        return None

    def _save_cache(self, data: pd.DataFrame, cache_type: str):
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(cache_type)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {str(e)}")

    def get_exchange_listed_companies(self) -> pd.DataFrame:
        """Get companies from major exchanges with caching."""
        # Try to load from cache
        cached_data = self._load_cache('exchanges')
        if cached_data is not None:
            return cached_data
            
        try:
            df = finagg.sec.api.exchanges.get()
            
            # Filter for major exchanges
            major_exchanges = ['NYSE', 'Nasdaq', 'NYSE Arca', 'NYSE American']
            df = df[df['exchange'].isin(major_exchanges)]
            
            # Basic filtering
            df = df[df['name'].notna()]
            df = df[df['ticker'].str.len() <= 5]
            
            # Cache the results
            self._save_cache(df, 'exchanges')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get exchange data: {str(e)}")
            raise

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get latest price with error handling."""
        try:
            price_df = finagg.yfinance.api.get(
                ticker,
                interval="1d",
                period="1d"
            )
            
            if not price_df.empty:
                return float(price_df['close'].iloc[-1])
                
        except Exception:
            self.error_counts['price_fetch_failed'] += 1
            
        return None

    def get_shares_outstanding(self, ticker: str) -> Optional[float]:
        """Get shares outstanding with multiple attempts."""
        for tag in self.SHARES_OUTSTANDING_TAGS:
            try:
                shares_data = finagg.sec.api.company_concept.get(
                    tag,
                    ticker=ticker,
                    units="shares"
                )
                
                if not shares_data.empty:
                    shares_data = shares_data.sort_values('end', ascending=False)
                    value = float(shares_data['value'].iloc[0])
                    # Validate the value
                    if value > 0 and value < 1e12:  # Basic sanity check
                        return value
                        
            except Exception as e:
                if "404" in str(e):
                    continue
                if "Connection pool is full" in str(e):
                    self.error_counts['connection_error'] += 1
                    time.sleep(0.1)  # Add small delay
                    continue
                    
        self.error_counts['shares_fetch_failed'] += 1
        return None

    def process_company(self, ticker: str) -> Optional[Dict]:
        """Process a single company with comprehensive error tracking."""
        try:
            self.processed_count += 1
            
            # Get latest price
            price = self.get_latest_price(ticker)
            if not price:
                return None
                
            # Get shares outstanding
            shares = self.get_shares_outstanding(ticker)
            if not shares:
                return None
                
            # Calculate market cap
            market_cap = price * shares
            
            # Validate market cap
            if market_cap <= 0 or market_cap > 1e15:  # Basic sanity check
                self.error_counts['invalid_market_cap'] += 1
                return None
                
            self.success_count += 1
            return {
                'ticker': ticker,
                'price': price,
                'shares_outstanding': shares,
                'market_cap': market_cap,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            self.error_counts['other_errors'] += 1
            return None

    def screen_small_caps(self, max_market_cap: float = 50_000_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Screen for small cap stocks with caching and comprehensive error tracking.
        
        Returns:
            Tuple of (small_caps_df, all_stocks_df)
        """
        # Try to load from cache
        cached_data = self._load_cache('market_caps')
        if cached_data is not None:
            print("Using cached market cap data...")
            all_stocks_df = cached_data
            small_caps = all_stocks_df[all_stocks_df['market_cap'] < max_market_cap].copy()
            return small_caps, all_stocks_df
        
        # Get exchange-listed companies
        print("\nFetching exchange-listed companies...")
        exchange_df = self.get_exchange_listed_companies()
        total_companies = len(exchange_df)
        print(f"Found {total_companies} companies on major exchanges")
        
        # Process companies in parallel
        print("\nCalculating market caps...")
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.process_company, ticker): ticker 
                for ticker in exchange_df['ticker']
            }
            
            with tqdm(total=total_companies, desc="Processing") as pbar:
                for future in as_completed(future_to_ticker):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        if not results:
            return pd.DataFrame(), pd.DataFrame()
        
        # Create results dataframe
        market_caps_df = pd.DataFrame(results)
        
        # Merge with exchange info
        all_stocks_df = pd.merge(market_caps_df, exchange_df, on='ticker', how='inner')
        
        # Cache all stocks data
        self._save_cache(all_stocks_df, 'market_caps')
        
        # Filter for small caps
        small_caps = all_stocks_df[all_stocks_df['market_cap'] < max_market_cap].copy()
        
        # Add millions column for display
        for df in [small_caps, all_stocks_df]:
            df['market_cap_millions'] = df['market_cap'] / 1_000_000
        
        # Sort by market cap
        small_caps = small_caps.sort_values('market_cap', ascending=True)
        
        # Print completion statistics
        self._print_completion_stats()
        
        return small_caps, all_stocks_df

    def _print_completion_stats(self):
        """Print detailed completion statistics."""
        print("\nProcessing Statistics:")
        print(f"Total companies processed: {self.processed_count}")
        print(f"Successfully processed: {self.success_count} ({(self.success_count/self.processed_count*100):.1f}%)")
        print("\nError Breakdown:")
        for error_type, count in self.error_counts.items():
            if count > 0:
                percentage = (count / self.processed_count) * 100
                print(f"- {error_type}: {count} ({percentage:.1f}%)")

def format_results(df: pd.DataFrame) -> pd.DataFrame:
    """Format results for display."""
    display_df = df.copy()
    
    # Format numeric columns
    display_df['market_cap_millions'] = display_df['market_cap_millions'].round(2)
    display_df['price'] = display_df['price'].round(2)
    display_df['shares_outstanding'] = display_df['shares_outstanding'].round(0)
    
    # Select and rename columns
    cols = [
        'ticker', 'name', 'exchange', 'market_cap_millions', 
        'price', 'shares_outstanding', 'last_updated'
    ]
    
    return display_df[cols]

def main():
    """Main execution function."""
    start_time = time.time()
    
    screener = MarketCapScreener(max_workers=20)
    
    try:
        # Screen for companies
        small_caps, all_stocks = screener.screen_small_caps(max_market_cap=50_000_000)
        
        if not small_caps.empty:
            # Save all results
            all_stocks.to_csv('all_stocks.csv', index=False)
            
            # Format and save small caps
            small_caps.to_csv('small_cap_stocks.csv', index=False)
            display_df = format_results(small_caps)
            
            print(f"\nFound {len(small_caps)} companies under $50,000,000 market cap")
            print("\nSmall Cap Companies Found:")
            print("=========================")
            pd.set_option('display.max_rows', None)
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            print(display_df.to_string(index=False))
            
            print(f"\nResults saved to:")
            print(f"- small_cap_stocks.csv (filtered results)")
            print(f"- all_stocks.csv (complete dataset)")
            
        else:
            print("\nNo companies found matching the criteria.")
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f"Screening failed: {str(e)}")

if __name__ == "__main__":
    main()
