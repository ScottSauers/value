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
    """Enhanced screener with fixed market cap calculation and additional metrics."""

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
    
    def get_stock_data(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive stock data including price, volume, and shares."""
        try:
            # Get detailed stock data
            stock_df = finagg.yfinance.api.get(
                ticker,
                interval="1d",
                period="5d"  # Get 5 days to calculate averages
            )
            
            if stock_df.empty:
                self.error_counts['price_fetch_failed'] += 1
                return None
                
            # Get latest data
            latest = stock_df.iloc[-1]
            
            # Calculate 5-day averages
            avg_volume = stock_df['volume'].mean()
            avg_price = stock_df['close'].mean()
            
            return {
                'price': float(latest['close']),
                'volume': float(latest['volume']),
                'avg_volume': float(avg_volume),
                'avg_price': float(avg_price),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'open': float(latest['open'])
            }
            
        except Exception as e:
            self.error_counts['price_fetch_failed'] += 1
            return None

    def get_shares_data(self, ticker: str) -> Optional[float]:
        """Get shares outstanding from multiple sources with better error handling."""
        try:
            # Try yfinance first for shares data
            import yfinance as yf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker_obj = yf.Ticker(ticker)
                try:
                    shares = ticker_obj.info.get('sharesOutstanding')
                    if shares and shares > 0:
                        return float(shares)
                except:
                    shares = ticker_obj.info.get('marketCap')
                    price = ticker_obj.info.get('currentPrice')
                    if shares and price and price > 0:
                        return float(shares / price)
        except:
            pass

        # Fallback to SEC data with better connection handling
        used_tags = set()
        for tag in self.SHARES_OUTSTANDING_TAGS:
            if tag in used_tags:
                continue
            used_tags.add(tag)
            
            try:
                with self.session.get(
                    f'https://data.sec.gov/api/xbrl/companyconcept/CIK{ticker}/{tag}.json',
                    timeout=5
                ) as response:
                    if response.status_code == 200:
                        data = response.json()
                        if 'units' in data and 'shares' in data['units']:
                            shares_data = pd.DataFrame(data['units']['shares'])
                            if not shares_data.empty:
                                shares_data = shares_data.sort_values('end', ascending=False)
                                shares = float(shares_data['val'].iloc[0])
                                if shares > 0:
                                    return shares
                    time.sleep(0.1)  # Rate limiting
            except:
                continue

        self.error_counts['shares_fetch_failed'] += 1
        return None

    def process_company(self, ticker: str) -> Optional[Dict]:
        """Process a single company with comprehensive data collection."""
        try:
            self.processed_count += 1
            
            # Get stock data
            stock_data = self.get_stock_data(ticker)
            if not stock_data:
                return None
                
            # Get shares outstanding
            shares = self.get_shares_data(ticker)
            if not shares:
                return None
                
            # Calculate market cap
            market_cap = stock_data['price'] * shares
            
            # Basic validation
            if market_cap <= 0 or market_cap > 1e15:
                self.error_counts['invalid_market_cap'] += 1
                return None
            
            # Calculate additional metrics
            daily_volume_usd = stock_data['volume'] * stock_data['price']
            avg_volume_usd = stock_data['avg_volume'] * stock_data['avg_price']
            
            self.success_count += 1
            return {
                'ticker': ticker,
                'price': stock_data['price'],
                'shares_outstanding': shares,
                'market_cap': market_cap,
                'volume': stock_data['volume'],
                'avg_volume': stock_data['avg_volume'],
                'daily_volume_usd': daily_volume_usd,
                'avg_volume_usd': avg_volume_usd,
                'high': stock_data['high'],
                'low': stock_data['low'],
                'open': stock_data['open'],
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            self.error_counts['other_errors'] += 1
            return None

    def format_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format results with better numeric handling."""
        if df.empty:
            return df
            
        display_df = df.copy()
        
        # Ensure numeric columns
        numeric_cols = [
            'market_cap_millions', 'price', 'shares_outstanding',
            'volume', 'avg_volume', 'daily_volume_usd', 'avg_volume_usd',
            'high', 'low', 'open'
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
        
        # Select columns that exist
        base_cols = [
            'ticker', 'name', 'exchange', 'market_cap_millions', 
            'price', 'shares_outstanding', 'volume', 'avg_volume',
            'daily_volume_usd', 'avg_volume_usd', 'high', 'low', 'open',
            'last_updated'
        ]
        cols = [col for col in base_cols if col in display_df.columns]
        
        return display_df[cols]

    def validate_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additional validation for market cap data."""
        validated = df.copy()
        
        # Remove obvious errors
        validated = validated[
            (validated['market_cap'] > 0) &  # Must be positive
            (validated['market_cap'] < 1e15) &  # Reasonable upper limit
            (validated['shares_outstanding'] > 1000) &  # Minimum shares
            (validated['shares_outstanding'] < 1e11) &  # Maximum shares
            (validated['price'] > 0.01) &  # Minimum price
            (validated['price'] < 1e5)  # Maximum price
        ]
        
        return validated

    def screen_small_caps(self, max_market_cap: float = 50_000_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Screen for small cap stocks with improved caching and validation."""
        try:
            # Try to load from cache
            cached_data = self._load_cache('market_caps')
            if cached_data is not None and not cached_data.empty:
                print("Using cached market cap data...")
                try:
                    all_stocks_df = self.validate_market_cap(cached_data)
                    small_caps = all_stocks_df[all_stocks_df['market_cap'] < max_market_cap].copy()
                    if not small_caps.empty:
                        return small_caps, all_stocks_df
                except Exception as e:
                    self.logger.warning(f"Cache validation failed: {str(e)}")
        
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
        
        # Validate data
        all_stocks_df = self.validate_market_cap(all_stocks_df)
        
        # Cache only if we have valid data
        if not all_stocks_df.empty:
            self._save_cache(all_stocks_df, 'market_caps')
        
        # Filter for small caps
        small_caps = all_stocks_df[all_stocks_df['market_cap'] < max_market_cap].copy()
        
        # Add millions column for display
        small_caps['market_cap_millions'] = small_caps['market_cap'] / 1_000_000
        all_stocks_df['market_cap_millions'] = all_stocks_df['market_cap'] / 1_000_000
        
        # Sort by market cap
        small_caps = small_caps.sort_values('market_cap', ascending=True)
        
        self._print_completion_stats()
        
        return small_caps, all_stocks_df
        
    except Exception as e:
        self.logger.error(f"Screening failed: {str(e)}")
            raise

    def validate_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive market cap validation."""
        if df.empty:
            return df
            
        validated = df.copy()
        
        # Convert numeric columns if needed
        numeric_cols = ['market_cap', 'price', 'shares_outstanding']
        for col in numeric_cols:
            if col in validated.columns:
                validated[col] = pd.to_numeric(validated[col], errors='coerce')
        
        # Basic validation filters
        validated = validated[
            (validated['market_cap'].notna()) &
            (validated['market_cap'] > 0) &
            (validated['market_cap'] < 1e15) &  # Trillion dollar cap
            (validated['shares_outstanding'].notna()) &
            (validated['shares_outstanding'] > 1000) &  # Minimum reasonable shares
            (validated['shares_outstanding'] < 1e11) &  # Maximum reasonable shares
            (validated['price'].notna()) &
            (validated['price'] > 0.01) &  # Penny stock minimum
            (validated['price'] < 1e5)  # Maximum reasonable price
        ].copy()
        
        # Verify market cap calculation
        validated['calc_market_cap'] = validated['price'] * validated['shares_outstanding']
        validated = validated[
            (validated['calc_market_cap'] / validated['market_cap']).between(0.9, 1.1)  # Allow 10% variance
        ]
        
        return validated
        
        
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

def main():
    """Main execution function."""
    start_time = time.time()
    
    screener = MarketCapScreener()
    
    try:
        small_caps, all_stocks = screener.screen_small_caps(max_market_cap=50_000_000)
        
        if not small_caps.empty:
            # Additional sorting and formatting
            small_caps = small_caps.sort_values('market_cap', ascending=True)
            
            # Save results
            all_stocks.to_csv('all_stocks.csv', index=False)
            small_caps.to_csv('small_cap_stocks.csv', index=False)
            
            # Display results
            display_df = screener.format_results(small_caps)
            
            print(f"\nFound {len(small_caps)} companies under $50,000,000 market cap")
            print("\nSmall Cap Companies Found:")
            print("=" * 100)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
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
