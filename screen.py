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

class MarketCapScreener:
    """Enhanced screener with fixed market cap calculation and additional metrics."""
    
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
        """Get shares outstanding from multiple sources."""
        try:
            # Try yfinance first for shares data
            import yfinance as yf
            ticker_obj = yf.Ticker(ticker)
            shares = ticker_obj.info.get('sharesOutstanding')
            if shares:
                return float(shares)
        except:
            pass
            
        # Fallback to SEC data
        for tag in self.SHARES_OUTSTANDING_TAGS:
            try:
                shares_data = finagg.sec.api.company_concept.get(
                    tag,
                    ticker=ticker,
                    units="shares"
                )
                
                if not shares_data.empty:
                    shares_data = shares_data.sort_values('end', ascending=False)
                    shares = float(shares_data['value'].iloc[0])
                    if shares > 0:
                        return shares
                        
            except Exception as e:
                if "Connection pool is full" in str(e):
                    time.sleep(0.1)
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
        """Format results with additional metrics."""
        display_df = df.copy()
        
        # Format numeric columns
        display_df['market_cap_millions'] = display_df['market_cap_millions'].round(2)
        display_df['price'] = display_df['price'].round(2)
        display_df['shares_outstanding'] = display_df['shares_outstanding'].round(0)
        display_df['avg_volume'] = display_df['avg_volume'].round(0)
        display_df['daily_volume_usd'] = (display_df['daily_volume_usd'] / 1_000_000).round(2)
        display_df['avg_volume_usd'] = (display_df['avg_volume_usd'] / 1_000_000).round(2)
        
        # Select and order columns
        cols = [
            'ticker', 'name', 'exchange', 
            'market_cap_millions', 'price', 'shares_outstanding',
            'volume', 'avg_volume', 'daily_volume_usd', 'avg_volume_usd',
            'high', 'low', 'open', 'last_updated'
        ]
        
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
