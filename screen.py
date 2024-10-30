import finagg
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import time
from functools import partial

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class MarketCapScreener:
    """Enhanced screener with parallel processing and better error handling."""
    
    # Alternative tags to try for shares outstanding
    SHARES_OUTSTANDING_TAGS = [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
        "SharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding"
    ]
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize the screener.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def get_exchange_listed_companies(self) -> pd.DataFrame:
        """Get companies from major exchanges with basic filtering."""
        try:
            df = finagg.sec.api.exchanges.get()
            
            # Filter for major exchanges
            major_exchanges = ['NYSE', 'Nasdaq', 'NYSE Arca', 'NYSE American']
            df = df[df['exchange'].isin(major_exchanges)]
            
            # Basic filtering to remove likely invalid entries
            df = df[df['name'].notna()]  # Must have a name
            df = df[df['ticker'].str.len() <= 5]  # Standard ticker length
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get exchange data: {str(e)}")
            raise

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get latest price data for a ticker."""
        try:
            price_df = finagg.yfinance.api.get(
                ticker,
                interval="1d",
                period="1d"
            )
            
            if not price_df.empty:
                return price_df['close'].iloc[-1]
                
        except Exception:
            pass
            
        return None

    def get_shares_outstanding(self, ticker: str) -> Optional[float]:
        """Try multiple tags to get shares outstanding."""
        for tag in self.SHARES_OUTSTANDING_TAGS:
            try:
                shares_data = finagg.sec.api.company_concept.get(
                    tag,
                    ticker=ticker,
                    units="shares"
                )
                
                if not shares_data.empty:
                    shares_data = shares_data.sort_values('end', ascending=False)
                    return shares_data['value'].iloc[0]
                    
            except Exception:
                continue
                
        return None

    def process_company(self, ticker: str) -> Optional[Dict]:
        """Process a single company for market cap calculation."""
        try:
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
            
            return {
                'ticker': ticker,
                'price': price,
                'shares_outstanding': shares,
                'market_cap': market_cap,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception:
            return None

    def screen_small_caps(self, max_market_cap: float = 50_000_000) -> pd.DataFrame:
        """
        Screen for small cap stocks with parallel processing.
        
        Args:
            max_market_cap: Maximum market cap in USD
            
        Returns:
            DataFrame of filtered companies
        """
        # Get exchange-listed companies
        print("\nFetching exchange-listed companies...")
        exchange_df = self.get_exchange_listed_companies()
        total_companies = len(exchange_df)
        print(f"Found {total_companies} companies on major exchanges")
        
        # Process companies in parallel with progress bar
        print("\nCalculating market caps (this may take a few minutes)...")
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.process_company, ticker): ticker 
                for ticker in exchange_df['ticker']
            }
            
            # Process results with progress bar
            with tqdm(total=total_companies, desc="Processing") as pbar:
                for future in as_completed(future_to_ticker):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        if not results:
            return pd.DataFrame()
        
        # Create results dataframe
        market_caps_df = pd.DataFrame(results)
        
        # Merge with exchange info
        result_df = pd.merge(market_caps_df, exchange_df, on='ticker', how='inner')
        
        # Filter for small caps
        small_caps = result_df[result_df['market_cap'] < max_market_cap].copy()
        small_caps['market_cap_millions'] = small_caps['market_cap'] / 1_000_000
        
        # Sort by market cap
        small_caps = small_caps.sort_values('market_cap', ascending=True)
        
        print(f"\nFound {len(small_caps)} companies under ${max_market_cap:,.0f} market cap")
        
        return small_caps

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
    
    # Initialize screener with desired number of workers
    screener = MarketCapScreener(max_workers=20)  # Adjust based on your CPU
    
    try:
        # Screen for companies under $50M market cap
        results = screener.screen_small_caps(max_market_cap=50_000_000)
        
        if not results.empty:
            # Format and display results
            display_df = format_results(results)
            
            print("\nSmall Cap Companies Found:")
            print("=========================")
            pd.set_option('display.max_rows', None)
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            print(display_df.to_string(index=False))
            
            # Save results
            results.to_csv('small_cap_stocks.csv', index=False)
            print(f"\nResults saved to small_cap_stocks.csv")
            
        else:
            print("\nNo companies found matching the criteria.")
        
        # Print execution time
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f"Screening failed: {str(e)}")

if __name__ == "__main__":
    main()
