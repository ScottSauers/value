import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import concurrent.futures
from typing import List, Dict, Set
import logging
from prices import PriceDataExtractor
import time
import os

class HistoricalDataProcessor:
    """Process and combine historical price data for multiple stocks."""
    
    def __init__(self, data_dir: str = "./data", max_workers: int = 4):
        """
        Initialize the processor with configuration.
        
        Args:
            data_dir: Directory for storing price data
            max_workers: Maximum number of parallel workers
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.extractor = PriceDataExtractor(output_dir=str(self.data_dir))
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'historical_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _is_cache_fresh(self, ticker: str, interval: str) -> bool:
        """
        Check if cached data for a ticker is fresh (less than 24 hours old).
        
        Args:
            ticker: Stock ticker symbol
            interval: Time interval (e.g., "1d", "1wk", "1mo")
            
        Returns:
            Boolean indicating if cache is fresh
        """
        pattern = f"{ticker}_prices_{interval}_*.tsv"
        existing_files = list(self.data_dir.glob(pattern))
        
        if not existing_files:
            return False
            
        # Get most recent file
        latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
        modified_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
        
        return (datetime.now() - modified_time) < timedelta(hours=24)

    def process_single_ticker(self, ticker: str, intervals: List[str]) -> Dict:
        """
        Process a single ticker, using cache if available.
        
        Args:
            ticker: Stock ticker symbol
            intervals: List of time intervals to process
            
        Returns:
            Dictionary of results per interval
        """
        results = {}
        
        for interval in intervals:
            if self._is_cache_fresh(ticker, interval):
                self.logger.info(f"Using cached data for {ticker} ({interval})")
                pattern = f"{ticker}_prices_{interval}_*.tsv"
                latest_file = max(
                    self.data_dir.glob(pattern),
                    key=lambda x: x.stat().st_mtime
                )
                results[interval] = (str(latest_file), '')
            else:
                try:
                    self.logger.info(f"Fetching fresh data for {ticker} ({interval})")
                    interval_results = self.extractor.process_ticker(ticker, [interval])
                    results.update(interval_results)
                except Exception as e:
                    self.logger.error(f"Failed to process {ticker} ({interval}): {str(e)}")
                    
        return results

    def _get_tickers_from_csv(self) -> List[str]:
        """
        Get list of tickers from small_cap_stocks_latest.csv
        
        Returns:
            List of ticker symbols
        """
        try:
            df = pd.read_csv('small_cap_stocks_latest.csv')
            tickers = df['ticker'].str.strip().tolist()
            self.logger.info(f"Found {len(tickers)} tickers in CSV")
            return tickers
        except Exception as e:
            self.logger.error(f"Failed to read tickers from CSV: {str(e)}")
            raise

    def _harmonize_dates(self, dfs: List[pd.DataFrame], interval: str) -> pd.DataFrame:
        """
        Harmonize dates across multiple dataframes.
        
        Args:
            dfs: List of dataframes to harmonize
            interval: Time interval of the data
            
        Returns:
            Combined DataFrame with harmonized dates
        """
        self.logger.info(f"Harmonizing {len(dfs)} dataframes for {interval} interval")
        
        # Convert all date columns to datetime
        for df in dfs:
            df['date'] = pd.to_datetime(df['date'])
            
        # Get the common date range
        min_date = max(df['date'].min() for df in dfs)
        max_date = min(df['date'].max() for df in dfs)
        
        self.logger.info(f"Common date range: {min_date.date()} to {max_date.date()}")
        
        # Filter each dataframe to common range and sort
        harmonized_dfs = []
        for df in dfs:
            mask = (df['date'] >= min_date) & (df['date'] <= max_date)
            harmonized_df = df[mask].sort_values('date')
            harmonized_dfs.append(harmonized_df)
            
        # Combine all dataframes
        result = pd.concat(harmonized_dfs, axis=0)
        result = result.sort_values(['date', 'ticker'])
        
        self.logger.info(f"Final harmonized shape: {result.shape}")
        return result

    def combine_price_files(self, interval: str) -> str:
        """
        Combine all price files for a given interval into one harmonized file.
        
        Args:
            interval: Time interval to combine
            
        Returns:
            Path to combined file
        """
        pattern = f"*_prices_{interval}_*.tsv"
        price_files = list(self.data_dir.glob(pattern))
        
        if not price_files:
            self.logger.warning(f"No price files found for interval {interval}")
            return None
            
        dfs = []
        for file in price_files:
            try:
                df = pd.read_csv(file, sep='\t')
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Failed to read {file}: {str(e)}")
                
        if not dfs:
            self.logger.error("No valid dataframes to combine")
            return None
            
        # Harmonize and combine
        combined_df = self._harmonize_dates(dfs, interval)
        
        # Save combined file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.data_dir / f"combined_prices_{interval}_{timestamp}.tsv"
        combined_df.to_csv(output_file, sep='\t', index=False)
        
        self.logger.info(f"Saved combined file to {output_file}")
        return str(output_file)

    def process_all_tickers(self, intervals: List[str] = None):
        """
        Process all tickers from CSV in parallel.
        
        Args:
            intervals: List of time intervals to process
        """
        if intervals is None:
            intervals = ["1d", "1wk", "1mo"]
            
        tickers = self._get_tickers_from_csv()
        total_tickers = len(tickers)
        processed_count = 0
        
        self.logger.info(f"Starting parallel processing of {total_tickers} tickers")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.process_single_ticker, ticker, intervals): ticker 
                for ticker in tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                processed_count += 1
                
                try:
                    results = future.result()
                    self.logger.info(
                        f"Processed {ticker} ({processed_count}/{total_tickers})"
                        f" - Success for intervals: {list(results.keys())}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to process {ticker}: {str(e)}")
                    
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Completed processing {total_tickers} tickers in {elapsed_time:.2f} seconds"
        )
        
        # Combine files for each interval
        self.logger.info("Starting file combination phase")
        combined_files = {}
        
        for interval in intervals:
            self.logger.info(f"Combining files for {interval} interval")
            combined_file = self.combine_price_files(interval)
            if combined_file:
                combined_files[interval] = combined_file
                
        self.logger.info("Processing completed successfully")
        return combined_files

def main():
    """Main execution function."""
    processor = HistoricalDataProcessor()
    intervals = ["1d", "1wk", "1mo"]
    
    print("\n=== Starting Historical Data Processing ===")
    print(f"Data directory: {processor.data_dir}")
    print(f"Processing intervals: {intervals}")
    print(f"Max parallel workers: {processor.max_workers}")
    
    try:
        combined_files = processor.process_all_tickers(intervals)
        
        print("\n=== Processing Complete ===")
        print("\nCombined files created:")
        for interval, filepath in combined_files.items():
            print(f"\n{interval} interval:")
            print(f"File: {filepath}")
            
            # Display sample statistics
            df = pd.read_csv(filepath, sep='\t')
            print(f"Total rows: {len(df):,}")
            print(f"Unique tickers: {df['ticker'].nunique():,}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
