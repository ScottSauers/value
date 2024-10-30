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
    
    def _get_latest_file(self, ticker: str, interval: str) -> Path:
        """Get the most recent file for a ticker and interval."""
        pattern = f"{ticker}_prices_{interval}_*.tsv"
        files = list(self.data_dir.glob(pattern))
        return max(files, key=lambda x: x.stat().st_mtime) if files else None

    def _is_cache_fresh(self, ticker: str, interval: str) -> tuple[bool, Path]:
        """
        Check if cached data for a ticker is fresh (less than 24 hours old).
        Returns (is_fresh, latest_file_path)
        """
        latest_file = self._get_latest_file(ticker, interval)
        
        if not latest_file:
            return False, None
            
        modified_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
        is_fresh = (datetime.now() - modified_time) < timedelta(hours=24)
        
        # Verify file has content
        if is_fresh:
            try:
                df = pd.read_csv(latest_file, sep='\t')
                if len(df) == 0:
                    self.logger.warning(f"Cache file for {ticker} ({interval}) is empty")
                    return False, None
                return True, latest_file
            except Exception:
                return False, None
                
        return False, None

    def process_single_ticker(self, ticker: str, intervals: List[str]) -> Dict:
        """Process a single ticker, using cache if available."""
        results = {}
        
        for interval in intervals:
            is_fresh, latest_file = self._is_cache_fresh(ticker, interval)
            
            if is_fresh and latest_file:
                self.logger.info(f"Using cached data for {ticker} ({interval})")
                results[interval] = (str(latest_file), '')
            else:
                try:
                    self.logger.info(f"Fetching fresh data for {ticker} ({interval})")
                    interval_results = self.extractor.process_ticker(ticker, [interval])
                    if interval_results:
                        results.update(interval_results)
                    else:
                        self.logger.error(f"No data returned for {ticker} ({interval})")
                except Exception as e:
                    self.logger.error(f"Failed to process {ticker} ({interval}): {str(e)}")
                    
        return results

    def _get_tickers_from_csv(self) -> List[str]:
        """Get list of tickers from small_cap_stocks_latest.csv"""
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
        Harmonize dates across multiple dataframes with a more flexible approach.
        Now keeps all dates and fills missing data with NaN.
        """
        self.logger.info(f"Harmonizing {len(dfs)} dataframes for {interval} interval")
        
        # Convert all date columns to datetime
        for df in dfs:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # Get all unique dates
        all_dates = sorted(set().union(*[set(df['date'].unique()) for df in dfs]))
        self.logger.info(f"Date range: {min(all_dates)} to {max(all_dates)}")
        
        # Create a multi-index with all combinations of dates and tickers
        tickers = [df['ticker'].iloc[0] for df in dfs]  # Assuming each df has one ticker
        multi_index = pd.MultiIndex.from_product([all_dates, tickers], names=['date', 'ticker'])
        
        # Initialize empty dataframe with all date-ticker combinations
        columns = ['open', 'high', 'low', 'close', 'volume']
        result = pd.DataFrame(index=multi_index, columns=columns)
        
        # Fill in data from each dataframe
        for df in dfs:
            ticker = df['ticker'].iloc[0]
            df_indexed = df.set_index(['date', 'ticker'])
            result.loc[pd.IndexSlice[:, ticker], :] = df_indexed[columns]
        
        # Reset index and sort
        result = result.reset_index()
        result = result.sort_values(['date', 'ticker'])
        
        self.logger.info(f"Final harmonized shape: {result.shape}")
        return result

    def combine_price_files(self, interval: str) -> str:
        """Combine all price files for a given interval into one harmonized file."""
        pattern = f"*_prices_{interval}_*.tsv"
        price_files = list(self.data_dir.glob(pattern))
        
        if not price_files:
            self.logger.warning(f"No price files found for interval {interval}")
            return None
            
        dfs = []
        for file in price_files:
            try:
                df = pd.read_csv(file, sep='\t')
                if len(df) > 0:  # Only include non-empty dataframes
                    dfs.append(df)
                else:
                    self.logger.warning(f"Skipping empty file: {file}")
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
        combined_df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
        
        self.logger.info(f"Saved combined file to {output_file}")
        return str(output_file)

    def process_all_tickers(self, intervals: List[str] = None):
        """Process all tickers from CSV in parallel."""
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
                    if results:
                        self.logger.info(
                            f"Processed {ticker} ({processed_count}/{total_tickers})"
                            f" - Success for intervals: {list(results.keys())}"
                        )
                    else:
                        self.logger.warning(
                            f"No data obtained for {ticker} ({processed_count}/{total_tickers})"
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
            print(f"Data completeness: {(df['close'].notna().sum() / len(df)) * 100:.1f}%")
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
