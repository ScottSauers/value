import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import logging
from datetime import datetime, timedelta
import time
import os
from tqdm import tqdm

class DataTransformer:
    """Transform price data from long to wide format."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "transformed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'transform_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_latest_file(self, pattern: str) -> Path:
        """Get most recent file matching pattern, excluding symlinks."""
        files = [f for f in self.data_dir.glob(pattern) if not f.is_symlink()]
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)

    def _read_price_file(self, file_path: Path) -> pd.DataFrame:
        """Read price file, handling both gzipped and uncompressed files."""
        self.logger.info(f"Reading {file_path}")
        
        if str(file_path).endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, sep='\t')
        else:
            df = pd.read_csv(file_path, sep='\t')
            
        self.logger.info(f"Read {len(df):,} rows, columns: {', '.join(df.columns)}")
        return df

    def _transform_to_wide(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Transform data from long to wide format with progress reporting."""
        self.logger.info("Starting wide format transformation")
        start_time = time.time()
        
        # Convert date to datetime if not already
        df['date'] = pd.to_datetime(df['date'])
        
        # Get unique dates and tickers
        dates = sorted(df['date'].unique())
        tickers = sorted(df['ticker'].unique())
        
        self.logger.info(f"Processing {len(dates):,} dates and {len(tickers):,} tickers")
        
        # Create empty result DataFrame with date as index
        wide_df = pd.DataFrame(index=pd.DatetimeIndex(dates))
        
        # Process each ticker with progress bar
        for ticker in tqdm(tickers, desc="Processing tickers"):
            # Filter data for current ticker
            ticker_data = df[df['ticker'] == ticker]
            
            # Set date as index for easy alignment
            ticker_data = ticker_data.set_index('date')
            
            # Add columns for each metric
            for col in ['close', 'open', 'high', 'low', 'volume']:
                col_name = f"{ticker}_{col}"
                wide_df[col_name] = ticker_data[col]
            
            if len(ticker_data) % 100 == 0:
                self.logger.info(
                    f"Added {ticker} data: {len(ticker_data):,} rows"
                )
        
        # Reset index to make date a regular column
        wide_df = wide_df.reset_index()
        wide_df.rename(columns={'index': 'date'}, inplace=True)
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"Transformation completed in {elapsed:.1f} seconds. "
            f"Shape: {wide_df.shape}"
        )
        
        return wide_df

    def _save_transformed(self, df: pd.DataFrame, interval: str):
        """Save transformed data as CSV and Parquet."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file_parquet = self.output_dir / f"price_data_{interval}_{timestamp}.parquet"
        output_file_csv = self.output_dir / f"price_data_{interval}_{timestamp}.csv"
        
        self.logger.info(f"Saving to {output_file_parquet} and {output_file_csv}")
        df.to_parquet(output_file_parquet)
        df.to_csv(output_file_csv, index=False, sep='\t')
        
        # Create symlink to latest Parquet file
        latest_link = self.output_dir / f"price_data_{interval}_latest.parquet"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(output_file_parquet.name)
        
        self.logger.info("Save completed")
        return output_file_csv, output_file_parquet

    def _get_latest_parquet_file_within_24h(self, interval: str) -> Path:
        """Check if a recent Parquet file exists within the last 24 hours."""
        latest_file = self._get_latest_file(f"price_data_{interval}_*.parquet")
        if latest_file:
            file_mtime = datetime.fromtimestamp(latest_file.stat().st_mtime)
            if (datetime.now() - file_mtime) < timedelta(hours=24):
                self.logger.info(f"Found recent Parquet file: {latest_file} (modified {file_mtime})")
                return latest_file
            else:
                self.logger.info(f"No Parquet file within last 24 hours for interval {interval}.")
        else:
            self.logger.info(f"No Parquet files found for interval {interval}.")
        return None

    def process_interval(self, interval: str):
        """Process data for a specific interval (e.g., '1d', '1wk')."""
        try:
            # Check for recent processed Parquet file
            recent_file = self._get_latest_parquet_file_within_24h(interval)
            if recent_file:
                self.logger.info(f"Using existing file: {recent_file}")
                print(f"Using existing file: {recent_file}")
                return

            # Find latest price file
            price_pattern = f"combined_prices_{interval}_*.tsv*"
            latest_price = self._get_latest_file(price_pattern)
            
            if not latest_price:
                raise FileNotFoundError(f"No price file found for interval {interval}")
            
            # Read and transform data
            df = self._read_price_file(latest_price)
            wide_df = self._transform_to_wide(df, interval)
            
            # Save results
            output_file_csv, output_file_parquet = self._save_transformed(wide_df, interval)
            
            # Print summary
            print(f"\nProcessed {interval} data:")
            print(f"Input rows: {len(df):,}")
            print(f"Output shape: {wide_df.shape}")
            print(f"Unique tickers: {len(df['ticker'].unique()):,}")
            print(f"Date range: {wide_df['date'].min()} to {wide_df['date'].max()}")
            print(f"Saved to: {output_file_csv}, {output_file_parquet}")
            
        except Exception as e:
            print(f"Error processing {interval} interval: {str(e)}")
            self.logger.error(f"Error processing {interval} interval: {str(e)}")
            raise

def main():
    processor = DataTransformer()
    
    print("\n=== Starting Processing ===")
    
    for interval in ['1d', '1wk']:
        print(f"\nProcessing {interval} interval data:")
        processor.process_interval(interval)

if __name__ == "__main__":
    main()
