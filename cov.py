import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import logging
from datetime import datetime
import time
import os
from tqdm import tqdm

class CovProcessor:
    """Process and transform price data into wide format with coverage analysis."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'coverage_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_latest_file(self, pattern: str) -> Path:
        """Get most recent file matching pattern."""
        files = list(self.data_dir.glob(pattern))
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
            
        self.logger.info(f"Read {len(df):,} rows")
        return df

    def _transform_to_wide(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Transform data from long to wide format with progress reporting.
        Creates columns like AAPL_close, AAPL_volume, etc.
        """
        self.logger.info("Starting wide format transformation")
        start_time = time.time()
        
        # Get unique dates and tickers for progress reporting
        dates = df['date'].unique()
        tickers = df['ticker'].unique()
        self.logger.info(f"Processing {len(dates):,} dates and {len(tickers):,} tickers")
        
        # Initialize with date column
        result_df = pd.DataFrame(index=dates)
        result_df.index.name = 'date'
        result_df = result_df.sort_index()
        
        # Track progress
        processed = 0
        total_tickers = len(tickers)
        
        for ticker in tqdm(tickers, desc="Processing tickers"):
            # Filter for current ticker
            ticker_data = df[df['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                self.logger.warning(f"No data for {ticker}")
                continue
                
            # Create ticker-specific columns
            for col in ['close', 'open', 'high', 'low', 'volume']:
                new_col = f"{ticker}_{col}"
                ticker_data.set_index('date', inplace=True)
                result_df[new_col] = ticker_data[col]
            
            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = (total_tickers - processed) / rate if rate > 0 else 0
                self.logger.info(
                    f"Processed {processed}/{total_tickers} tickers "
                    f"({processed/total_tickers*100:.1f}%) "
                    f"- Est. remaining time: {remaining/60:.1f} minutes"
                )
        
        self.logger.info(f"Transformation completed in {time.time() - start_time:.1f} seconds")
        return result_df.reset_index()

    def _save_with_cache(self, df: pd.DataFrame, interval: str):
        """Save transformed data with caching."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cache_file = self.cache_dir / f"wide_format_{interval}_{timestamp}.parquet"
        
        self.logger.info(f"Saving to {cache_file}")
        df.to_parquet(cache_file)
        
        # Create symlink to latest
        latest_link = self.cache_dir / f"wide_format_{interval}_latest.parquet"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(cache_file.name)
        
        self.logger.info("Save completed")
        return cache_file

    def process_interval(self, interval: str) -> pd.DataFrame:
        """
        Process data for a specific interval (e.g., '1d', '1wk').
        Handles caching and returns transformed DataFrame.
        """
        # Check cache first
        cache_pattern = f"wide_format_{interval}_*.parquet"
        latest_cache = self._get_latest_file(str(self.cache_dir / cache_pattern))
        
        if latest_cache and (datetime.now() - datetime.fromtimestamp(latest_cache.stat().st_mtime)).days < 1:
            self.logger.info(f"Using cached data from {latest_cache}")
            return pd.read_parquet(latest_cache)
        
        # Find latest price file
        price_pattern = f"combined_prices_{interval}_*.tsv*"
        latest_price = self._get_latest_file(price_pattern)
        
        if not latest_price:
            raise FileNotFoundError(f"No price file found for interval {interval}")
            
        # Read and transform data
        df = self._read_price_file(latest_price)
        wide_df = self._transform_to_wide(df, interval)
        
        # Cache results
        self._save_with_cache(wide_df, interval)
        
        return wide_df

    def analyze_coverage(self, df: pd.DataFrame) -> dict:
        """Analyze data coverage and completeness."""
        self.logger.info("Analyzing data coverage")
        
        # Get all ticker columns (exclude date)
        ticker_cols = [col for col in df.columns if col != 'date']
        unique_tickers = len(set(col.split('_')[0] for col in ticker_cols))
        
        # Calculate coverage stats
        stats = {
            'total_rows': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'unique_tickers': unique_tickers,
            'total_columns': len(ticker_cols),
            'missing_values': df[ticker_cols].isna().sum().sum(),
            'coverage_pct': (1 - df[ticker_cols].isna().sum().sum() / (len(df) * len(ticker_cols))) * 100
        }
        
        self.logger.info("\nCoverage Analysis:")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
            
        return stats

def main():
    processor = CovProcessor()
    
    print("\n=== Starting Processing ===")
    
    for interval in ['1d', '1wk']:
        print(f"\nProcessing {interval} interval data:")
        try:
            start_time = time.time()
            df = processor.process_interval(interval)
            stats = processor.analyze_coverage(df)
            
            print(f"\n{interval.upper()} Interval Statistics:")
            print(f"Total Rows: {stats['total_rows']:,}")
            print(f"Date Range: {stats['date_range']}")
            print(f"Unique Tickers: {stats['unique_tickers']:,}")
            print(f"Total Columns: {stats['total_columns']:,}")
            print(f"Data Coverage: {stats['coverage_pct']:.1f}%")
            print(f"Processing Time: {time.time() - start_time:.1f} seconds")
            
        except Exception as e:
            print(f"Error processing {interval} interval: {str(e)}")
            continue

if __name__ == "__main__":
    main()
