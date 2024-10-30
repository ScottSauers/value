import finagg
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings

# Filter out yfinance FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

class PriceDataExtractor:
    """Extracts historical price data using yfinance integration."""

    def __init__(self, output_dir: str = "./data"):
        """Initialize the price data extractor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'price_data.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _preprocess_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Preprocess the price dataframe to ensure consistent format.
        
        Args:
            df: Raw price DataFrame
            ticker: Stock ticker symbol
            
        Returns:
            Processed DataFrame with consistent format
        """
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure all numeric columns are float64
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        
        # Handle volume as int64
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(np.int64)
        
        # Sort by date
        df = df.sort_values('date', ascending=True)
        
        # Remove any duplicates
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        return df

    def get_price_data(self, ticker: str, interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            interval: Time interval for the data
            
        Returns:
            DataFrame containing historical price data
        """
        try:
            # Only try refined data for daily interval
            if interval == "1d":
                try:
                    df = finagg.yfinance.feat.daily.from_refined(ticker)
                    self.logger.info(f"Retrieved refined daily data for {ticker}")
                    return self._preprocess_dataframe(df, ticker)
                except Exception as e:
                    self.logger.debug(f"No refined data available for {ticker}: {str(e)}")

            # Get data from API
            df = finagg.yfinance.api.get(
                ticker,
                interval=interval,
                period="max"  # Get all available historical data
            )
            
            if df.empty:
                raise ValueError(f"No price data found for {ticker}")
            
            df = self._preprocess_dataframe(df, ticker)
            self.logger.info(f"Retrieved API data for {ticker} with interval {interval}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve price data for {ticker}: {str(e)}")
            raise

    def save_data(self, df: pd.DataFrame, ticker: str, interval: str) -> Tuple[str, str]:
        """
        Save the price data and metadata to files.
        
        Args:
            df: DataFrame to save
            ticker: Stock ticker symbol
            interval: Time interval of the data
            
        Returns:
            Tuple of (data filepath, metadata filepath)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare statistics for metadata
        stats = {
            'ticker': ticker,
            'interval': interval,
            'extraction_date': datetime.now().isoformat(),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            },
            'rows': len(df),
            'trading_days': len(df['date'].unique()),
            'price_range': {
                'min': float(df['low'].min()),
                'max': float(df['high'].max())
            },
            'volume_stats': {
                'total': int(df['volume'].sum()),
                'mean': float(df['volume'].mean()),
                'median': float(df['volume'].median())
            }
        }
        
        # Save price data
        data_filename = f"{ticker}_prices_{interval}_{timestamp}.tsv"
        data_filepath = self.output_dir / data_filename
        
        # Convert date to ISO format before saving
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.to_csv(data_filepath, sep='\t', index=False, float_format='%.8f')
        self.logger.info(f"Price data saved to {data_filepath}")
        
        # Save metadata
        metadata_filename = f"{ticker}_prices_{interval}_{timestamp}.json"
        metadata_filepath = self.output_dir / metadata_filename
        pd.Series(stats).to_json(metadata_filepath)
        self.logger.info(f"Metadata saved to {metadata_filepath}")
        
        return str(data_filepath), str(metadata_filepath)

    def process_ticker(self, ticker: str, intervals: Optional[list[str]] = None) -> dict[str, Tuple[str, str]]:
        """
        Process a single ticker to extract and save price data for specified intervals.
        
        Args:
            ticker: Stock ticker symbol
            intervals: List of time intervals to collect. If None, uses daily data only.
            
        Returns:
            Dictionary mapping intervals to (data_filepath, metadata_filepath) tuples
        """
        if intervals is None:
            intervals = ["1d"]  # Default to daily data only
            
        results = {}
        
        for interval in intervals:
            try:
                # Get and process price data
                df = self.get_price_data(ticker, interval)
                
                # Save data and metadata
                data_file, metadata_file = self.save_data(df, ticker, interval)
                results[interval] = (data_file, metadata_file)
                
            except Exception as e:
                self.logger.error(f"Failed to process {ticker} for interval {interval}: {str(e)}")
                continue
                
        return results

def main():
    """Main execution function."""
    extractor = PriceDataExtractor()
    
    # Configure intervals - stick to the most reliable ones
    intervals = ["1d", "1wk", "1mo"]
    
    # List of tickers to process
    tickers = ["AAPL"]  # Expand this list as needed
    
    for ticker in tickers:
        try:
            results = extractor.process_ticker(ticker, intervals)
            print(f"\nSuccessfully processed {ticker}:")
            for interval, (data_file, metadata_file) in results.items():
                print(f"\nInterval: {interval}")
                print(f"Data file: {data_file}")
                print(f"Metadata file: {metadata_file}")
        except Exception as e:
            print(f"\nFailed to process {ticker}: {str(e)}")

if __name__ == "__main__":
    main()
