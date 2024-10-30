import finagg
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class TimeInterval:
    """Represents a time interval for price data collection."""
    interval: str
    description: str

class PriceDataExtractor:
    """Extracts historical price data using multiple sources and intervals."""
    
    # Define available time intervals, fix later
  
    INTERVALS = [
        TimeInterval("1d", "Daily data"),
        TimeInterval("1wk", "Weekly data"),
        TimeInterval("1mo", "Monthly data"),
        TimeInterval("1h", "Hourly data"),
        TimeInterval("5m", "5-minute data"),
        TimeInterval("2m", "2-minute data"),
        TimeInterval("1m", "1-minute data")
    ]

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
            # First try the refined data
            try:
                if interval == "1d":
                    df = finagg.yfinance.feat.daily.from_refined(ticker)
                    self.logger.info(f"Retrieved refined daily data for {ticker}")
                    return df
            except Exception as e:
                self.logger.warning(f"Failed to get refined data for {ticker}: {str(e)}")

            # Then try raw data
            try:
                df = finagg.yfinance.feat.prices.from_raw(ticker)
                self.logger.info(f"Retrieved raw data for {ticker}")
                return df
            except Exception as e:
                self.logger.warning(f"Failed to get raw data for {ticker}: {str(e)}")

            # Finally, try API directly
            df = finagg.yfinance.api.get(
                ticker,
                interval=interval,
                period="max"  # Get all available historical data
            )
            
            if df.empty:
                raise ValueError(f"No price data found for {ticker}")
                
            self.logger.info(f"Retrieved API data for {ticker} with interval {interval}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve price data for {ticker}: {str(e)}")
            raise

    def save_data(self, df: pd.DataFrame, ticker: str, interval: str, metadata: Dict) -> Tuple[str, str]:
        """
        Save the price data and metadata to files.
        
        Args:
            df: DataFrame to save
            ticker: Stock ticker symbol
            interval: Time interval of the data
            metadata: Dictionary of metadata about the extraction
            
        Returns:
            Tuple of (data filepath, metadata filepath)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save price data
        data_filename = f"{ticker}_prices_{interval}_{timestamp}.tsv"
        data_filepath = self.output_dir / data_filename
        df.to_csv(data_filepath, sep='\t', index=False)
        self.logger.info(f"Price data saved to {data_filepath}")
        
        # Save metadata
        metadata_filename = f"{ticker}_prices_{interval}_{timestamp}.json"
        metadata_filepath = self.output_dir / metadata_filename
        pd.Series(metadata).to_json(metadata_filepath)
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
                # Get price data
                df = self.get_price_data(ticker, interval)
                
                # Prepare metadata
                metadata = {
                    'ticker': ticker,
                    'interval': interval,
                    'extraction_date': datetime.now().isoformat(),
                    'start_date': df['date'].min(),
                    'end_date': df['date'].max(),
                    'total_rows': len(df),
                    'columns': list(df.columns),
                    'data_sources_tried': [
                        'refined' if interval == "1d" else None,
                        'raw',
                        'api'
                    ]
                }
                
                # Save data and metadata
                data_file, metadata_file = self.save_data(df, ticker, interval, metadata)
                results[interval] = (data_file, metadata_file)
                
            except Exception as e:
                self.logger.error(f"Failed to process {ticker} for interval {interval}: {str(e)}")
                continue
                
        return results

def main():
    """Main execution function."""
    extractor = PriceDataExtractor()
    
    # List of tickers to process
    tickers = ["AAPL"]
    
    # List of intervals to collect
    intervals = ["1d", "1wk", "1mo"]
    
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
