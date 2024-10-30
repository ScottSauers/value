import concurrent.futures
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Optional
from fundamentals import SECDataExtractor

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'parallel_fundamentals.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_ticker(ticker: str, output_dir: Path) -> dict:
    """Process a single ticker and return results."""
    try:
        extractor = SECDataExtractor(str(output_dir))
        data_file, metadata_file = extractor.process_ticker(ticker)
        return {
            'ticker': ticker,
            'status': 'success',
            'data_file': data_file,
            'metadata_file': metadata_file
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'status': 'failed',
            'error': str(e)
        }

def parallel_process_tickers(
    input_file: str,
    output_dir: str,
    max_workers: Optional[int] = None,
    ticker_column: str = 'ticker'
) -> pd.DataFrame:
    """
    Process multiple tickers in parallel to collect SEC fundamental data.
    
    Args:
        input_file: Path to CSV file containing ticker symbols
        output_dir: Directory to save output files
        max_workers: Maximum number of parallel workers (None for CPU count)
        ticker_column: Name of column containing ticker symbols
    
    Returns:
        DataFrame with processing results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_path)
    
    # Read ticker list
    try:
        df = pd.read_csv(input_file)
        tickers = df[ticker_column].unique().tolist()
        logger.info(f"Found {len(tickers)} unique tickers to process")
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        raise
    
    # Process tickers in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_ticker, ticker, output_path): ticker 
            for ticker in tickers
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed processing {ticker}: {result['status']}")
            except Exception as e:
                logger.error(f"Exception processing {ticker}: {str(e)}")
                results.append({
                    'ticker': ticker,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Save summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = output_path / f'processing_summary_{timestamp}.csv'
    results_df.to_csv(summary_file, index=False)
    logger.info(f"Processing summary saved to {summary_file}")
    
    return results_df

def main():
    """Main execution function."""
    # Configuration
    INPUT_FILE = 'small_cap_stocks_latest.csv'
    OUTPUT_DIR = './data/fundamentals'
    MAX_WORKERS = None  # None will use CPU count
    TICKER_COLUMN = 'ticker'
    
    try:
        results = parallel_process_tickers(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            max_workers=MAX_WORKERS,
            ticker_column=TICKER_COLUMN
        )
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total tickers processed: {len(results)}")
        print(f"Successful: {len(results[results['status'] == 'success'])}")
        print(f"Failed: {len(results[results['status'] == 'failed'])}")
        
    except Exception as e:
        print(f"Failed to complete processing: {str(e)}")

if __name__ == "__main__":
    main()
