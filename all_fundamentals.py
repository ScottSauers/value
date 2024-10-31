import os
import concurrent.futures
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Optional
import finagg
from dotenv import load_dotenv

def setup_environment():
    """Set up environment variables and configurations."""
    load_dotenv()
    
    # Check if SEC_API_USER_AGENT is properly configured
    sec_user_agent = os.getenv('SEC_API_USER_AGENT')
    if not sec_user_agent or sec_user_agent == "Your Name your-email@example.com":
        raise EnvironmentError(
            "SEC_API_USER_AGENT is not properly configured. "
            "Please update your .env file with your actual name and email"
        )
    
    # Configure finagg with SEC user agent
    finagg.sec.api.company_concept.SEC_USER_AGENT = sec_user_agent

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
        from fundamentals import SECDataExtractor  # Import here to avoid circular imports
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
    ticker_column: str = 'ticker',
    batch_size: int = 100  # Process tickers in batches
) -> pd.DataFrame:
    """Process multiple tickers in parallel to collect SEC fundamental data."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup environment and logging
    setup_environment()
    logger = setup_logging(output_path)
    
    # Read ticker list
    try:
        df = pd.read_csv(input_file)
        tickers = df[ticker_column].unique().tolist()
        logger.info(f"Found {len(tickers)} unique tickers to process")
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        raise
    
    # Process tickers in batches
    results = []
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch_tickers)} tickers)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(process_ticker, ticker, output_path): ticker 
                for ticker in batch_tickers
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
    
    # Print summary statistics
    success_count = len(results_df[results_df['status'] == 'success'])
    fail_count = len(results_df[results_df['status'] == 'failed'])
    logger.info(f"\nProcessing Summary:")
    logger.info(f"Total tickers processed: {len(results_df)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    
    return results_df

def main():
    """Main execution function."""
    # Configuration
    INPUT_FILE = 'small_cap_stocks_latest.csv'
    OUTPUT_DIR = './data/fundamentals'
    MAX_WORKERS = None  # None will use CPU count
    TICKER_COLUMN = 'ticker'
    BATCH_SIZE = 100
    
    try:
        results = parallel_process_tickers(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            max_workers=MAX_WORKERS,
            ticker_column=TICKER_COLUMN,
            batch_size=BATCH_SIZE
        )
        
    except Exception as e:
        print(f"Failed to complete processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
