import os
import concurrent.futures
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import finagg
from dotenv import load_dotenv
from fundamentals import SECDataExtractor
import time
import hashlib
import sqlite3
from tqdm import tqdm
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.logging import RichHandler

class CacheManager:
    """Manages caching of processed tickers using SQLite."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / 'ticker_cache.db'
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with necessary tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processed_tickers (
                    ticker TEXT PRIMARY KEY,
                    last_processed TIMESTAMP,
                    status TEXT,
                    data_file TEXT,
                    metadata_file TEXT,
                    error TEXT,
                    data_hash TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_stats (
                    batch_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_tickers INTEGER,
                    successful INTEGER,
                    failed INTEGER
                )
            ''')

    def get_cached_result(self, ticker: str) -> Optional[Dict]:
        """Retrieve cached result for a ticker if it exists and is recent."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'SELECT * FROM processed_tickers WHERE ticker = ? AND last_processed > datetime("now", "-7 days")',
                (ticker,)
            )
            result = cursor.fetchone()
            
        if result:
            return {
                'ticker': result[0],
                'last_processed': result[1],
                'status': result[2],
                'data_file': result[3],
                'metadata_file': result[4],
                'error': result[5],
                'data_hash': result[6]
            }
        return None

    def cache_result(self, ticker: str, result: Dict):
        """Cache the processing result for a ticker."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                '''
                INSERT OR REPLACE INTO processed_tickers 
                (ticker, last_processed, status, data_file, metadata_file, error, data_hash)
                VALUES (?, datetime('now'), ?, ?, ?, ?, ?)
                ''',
                (
                    ticker,
                    result['status'],
                    result.get('data_file'),
                    result.get('metadata_file'),
                    result.get('error'),
                    result.get('data_hash')
                )
            )

    def save_batch_stats(self, batch_id: str, stats: Dict):
        """Save statistics for a processing batch."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                '''
                INSERT OR REPLACE INTO processing_stats 
                (batch_id, start_time, end_time, total_tickers, successful, failed)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    batch_id,
                    stats['start_time'],
                    stats['end_time'],
                    stats['total_tickers'],
                    stats['successful'],
                    stats['failed']
                )
            )

class ProgressTracker:
    """Tracks and displays processing progress using rich console."""
    
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        )
        self.total_task = None
        self.current_batch_task = None

    def init_progress(self, total_tickers: int):
        """Initialize progress bars."""
        self.progress.start()
        self.total_task = self.progress.add_task(
            "[blue]Overall Progress", total=total_tickers
        )
        return self.progress

    def update_progress(self, completed: int, batch_completed: int = None):
        """Update progress bars."""
        self.progress.update(self.total_task, completed=completed)
        if batch_completed is not None and self.current_batch_task:
            self.progress.update(self.current_batch_task, completed=batch_completed)

    def new_batch(self, batch_size: int):
        """Create a new progress bar for the current batch."""
        if self.current_batch_task:
            self.progress.remove_task(self.current_batch_task)
        self.current_batch_task = self.progress.add_task(
            "[green]Current Batch", total=batch_size
        )

    def finish(self):
        """Complete progress tracking."""
        self.progress.stop()

def setup_environment():
    """Set up environment variables and configurations."""
    load_dotenv()
    
    sec_user_agent = os.getenv('SEC_API_USER_AGENT')
    if not sec_user_agent or sec_user_agent == "Your Name your-email@example.com":
        raise EnvironmentError(
            "SEC_API_USER_AGENT is not properly configured. "
            "Please update your .env file with your actual name and email"
        )
    
    finagg.sec.api.company_concept.SEC_USER_AGENT = sec_user_agent

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up rich logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(output_dir / 'parallel_fundamentals.log')
        ]
    )
    return logging.getLogger(__name__)

def calculate_data_hash(data_file: Path) -> str:
    """Calculate SHA-256 hash of data file for cache validation."""
    if not data_file.exists():
        return ""
    
    sha256_hash = hashlib.sha256()
    with open(data_file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def process_ticker_batch(
    tickers: List[str],
    output_dir: Path,
    cache_manager: CacheManager,
    progress_tracker: ProgressTracker
) -> List[dict]:
    """Process a batch of tickers with improved caching and progress tracking."""
    results = []
    extractor = SECDataExtractor(str(output_dir))
    
    progress_tracker.new_batch(len(tickers))
    completed_in_batch = 0
    
    for ticker in tickers:
        try:
            # Check cache first
            cached_result = cache_manager.get_cached_result(ticker)
            if cached_result:
                # Validate cache with hash if data file exists
                if cached_result['data_file']:
                    data_file = Path(cached_result['data_file'])
                    if data_file.exists():
                        current_hash = calculate_data_hash(data_file)
                        if current_hash == cached_result['data_hash']:
                            results.append(cached_result)
                            completed_in_batch += 1
                            progress_tracker.update_progress(None, completed_in_batch)
                            continue
            
            # Add delay between requests to respect rate limits
            time.sleep(0.1)
            
            # Process ticker with retries
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    data_file, metadata_file = extractor.process_ticker(ticker)
                    
                    result = {
                        'ticker': ticker,
                        'status': 'success',
                        'data_file': str(data_file),
                        'metadata_file': str(metadata_file),
                        'data_hash': calculate_data_hash(data_file)
                    }
                    
                    # Cache successful result
                    cache_manager.cache_result(ticker, result)
                    results.append(result)
                    
                    logging.info(f"✅ Successfully processed {ticker}")
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        error_result = {
                            'ticker': ticker,
                            'status': 'failed',
                            'error': str(e)
                        }
                        cache_manager.cache_result(ticker, error_result)
                        results.append(error_result)
                        logging.error(f"❌ Failed to process {ticker}: {str(e)}")
                    else:
                        time.sleep(retry_delay * (2 ** attempt))
                        continue
                    
        except Exception as e:
            error_result = {
                'ticker': ticker,
                'status': 'failed',
                'error': str(e)
            }
            cache_manager.cache_result(ticker, error_result)
            results.append(error_result)
            logging.error(f"❌ Failed to process {ticker}: {str(e)}")
        
        completed_in_batch += 1
        progress_tracker.update_progress(None, completed_in_batch)
            
    return results

def parallel_process_tickers(
    input_file: str,
    output_dir: str,
    max_workers: Optional[int] = None,
    ticker_column: str = 'ticker',
    batch_size: int = 50
) -> pd.DataFrame:
    """Process multiple tickers in parallel with enhanced progress tracking and caching."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup environment, logging, and managers
    setup_environment()
    logger = setup_logging(output_path)
    cache_manager = CacheManager(output_path / 'cache')
    progress_tracker = ProgressTracker()
    
    # Generate batch ID
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Read ticker list
    try:
        df = pd.read_csv(input_file)
        tickers = df[ticker_column].unique().tolist()
        logger.info(f"📋 Found {len(tickers)} unique tickers to process")
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        raise
    
    # Split tickers into batches
    ticker_batches = [
        tickers[i:i + batch_size] 
        for i in range(0, len(tickers), batch_size)
    ]
    
    # Initialize progress tracking
    progress_tracker.init_progress(len(tickers))
    
    # Process batches in parallel
    start_time = datetime.now()
    results = []
    completed_total = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit batch processing jobs
        future_to_batch = {
            executor.submit(
                process_ticker_batch,
                batch,
                output_path,
                cache_manager,
                progress_tracker
            ): batch 
            for batch in ticker_batches
        }
        
        # Collect results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_batch)):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
                completed_total += len(batch)
                progress_tracker.update_progress(completed_total)
                
                # Calculate and display completion percentage
                completion_percentage = (completed_total / len(tickers)) * 100
                logger.info(
                    f"📊 Completed batch {i+1}/{len(ticker_batches)} "
                    f"({completion_percentage:.1f}% overall)"
                )
                
            except Exception as e:
                logger.error(f"Exception processing batch: {str(e)}")
                for ticker in batch:
                    error_result = {
                        'ticker': ticker,
                        'status': 'failed',
                        'error': str(e)
                    }
                    cache_manager.cache_result(ticker, error_result)
                    results.append(error_result)
                completed_total += len(batch)
                progress_tracker.update_progress(completed_total)
    
    # Complete progress tracking
    progress_tracker.finish()
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Save summary
    summary_file = output_path / f'processing_summary_{batch_id}.csv'
    results_df.to_csv(summary_file, index=False)
    
    # Calculate final statistics
    end_time = datetime.now()
    success_count = len(results_df[results_df['status'] == 'success'])
    fail_count = len(results_df[results_df['status'] == 'failed'])
    
    # Save batch statistics
    batch_stats = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_tickers': len(results_df),
        'successful': success_count,
        'failed': fail_count
    }
    cache_manager.save_batch_stats(batch_id, batch_stats)
    
    # Print final summary
    logger.info("\n📋 Processing Summary:")
    logger.info(f"Total tickers processed: {len(results_df)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed: {fail_count}")
    logger.info(f"⏱️ Total time: {end_time - start_time}")
    logger.info(f"📁 Summary saved to: {summary_file}")
    
    return results_df

def main():
    """Main execution function with configuration."""
    # Configuration
    INPUT_FILE = 'small_cap_stocks_latest.csv'
    OUTPUT_DIR = './data/fundamentals'
    MAX_WORKERS = 4
    TICKER_COLUMN = 'ticker'
    BATCH_SIZE = 50
    
    try:
        console = Console()
        console.print("\n[bold blue]🚀 Starting SEC Data Processing[/bold blue]\n")
        
        results = parallel_process_tickers(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            max_workers=MAX_WORKERS,
            ticker_column=TICKER_COLUMN,
            batch_size=BATCH_SIZE
        )
        
        # Calculate success rate
        success_rate = (len(results[results['status'] == 'success']) / len(results)) * 100
        
        # Display final summary with rich formatting
        console.print("\n[bold green]✨ Processing Complete![/bold green]")
        console.print("\n[bold]Final Statistics:[/bold]")
        console.print(f"📊 Success Rate: [green]{success_rate:.1f}%[/green]")
        console.print(f"📁 Output Directory: [blue]{OUTPUT_DIR}[/blue]")
        
        # Display error summary if there were failures
        failed_results = results[results['status'] == 'failed']
        if not failed_results.empty:
            console.print("\n[bold red]⚠️ Failed Tickers:[/bold red]")
            for _, row in failed_results.iterrows():
                console.print(f"❌ {row['ticker']}: {row['error']}")
                
        console.print("\n[bold green]🎉 Process completed successfully![/bold green]\n")
        
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Process interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Fatal error: {str(e)}[/bold red]")
        console.print_exception()
        sys.exit(1)

if __name__ == "__main__":
    main()
