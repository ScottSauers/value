# all_fundamentals.py

import os
import concurrent.futures
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import finagg
from dotenv import load_dotenv
from fundamentals import SECDataExtractor, CacheManager
import time
import hashlib
import sqlite3
from tqdm import tqdm
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.logging import RichHandler
import random
import gc
import psutil
import threading

class ProgressTracker:
    """Tracks and displays processing progress using rich console."""
    
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "•",
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
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

    def update_progress(self, completed: Optional[int], batch_completed: Optional[int] = None):
        """Update progress bars."""
        if completed is not None and self.total_task:
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
        try:
            if self.progress:
                self.progress.stop()
                self.progress = None
        except:
            pass  # Don't hang if progress bar is already stopped
        finally:
            # Aggressively clear all references
            self.total_task = None
            self.current_batch_task = None
            self.console = None

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
            logging.FileHandler(output_dir / 'all_fundamentals.log')
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

MAX_RETRIES = 3
BASE_RETRY_DELAY = 1

def process_ticker_batch(tickers: List[str], output_dir: Path, cache_manager: CacheManager, progress_tracker: ProgressTracker) -> List[dict]:
    """Process a batch of tickers."""
    results = []
    completed_in_batch = 0
    extractor = SECDataExtractor(cache_manager=cache_manager, output_dir=str(output_dir))

    for ticker in tickers:
        try:
            # Retrieve cached result
            cached_result = cache_manager.get_cached_tickers_count(ticker)
            if cached_result and cached_result['status'] == 'success':
                concept_count = cached_result.get('concept_count', 0)
                if concept_count >= cache_manager.MIN_CONCEPT_THRESHOLD:
                    # Sufficient concepts present, skip processing
                    results.append(cached_result)
                    completed_in_batch += 1
                    progress_tracker.update_progress(completed=None, batch_completed=completed_in_batch)
                    continue
                else:
                    # Insufficient concepts, proceed to fetch missing data
                    cache_manager.logger.info(f"Ticker {ticker} has {concept_count} concepts. Proceeding to fetch missing data.")
        
            # Process ticker with retry only for rate limits
            for attempt in range(MAX_RETRIES):
                try:
                    data_file, metadata_file = extractor.process_ticker(ticker)
                    
                    if data_file == "N/A" and metadata_file == "N/A":
                        result = {
                            'ticker': ticker,
                            'status': 'N/A',
                            'data_file': "N/A",
                            'metadata_file': "N/A"
                        }
                    else:
                        result = {
                            'ticker': ticker,
                            'status': 'success',
                            'data_file': str(data_file),
                            'metadata_file': str(metadata_file)
                        }
                    
                    cache_manager.cache_result(ticker, result)
                    results.append(result)
                    break

                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        if attempt < MAX_RETRIES - 1:
                            delay = BASE_RETRY_DELAY * (2 ** attempt)
                            cache_manager.logger.warning(
                                f"Rate limit hit for {ticker}, attempt {attempt + 1}/{MAX_RETRIES}. "
                                f"Waiting {delay}s"
                            )
                            time.sleep(delay)
                            continue
                    raise

        except Exception as e:
            error_result = {
                'ticker': ticker,
                'status': 'failed', 
                'error': str(e)
            }
            cache_manager.cache_result(ticker, error_result)
            results.append(error_result)
            cache_manager.logger.error(f"Failed to process {ticker}: {str(e)}")

        completed_in_batch += 1
        progress_tracker.update_progress(completed=None, batch_completed=completed_in_batch)

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
    
    # Setup environment and logging
    setup_environment()
    logger = setup_logging(output_path)
    
    # Initialize CacheManager
    cache_manager = CacheManager(output_path / 'cache')
    cache_manager.cleanup_old_entries()
    
    # Initialize ProgressTracker
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
    
    # Determine optimal batch size based on available memory
    available_memory = psutil.virtual_memory().available
    estimated_memory_per_ticker = 1024 * 1024  # 1MB estimate
    optimal_batch_size = min(
        batch_size,
        max(10, int(available_memory / (estimated_memory_per_ticker * 2)))
    )
    logger.info(f"Optimal batch size set to {optimal_batch_size} based on available memory.")
    
    # Split tickers into batches
    ticker_batches = [
        tickers[i:i + optimal_batch_size] 
        for i in range(0, len(tickers), optimal_batch_size)
    ]
    
    # Get count of already processed tickers
    already_processed_count = cache_manager.get_processed_tickers_count()
    
    # Get specific tickers from cache for comparison
    cached_tickers = set()
    with sqlite3.connect(str(cache_manager.db_path)) as conn:
        cursor = conn.execute('''
            SELECT ticker FROM concept_cache
            WHERE last_updated > datetime('now', '-7 days')
            GROUP BY ticker
            HAVING COUNT(DISTINCT concept_tag) >= ?
        ''', (cache_manager.MIN_CONCEPT_THRESHOLD,))
        cached_tickers.update(row[0] for row in cursor.fetchall())
    
    # Compare input vs cache
    input_tickers = set(tickers)
    extra_cached = cached_tickers - input_tickers
    missing_from_cache = input_tickers - cached_tickers
    
    if extra_cached:
        logger.warning(f"Cache contains {len(extra_cached)} tickers not in input file: {sorted(extra_cached)}")
    if missing_from_cache:
        logger.info(f"Input file contains {len(missing_from_cache)} new tickers: {sorted(missing_from_cache)}")

    logger.info(f"📋 Found {len(tickers)} unique tickers to process "
                f"({already_processed_count} already processed, "
                f"{len(tickers) - already_processed_count} remaining)")
    
    # Initialize progress tracking with already processed count
    progress_tracker.init_progress(len(tickers))
    if already_processed_count > 0:
        progress_tracker.update_progress(completed=None, batch_completed=already_processed_count)
        completed_total = already_processed_count
    else:
        completed_total = 0

    # Process tickers in batches with ThreadPoolExecutor
    start_time = datetime.now()
    results = []
    MAX_CONCURRENT_BATCHES = min(4, max_workers if max_workers else 4)
    active_futures = set()
    
    def save_batch_results(batch_results, batch_num):
        batch_df = pd.DataFrame(batch_results)
        batch_file = output_path / f'batch_results_{batch_id}_part{batch_num}.csv'
        batch_df.to_csv(batch_file, index=False)
        logger.info(f"Batch {batch_num} results saved to {batch_file}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_num, batch in enumerate(ticker_batches, start=1):
            # Wait if too many active futures
            while len(active_futures) >= MAX_CONCURRENT_BATCHES:
                done, active_futures = concurrent.futures.wait(
                    active_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        completed_total += len(batch_results)
                        progress_tracker.update_progress(completed=None, batch_completed=completed_total)
                        
                        # Save and clear results periodically
                        if len(results) > 1000:
                            results = save_batch_results(results, batch_num)
                    except Exception as e:
                        logger.error(f"Batch processing error: {str(e)}")
            
            # Monitor memory usage
            memory_percent = psutil.Process().memory_percent()
            if memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
                # Wait for all current batches to complete
                concurrent.futures.wait(active_futures)
                # Save remaining results
                batch_results = [future.result() for future in active_futures if future.done()]
                results.extend(batch_results)
                active_futures.clear()
                gc.collect()
            
            # Submit new batch
            future = executor.submit(
                process_ticker_batch,
                batch,
                output_path,
                cache_manager,
                progress_tracker
            )
            active_futures.add(future)

    # Wait for remaining futures
    remaining_futures = list(active_futures)
    while remaining_futures:
        done, remaining_futures = concurrent.futures.wait(
            remaining_futures, 
            timeout=10.0,  # 10 second timeout
            return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            try:
                batch_results = future.result(timeout=5.0)  # 5 second timeout
                results.extend(batch_results)
                completed_total += len(batch_results)
                progress_tracker.update_progress(completed=None, batch_completed=completed_total)
            except concurrent.futures.TimeoutError:
                logger.error("Timeout waiting for batch results")
                # Cancel the future if possible
                future.cancel()
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
        
        # Force cleanup
        gc.collect()

    # Complete progress tracking
    progress_tracker.finish()
    gc.collect()
    
    # Combine all results
    final_results_df = pd.DataFrame(results)
    
    # Save summary
    summary_file = output_path / f'processing_summary_{batch_id}.csv'
    final_results_df.to_csv(summary_file, index=False)
    logger.info(f"📁 Summary saved to: {summary_file}")
    
    # Calculate final statistics
    end_time = datetime.now()
    success_count = len(final_results_df[final_results_df['status'] == 'success'])
    fail_count = len(final_results_df[final_results_df['status'] == 'failed'])
    
    # Save batch statistics
    batch_stats = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_tickers': len(final_results_df),
        'successful': success_count,
        'failed': fail_count
    }
    cache_manager.save_batch_stats(batch_id, batch_stats)
    
    # Print final summary
    logger.info("\n📋 Processing Summary:")
    logger.info(f"Total tickers processed: {len(final_results_df)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed: {fail_count}")
    logger.info(f"⏱️ Total time: {end_time - start_time}")
    logger.info(f"📁 Summary saved to: {summary_file}")
    
    # Display final summary with rich formatting
    success_rate = (success_count / len(final_results_df)) * 100 if len(final_results_df) > 0 else 0
    console = Console()
    console.print("\n[bold green]✨ Processing Complete![/bold green]")
    console.print("\n[bold]Final Statistics:[/bold]")
    console.print(f"📊 Success Rate: [green]{success_rate:.1f}%[/green]")
    console.print(f"📁 Output Directory: [blue]{output_dir}[/blue]")
    
    # Display error summary if there were failures
    if fail_count > 0:
        failed_results = final_results_df[final_results_df['status'] == 'failed']
        console.print("\n[bold red]⚠️ Failed Tickers:[/bold red]")
        for _, row in failed_results.iterrows():
            console.print(f"❌ {row['ticker']}: {row['error']}")
            
    console.print("\n[bold green]🎉 Process completed successfully![/bold green]\n")

def main():
    """Main execution function."""
    # Configuration
    INPUT_FILE = 'small_cap_stocks_latest.csv'
    OUTPUT_DIR = './data/fundamentals'
    MAX_WORKERS = 8
    TICKER_COLUMN = 'ticker'
    BATCH_SIZE = 16
    
    try:
        console = Console()
        console.print("\n[bold blue]🚀 Starting All Fundamentals Data Processing[/bold blue]\n")
        
        results = parallel_process_tickers(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            max_workers=MAX_WORKERS,
            ticker_column=TICKER_COLUMN,
            batch_size=BATCH_SIZE
        )
        
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Process interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]❌ Fatal error: {str(e)}[/bold red]")
        console.print_exception()
        sys.exit(1)

if __name__ == "__main__":
    main()
