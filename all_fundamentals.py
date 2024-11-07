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
from fundamentals import SECDataExtractor
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

class CacheManager:
    MIN_CONCEPT_THRESHOLD = 24  # Minimum concepts required for a ticker to be considered fully processed
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / 'ticker_cache.db'
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._init_db()
        self._check_schema()
        self.resync_ticker_cache()
    

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

    def resync_ticker_cache(self):
        """Rebuild ticker_cache.db by identifying tickers with .tsv files modified in the last 24 hours."""
        ticker_stats_dict = {}
    
        # First, get existing processed tickers from ticker_cache.db
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute('''
                SELECT ticker, last_processed, status
                FROM processed_tickers
                WHERE last_processed > datetime('now', '-7 days')
                AND status = 'success'
            ''')
            existing_tickers = {row[0]: row[1] for row in cursor.fetchall()}
            ticker_stats_dict.update(existing_tickers)
    
        # Then get tickers from granular_cache.db
        granular_cache_path = self.cache_dir / 'granular_cache.db'
        if granular_cache_path.exists():
            with sqlite3.connect(str(granular_cache_path)) as gconn:
                cursor = gconn.execute('''
                    SELECT ticker, COUNT(DISTINCT concept_tag) as concept_count,
                           MAX(last_updated) as last_update
                    FROM concept_cache 
                    WHERE concept_value IS NOT NULL
                    AND last_updated > datetime('now', '-7 days')
                    GROUP BY ticker
                    HAVING concept_count > 65
                ''')
                ticker_stats = cursor.fetchall()
                for ticker, _, last_update in ticker_stats:
                    if ticker not in ticker_stats_dict:
                        ticker_stats_dict[ticker] = last_update
    
        # Now scan for recent .tsv files
        data_dir = self.cache_dir.parent.parent
        recent_files = list(data_dir.glob('**/*sec_data_*.tsv'))
        
        for file_path in recent_files:
            try:
                modification_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if modification_time >= (datetime.now() - timedelta(days=7)):
                    ticker = file_path.stem.split('_')[0]
                    mod_time_str = modification_time.strftime('%Y-%m-%d %H:%M:%S')
                    ticker_stats_dict[ticker] = mod_time_str
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {e}")
    
        # Write all to ticker_cache.db
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                # Clear existing processed tickers
                conn.execute('DELETE FROM processed_tickers')
    
                # Insert updated processed tickers
                conn.executemany('''
                    INSERT INTO processed_tickers 
                    (ticker, last_processed, status, data_file, metadata_file, error, data_hash)
                    VALUES (?, ?, 'success', NULL, NULL, NULL, NULL)
                ''', [(t, v) for t, v in ticker_stats_dict.items()])
    
                conn.commit()
                self.logger.info(f"Resynced ticker cache with {len(ticker_stats_dict)} tickers.")
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error during resync_ticker_cache: {e}")
                raise

    def cleanup_old_entries(self):
        """Remove entries older than 7 days."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                DELETE FROM processed_tickers 
                WHERE last_processed < datetime('now', '-7 days')
            ''')

    def get_cached_result(self, ticker: str) -> Optional[Dict]:
        """Retrieve cached result for a ticker if it exists and is recent, along with concept count."""
        with self._lock:
            # First check ticker_cache.db for processed status
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('''
                    SELECT * FROM processed_tickers 
                    WHERE ticker = ? AND last_processed > datetime("now", "-7 days")
                ''', (ticker,))
                result = cursor.fetchone()
    
                if result:
                    # Now check granular_cache.db for concept count
                    granular_db_path = self.cache_dir / 'granular_cache.db'
                    with sqlite3.connect(str(granular_db_path)) as gconn:
                        concept_count_cursor = gconn.execute('''
                            SELECT COUNT(DISTINCT concept_tag) 
                            FROM concept_cache 
                            WHERE ticker = ? 
                              AND last_updated > datetime("now", "-7 days")
                        ''', (ticker,))
                        concept_count = concept_count_cursor.fetchone()[0] if concept_count_cursor else 0
    
                    return {
                        'ticker': result[0],
                        'last_processed': result[1],
                        'status': result[2],
                        'data_file': result[3],
                        'metadata_file': result[4],
                        'error': result[5],
                        'data_hash': result[6],
                        'concept_count': concept_count
                    }
    
                # If not found, mark as processing
                conn.execute('''
                    INSERT OR REPLACE INTO processed_tickers 
                    (ticker, last_processed, status)
                    VALUES (?, datetime('now'), 'processing')
                ''', (ticker,))
                conn.commit()
                return None
    
    def get_processed_tickers_count(self) -> int:
        """Get count of successfully processed tickers within the last 7 days that meet the concept threshold."""
        # Get count from processed_tickers in ticker_cache.db
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute('''
                SELECT COUNT(DISTINCT ticker) FROM processed_tickers
                WHERE status = 'success'
                AND last_processed > datetime("now", "-7 days")
            ''')
            ticker_count = cursor.fetchone()[0]
    
        # Get count from concept_cache in granular_cache.db
        try:
            granular_db_path = self.cache_dir / 'granular_cache.db'
            if granular_db_path.exists():
                with sqlite3.connect(str(granular_db_path)) as conn:
                    # Count tickers with sufficient concepts
                    cursor = conn.execute('''
                        SELECT COUNT(DISTINCT ticker) 
                        FROM (
                            SELECT ticker
                            FROM concept_cache 
                            WHERE last_updated > datetime("now", "-7 days")
                            GROUP BY ticker 
                            HAVING COUNT(DISTINCT concept_tag) >= ?
                        )
                    ''', (self.MIN_CONCEPT_THRESHOLD,))
                    concept_count = cursor.fetchone()[0]
            else:
                concept_count = 0
        except Exception as e:
            self.logger.warning(f"Error checking concept cache: {e}")
            concept_count = 0
    
        return max(ticker_count, concept_count)


    def _check_schema(self):
        """Check that the database schema is up to date."""
        granular_db_path = self.cache_dir / 'granular_cache.db'
        if granular_db_path.exists():
            with sqlite3.connect(str(granular_db_path)) as conn:
                # Check if last_updated column exists
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM pragma_table_info('concept_cache')
                    WHERE name = 'last_updated'
                ''')
                has_timestamp = cursor.fetchone()[0] > 0
    
                if not has_timestamp:
                    conn.execute('''
                        ALTER TABLE concept_cache
                        ADD COLUMN last_updated TIMESTAMP
                    ''')
                    conn.execute('''
                        UPDATE concept_cache 
                        SET last_updated = datetime('now')
                        WHERE last_updated IS NULL
                    ''')
    
                # Add fetch_status if it doesn't exist
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM pragma_table_info('concept_cache')
                    WHERE name = 'fetch_status'
                ''')
                has_fetch_status = cursor.fetchone()[0] > 0
    
                if not has_fetch_status:
                    conn.execute('''
                        ALTER TABLE concept_cache
                        ADD COLUMN fetch_status TEXT DEFAULT 'unknown'
                    ''')


    def get_processed_tickers_in_batch(self, batch_tickers: List[str]) -> int:
        """Get count of successfully processed tickers from a specific batch."""
        placeholders = ','.join(['?' for _ in batch_tickers])
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(f'''
                SELECT COUNT(*) FROM processed_tickers 
                WHERE ticker IN ({placeholders})
                AND status = 'success' 
                AND last_processed > datetime("now", "-7 days")
            ''', batch_tickers)
            return cursor.fetchone()[0]

    def cache_result(self, ticker: str, result: Dict):
        """Cache the processing result for a ticker with transaction safety."""
        with self._lock:  # Use the existing lock
            with sqlite3.connect(str(self.db_path)) as conn:
                try:
                    conn.execute('BEGIN IMMEDIATE')  # Ensure atomic operation
                    conn.execute('''
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
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise


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
        level=logging.DEBUG,
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

MAX_RETRIES = 3
BASE_RETRY_DELAY = 1

def process_ticker_batch(tickers: List[str], extractor: SECDataExtractor, cache_manager: CacheManager, progress_tracker: ProgressTracker) -> List[dict]:
    """Process a batch of tickers."""
    results = []
    completed_in_batch = 0

    for ticker in tickers:
        try:
            # Retrieve cached result
            cached_result = cache_manager.get_cached_result(ticker)
            if cached_result and cached_result['status'] == 'success':
                concept_count = cached_result.get('concept_count', 0)
                if concept_count >= cache_manager.MIN_CONCEPT_THRESHOLD:
                    # Sufficient concepts present, skip processing
                    results.append(cached_result)
                    completed_in_batch += 1
                    progress_tracker.update_progress(None, completed_in_batch)
                    continue
                else:
                    # Insufficient concepts, proceed to fetch missing data
                    extractor.logger.info(f"Ticker {ticker} has {concept_count} concepts. Proceeding to fetch missing data.")
                    pass  # Continue to processing below

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
                            logging.warning(
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
            logging.error(f"Failed to process {ticker}: {str(e)}")

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
    extractor = SECDataExtractor(str(output_dir))
    cache_manager = CacheManager(output_path / 'cache')
    progress_tracker = ProgressTracker()
    
    # Generate batch ID
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Read ticker list
    try:
        df = pd.read_csv(input_file)
        df[ticker_column] = df[ticker_column].astype(str).str.strip()  # Convert to string and strip whitespace
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
            SELECT ticker FROM processed_tickers
            WHERE status = 'success'
            AND last_processed > datetime('now', '-7 days')
        ''')
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
        progress_tracker.update_progress(already_processed_count)
        completed_total = already_processed_count
    else:
        completed_total = 0

    # Process batches in parallel with memory management
    start_time = datetime.now()
    results = []
    MAX_CONCURRENT_BATCHES = min(4, max_workers if max_workers else 4)
    active_futures = set()
    
    def save_batch_results(batch_results, batch_num):
        batch_df = pd.DataFrame(batch_results)
        batch_file = output_path / f'batch_results_{batch_id}_part{batch_num}.csv'
        batch_df.to_csv(batch_file, index=False)
        return []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_num, batch in enumerate(ticker_batches):
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
                        progress_tracker.update_progress(completed_total)
                        
                        # Save and clear results periodically
                        if len(results) > 100:
                            results = save_batch_results(results, batch_num)
                    except Exception as e:
                        logger.error(f"Batch processing error: {str(e)}")
                
                # Force garbage collection
                gc.collect()
            
            # Monitor memory usage
            memory_percent = psutil.Process().memory_percent()
            if memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
                # Wait for all current batches to complete
                concurrent.futures.wait(active_futures)
                results = save_batch_results(results, batch_num)
                gc.collect()
            
            # Submit new batch
            future = executor.submit(
                process_ticker_batch,
                batch,
                extractor,
                cache_manager,
                progress_tracker
            )
            active_futures.add(future)
        
        # Wait for remaining futures with a timeout
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
                    progress_tracker.update_progress(completed_total)
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
    progress_tracker.progress = None
    gc.collect()
    
    # Combine all results
    final_results_df = pd.DataFrame(results)
    
    # Save summary
    summary_file = output_path / f'processing_summary_{batch_id}.csv'
    final_results_df.to_csv(summary_file, index=False)
    
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
    
    return final_results_df

def main():
    """Main execution function with configuration."""
    # Configuration
    INPUT_FILE = 'small_cap_stocks_latest.csv'
    OUTPUT_DIR = './data/fundamentals'
    MAX_WORKERS = 8
    TICKER_COLUMN = 'ticker'
    BATCH_SIZE = 16
    
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
