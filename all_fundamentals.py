import os
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
import concurrent.futures
from dotenv import load_dotenv

class SECCache:
    """Handles caching of SEC API responses to disk."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / 'sec_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self._load_metadata()

    def _load_metadata(self):
        """Load or initialize cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self._save_metadata()

    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def _get_cache_path(self, ticker: str, concept: str) -> Path:
        """Generate cache file path for a specific ticker and concept."""
        return self.cache_dir / f"{ticker}_{concept}.json"

    def is_cached_valid(self, ticker: str, concept: str, max_age_days: int = 7) -> bool:
        """Check if cached data exists and is still valid."""
        cache_key = f"{ticker}_{concept}"
        if cache_key in self.metadata:
            cache_date = datetime.fromisoformat(self.metadata[cache_key]['timestamp'])
            age = datetime.now() - cache_date
            return age.days < max_age_days
        return False

    def get_cached_data(self, ticker: str, concept: str) -> Optional[Dict]:
        """Retrieve cached data if it exists."""
        cache_path = self._get_cache_path(ticker, concept)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def cache_data(self, ticker: str, concept: str, data: Dict):
        """Cache data to disk."""
        cache_path = self._get_cache_path(ticker, concept)
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        
        self.metadata[f"{ticker}_{concept}"] = {
            'timestamp': datetime.now().isoformat(),
            'path': str(cache_path)
        }
        self._save_metadata()

class SECRateLimiter:
    """Handles SEC API rate limiting."""
    
    def __init__(self, requests_per_second: float = 10):
        self.period = 1.0 / requests_per_second

    @sleep_and_retry
    @limits(calls=10, period=1)
    def wait(self):
        """Wait to comply with rate limits."""
        pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def fetch_sec_data(url: str, headers: Dict[str, str]) -> Dict:
    """Fetch data from SEC API with retries and exponential backoff."""
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

class EnhancedSECDataExtractor:
    """Enhanced SEC data extractor with caching and rate limiting."""
    
    def __init__(self, output_dir: str, sec_user_agent: str):
        self.output_dir = Path(output_dir)
        self.cache = SECCache(output_dir)
        self.rate_limiter = SECRateLimiter()
        self.headers = {'User-Agent': sec_user_agent}
        self.logger = logging.getLogger(__name__)

    def _get_cik(self, ticker: str) -> str:
        """Get CIK for a ticker with caching."""
        # First check cache
        cik_cache_path = self.cache._get_cache_path(ticker, "cik")
        if cik_cache_path.exists():
            with open(cik_cache_path, 'r') as f:
                return json.load(f)['cik']
        
        # If not in cache, fetch from SEC
        try:
            self.rate_limiter.wait()
            url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            cik = data['cik']
            
            # Cache the CIK
            with open(cik_cache_path, 'w') as f:
                json.dump({'cik': cik, 'timestamp': datetime.now().isoformat()}, f)
            
            return cik
        except Exception as e:
            self.logger.error(f"Failed to get CIK for {ticker}: {str(e)}")
            raise

    def fetch_concept_data(self, ticker: str, concept: str) -> Optional[Dict]:
        """Fetch concept data with caching and rate limiting."""
        # Check cache first
        if self.cache.is_cached_valid(ticker, concept):
            self.logger.info(f"Using cached data for {ticker} {concept}")
            return self.cache.get_cached_data(ticker, concept)

        # If not in cache, fetch from API
        try:
            self.rate_limiter.wait()
            cik = self._get_cik(ticker)
            url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
            data = fetch_sec_data(url, self.headers)
            
            # Cache the response
            self.cache.cache_data(ticker, concept, data)
            self.logger.info(f"Successfully retrieved {concept} data for {ticker}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                self.logger.warning(f"Rate limit exceeded for {ticker} {concept}. Backing off...")
                raise
            self.logger.error(f"Failed to retrieve {concept} data for {ticker}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {ticker} {concept}: {str(e)}")
            return None

    def process_ticker(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """Process a single ticker with enhanced error handling and caching."""
        try:
            # List of concepts to fetch
            concepts = [
                'Assets', 'AssetsCurrent', 'AssetsNoncurrent',
                'CashAndCashEquivalentsAtCarryingValue', 'MarketableSecurities',
                'AvailableForSaleSecurities', 'AccountsReceivableNetCurrent'
            ]
            
            ticker_data = {}
            for concept in concepts:
                concept_data = self.fetch_concept_data(ticker, concept)
                if concept_data:
                    ticker_data[concept] = concept_data

            if not ticker_data:
                raise Exception("No data could be retrieved for ticker")

            # Save processed data
            output_file = self.output_dir / f"{ticker}_fundamentals.json"
            metadata_file = self.output_dir / f"{ticker}_metadata.json"
            
            with open(output_file, 'w') as f:
                json.dump(ticker_data, f)
            
            metadata = {
                'ticker': ticker,
                'processed_date': datetime.now().isoformat(),
                'concepts_retrieved': list(ticker_data.keys())
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)

            return str(output_file), str(metadata_file)

        except Exception as e:
            self.logger.error(f"Failed to process ticker {ticker}: {str(e)}")
            return None, None

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'sec_fundamentals.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_ticker_parallel(ticker: str, output_dir: Path, sec_user_agent: str) -> dict:
    """Process a single ticker using the enhanced extractor."""
    try:
        extractor = EnhancedSECDataExtractor(str(output_dir), sec_user_agent)
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
    sec_user_agent: str,
    max_workers: Optional[int] = None,
    ticker_column: str = 'ticker',
    batch_size: int = 50  # Reduced batch size for better rate limit management
) -> pd.DataFrame:
    """Process multiple tickers in parallel with improved rate limiting and caching."""
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
    
    # Process tickers in batches
    results = []
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch_tickers)} tickers)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    process_ticker_parallel, 
                    ticker, 
                    output_path,
                    sec_user_agent
                ): ticker 
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
    # Load environment variables
    load_dotenv()
    
    # Configuration
    INPUT_FILE = 'small_cap_stocks_latest.csv'
    OUTPUT_DIR = './data/fundamentals'
    MAX_WORKERS = None  # None will use CPU count
    TICKER_COLUMN = 'ticker'
    BATCH_SIZE = 50
    
    # Get SEC API user agent from environment
    sec_user_agent = os.getenv('SEC_API_USER_AGENT')
    if not sec_user_agent:
        raise EnvironmentError("SEC_API_USER_AGENT environment variable not set")
    
    try:
        results = parallel_process_tickers(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            sec_user_agent=sec_user_agent,
            max_workers=MAX_WORKERS,
            ticker_column=TICKER_COLUMN,
            batch_size=BATCH_SIZE
        )
        
    except Exception as e:
        print(f"Failed to complete processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
