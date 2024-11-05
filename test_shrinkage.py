"""
Covariance Matrix Shrinkage Evaluation Framework.
Implements rolling window evaluation of various shrinkage methods with proper time series handling.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
from shrinkage import (
    linear_shrinkage_identity, 
    linear_shrinkage_constant_correlation,
    linear_shrinkage_single_factor, 
    nonlinear_analytical_shrinkage
)
import sys
from scipy import linalg
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

class ResultLogger:
    """Handles formatted printing of evaluation results."""
    
    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path
        self.log_entries = []
        
    def print_and_log(self, msg: str, data: Optional[Union[pd.DataFrame, dict, str]] = None):
        """Print message and optionally associated data with formatting."""
        border = "=" * 80
        print(f"\n{border}")
        print(msg)
        
        if data is not None:
            if isinstance(data, pd.DataFrame):
                print(f"\n{data}")
                print(f"\nShape: {data.shape}")
            elif isinstance(data, dict):
                for k, v in data.items():
                    print(f"{k}: {v}")
            else:
                print(data)
        
        print(f"{border}\n")
        
        # Store for potential file output
        self.log_entries.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': msg,
            'data': data
        })
    
    def save_to_file(self):
        """Save logged results to file if output path specified."""
        if self.output_path:
            with open(self.output_path, 'w') as f:
                for entry in self.log_entries:
                    f.write(f"\n{entry['timestamp']} - {entry['message']}\n")
                    if entry['data'] is not None:
                        f.write(f"{str(entry['data'])}\n")
                        f.write("-" * 80 + "\n")

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def determine_separator(file_path: str) -> str:
        """Determine file separator by analyzing first few lines."""
        with open(file_path, 'r') as f:
            first_lines = [next(f) for _ in range(5)]
        
        delimiters = {
            '\t': [len(line.split('\t')) for line in first_lines],
            ',': [len(line.split(',')) for line in first_lines]
        }
        
        consistent_counts = {
            d: len(set(counts)) == 1 and counts[0] > 1
            for d, counts in delimiters.items()
        }
        
        if consistent_counts.get('\t'):
            return '\t'
        elif consistent_counts.get(','):
            return ','
        else:
            raise ValueError(f"Could not determine consistent delimiter for {file_path}")
    
    @staticmethod
    def load_price_data(
        file_path: str,
        start_date: str,
        min_market_cap_pct: float = 0.5,  # Minimum percentile of average price
        max_missing_pct: float = 0.01     # Maximum allowed missing data
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load and clean price data with detailed statistics.
        
        Args:
            file_path: Path to price data file
            start_date: Start date for analysis
            min_market_cap_pct: Minimum percentile of avg price to include
            max_missing_pct: Maximum fraction of missing data allowed
        
        Returns:
            Cleaned price DataFrame, statistics dictionary
        """
        stats = {}
        
        # Load data
        sep = DataProcessor.determine_separator(str(file_path))
        df = pd.read_csv(
            file_path,
            sep=sep,
            skipinitialspace=True,
            dtype={'date': str}
        )
        
        stats['initial_shape'] = df.shape
        
        # Process dates
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= start_date].copy()
        df.set_index('date', inplace=True)
        
        # Get price columns
        price_cols = [col for col in df.columns if col.endswith('_close')]
        if not price_cols:
            raise ValueError("No price columns found in data")
        
        df_prices = df[price_cols].copy()
        stats['price_columns'] = len(price_cols)
        
        # Calculate missing data percentages
        missing_pct = df_prices.isnull().mean()
        valid_cols = missing_pct[missing_pct < max_missing_pct].index
        df_prices = df_prices[valid_cols]
        
        stats['removed_missing'] = len(price_cols) - len(valid_cols)
        
        # Select stocks above market cap threshold
        avg_prices = df_prices.mean()
        cutoff = np.percentile(avg_prices, min_market_cap_pct * 100)
        large_caps = avg_prices[avg_prices >= cutoff].index
        df_prices = df_prices[large_caps]
        
        stats['final_stocks'] = len(df_prices.columns)
        
        # Forward fill then backward fill remaining missing values
        df_prices = df_prices.ffill().bfill()
        
        # Final checks
        stats['final_shape'] = df_prices.shape
        stats['missing_data_final'] = df_prices.isnull().sum().sum()
        
        return df_prices, stats
    
    @staticmethod
    def calculate_returns(
        prices: pd.DataFrame,
        winsorize_pct: Optional[float] = 0.01  # Winsorize extreme returns
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Calculate returns with proper handling of outliers and missing data.
        
        Args:
            prices: Price DataFrame
            winsorize_pct: Percentile for winsorizing returns (None for no winsorization)
            
        Returns:
            Returns DataFrame, statistics dictionary
        """
        stats = {}
        
        # Calculate returns
        returns = prices.pct_change(fill_method=None)
        returns.columns = [col.replace('_close', '') for col in returns.columns]
        
        # Remove first row (NaN from pct_change)
        returns = returns.iloc[1:]
        
        # Handle extremes
        if winsorize_pct:
            lower = np.percentile(returns, winsorize_pct)
            upper = np.percentile(returns, 100 - winsorize_pct)
            returns = returns.clip(lower=lower, upper=upper)
            
            stats['winsorized_lower'] = lower
            stats['winsorized_upper'] = upper
        
        # Remove any remaining invalid values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        initial_rows = len(returns)
        returns = returns.dropna(how='any')
        
        stats['rows_removed'] = initial_rows - len(returns)
        stats['final_shape'] = returns.shape
        
        # Calculate basic statistics
        stats['mean_return'] = returns.mean().mean()
        stats['std_return'] = returns.std().mean()
        
        return returns, stats

class CovarianceEvaluator:
    """Evaluates covariance matrix estimation methods using rolling windows."""
    
    def __init__(self, logger: ResultLogger):
        self.logger = logger
        
    def frobenius_norm(self, est_cov: np.ndarray, true_cov: np.ndarray) -> float:
        """Calculate Frobenius norm between estimated and realized covariance."""
        diff = est_cov - true_cov
        return np.sqrt(np.sum(diff * diff))
    
    def evaluate_estimation(
        self,
        est_cov: np.ndarray,
        true_cov: np.ndarray,
        est_returns: pd.DataFrame,
        val_returns: pd.DataFrame
    ) -> dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Matrix distance metrics
        metrics['frobenius'] = self.frobenius_norm(est_cov, true_cov)
        
        # Portfolio-based metrics
        try:
            # Construct minimum variance portfolio with improved conditioning
            n = len(est_cov)
            ones = np.ones(n)
            
            # More aggressive regularization
            epsilon = 1e-6 * np.trace(est_cov) / n
            reg_est_cov = est_cov + epsilon * np.eye(n)
            
            try:
                # First try Cholesky decomposition
                L = linalg.cholesky(reg_est_cov, lower=True)
                weights = linalg.solve_triangular(L, ones, lower=True)
                weights = linalg.solve_triangular(L.T, weights, lower=False)
            except:
                # If Cholesky fails, try eigen-decomposition based regularization
                eigenvals, eigenvecs = linalg.eigh(reg_est_cov)
                min_eig = np.maximum(eigenvals.min(), 1e-8)
                reg_est_cov = reg_est_cov + (min_eig * np.eye(n))
                weights = linalg.solve(reg_est_cov, ones, assume_a='pos')
                
            weights = weights / weights.sum()
            
            # Check for numerical stability
            if np.any(np.abs(weights) > 10) or np.any(np.isnan(weights)):
                raise ValueError("Unstable weights detected")
            
            # Calculate predicted vs realized risk
            pred_var = weights @ est_cov @ weights
            real_var = weights @ true_cov @ weights
            
            metrics['pred_var'] = pred_var
            metrics['real_var'] = real_var
            metrics['var_ratio'] = real_var / pred_var

            print(f"Predicted portfolio variance: {pred_var:.6f}")
            print(f"Realized portfolio variance: {real_var:.6f}")
            print(f"Variance ratio (realized/predicted): {metrics['var_ratio']:.3f}")
            print(f"Perfect prediction would be 1.000\n")

            # Add portfolio properties
            metrics['max_weight'] = np.abs(weights).max()
            metrics['min_weight'] = np.abs(weights).min()
            metrics['weight_std'] = weights.std()
            
        except Exception as e:
            self.logger.print_and_log(f"Error in portfolio calculations: {str(e)}")
            metrics.update({
                'pred_var': np.nan,
                'real_var': np.nan,
                'var_ratio': np.nan,
                'max_weight': np.nan,
                'min_weight': np.nan,
                'weight_std': np.nan
            })
        
        return metrics
    
    def evaluate_rolling_windows(
        self,
        returns: pd.DataFrame,
        train_window: int = 252,  # 1 year
        test_window: int = 252,   # 1 year test
        min_periods: int = 200,
        methods: list = ['identity', 'const_corr', 'single_factor', 'nonlinear']
    ) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
        """
        Evaluate methods using rolling windows of train/test data.
        
        Args:
            returns: Returns DataFrame
            train_window: Training window length
            test_window: Testing window length
            min_periods: Minimum required periods
            methods: List of methods to evaluate
            
        Returns:
            Summary DataFrame, detailed results dictionary
        """
        self.logger.print_and_log("Starting rolling window evaluation")
        
        results = {method: [] for method in methods + ['sample']}
        window_info = []
        
        # Rolling windows
        for t in range(0, len(returns) - train_window - test_window + 1, test_window):
            train_start = returns.index[t]
            train_end = returns.index[t + train_window - 1]
            test_end = returns.index[min(t + train_window + test_window - 1, len(returns) - 1)]
            
            window_info.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_end': test_end
            })
            
            self.logger.print_and_log(
                f"Window {len(window_info)}: "
                f"Train {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, "
                f"Test until {test_end.strftime('%Y-%m-%d')}"
            )
            
            # Split data
            train_rets = returns.loc[train_start:train_end]
            test_rets = returns.loc[train_end:test_end]
            
            if len(train_rets) < min_periods:
                self.logger.print_and_log("Insufficient data in window")
                continue
            
            # Calculate realized covariance
            true_cov = test_rets.cov().values
            
            # Evaluate each method
            for method in ['sample'] + methods:
                try:
                    # Get covariance estimate
                    if method == 'sample':
                        est_cov = train_rets.cov().values
                    else:
                        est_cov = self.get_shrinkage_estimate(train_rets, method)
                    
                    # Calculate metrics
                    metrics = self.evaluate_estimation(
                        est_cov, true_cov, train_rets, test_rets
                    )
                    
                    metrics['window'] = len(window_info)
                    metrics['method'] = method
                    results[method].append(metrics)
                    if 'var_ratio' in metrics:
                        print(f"\n{method.upper()} METHOD:")
                        print(f"Predicted portfolio variance: {metrics['pred_var']:.6f}")
                        print(f"Realized portfolio variance:  {metrics['real_var']:.6f}")
                        print(f"Variance ratio:              {metrics['var_ratio']:.3f}")
                        print(f"(Perfect prediction would be 1.000)")
                    
                except Exception as e:
                    self.logger.print_and_log(f"Error evaluating {method}: {str(e)}")
        
        # Create summary
        summary = self.create_summary(results, window_info)
        
        return summary, results
    
    def get_shrinkage_estimate(
        self,
        returns: pd.DataFrame,
        method: str,
        demean: bool = True
    ) -> np.ndarray:
        """Get covariance estimate using specified shrinkage method."""
        returns_array = returns.values
        
        if method == 'identity':
            est_cov, _ = linear_shrinkage_identity(returns_array, demean)
        elif method == 'const_corr':
            est_cov, _ = linear_shrinkage_constant_correlation(returns_array, demean)
        elif method == 'single_factor':
            est_cov, _ = linear_shrinkage_single_factor(returns_array, None, demean)
        elif method == 'nonlinear':
            est_cov = nonlinear_analytical_shrinkage(returns_array, demean)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return est_cov
    
    def create_summary(
        self,
        results: Dict[str, List[dict]],
        window_info: List[dict]
    ) -> pd.DataFrame:
        """Create summary DataFrame from detailed results."""
        summary_data = []
        
        metrics = [
            'frobenius', 'var_ratio',
            'pred_var', 'real_var'
        ]
        
        for method, method_results in results.items():
            method_summary = {'method': method}
            
            for metric in metrics:
                values = [r[metric] for r in method_results if metric in r]
                values = np.array(values)[~np.isnan(values)]
                
                if len(values) > 0:
                    method_summary[f'{metric}_mean'] = np.mean(values)
                    method_summary[f'{metric}_std'] = np.std(values)
                else:
                    method_summary[f'{metric}_mean'] = np.nan
                    method_summary[f'{metric}_std'] = np.nan
            
            summary_data.append(method_summary)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df.set_index('method')

def main():
    """Main execution function."""
    try:
        # Initialize logger
        logger = ResultLogger(Path("data/transformed/evaluation_log.txt"))
        logger.print_and_log("Starting covariance estimation evaluation")
        
        # Find latest data file
        base_dir = Path("data/transformed")
        daily_files = list(base_dir.glob("price_data_1d_*.csv"))
        
        if not daily_files:
            raise ValueError("No daily price data files found")
        
        latest_daily = max(daily_files)
        logger.print_and_log(f"Using data file: {latest_daily}")
        
        # Load and process data
        start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')  # 5 years
        df_prices, price_stats = DataProcessor.load_price_data(
            str(latest_daily),
            start_date,
            min_market_cap_pct=0.5,
            max_missing_pct=0.01
        )
        
        logger.print_and_log("Price data statistics:", price_stats)
        
        returns, return_stats = DataProcessor.calculate_returns(
            df_prices,
            winsorize_pct=0.01
        )
        
        logger.print_and_log("Returns statistics:", return_stats)
        
        # Initialize evaluator
        evaluator = CovarianceEvaluator(logger)
        
        # Run evaluation
        methods = ['identity', 'const_corr', 'single_factor', 'nonlinear']
        summary, detailed_results = evaluator.evaluate_rolling_windows(
            returns,
            train_window=252,    # 1 year training
            test_window=252,     # 1 year testing
            min_periods=200,
            methods=methods
        )
        
        # Print and save results
        logger.print_and_log("\nEvaluation Summary:", summary)

        print("\n" + "="*80)
        print("VARIANCE RATIO SUMMARY BY METHOD")
        print("="*80)
        for idx in summary.index:
            print(f"{idx:15} - Mean: {summary.loc[idx,'var_ratio_mean']:6.3f}  Std: {summary.loc[idx,'var_ratio_std']:6.3f}")
        print("\nMethod with ratio closest to 1.000 is best")
        print("Low std.dev. indicates consistency across time periods")
        print("="*80)
        logger.print_and_log("\nFull Summary Statistics:", summary)
        
        # Save detailed results
        results_file = base_dir / 'covariance_evaluation.csv'
        summary.to_csv(results_file)
        logger.print_and_log(f"\nDetailed results saved to: {results_file}")
        
        # Save log
        logger.save_to_file()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
