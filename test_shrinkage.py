"""
Covariance Matrix Shrinkage Evaluation Framework.
Implements rolling window evaluation of various shrinkage methods.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
from shrinkage import (
    linear_shrinkage_identity, 
    linear_shrinkage_constant_correlation,
    linear_shrinkage_single_factor, 
    rscm_shrinkage,
    dual_shrinkage,
    nonlinear_analytical_shrinkage
)
import sys
from scipy import linalg
from datetime import datetime, timedelta
import warnings
from weights import optimize_portfolio, print_portfolio_weights

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
        max_missing_pct: float = 0.01,     # Maximum allowed missing data
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load and clean price data with detailed statistics.
        
        Args:
            file_path: Path to price data file
            start_date: Start date for analysis
            max_missing_pct: Maximum fraction of missing data allowed
            max_stocks_display: Maximum number of stocks to display in warnings
            
        Returns:
            Cleaned price DataFrame, statistics dictionary
        """
        stats = {}
        max_stocks_display = 50
        
        # Load data
        sep = DataProcessor.determine_separator(str(file_path))
        df = pd.read_csv(
            file_path,
            sep=sep,
            skipinitialspace=True,
            dtype={'date': str}
        )
        
        print(f"\nLoading data from {file_path}")
        stats['initial_shape'] = df.shape
        print(f"Initial data shape: {df.shape}")
        
        # Process dates
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y') 
        df = df[df['date'] >= start_date].copy()
        df.set_index('date', inplace=True)
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Get price columns
        price_cols = [col for col in df.columns if col.endswith('_close')]
        if not price_cols:
            raise ValueError("No price columns found in data")
        
        df_prices = df[price_cols].copy()
        stats['initial_stocks'] = len(price_cols)
        print(f"\nFound {len(price_cols)} price columns")
        
        # Analyze missing data for each stock
        print(f"\nAnalyzing missing data (max allowed: {max_missing_pct:.1%})")
        missing_pct = df_prices.isnull().mean()
        stocks_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(stocks_with_missing) > 0:
            print("\nStocks with missing data:")
            for i, (col, pct) in enumerate(stocks_with_missing.items()):
                if i < max_stocks_display:
                    print(f"{col}: {pct:.2%} missing")
                elif i == max_stocks_display:
                    remaining = len(stocks_with_missing) - max_stocks_display
                    print(f"... and {remaining} more stocks with missing data")
        
        # Remove stocks with too much missing data
        valid_cols = missing_pct[missing_pct < max_missing_pct].index
        removed_stocks = set(price_cols) - set(valid_cols)
        
        if removed_stocks:
            print(f"\nRemoving {len(removed_stocks)} stocks exceeding {max_missing_pct:.1%} missing data:")
            for i, stock in enumerate(sorted(removed_stocks)):
                if i < max_stocks_display:
                    print(f"{stock}: {missing_pct[stock]:.2%} missing")
                elif i == max_stocks_display:
                    remaining = len(removed_stocks) - max_stocks_display
                    print(f"... and {remaining} more stocks removed")
        
        df_prices = df_prices[valid_cols]
        
        stats.update({
            'final_stocks': len(df_prices.columns)
        })
        
        # Fill missing data using linear interpolation
        print("\nFilling remaining missing values with linear interpolation...")
        df_prices = df_prices.interpolate(method='linear', limit_direction='both')
        
        # Report any remaining missing data after interpolation
        remaining_missing = df_prices.isnull().sum()
        stocks_still_missing = remaining_missing[remaining_missing > 0]
        
        if len(stocks_still_missing) > 0:
            print("\nWARNING: Some missing data could not be interpolated:")
            for i, (col, count) in enumerate(stocks_still_missing.items()):
                if i < max_stocks_display:
                    print(f"{col}: {count} values still missing")
                elif i == max_stocks_display:
                    remaining = len(stocks_still_missing) - max_stocks_display
                    print(f"... and {remaining} more stocks with uninterpolated values")
            
            # Remove any stocks with remaining missing data
            df_prices = df_prices.dropna(axis=1)
            print(f"Removed {len(stocks_still_missing)} stocks with uninterpolatable missing data")
        
        # Final statistics
        stats.update({
            'final_shape': df_prices.shape,
            'missing_data_final': df_prices.isnull().sum().sum()
        })
        
        print(f"\nFinal dataset: {df_prices.shape[1]} stocks, {df_prices.shape[0]} days")
        
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
        val_returns: pd.DataFrame,
        method: str
    ) -> dict:
        """Calculate evaluation metrics with optimized portfolio."""
        metrics = {}
        
        # Matrix distance metrics
        metrics['frobenius'] = self.frobenius_norm(est_cov, true_cov)
        
        try:
            # Get optimized portfolio weights
            weights, stats = optimize_portfolio(
                est_returns,
                est_cov,
                target_return=0.20,  # 20% annual return target
                position_limit=0.20   # 20% maximum position size
            )
            
            # Calculate predicted vs realized risk
            pred_var = weights @ est_cov @ weights
            real_var = weights @ true_cov @ weights
            
            # Calculate realized returns on test set
            test_returns = val_returns @ weights  # Portfolio returns in test period
            realized_annual_return = test_returns.mean() * 252  # Annualize
            realized_annual_vol = test_returns.std() * np.sqrt(252)
            realized_sharpe = realized_annual_return / realized_annual_vol
            
            metrics.update({
                'pred_var': pred_var,
                'real_var': real_var,
                'var_ratio': real_var / pred_var,
                'portfolio_weights': weights,
                'portfolio_stats': stats,
                'realized_return': realized_annual_return,
                'realized_vol': realized_annual_vol,
                'realized_sharpe': realized_sharpe
            })
            
            # Print detailed results
            print(f"\nEvaluation results for {method}:")
            print(f"Frobenius norm: {metrics['frobenius']:.6f}")
            print(f"Predicted portfolio variance: {metrics['pred_var']:.6f}")
            print(f"Realized portfolio variance: {metrics['real_var']:.6f}")
            print(f"Variance ratio (realized/predicted): {metrics['var_ratio']:.3f}")
            print(f"TEST SET PERFORMANCE:")
            print(f"Realized Annual Return: {realized_annual_return:.2%}")
            print(f"Realized Annual Volatility: {realized_annual_vol:.2%}")
            print(f"Realized Sharpe Ratio: {realized_sharpe:.2f}")
            
            # Print portfolio weights
            print_portfolio_weights(
                weights,
                est_returns.columns,
                stats
            )
            
        except Exception as e:
            self.logger.print_and_log(f"Error in variance calculations for method {method}: {str(e)}")
            metrics.update({
                'pred_var': np.nan,
                'real_var': np.nan,
                'var_ratio': np.nan,
                'portfolio_weights': None,
                'portfolio_stats': None,
                'realized_return': np.nan,
                'realized_vol': np.nan,
                'realized_sharpe': np.nan
            })
        
        return metrics


    def save_window_statistics(self, results: Dict[str, List[dict]], window_info: List[dict], base_dir: Path):
        """Save per-window statistics to CSV and create visualizations."""
        # Create directory for outputs if it doesn't exist
        output_dir = base_dir / 'evaluation_results'
        output_dir.mkdir(exist_ok=True)
        
        # Prepare data for per-window statistics
        window_stats = []
        for method, method_results in results.items():
            for result in method_results:
                window_data = {
                    'method': method,
                    'window': result['window'],
                    'train_start': window_info[result['window']-1]['train_start'],
                    'train_end': window_info[result['window']-1]['train_end'],
                    'test_end': window_info[result['window']-1]['test_end'],
                    'realized_return': result.get('realized_return', np.nan),
                    'realized_sharpe': result.get('realized_sharpe', np.nan),
                    'var_ratio': result.get('var_ratio', np.nan),
                    'frobenius': result.get('frobenius', np.nan)
                }
                window_stats.append(window_data)
        
        # Convert to DataFrame and save
        window_df = pd.DataFrame(window_stats)
        window_df.to_csv(output_dir / 'window_statistics.csv', index=False)
        self.logger.print_and_log(f"Saved window statistics to: {output_dir / 'window_statistics.csv'}")
        
        # Create visualizations
        self.create_performance_charts(window_df, output_dir)
        
    def create_performance_charts(self, window_df: pd.DataFrame, output_dir: Path):
        """Create and save performance visualization charts."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        window_df['test_end'] = pd.to_datetime(window_df['test_end'])
        
        for method in window_df['method'].unique():
            method_data = window_df[window_df['method'] == method]
            
            # Returns
            ax1.plot(method_data['test_end'], method_data['realized_return'], 
                    label=method, marker='o', markersize=4, alpha=0.7)
            
            # Portfolio Value
            returns = method_data['realized_return'].values
            cumulative_returns = (1 + returns).cumprod()
            ax2.plot(method_data['test_end'], cumulative_returns, 
                    label=method, marker='o', markersize=4, alpha=0.7)
            
            # Variance Ratio
            ax3.plot(method_data['test_end'], method_data['var_ratio'], 
                    label=method, marker='o', markersize=4, alpha=0.7)
            
            # Frobenius
            ax4.plot(method_data['test_end'], method_data['frobenius'], 
                    label=method, marker='o', markersize=4, alpha=0.7)
        
        # Configure subplots
        ax1.set_title('Test Period Returns Over Time')
        ax2.set_title('Cumulative Portfolio Value')
        ax3.set_title('Variance Ratio Over Time')
        ax4.set_title('Frobenius Norm Over Time')
        
        ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_metrics.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Summary statistics
        summary_stats = []
        for method in window_df['method'].unique():
            method_data = window_df[window_df['method'] == method]
            stats = {
                'method': method,
                'var_ratio_mean': method_data['var_ratio'].mean(),
                'var_ratio_std': method_data['var_ratio'].std(),
                'frobenius_mean': method_data['frobenius'].mean(),
                'frobenius_std': method_data['frobenius'].std(),
                'return_mean': method_data['realized_return'].mean(),
                'return_std': method_data['realized_return'].std()
            }
            summary_stats.append(stats)
            
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY BY METHOD")
        print("="*80)
        print(f"{'Method':20} {'Var Ratio':>10} {'Var Std':>10} {'Frob Mean':>10} {'Frob Std':>10} {'Ret Mean':>10} {'Ret Std':>10}")
        print("-"*80)
        for stats in summary_stats:
            print(f"{stats['method']:20} {stats['var_ratio_mean']:10.3f} {stats['var_ratio_std']:10.3f} "
                  f"{stats['frobenius_mean']:10.3f} {stats['frobenius_std']:10.3f} "
                  f"{stats['return_mean']:10.3f} {stats['return_std']:10.3f}")
        print("="*80)
    
    
    def evaluate_rolling_windows(
        self,
        returns: pd.DataFrame,
        lookback_days: int = 378,  # 1.5 years
        test_window: int = 63,     # 3 months
        min_periods: int = 252,    # Minimum required periods
        methods: list = ['identity', 'const_corr', 'single_factor', 'rscm', 'dual_shrinkage', 'nonlinear']
    ) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
        """Evaluate methods using rolling windows with fixed lookback."""
        
        results = {method: [] for method in methods + ['sample']}
        window_info = []
        
        # Get all dates
        dates = returns.index.sort_values()
        end_date = dates[-1]
        
        # Roll forward test windows from start to end
        test_end = dates[-1]
        while test_end > dates[0]:
            test_start = test_end - pd.Timedelta(days=test_window)
            train_end = test_start
            train_start = train_end - pd.Timedelta(days=lookback_days)
            
            # Get data for this window
            train_rets = returns.loc[train_start:train_end]
            test_rets = returns.loc[test_start:test_end]
            
            if len(train_rets) < min_periods:
                test_end = test_start
                continue
                
            # Store window info
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
            test_end = test_start
            
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
                        est_cov, true_cov, train_rets, test_rets, method
                    )
                    
                    metrics['window'] = len(window_info)
                    metrics['method'] = method
                    results[method].append(metrics)
                    if 'var_ratio' in metrics:
                        print(f"\n{method.upper()} METHOD - Window {len(window_info)}:")
                        print(f"Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
                        print(f"Test:  {train_end.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
                        print(f"Predicted portfolio variance: {metrics['pred_var']:.6f}")
                        print(f"Realized portfolio variance:  {metrics['real_var']:.6f}")
                        print(f"Variance ratio:              {metrics['var_ratio']:.3f}")
                        print(f"(Perfect prediction would be 1.000)")
                    
                except Exception as e:
                    self.logger.print_and_log(f"Error evaluating {method}: {str(e)}")
        
        # Save detailed window statistics and create visualizations
        self.save_window_statistics(results, window_info, Path("data/transformed"))
        
        # Create summary
        summary = self.create_summary(results, window_info)
        
        return summary, results
    
    def get_ensemble_estimate(
        self,
        returns: pd.DataFrame,
        base_methods: list = ['sample', 'identity', 'const_corr', 'single_factor', 'rscm', 'dual_shrinkage', 'nonlinear'],
        ensemble_type: str = 'smallest'
    ) -> np.ndarray:
        """Get ensemble covariance estimate using element-wise combination."""
        n = returns.shape[1]
        estimates = []
        
        for method in base_methods:
            try:
                if method == 'sample':
                    est = returns.cov().values
                else:
                    est = self.get_shrinkage_estimate(returns, method)
                estimates.append(est)
            except Exception as e:
                self.logger.print_and_log(f"Error in ensemble with {method}: {str(e)}")
                
        if not estimates:
            raise ValueError("No valid estimates for ensemble")
            
        result = np.zeros((n, n))
        estimates_array = np.array(estimates)
        
        for i in range(n):
            for j in range(i + 1):
                values = estimates_array[:, i, j]
                values = values[~np.isnan(values)]
                
                if len(values) == 0:
                    raise ValueError(f"No valid estimates for position ({i}, {j})")
                
                sorted_vals = np.sort(values)
    
                if ensemble_type == 'third_smallest':
                    val = sorted_vals[2] if len(sorted_vals) > 2 else sorted_vals[-1]
                elif ensemble_type == 'trimmed_mean':
                    if len(values) >= 4:
                        q1, q3 = np.percentile(values, [25, 75])
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        trimmed = values[(values >= lower) & (values <= upper)]
                        val = np.mean(trimmed)
                    else:
                        val = np.mean(values)
                elif ensemble_type == 'third_and_mean':
                    # Get third smallest
                    third = sorted_vals[2] if len(sorted_vals) > 2 else sorted_vals[-1]
                    
                    # Get trimmed mean
                    if len(values) >= 4:
                        q1, q3 = np.percentile(values, [25, 75])
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        trimmed = values[(values >= lower) & (values <= upper)]
                        mean_val = np.mean(trimmed)
                    else:
                        mean_val = np.mean(values)
                    
                    # Average them
                    val = (third + mean_val) / 2
                        
                result[i, j] = val
                if i != j:
                    result[j, i] = val
    
        eigenvals = linalg.eigvalsh(result)
        if np.min(eigenvals) < 1e-10:
            min_eig = np.maximum(eigenvals.min(), 1e-10)
            result += min_eig * np.eye(n)
            
        return result

        
        
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
        elif method == 'rscm':
            est_cov = rscm_shrinkage(returns_array)
        elif method == 'dual_shrinkage':
            est_cov = dual_shrinkage(returns)
        elif method == 'nonlinear':
            est_cov = nonlinear_analytical_shrinkage(returns_array, demean)
        elif method == 'ensemble_third':
            est_cov = self.get_ensemble_estimate(returns, ensemble_type='third_smallest')
        elif method == 'ensemble_mean':
            est_cov = self.get_ensemble_estimate(returns, ensemble_type='trimmed_mean')
        elif method == 'ensemble_third_mean':
            est_cov = self.get_ensemble_estimate(returns, ensemble_type='third_and_mean')
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
            'frobenius', 'var_ratio', 'pred_var', 'real_var',
            'realized_return', 'realized_vol', 'realized_sharpe'
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
        methods = [
            'identity', 'const_corr', 'single_factor', 'rscm', 'dual_shrinkage', 
            'nonlinear', 'ensemble_third',
            'ensemble_mean', 'ensemble_third_mean'
        ]    
        summary, detailed_results = evaluator.evaluate_rolling_windows(
            returns,
            lookback_days=378,    # 252 is 1 year training
            test_window=63,     # 252 is 1 year testing
            min_periods=252,
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
