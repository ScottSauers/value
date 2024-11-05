from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict
from shrinkage import linear_shrinkage_identity, linear_shrinkage_constant_correlation, linear_shrinkage_single_factor, nonlinear_analytical_shrinkage
import sys
from scipy import linalg

def debug_print(msg, data=None, max_lines=10):
    """Debug printing function for monitoring execution."""
    print(f"\nDEBUG: {msg}")
    if data is not None:
        if isinstance(data, pd.DataFrame):
            print(f"Shape: {data.shape}")
            print(f"First {max_lines} rows:")
            print(data.head(max_lines))
            print("\nData Info:")
            print(data.info())
        else:
            print(data)
    sys.stdout.flush()

def determine_separator(file_path: str) -> str:
    """Determine the correct separator by comparing field counts."""
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

def select_largest_stocks(df_prices: pd.DataFrame, n_stocks: int = 100) -> pd.DataFrame:
    """Select the n largest stocks by average market value."""
    avg_prices = df_prices.mean()
    largest_stocks = avg_prices.nlargest(n_stocks).index
    return df_prices[largest_stocks]

def load_price_data(file_path: str, start_date: str, n_stocks: int = 100) -> pd.DataFrame:
    """Load and preprocess price data from file."""
    debug_print(f"Reading file: {file_path}")
    
    sep = determine_separator(str(file_path))
    debug_print(f"Detected separator: {repr(sep)}")
    
    df = pd.read_csv(
        file_path,
        sep=sep,
        skipinitialspace=True,
        dtype={'date': str}
    )
    
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start_date].copy()
    df.set_index('date', inplace=True)
    
    # Get price columns
    price_cols = [col for col in df.columns if col.endswith('_close')]
    if not price_cols:
        raise ValueError("No price columns found in data")
    
    df_prices = df[price_cols].copy()
    debug_print("Initial price data shape", df_prices.shape)
    
    # Remove companies with too many missing values
    pct_missing = df_prices.isnull().mean()
    valid_cols = pct_missing[pct_missing < 0.01].index
    df_prices = df_prices[valid_cols]
    
    # Select largest stocks
    df_prices = select_largest_stocks(df_prices, n_stocks)
    
    # Forward fill remaining missing values
    df_prices = df_prices.ffill().bfill()
    
    debug_print("Price data after cleaning", df_prices.shape)
    return df_prices

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns from price data."""
    returns = df.pct_change(fill_method=None)
    returns.columns = [col.replace('_close', '') for col in returns.columns]
    
    # Remove any infinite values and first row (which will be NaN)
    returns = returns.iloc[1:]
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how='any')
    
    debug_print("Returns data shape", returns.shape)
    debug_print("Returns summary stats", returns.describe())
    
    return returns

def calculate_shrinkage_covariance(
    returns: pd.DataFrame,
    method: str = 'nonlinear',
    market_returns: Optional[np.ndarray] = None,
    demean: bool = True
) -> Union[Tuple[pd.DataFrame, float], pd.DataFrame]:
    """Calculate covariance matrix using shrinkage estimation."""
    returns_array = returns.values
    
    # Choose appropriate shrinkage method
    if method == 'identity':
        result = linear_shrinkage_identity(returns_array, demean)
    elif method == 'const_corr':
        result = linear_shrinkage_constant_correlation(returns_array, demean)
    elif method == 'single_factor':
        result = linear_shrinkage_single_factor(returns_array, market_returns, demean)
    else:  # nonlinear
        result = nonlinear_analytical_shrinkage(returns_array, demean)
    
    # Convert result to DataFrame
    if isinstance(result, tuple):
        cov_matrix, shrinkage = result
        cov_df = pd.DataFrame(
            cov_matrix, 
            index=returns.columns,
            columns=returns.columns
        )
        return cov_df, shrinkage
    else:
        cov_df = pd.DataFrame(
            result,
            index=returns.columns,
            columns=returns.columns
        )
        return cov_df

def minimum_variance_portfolio(cov_matrix: pd.DataFrame) -> pd.Series:
    """Calculate minimum variance portfolio weights with improved numerical stability."""
    n = len(cov_matrix)
    
    try:
        # Add small diagonal matrix to ensure positive definiteness
        epsilon = 1e-8 * np.trace(cov_matrix) / n
        regularized_cov = cov_matrix.values + epsilon * np.eye(n)
        
        # Solve the system with regularization
        ones = np.ones(n)
        weights = linalg.solve(regularized_cov, ones, assume_a='pos')
        weights = weights / np.sum(weights)
        
        # Additional checks
        if np.any(np.abs(weights) > 5) or np.any(np.isnan(weights)):
            raise ValueError("Unstable weights detected")
        
        return pd.Series(weights, index=cov_matrix.index)
        
    except Exception as e:
        debug_print(f"Error in minimum_variance_portfolio: {str(e)}")
        # Return equal weights portfolio
        return pd.Series(np.ones(n) / n, index=cov_matrix.index)

def portfolio_variance(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Calculate portfolio variance with error checking."""
    try:
        # Ensure alignment
        weights = weights.reindex(cov_matrix.index)
        var = weights @ cov_matrix @ weights
        
        # Sanity checks
        if np.isnan(var) or np.isinf(var) or var < 0:
            raise ValueError("Invalid variance calculated")
        
        return var
    except Exception as e:
        debug_print(f"Error calculating portfolio variance: {str(e)}")
        return np.nan

def evaluate_out_of_sample(
    returns: pd.DataFrame,
    estimation_window: int = 252,  # 1 year
    prediction_window: int = 21,   # 1 month
    min_periods: int = 200,        # Minimum required periods
    methods: list = ['identity', 'const_corr', 'single_factor', 'nonlinear']
) -> Dict[str, list]:
    """Evaluate shrinkage methods using out-of-sample portfolio variance."""
    results = {method: [] for method in methods}
    results['sample'] = []
    
    # Ensure we have enough data
    if len(returns) < estimation_window + prediction_window:
        raise ValueError("Not enough data for out-of-sample evaluation")
    
    debug_print(f"Starting evaluation with {len(returns)} periods of data")
    
    for t in range(0, len(returns) - estimation_window - prediction_window, prediction_window):
        debug_print(f"Evaluation window starting at period {t}")
        
        # Split data
        est_returns = returns.iloc[t:t+estimation_window]
        val_returns = returns.iloc[t+estimation_window:t+estimation_window+prediction_window]
        
        # Skip if we have insufficient data
        if len(est_returns) < min_periods:
            debug_print("Skipping window due to insufficient data")
            continue
        
        # Calculate realized covariance
        realized_cov = val_returns.cov()
        
        # Evaluate each method
        for method in ['sample'] + methods:
            try:
                # Get covariance estimate
                if method == 'sample':
                    cov_est = est_returns.cov()
                else:
                    cov_est = calculate_shrinkage_covariance(est_returns, method=method)
                    if isinstance(cov_est, tuple):
                        cov_est = cov_est[0]
                
                # Calculate portfolio weights and realized variance
                weights = minimum_variance_portfolio(cov_est)
                realized_var = portfolio_variance(weights, realized_cov)
                
                # Store result
                results[method].append(realized_var)
                debug_print(f"Method {method} - realized variance: {realized_var}")
                
            except Exception as e:
                debug_print(f"Error with method {method}: {str(e)}")
                results[method].append(np.nan)
    
    return results

def main():
    try:
        debug_print("Starting main execution")
        base_dir = Path("data/transformed")
        
        daily_files = list(base_dir.glob("price_data_1d_*.csv"))
        if not daily_files:
            raise ValueError("No daily price data files found")
            
        latest_daily = max(daily_files)
        
        # Use 3 years of data
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=1095)).strftime('%Y-%m-%d')
        
        # Load and process data (using only top 100 stocks)
        df_prices = load_price_data(str(latest_daily), start_date, n_stocks=100)
        returns = calculate_returns(df_prices)
        
        debug_print("Final returns data shape before evaluation", returns.shape)
        
        # Evaluate methods
        methods = ['identity', 'const_corr', 'single_factor', 'nonlinear']
        results = evaluate_out_of_sample(returns, methods=methods)
        
        # Calculate summary statistics
        summary = pd.DataFrame({
            method: {
                'Mean Variance': np.nanmean(variances),
                'Std Variance': np.nanstd(variances),
                'Success Rate': np.mean(~np.isnan(variances)),
                'Sharpe Ratio': np.nanmean(variances) / np.nanstd(variances)
            }
            for method, variances in results.items()
        })
        
        print("\nOut-of-Sample Performance Summary:")
        print(summary)
        
        # Save results
        summary.to_csv(base_dir / 'shrinkage_comparison.csv')
        print(f"\nResults saved to {base_dir / 'shrinkage_comparison.csv'}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
