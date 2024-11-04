from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict
from shrinkage import shrinkage_estimation
import sys

def debug_print(msg, data=None, max_lines=10):
    """Debug printing function for monitoring execution."""
    print(f"\nDEBUG: {msg}")
    if data is not None:
        if isinstance(data, pd.DataFrame):
            print(f"Shape: {data.shape}")
            print(f"First {max_lines} rows:")
            print(data.head(max_lines))
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

def load_price_data(file_path: str, start_date: str) -> pd.DataFrame:
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
    
    # Remove companies with too many missing values
    pct_missing = df_prices.isnull().mean()
    valid_cols = pct_missing[pct_missing < 0.01].index
    df_prices = df_prices[valid_cols]
    
    return df_prices

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns from price data."""
    returns = df.pct_change().dropna()
    returns.columns = [col.replace('_close', '') for col in returns.columns]
    return returns

def calculate_shrinkage_covariance(
    returns: pd.DataFrame,
    method: str = 'nonlinear',
    market_returns: Optional[np.ndarray] = None,
    demean: bool = True
) -> Union[Tuple[pd.DataFrame, float], pd.DataFrame]:
    """Calculate covariance matrix using shrinkage estimation."""
    returns_array = returns.values
    
    result = shrinkage_estimation(
        returns=returns_array,
        method=method,
        market_returns=market_returns,
        demean=demean
    )
    
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
    """Calculate minimum variance portfolio weights."""
    n = len(cov_matrix)
    ones = np.ones(n)
    inv_cov = np.linalg.inv(cov_matrix.values)
    weights = inv_cov @ ones / (ones @ inv_cov @ ones)
    return pd.Series(weights, index=cov_matrix.index)

def portfolio_variance(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Calculate portfolio variance."""
    return weights @ cov_matrix @ weights

def evaluate_out_of_sample(
    returns: pd.DataFrame,
    estimation_window: int = 252,  # 1 year of daily data
    prediction_window: int = 63,   # 3 months
    methods: list = ['identity', 'const_corr', 'single_factor', 'nonlinear']
) -> Dict[str, list]:
    """
    Evaluate shrinkage methods using out-of-sample portfolio variance.
    
    Uses a rolling window approach:
    1. Estimate covariance using estimation_window
    2. Form minimum variance portfolio
    3. Calculate realized variance in prediction_window
    4. Roll forward and repeat
    """
    results = {method: [] for method in methods}
    results['sample'] = []  # Also track regular sample covariance
  
    for t in range(0, len(returns) - estimation_window - prediction_window, prediction_window):
        # Split data into estimation and validation periods
        est_returns = returns.iloc[t:t+estimation_window]
        val_returns = returns.iloc[t+estimation_window:t+estimation_window+prediction_window]
        
        # Calculate realized covariance for validation period
        realized_cov = val_returns.cov()
        
        # For each method:
        # 1. Estimate covariance
        # 2. Calculate minimum variance portfolio
        # 3. Calculate realized variance
        
        # Regular sample covariance
        sample_cov = est_returns.cov()
        w_sample = minimum_variance_portfolio(sample_cov)
        realized_var_sample = portfolio_variance(w_sample, realized_cov)
        results['sample'].append(realized_var_sample)
        
        # Shrinkage methods
        for method in methods:
            try:
                cov_est = calculate_shrinkage_covariance(est_returns, method=method)
                if isinstance(cov_est, tuple):
                    cov_est = cov_est[0]  # Take just the matrix for linear methods
                
                weights = minimum_variance_portfolio(cov_est)
                realized_var = portfolio_variance(weights, realized_cov)
                results[method].append(realized_var)
            except Exception as e:
                debug_print(f"Error with method {method} at time {t}: {str(e)}")
                results[method].append(np.nan)
    
    return results

def main():
    try:
        debug_print("Starting main execution")
        base_dir = Path("data/transformed")
        
        # Find latest price files
        weekly_files = list(base_dir.glob("price_data_1wk_*.tsv"))
        daily_files = list(base_dir.glob("price_data_1d_*.csv"))
        
        if not weekly_files and not daily_files:
            print("Error: No price data files found")
            sys.exit(1)
        
        latest_daily = max(daily_files, default=None)
        
        if latest_daily is None:
            print("Error: No daily price data found")
            sys.exit(1)
        
        # Use 6 years of data (5 for estimation, 1 for validation)
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=2190)).strftime('%Y-%m-%d')
        
        # Load and process data
        df_prices = load_price_data(str(latest_daily), start_date)
        returns = calculate_returns(df_prices)
        
        # Evaluate all methods out-of-sample
        methods = ['identity', 'const_corr', 'single_factor', 'nonlinear']
        results = evaluate_out_of_sample(returns, methods=methods)
        
        # Calculate summary statistics
        summary = pd.DataFrame({
            method: {
                'Mean Variance': np.nanmean(variances),
                'Std Variance': np.nanstd(variances),
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
