from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict
from shrinkage import shrinkage_estimation
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
    debug_print("Initial price data shape", df_prices.shape)
    
    # Remove companies with too many missing values
    pct_missing = df_prices.isnull().mean()
    valid_cols = pct_missing[pct_missing < 0.01].index
    df_prices = df_prices[valid_cols]
    
    # Forward fill remaining missing values
    df_prices = df_prices.fillna(method='ffill')
    
    debug_print("Price data after cleaning", df_prices.shape)
    return df_prices

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns from price data."""
    # Fill any remaining NAs before calculating returns
    df_filled = df.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate returns with explicit fill_method=None since we pre-filled
    returns = df_filled.pct_change(fill_method=None).dropna()
    returns.columns = [col.replace('_close', '') for col in returns.columns]
    
    # Remove any infinite values
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna()
    
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
    """
    Calculate minimum variance portfolio weights with improved numerical stability.
    """
    try:
        n = len(cov_matrix)
        ones = np.ones(n)
        
        # Use scipy's more stable solver instead of direct inversion
        weights = linalg.solve(cov_matrix.values, ones, assume_a='pos')
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        # Check for reasonable weights
        if np.any(np.abs(weights) > 10):  # Arbitrary threshold for "too extreme"
            raise ValueError("Extreme weights detected")
            
        return pd.Series(weights, index=cov_matrix.index)
        
    except Exception as e:
        debug_print(f"Error in minimum_variance_portfolio: {str(e)}")
        # Fall back to equal weights if optimization fails
        n = len(cov_matrix)
        return pd.Series(np.ones(n) / n, index=cov_matrix.index)

def portfolio_variance(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Calculate portfolio variance with error checking."""
    try:
        var = weights @ cov_matrix @ weights
        if np.isnan(var) or np.isinf(var):
            raise ValueError("Invalid variance calculated")
        return var
    except Exception as e:
        debug_print(f"Error calculating portfolio variance: {str(e)}")
        return np.nan

def evaluate_out_of_sample(
    returns: pd.DataFrame,
    estimation_window: int = 252,  # 1 year of daily data
    prediction_window: int = 63,   # 3 months
    methods: list = ['identity', 'const_corr', 'single_factor', 'nonlinear']
) -> Dict[str, list]:
    """
    Evaluate shrinkage methods using out-of-sample portfolio variance.
    """
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
        
        # Skip if we have too few observations in either window
        if len(est_returns) < estimation_window/2 or len(val_returns) < prediction_window/2:
            debug_print("Skipping window due to insufficient data")
            continue
            
        # Calculate realized covariance
        realized_cov = val_returns.cov()
        
        # Sample covariance
        try:
            sample_cov = est_returns.cov()
            w_sample = minimum_variance_portfolio(sample_cov)
            realized_var_sample = portfolio_variance(w_sample, realized_cov)
            results['sample'].append(realized_var_sample)
        except Exception as e:
            debug_print(f"Error with sample covariance: {str(e)}")
            results['sample'].append(np.nan)
        
        # Shrinkage methods
        for method in methods:
            try:
                cov_est = calculate_shrinkage_covariance(est_returns, method=method)
                if isinstance(cov_est, tuple):
                    cov_est = cov_est[0]
                
                weights = minimum_variance_portfolio(cov_est)
                realized_var = portfolio_variance(weights, realized_cov)
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
        
        # Find latest price files
        daily_files = list(base_dir.glob("price_data_1d_*.csv"))
        
        if not daily_files:
            raise ValueError("No daily price data files found")
        
        latest_daily = max(daily_files)
        
        # Use 6 years of data (5 for estimation, 1 for validation)
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=2190)).strftime('%Y-%m-%d')
        
        # Load and process data
        df_prices = load_price_data(str(latest_daily), start_date)
        returns = calculate_returns(df_prices)
        
        debug_print("Final returns data shape before evaluation", returns.shape)
        
        # Evaluate all methods out-of-sample
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
