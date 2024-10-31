import pandas as pd
import numpy as np
from pathlib import Path
import sys

def debug_print(msg, data=None, max_lines=10):
    print(f"\nDEBUG: {msg}")
    if data is not None:
        if isinstance(data, pd.DataFrame):
            print(f"Shape: {data.shape}")
            print("First {max_lines} rows:")
            print(data.head(max_lines))
        elif isinstance(data, (list, set)):
            data_list = list(data)
            print(f"Length: {len(data_list)}")
            if len(data_list) > 20:  # If more than 20 items, show first and last 10
                print("First 10 items:")
                print(data_list[:10])
                print("Last 10 items:")
                print(data_list[-10:])
            else:
                print(f"All items:")
                print(data_list)
        else:
            print(data)
    sys.stdout.flush()

def load_and_process_data(file_path: str, start_date: str) -> tuple[pd.DataFrame, dict]:
    try:
        debug_print(f"Reading file: {file_path}")
        
        if str(file_path).endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            df = pd.read_csv(file_path)
            
        # Convert date explicitly
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df[df['date'] >= start_date].copy()
        debug_print(f"After filtering to start_date {start_date}:", df)
        
        if df.empty:
            raise ValueError(f"No data found after {start_date}")
        
        # Get all price columns
        price_cols = [col for col in df.columns if col.endswith('_close')]
        debug_print("Price columns found:", price_cols)
        
        # Get only companies with data
        companies_with_data = []
        min_valid_points = 30  # Minimum number of valid data points required
        for col in price_cols:
            valid_data = df[col].notna()
            valid_count = valid_data.sum()
            if valid_count >= min_valid_points:
                companies_with_data.append(col)
                debug_print(f"Company {col}: {valid_count} valid data points")
        
        debug_print(f"Companies with any data: {len(companies_with_data)}", companies_with_data)
        debug_print("Sample of data:", df[companies_with_data].head())
        
        if not companies_with_data:
            raise ValueError("No companies with valid price data found")
            
        # Create output dataframe and stats
        df_clean = df[['date'] + companies_with_data].copy()
        
        stats = {
            'initial_companies': len(companies_with_data),
            'removed_companies': [],
            'final_companies': len(companies_with_data),
            'removal_percentage': 0.0,
            'date_range': (df['date'].min(), df['date'].max()),
            'total_dates': len(df)
        }
        
        return df_clean, stats
        
    except Exception as e:
        print(f"Error in load_and_process_data: {str(e)}")
        raise

def calculate_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate covariance matrix with improved handling of missing data.
    """
    debug_print("Starting covariance calculation")
    
    # Get price columns
    price_columns = [col for col in df.columns if col.endswith('_close')]
    debug_print("Computing covariance for columns:", price_columns)
    
    if not price_columns:
        raise ValueError("No valid price columns found after cleaning")
    
    # Calculate returns with explicit fill_method=None
    returns = df[price_columns].pct_change(fill_method=None)
    debug_print(f"Returns shape before processing: {returns.shape}")
    debug_print("Sample returns:", returns.head())
    
    # Instead of dropping all NA rows, we'll calculate pairwise covariances
    cov_matrix = pd.DataFrame(index=price_columns, columns=price_columns)
    min_periods = 30  # Minimum number of overlapping periods required
    
    total_pairs = len(price_columns) * (len(price_columns) - 1) // 2
    processed_pairs = 0
    
    for i, col1 in enumerate(price_columns):
        for j, col2 in enumerate(price_columns[i:], i):
            # Get paired data without NaNs
            paired_data = pd.DataFrame({
                'col1': returns[col1],
                'col2': returns[col2]
            }).dropna()
            
            if len(paired_data) >= min_periods:
                cov = paired_data['col1'].cov(paired_data['col2'])
                cov_matrix.loc[col1, col2] = cov
                cov_matrix.loc[col2, col1] = cov  # Matrix is symmetric
            else:
                cov_matrix.loc[col1, col2] = np.nan
                cov_matrix.loc[col2, col1] = np.nan
            
            processed_pairs += 1
            if processed_pairs % 1000 == 0:
                print(f"Processed {processed_pairs}/{total_pairs} pairs...")
    
    debug_print(f"Covariance matrix shape: {cov_matrix.shape}")
    debug_print("Sample of covariance matrix:", cov_matrix.iloc[:5, :5])
    
    # Clean up column names
    cov_matrix.columns = [col.replace('_close', '') for col in cov_matrix.columns]
    cov_matrix.index = [col.replace('_close', '') for col in cov_matrix.index]
    
    # Calculate the percentage of non-null values
    non_null_percent = (cov_matrix.notna().sum().sum() / 
                       (cov_matrix.shape[0] * cov_matrix.shape[1]) * 100)
    debug_print(f"Percentage of non-null values in covariance matrix: {non_null_percent:.2f}%")
    
    return cov_matrix

def main():
    try:
        debug_print("Starting main execution")
        base_dir = Path("data/transformed")
        
        weekly_files = list(base_dir.glob("price_data_1wk_*.tsv"))
        daily_files = list(base_dir.glob("price_data_1d_*.csv"))
        
        debug_print("Files found:", {
            'weekly': [f.name for f in weekly_files],
            'daily': [f.name for f in daily_files]
        })
        
        if not weekly_files and not daily_files:
            print("Error: No price data files found in data/transformed directory")
            sys.exit(1)
        
        latest_weekly = max(weekly_files, default=None)
        latest_daily = max(daily_files, default=None)
        
        for file_path in [f for f in [latest_weekly, latest_daily] if f is not None]:
            print(f"\nProcessing {file_path.name}...")
            
            days_lookback = 1825
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=days_lookback)).strftime('%Y-%m-%d')
            debug_print(f"Using start date: {start_date}")
            
            try:
                df, stats = load_and_process_data(str(file_path), start_date)
                
                print("\nProcessing Statistics:")
                print(f"Start date: {start_date}")
                print(f"Date range: {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
                print(f"Total dates in range: {stats['total_dates']}")
                print(f"Total companies: {stats['initial_companies']}")
                
                print("\nCalculating covariance matrix...")
                cov_matrix = calculate_covariance(df)
                
                output_name = f"covariance_matrix_{file_path.stem.split('_')[2]}.csv"
                output_path = base_dir / output_name
                
                cov_matrix.to_csv(output_path)
                print(f"Covariance matrix saved to {output_path}")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
