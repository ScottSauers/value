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
            if len(data_list) > 20:
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
        
        # Read the file
        if str(file_path).endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            df = pd.read_csv(file_path)
            
        # Ensure date column exists
        if 'date' not in df.columns:
            raise ValueError("No 'date' column found in the data")
            
        # Convert date and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= start_date].copy()
        df.set_index('date', inplace=True)
        debug_print(f"After filtering to start_date {start_date}:", df)
        
        if df.empty:
            raise ValueError(f"No data found after {start_date}")
        
        # Get all price columns
        price_cols = [col for col in df.columns if col.endswith('_close')]
        debug_print("Price columns found:", price_cols)
        
        # Get only companies with sufficient data
        companies_with_data = []
        min_valid_points = 30  # Minimum number of valid data points required
        for col in price_cols:
            valid_data = df[col].notna()
            valid_count = valid_data.sum()
            if valid_count >= min_valid_points:
                companies_with_data.append(col)
                debug_print(f"Company {col}: {valid_count} valid data points")
        
        debug_print(f"Companies with sufficient data: {len(companies_with_data)}", companies_with_data)
        
        if not companies_with_data:
            raise ValueError("No companies with sufficient valid price data found")
            
        # Keep only price columns with sufficient data
        df_clean = df[companies_with_data].copy()
        debug_print("Sample of cleaned data:", df_clean.head())
        
        stats = {
            'initial_companies': len(price_cols),
            'final_companies': len(companies_with_data),
            'removed_companies': len(price_cols) - len(companies_with_data),
            'removal_percentage': (len(price_cols) - len(companies_with_data)) / len(price_cols) * 100,
            'date_range': (df.index.min(), df.index.max()),
            'total_dates': len(df)
        }
        
        return df_clean, stats
        
    except Exception as e:
        print(f"Error in load_and_process_data: {str(e)}")
        raise

def calculate_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate covariance matrix using pandas built-in cov method.
    """
    debug_print("Starting covariance calculation")
    
    if df.empty:
        raise ValueError("Empty DataFrame provided")
    
    # Calculate returns (percentage change)
    debug_print("Original data sample:", df.head())
    returns = df.pct_change()
    debug_print(f"Returns shape: {returns.shape}")
    debug_print("Returns sample:", returns.head())
    
    # Calculate covariance matrix
    cov_matrix = returns.cov(min_periods=30)
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
                print(f"Initial companies: {stats['initial_companies']}")
                print(f"Final companies: {stats['final_companies']}")
                print(f"Removed companies: {stats['removed_companies']} ({stats['removal_percentage']:.1f}%)")
                
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
