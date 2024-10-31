import pandas as pd
import numpy as np
from pathlib import Path
import sys

def debug_print(msg, data=None, max_lines=10):
    print(f"\nDEBUG: {msg}")
    if data is not None:
        if isinstance(data, pd.DataFrame):
            print(f"Shape: {data.shape}")
            print(f"First {max_lines} rows:")
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
                print("All items:")
                print(data_list)
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

def filter_companies_by_data_quality(df: pd.DataFrame, threshold: float = 0.99) -> tuple[pd.DataFrame, dict]:
    """
    Filter companies based on data availability when other companies have data.
    First removes entirely empty columns, then applies threshold-based filtering.
    
    Args:
        df: DataFrame with companies as columns and dates as index
        threshold: Minimum proportion of companies required to have data for a row to be considered
    
    Returns:
        Tuple of (filtered DataFrame, statistics dictionary)
    """
    debug_print("Starting company filtering based on data quality")
    initial_companies = df.columns.tolist()
    initial_count = len(initial_companies)
    
    # First, identify and remove completely empty columns
    empty_columns = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=empty_columns)
    debug_print(f"Removed {len(empty_columns)} completely empty columns", empty_columns)
    
    # Now calculate the minimum number of companies needed to consider a row valid
    # using the count of non-empty columns
    min_companies_threshold = int(len(df.columns) * threshold)
    
    # For each row, count how many companies have data
    companies_with_data_per_row = df.notna().sum(axis=1)
    
    # Identify rows where enough companies have data
    valid_rows = companies_with_data_per_row >= min_companies_threshold
    debug_print(f"Found {valid_rows.sum()} rows with >= {threshold*100}% companies having data")
    
    # For these rows, any company with missing data should be removed
    companies_to_remove = set()
    
    for company in df.columns:
        # Check if company has any missing data in rows where most others have data
        missing_in_valid_rows = df[company][valid_rows].isna().any()
        if missing_in_valid_rows:
            companies_to_remove.add(company)
    
    debug_print(f"Companies to be removed due to missing data: {len(companies_to_remove)}", companies_to_remove)
    
    # Remove the identified companies
    remaining_companies = [col for col in df.columns if col not in companies_to_remove]
    df_filtered = df[remaining_companies].copy()
    
    # Calculate statistics
    stats = {
        'initial_company_count': initial_count,
        'empty_columns': empty_columns,
        'empty_columns_count': len(empty_columns),
        'removed_companies': list(companies_to_remove),
        'removed_count': len(companies_to_remove),
        'total_removed': len(empty_columns) + len(companies_to_remove),
        'removal_percentage': ((len(empty_columns) + len(companies_to_remove)) / initial_count) * 100,
        'remaining_count': len(remaining_companies)
    }
    
    debug_print("After filtering:", df_filtered)
    return df_filtered, stats

def load_and_process_data(file_path: str, start_date: str) -> tuple[pd.DataFrame, dict]:
    try:
        debug_print(f"Reading file: {file_path}")
        
        # Determine correct separator
        sep = determine_separator(str(file_path))
        debug_print(f"Detected separator: {repr(sep)}")
        
        # Read the file with detected separator
        df = pd.read_csv(
            file_path,
            sep=sep,
            skipinitialspace=True,
            dtype={'date': str}
        )
        
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= start_date].copy()
        df.set_index('date', inplace=True)
        debug_print(f"After filtering to start_date {start_date}:", df)
        
        if df.empty:
            raise ValueError(f"No data found after {start_date}")
        
        # Get all price columns
        price_cols = [col for col in df.columns if col.endswith('_close')]
        debug_print("Price columns found:", price_cols)
        
        if not price_cols:
            raise ValueError("No price columns found in data")
        
        # Keep only price columns
        df_prices = df[price_cols].copy()
        
        # Apply the new filtering based on data quality
        df_filtered, quality_stats = filter_companies_by_data_quality(df_prices, threshold=0.99)
        
        if df_filtered.empty:
            raise ValueError("No companies remaining after filtering")
        
        stats = {
            'date_range': (df_filtered.index.min(), df_filtered.index.max()),
            'total_dates': len(df_filtered),
            **quality_stats
        }
        
        return df_filtered, stats
        
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
    cov_matrix = returns.cov()
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
            
            # Use command line argument for start_date if provided, otherwise use 5 years
            if len(sys.argv) > 1:
                start_date = sys.argv[1]
            else:
                days_lookback = 1825  # 5 years
                start_date = (pd.Timestamp.now() - pd.Timedelta(days=days_lookback)).strftime('%Y-%m-%d')
            
            debug_print(f"Using start date: {start_date}")
            
            try:
                df, stats = load_and_process_data(str(file_path), start_date)
                
                print("\nProcessing Statistics:")
                print(f"Start date: {start_date}")
                print(f"Date range: {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
                print(f"Total dates in range: {stats['total_dates']}")
                print(f"Initial companies: {stats['initial_company_count']}")
                print(f"Empty columns removed: {stats['empty_columns_count']}")
                print(f"Companies removed due to missing data: {stats['removed_count']}")
                print(f"Total removed: {stats['total_removed']} ({stats['removal_percentage']:.1f}%)")
                print(f"Remaining companies: {stats['remaining_count']}")
                
                if stats['empty_columns_count'] > 0:
                    print("\nCompletely empty columns removed:")
                    for company in stats['empty_columns']:
                        print(f"- {company.replace('_close', '')}")
                
                if stats['removed_count'] > 0:
                    print("\nCompanies removed due to missing data:")
                    for company in stats['removed_companies']:
                        print(f"- {company.replace('_close', '')}")
                
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
