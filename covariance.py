import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

def load_and_process_data(file_path: str, start_date: str) -> tuple[pd.DataFrame, dict]:
    """
    Load price data and process it according to specifications.
    
    Args:
        file_path: Path to the price data CSV/TSV file
        start_date: Date string to start analysis from (YYYY-MM-DD)
        
    Returns:
        tuple: (Processed DataFrame with complete data, Statistics dictionary)
    """
    try:
        # Read the file with explicit date column
        if str(file_path).endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t', parse_dates=['date'])
        else:
            df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Filter data after start date
        df = df[df['date'] >= start_date].copy()
        
        if df.empty:
            raise ValueError(f"No data found after {start_date}")
        
        # Get list of all companies (columns that end with '_close')
        companies = [col for col in df.columns if col.endswith('_close')]
        initial_company_count = len(companies)
        
        if not companies:
            raise ValueError(f"No price columns found (columns ending with '_close')")
        
        # Initialize statistics
        stats = {
            'initial_companies': initial_company_count,
            'removed_companies': [],
            'final_companies': 0,
            'removal_percentage': 0.0,
            'date_range': (df['date'].min(), df['date'].max()),
            'total_dates': len(df)
        }
        
        # Find rows where at least one company has data
        valid_rows = df[companies].notna().any(axis=1)
        
        # For each valid row, check which companies have missing data
        companies_to_remove = []
        for company in companies:
            other_companies = [c for c in companies if c != company]
            
            # For each row where other companies have data
            for idx in df[valid_rows].index:
                if pd.isna(df.at[idx, company]) and df.loc[idx, other_companies].notna().any():
                    companies_to_remove.append(company)
                    break
        
        # Remove duplicates
        companies_to_remove = list(set(companies_to_remove))
        
        # Keep only clean companies
        clean_companies = [c for c in companies if c not in companies_to_remove]
        if clean_companies:
            df_clean = df[['date'] + clean_companies].copy()
        else:
            df_clean = df[['date']].copy()
        
        # Update statistics
        stats['removed_companies'] = companies_to_remove
        stats['final_companies'] = len(clean_companies)
        stats['removal_percentage'] = (len(companies_to_remove) / initial_company_count) * 100
        
        return df_clean, stats
        
    except Exception as e:
        print(f"Error in load_and_process_data: {str(e)}")
        raise

def calculate_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the covariance matrix for the cleaned price data.
    
    Args:
        df: Cleaned DataFrame with only complete data
        
    Returns:
        pd.DataFrame: Covariance matrix
    """
    # Get price columns (ending with _close)
    price_columns = [col for col in df.columns if col.endswith('_close')]
    
    if not price_columns:
        raise ValueError("No valid price columns found after cleaning")
    
    # Calculate returns (handle NA values appropriately)
    returns = df[price_columns].pct_change().dropna()
    
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    
    # Clean up column names for the output
    cov_matrix.columns = [col.replace('_close', '') for col in cov_matrix.columns]
    cov_matrix.index = [col.replace('_close', '') for col in cov_matrix.index]
    
    return cov_matrix

def main():
    try:
        # Define base directory
        base_dir = Path("data/transformed")
        
        # Find most recent weekly and daily files
        weekly_files = list(base_dir.glob("price_data_1wk_*.tsv"))
        daily_files = list(base_dir.glob("price_data_1d_*.csv"))
        
        if not weekly_files and not daily_files:
            print("Error: No price data files found in data/transformed directory")
            sys.exit(1)
        
        # Get most recent files
        latest_weekly = max(weekly_files, default=None)
        latest_daily = max(daily_files, default=None)
        
        # Process both weekly and daily data if available
        for file_path in [f for f in [latest_weekly, latest_daily] if f is not None]:
            print(f"\nProcessing {file_path.name}...")
            
            # Use 5-year lookback for both daily and weekly
            days_lookback = 1825  # ~5 years
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=days_lookback)).strftime('%Y-%m-%d')
            
            try:
                # Process data
                df, stats = load_and_process_data(str(file_path), start_date)
                
                # Print statistics
                print("\nProcessing Statistics:")
                print(f"Start date: {start_date}")
                print(f"Date range: {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
                print(f"Total dates in range: {stats['total_dates']}")
                print(f"Initial number of companies: {stats['initial_companies']}")
                print(f"Companies removed: {len(stats['removed_companies'])} ({stats['removal_percentage']:.1f}%)")
                print(f"Final number of companies: {stats['final_companies']}")
                
                if stats['removed_companies']:
                    print("\nRemoved companies:")
                    for company in sorted(stats['removed_companies']):
                        print(f"  - {company.replace('_close', '')}")
                
                if stats['final_companies'] == 0:
                    print("\nWarning: No companies remaining after filtering!")
                    continue
                
                # Calculate covariance matrix
                print("\nCalculating covariance matrix...")
                cov_matrix = calculate_covariance(df)
                
                # Create output filename based on input file
                output_name = f"covariance_matrix_{file_path.stem.split('_')[2]}.csv"
                output_path = base_dir / output_name
                
                # Save to CSV
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
