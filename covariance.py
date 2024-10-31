import pandas as pd
import numpy as np
from pathlib import Path
import argparse
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
    # Determine file type and read accordingly
    if file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data after start date
    df = df[df['date'] >= start_date].copy()
    
    if df.empty:
        raise ValueError(f"No data found after {start_date}")
    
    # Get list of all companies (columns that end with '_close')
    companies = [col for col in df.columns if col.endswith('_close')]
    initial_company_count = len(companies)
    
    # Initialize statistics
    stats = {
        'initial_companies': initial_company_count,
        'removed_companies': [],
        'final_companies': 0,
        'removal_percentage': 0.0,
        'date_range': (df['date'].min(), df['date'].max()),
        'total_dates': len(df)
    }
    
    # Find companies with any missing data in rows where others have data
    companies_to_remove = set()
    
    for company in companies:
        # Check if company has missing data in rows where other companies have data
        company_data = df[company]
        has_missing = company_data.isna()
        
        if has_missing.any():
            # Check if these missing values are in rows where other companies have data
            other_companies_data = df[[c for c in companies if c != company]]
            rows_with_other_data = other_companies_data.notna().any(axis=1)
            
            if (has_missing & rows_with_other_data).any():
                companies_to_remove.add(company)
    
    # Remove companies with missing data
    clean_companies = [c for c in companies if c not in companies_to_remove]
    df_clean = df[['date'] + clean_companies]
    
    # Update statistics
    stats['removed_companies'] = list(companies_to_remove)
    stats['final_companies'] = len(clean_companies)
    stats['removal_percentage'] = (len(companies_to_remove) / initial_company_count) * 100
    
    return df_clean, stats

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
    
    # Calculate returns
    returns = df[price_columns].pct_change()
    
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    
    # Clean up column names for the output
    cov_matrix.columns = [col.replace('_close', '') for col in cov_matrix.columns]
    cov_matrix.index = [col.replace('_close', '') for col in cov_matrix.index]
    
    return cov_matrix

def main():
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
        
        # Set start date to 30 days ago for daily data, 180 days for weekly
        days_lookback = 30 if 'price_data_1d_' in file_path.name else 180
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
            print(f"Error processing {file_path.name}: {str(e)}", file=sys.stderr)
            continue

if __name__ == "__main__":
    main()
