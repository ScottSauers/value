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
            print(f"Length: {len(data)}")
            print("First {max_lines} items:")
            print(list(data)[:max_lines])
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
        
        df['date'] = pd.to_datetime(df['date'])
        
        debug_print("Initial dataframe:", df)
        debug_print("Columns:", df.columns.tolist())
        
        df = df[df['date'] >= start_date].copy()
        debug_print(f"After filtering to start_date {start_date}:", df)
        
        if df.empty:
            raise ValueError(f"No data found after {start_date}")
        
        price_cols = [col for col in df.columns if col.endswith('_close')]
        debug_print("Price columns found:", price_cols)
        
        companies_with_data = []
        for col in price_cols:
            valid_data = df[col].notna()
            if valid_data.any():
                companies_with_data.append(col)
                debug_print(f"Company {col}: {valid_data.sum()} valid data points")
        
        debug_print(f"Companies with any data: {len(companies_with_data)}", companies_with_data)
        
        if not companies_with_data:
            raise ValueError("No companies with valid price data found")
            
        stats = {
            'initial_companies': len(companies_with_data),
            'removed_companies': [],
            'final_companies': 0,
            'date_range': (df['date'].min(), df['date'].max()),
            'total_dates': len(df)
        }
        
        companies_to_remove = set()
        total_companies = len(companies_with_data)
        threshold = 0.1
        
        for company in companies_with_data:
            missing_ratios = []
            for date in df['date'].unique():
                day_data = df[df['date'] == date][companies_with_data]
                present_companies = day_data.notna().sum()
                max_present = present_companies.max()
                if max_present > 0:
                    company_present = day_data[company].notna().sum()
                    if company_present == 0 and (present_companies / total_companies > (1 - threshold)):
                        companies_to_remove.add(company)
                        break
        
        debug_print(f"Companies to remove: {len(companies_to_remove)}")
        
        clean_companies = [c for c in companies_with_data if c not in companies_to_remove]
        debug_print("Clean companies remaining:", clean_companies)
        
        if clean_companies:
            df_clean = df[['date'] + clean_companies].copy()
            debug_print("Clean dataframe shape:", df_clean.shape)
        else:
            df_clean = df[['date']].copy()
            debug_print("WARNING: No clean companies remain!")
        
        stats['removed_companies'] = list(companies_to_remove)
        stats['final_companies'] = len(clean_companies)
        stats['removal_percentage'] = (len(companies_to_remove) / stats['initial_companies'] * 100)
        
        debug_print("Final statistics:", stats)
        
        return df_clean, stats
        
    except Exception as e:
        print(f"Error in load_and_process_data: {str(e)}")
        raise

def calculate_covariance(df: pd.DataFrame) -> pd.DataFrame:
    debug_print("Starting covariance calculation")
    
    price_columns = [col for col in df.columns if col.endswith('_close')]
    debug_print("Price columns for covariance:", price_columns)
    
    if not price_columns:
        raise ValueError("No valid price columns found after cleaning")
    
    returns = df[price_columns].pct_change()
    debug_print("Returns calculation:", returns)
    
    returns_clean = returns.dropna()
    debug_print("Clean returns shape:", returns_clean.shape)
    
    cov_matrix = returns_clean.cov()
    debug_print("Covariance matrix shape:", cov_matrix.shape)
    
    cov_matrix.columns = [col.replace('_close', '') for col in cov_matrix.columns]
    cov_matrix.index = [col.replace('_close', '') for col in cov_matrix.index]
    
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
