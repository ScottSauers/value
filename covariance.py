def calculate_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate covariance matrix with improved handling of missing data.
    
    Args:
        df: DataFrame with date index and price columns ending in '_close'
        
    Returns:
        DataFrame containing the covariance matrix
    """
    debug_print("Starting covariance calculation")
    
    # Get price columns
    price_columns = [col for col in df.columns if col.endswith('_close')]
    debug_print("Computing covariance for columns:", price_columns)
    
    if not price_columns:
        raise ValueError("No valid price columns found after cleaning")
    
    # Calculate returns
    returns = df[price_columns].pct_change()
    debug_print(f"Returns shape before processing: {returns.shape}")
    debug_print("Sample returns:", returns.head())
    
    # Instead of dropping all NA rows, we'll calculate pairwise covariances
    cov_matrix = pd.DataFrame(index=price_columns, columns=price_columns)
    
    for i, col1 in enumerate(price_columns):
        for j, col2 in enumerate(price_columns[i:], i):
            # Get paired data without NaNs
            paired_data = returns[[col1, col2]].dropna()
            
            if len(paired_data) > 1:  # Need at least 2 points for covariance
                cov = paired_data[col1].cov(paired_data[col2])
                cov_matrix.loc[col1, col2] = cov
                cov_matrix.loc[col2, col1] = cov  # Matrix is symmetric
            else:
                cov_matrix.loc[col1, col2] = np.nan
                cov_matrix.loc[col2, col1] = np.nan
    
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
