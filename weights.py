"""
Portfolio optimization utilities implementing minimum variance optimization
with target return and position limit constraints.
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd

def get_portfolio_stats(weights: np.ndarray, 
                       returns: np.ndarray,
                       cov_matrix: np.ndarray) -> tuple:
    """
    Calculate portfolio statistics (return, volatility, Sharpe ratio).
    
    Args:
        weights: Array of portfolio weights
        returns: Array of asset returns
        cov_matrix: Covariance matrix of returns
        
    Returns:
        tuple: (portfolio return, portfolio variance)
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    return portfolio_return, portfolio_variance

def optimize_portfolio(returns: pd.DataFrame,
                      cov_matrix: np.ndarray,
                      target_return: float = 0.20,  # 20% annual return
                      position_limit: float = 0.20,  # 20% max position
                      risk_free_rate: float = 0.02) -> tuple:
    """
    Optimize portfolio for minimum variance subject to constraints.
    
    Args:
        returns: DataFrame of asset returns
        cov_matrix: Covariance matrix of returns
        target_return: Target annual portfolio return
        position_limit: Maximum allowed position size
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        
    Returns:
        tuple: (optimal weights, portfolio stats)
    """
    n_assets = returns.shape[1]
    
    # Initial guess: equal weights
    init_weights = np.array([1.0/n_assets] * n_assets)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        # Target return constraint (annualized)
        {'type': 'eq', 
         'fun': lambda w: np.sum(returns.mean() * w) * 252 - target_return}
    ]
    
    # Position size bounds
    bounds = tuple((0, position_limit) for _ in range(n_assets))
    
    # Objective: minimize portfolio variance
    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Optimize
    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    optimal_weights = result.x
    
    # Calculate portfolio statistics
    port_return, port_variance = get_portfolio_stats(
        optimal_weights, returns.values, cov_matrix
    )
    
    stats = {
        'return': port_return,
        'variance': port_variance,
        'volatility': np.sqrt(port_variance) * np.sqrt(252),  # Annualized
        'sharpe': (port_return - risk_free_rate) / (np.sqrt(port_variance) * np.sqrt(252))
    }
    
    return optimal_weights, stats

def print_portfolio_weights(weights: np.ndarray, 
                          asset_names: list,
                          stats: dict):
    """Print formatted portfolio weights and statistics."""
    print("\nOptimal Portfolio Weights:")
    print("-" * 50)
    for asset, weight in zip(asset_names, weights):
        if weight > 0.0001:  # Only print non-zero weights
            print(f"{asset:20s}: {weight:8.4f}")
    
    print("\nPortfolio Statistics:")
    print("-" * 50)
    print(f"Expected Annual Return: {stats['return']:8.4f}")
    print(f"Annual Volatility:     {stats['volatility']:8.4f}")
    print(f"Sharpe Ratio:          {stats['sharpe']:8.4f}")
    print(f"Sum of weights:        {np.sum(weights):8.4f}")
    print(f"Max position:          {np.max(weights):8.4f}")
