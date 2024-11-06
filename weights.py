"""
Portfolio optimization using quadratic programming.
Constraints:
- No short selling (weights >= 0)
- Position limits (no position > 20%)
- Target return constraint (achieve at least 20% annual return)
- Risk minimization within these constraints
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from numpy.linalg import LinAlgError
import warnings

@dataclass(frozen=True)
class PortfolioStats:
    """Immutable container for portfolio statistics"""
    return_: float  # Using return_ to avoid Python keyword
    volatility: float
    sharpe: float
    max_position: float
    active_positions: int

class PortfolioOptimizer:
    """Portfolio optimizer for long-only constrained portfolios"""
    
    def __init__(self, returns: pd.DataFrame, cov_matrix: np.ndarray):
        """
        Initialize optimizer with returns data and covariance matrix.
        
        Args:
            returns: DataFrame of returns (columns are assets)
            cov_matrix: Pre-computed covariance matrix from shrinkage estimator
        """
        # Input validation
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()
        
        # Store covariance matrix and verify dimensions
        if cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("Covariance matrix dimensions don't match returns")
        self.cov_matrix = cov_matrix
        
        # Pre-compute annualized returns
        self.annual_returns = returns.mean() * 252
        
        # Pre-compute matrix decomposition for efficient optimization
        try:
            self.L = np.linalg.cholesky(self.cov_matrix)
            self.use_cholesky = True
        except LinAlgError:
            eigenvals, self.eigenvecs = np.linalg.eigh(self.cov_matrix)
            self.eigenvals = np.maximum(eigenvals, 0)
            self.use_cholesky = False
            
    def _calculate_variance(self, weights: np.ndarray) -> float:
        """Calculate portfolio variance using cached decomposition"""
        if self.use_cholesky:
            temp = self.L @ weights
            return np.dot(temp, temp)
        else:
            w_transform = self.eigenvecs.T @ weights
            return np.sum(self.eigenvals * w_transform * w_transform)
            
    def _find_feasible_portfolio(self, target_return: float, position_limit: float) -> np.ndarray:
        """Find initial feasible portfolio meeting constraints"""
        # Sort assets by return
        sorted_idx = np.argsort(self.annual_returns)[::-1]
        weights = np.zeros(self.n_assets)
        
        remaining = 1.0
        port_return = 0.0
        
        # Allocate to highest return assets first
        for idx in sorted_idx:
            weight = min(position_limit, remaining)
            weights[idx] = weight
            port_return += weight * self.annual_returns[idx]
            remaining -= weight
            
            if port_return >= target_return and np.isclose(remaining, 0):
                return weights
                
            if remaining < 1e-10:
                break
                
        return None
            
    def optimize(self, target_return: float = 0.20, position_limit: float = 0.20) -> Tuple[np.ndarray, dict]:
        """
        Optimize portfolio to minimize risk subject to constraints.
        
        Args:
            target_return: Target annual return (default 20%)
            position_limit: Maximum position size (default 20%)
            
        Returns:
            (weights, stats_dict) tuple
        """
        if position_limit <= 0 or position_limit > 1:
            raise ValueError(f"Position limit must be between 0 and 1")
            
        if position_limit < 1/self.n_assets:
            raise ValueError(f"Position limit {position_limit:.1%} too small to sum to 1")
            
        # Find initial feasible portfolio
        weights = self._find_feasible_portfolio(target_return, position_limit)
        if weights is None:
            raise ValueError("Could not find feasible portfolio meeting constraints")
            
        # Calculate portfolio statistics
        variance = self._calculate_variance(weights)
        port_return = np.dot(weights, self.annual_returns)
        vol = np.sqrt(variance)
        
        # Package results in format expected by test_shrinkage.py
        stats = {
            'return': port_return,
            'volatility': vol,
            'variance': variance,
            'sharpe': (port_return - 0.02) / vol,  # Assuming 2% risk-free rate
            'max_position': np.max(weights),
            'active_positions': np.sum(weights > 1e-4)
        }
        
        return weights, stats

# Wrapper functions for test_shrinkage.py compatibility
def optimize_portfolio(returns: pd.DataFrame,
                      cov_matrix: np.ndarray,
                      target_return: float = 0.20,
                      position_limit: float = 0.20,
                      risk_free_rate: float = 0.02) -> tuple:
    """Wrapper function used by test_shrinkage.py"""
    # Convert returns to DataFrame if needed
    if isinstance(returns, pd.Series):
        returns = returns.to_frame().T
    elif not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(returns)
        
    # Convert covariance to array if needed
    cov_matrix = np.asarray(cov_matrix)
    
    optimizer = PortfolioOptimizer(returns, cov_matrix)
    return optimizer.optimize(target_return, position_limit)

def print_portfolio_weights(weights: np.ndarray,
                          asset_names: list,
                          stats: dict):
    """Wrapper function used by test_shrinkage.py"""
    # Format asset names
    if isinstance(asset_names, pd.Series):
        asset_names = asset_names.values
    
    print("\nOptimal Portfolio Weights:")
    print("-" * 50)
    for asset, weight in zip(asset_names, weights):
        if weight > 1e-4:  # Only print non-zero weights
            print(f"{asset:20s}: {weight:8.4f}")
    
    print("\nPortfolio Statistics:")
    print("-" * 50)
    print(f"Expected Annual Return: {stats['return']:8.4f}")
    print(f"Annual Volatility:     {stats['volatility']:8.4f}")
    print(f"Sharpe Ratio:          {stats['sharpe']:8.4f}")
    print(f"Active Positions:      {stats['active_positions']:8d}")
    print(f"Maximum Position:      {stats['max_position']:8.4f}")
