"""
Requirements:
- No short selling (weights >= 0)
- Position limits (no position > 20%)
- Target return constraint (achieve at least 20% annual return)
- Risk minimization within these constraints
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Set
from numpy.linalg import LinAlgError
import warnings

@dataclass(frozen=True)
class PortfolioStats:
    """Immutable container for portfolio statistics"""
    annual_return: float
    annual_volatility: float 
    sharpe_ratio: float
    max_position: float
    active_positions: int
    
class PortfolioOptimizer:
    """Fast portfolio optimizer for long-only constrained portfolios"""
    
    def __init__(self, returns: pd.DataFrame, cov_matrix: Optional[np.ndarray] = None):
        """
        Initialize optimizer with returns data and covariance matrix.
        
        Args:
            returns: DataFrame of asset returns (columns are assets)
            cov_matrix: Pre-computed covariance matrix
        """
        # Convert returns to DataFrame if needed
        if isinstance(returns, pd.Series):
            returns = returns.to_frame().T
        elif not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)
        
        self._validate_inputs(returns)
        
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()
        
        # Pre-compute annualized returns
        self.annual_returns = returns.mean() * 252
        

        # numpy array
        self.cov_matrix = np.asarray(cov_matrix)
        
        # Check dimensions
        if self.cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("Covariance matrix dimensions don't match returns")
            
        # Symmetry
        self.cov_matrix = (self.cov_matrix + self.cov_matrix.T) / 2
        
        # Check positive definiteness and fix if needed
        try:
            self.L = np.linalg.cholesky(self.cov_matrix)
            self.use_cholesky = True
        except np.linalg.LinAlgError:
            # If not positive definite, use eigendecomposition with cleaning
            eigenvals, self.eigenvecs = np.linalg.eigh(self.cov_matrix)
            self.eigenvals = np.maximum(eigenvals, 1e-8)  # Force positive eigenvalues
            self.use_cholesky = False
            
            # Reconstruct cleaned covariance matrix
            self.cov_matrix = self.eigenvecs @ np.diag(self.eigenvals) @ self.eigenvecs.T
            
            warnings.warn("Covariance matrix was not positive definite. Eigenvalues were clipped for positive definiteness.")

    @staticmethod
    def _validate_inputs(returns: pd.DataFrame) -> None:
        """Validate input data"""
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
        if returns.isnull().any().any():
            raise ValueError("returns contain NaN values")
        if returns.shape[1] < 2:
            raise ValueError("need at least 2 assets for optimization")
            
    def _calculate_portfolio_variance(self, weights: np.ndarray) -> float:
        """Calculate portfolio variance using cached decomposition"""
        if self.use_cholesky:
            # Use Cholesky decomposition
            temp = self.L @ weights
            return np.dot(temp, temp)
        else:
            # Use eigendecomposition
            w_transform = self.eigenvecs.T @ weights
            return np.sum(self.eigenvals * w_transform * w_transform)

    def _find_feasible_portfolio(self, target_return: float, 
                               position_limit: float) -> Optional[np.ndarray]:
        """Find initial feasible portfolio satisfying all constraints"""
        # Sort assets by return
        sorted_idx = np.argsort(self.annual_returns)[::-1]
        weights = np.zeros(self.n_assets)
        
        remaining_weight = 1.0
        portfolio_return = 0.0
        
        # Allocate to highest return assets until target return met
        for idx in sorted_idx:
            weight = min(position_limit, remaining_weight)
            weights[idx] = weight
            portfolio_return += weight * self.annual_returns[idx]
            remaining_weight -= weight
            
            if portfolio_return >= target_return and np.isclose(remaining_weight, 0):
                return weights
                
            if remaining_weight < 1e-10:
                break
                
        return None

    def _optimize_with_active_set(self, 
                                target_return: float,
                                position_limit: float,
                                active_set: Set[int]) -> Tuple[np.ndarray, Set[int]]:
        """Optimize over active set of assets"""
        n_active = len(active_set)
        active_list = sorted(active_set)
        
        # Extract active portions of matrices
        cov_active = self.cov_matrix[np.ix_(active_list, active_list)]
        returns_active = self.annual_returns[active_list]
        
        # Solve KKT system for active set
        ones = np.ones(n_active)
        
        M = np.block([
            [cov_active, ones.reshape(-1, 1), returns_active.reshape(-1, 1)],
            [ones, 0, 0],
            [returns_active, 0, 0]
        ])
        
        b = np.zeros(n_active + 2)
        b[-2] = 1  # Sum to 1 constraint
        b[-1] = target_return  # Return constraint
        
        try:
            sol = np.linalg.solve(M, b)
        except LinAlgError:
            return None, active_set
            
        weights_active = sol[:n_active]
        
        # Check if solution violates any constraints
        if np.any(weights_active < -1e-8) or np.any(weights_active > position_limit + 1e-8):
            return None, active_set
            
        # Construct full weight vector
        weights = np.zeros(self.n_assets)
        for idx, active_idx in enumerate(active_list):
            weights[active_idx] = weights_active[idx]
            
        return weights, active_set

    def optimize(self, 
                target_return: float = 0.20,
                position_limit: float = 0.20) -> Tuple[np.ndarray, PortfolioStats]:
        """
        Optimize portfolio using Critical Line Algorithm.
        
        Args:
            target_return: Target annual portfolio return (default 20%)
            position_limit: Maximum position size (default 20%)
            
        Returns:
            Tuple of (optimal weights, portfolio stats)
            
        Raises:
            ValueError: If constraints cannot be satisfied
        """
        # Validate constraints
        if position_limit <= 0 or position_limit > 1:
            raise ValueError(f"Position limit must be between 0 and 1, got {position_limit}")
            
        if position_limit < 1/self.n_assets:
            raise ValueError(f"Position limit {position_limit:.1%} too small to sum to 1 "
                           f"(minimum {1/self.n_assets:.1%})")
            
        max_possible_return = np.sum(
            np.sort(self.annual_returns)[::-1] * 
            np.minimum(position_limit, np.ones(self.n_assets)/np.arange(1, self.n_assets + 1))
        )
        
        if target_return > max_possible_return:
            raise ValueError(f"Target return {target_return:.1%} impossible with "
                           f"position limit {position_limit:.1%}. "
                           f"Maximum possible: {max_possible_return:.1%}")
        
        # Find initial feasible portfolio
        initial_weights = self._find_feasible_portfolio(target_return, position_limit)
        if initial_weights is None:
            raise ValueError("Could not find feasible portfolio satisfying constraints")
        
        # Initialize with assets that have non-zero weights
        active_set = {i for i, w in enumerate(initial_weights) if w > 1e-8}
        best_weights = initial_weights
        best_variance = self._calculate_portfolio_variance(initial_weights)
        
        # Iteratively improve solution
        for _ in range(100):  # Usually converges in <10 iterations
            weights, active_set = self._optimize_with_active_set(
                target_return, position_limit, active_set
            )
            
            if weights is None:
                break
                
            variance = self._calculate_portfolio_variance(weights)
            
            if variance < best_variance:
                best_weights = weights
                best_variance = variance
            else:
                break
            
            # Update active set based on KKT conditions
            gradients = 2 * (self.cov_matrix @ weights)
            min_gradient = np.min(gradients[list(active_set)])
            
            # Look for assets to add to active set
            for i in range(self.n_assets):
                if i not in active_set and gradients[i] < min_gradient - 1e-8:
                    active_set.add(i)
                    break
            else:
                break  # No more assets to add
        
        # Calculate final portfolio statistics
        portfolio_return = np.dot(best_weights, self.annual_returns)
        portfolio_vol = np.sqrt(best_variance)
        active_positions = np.sum(best_weights > 1e-4)
        
        stats = PortfolioStats(
            annual_return=portfolio_return,
            annual_volatility=portfolio_vol,
            sharpe_ratio=(portfolio_return - 0.02) / portfolio_vol,  # Assume 2% risk-free
            max_position=np.max(best_weights),
            active_positions=active_positions
        )
        
        return best_weights, stats

    def generate_report(self, 
                       weights: np.ndarray,
                       stats: PortfolioStats,
                       min_weight: float = 1e-4) -> str:
        """Generate detailed portfolio report"""
        lines = [
            "Portfolio Optimization Report",
            "=" * 50,
            "\nPortfolio Statistics:",
            "-" * 20,
            f"Expected Annual Return: {stats.annual_return:>10.2%}",
            f"Annual Volatility:     {stats.annual_volatility:>10.2%}",
            f"Sharpe Ratio:          {stats.sharpe_ratio:>10.2f}",
            f"Active Positions:      {stats.active_positions:>10d}",
            f"Maximum Position:      {stats.max_position:>10.2%}",
            "\nPortfolio Weights:",
            "-" * 20
        ]
        
        # Sort positions by size
        sorted_idx = np.argsort(weights)[::-1]
        for idx in sorted_idx:
            weight = weights[idx]
            if weight > min_weight:
                lines.append(
                    f"{self.asset_names[idx]:<20s}: {weight:>8.2%}"
                )
        
        return "\n".join(lines)

def optimize_portfolio(returns: pd.DataFrame,
                      cov_matrix: np.ndarray,
                      target_return: float = 0.20,
                      position_limit: float = 0.20,
                      risk_free_rate: float = 0.02) -> tuple:
    """Wrapper function for backward compatibility"""
    # Convert returns to DataFrame if needed
    if isinstance(returns, pd.Series):
        returns = returns.to_frame().T
    elif not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(returns)
    
    # Convert covariance matrix to numpy array
    cov_matrix = np.asarray(cov_matrix)
    
    optimizer = PortfolioOptimizer(returns, cov_matrix=cov_matrix)
    return optimizer.optimize(target_return, position_limit)

def print_portfolio_weights(weights: np.ndarray, 
                          asset_names: list,
                          stats: dict):
    """Wrapper function for backward compatibility"""
    # Convert asset_names to proper format
    if isinstance(asset_names, pd.Series):
        df = pd.DataFrame(columns=asset_names.values)
    else:
        df = pd.DataFrame(columns=asset_names)
    
    # Convert stats dict to PortfolioStats
    portfolio_stats = PortfolioStats(
        annual_return=stats.get('return', 0.0),
        annual_volatility=stats.get('volatility', 0.0),
        sharpe_ratio=stats.get('sharpe', 0.0),
        max_position=np.max(weights),
        active_positions=np.sum(weights > 1e-4)
    )
    
    optimizer = PortfolioOptimizer(df)
    print(optimizer.generate_report(weights, portfolio_stats))
