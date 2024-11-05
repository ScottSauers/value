"""
Covariance Matrix Shrinkage Estimators.

This module implements various shrinkage methods for estimating large-dimensional 
covariance matrices.

References:
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
- Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices.
"""

from typing import Tuple, Optional, Union
import numpy as np
from scipy import linalg
from scipy import stats

def preprocess_returns(returns: np.ndarray, 
                      demean: bool = True) -> Tuple[np.ndarray, int, int]:
    """
    Preprocess returns with proper centering and handling of samples.
    """
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
        
    if returns.ndim != 2:
        raise ValueError("Returns must be a 2-dimensional array") 
        
    T, N = returns.shape
    
    # Handle demeaning
    if demean:
        # Use ddof=1 for unbiased estimation
        means = returns.mean(axis=0)
        returns = returns - means
        
    # Check for nan/inf
    if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
        raise ValueError("Returns contain nan/inf values")
        
    # Warning if T < N 
    if T < N:
        print(f"Warning: More variables ({N}) than observations ({T})")
        
    return returns, T, N

def linear_shrinkage_identity(returns: np.ndarray,
                            demean: bool = True) -> Tuple[np.ndarray, float]:
    """
    Compute linear shrinkage estimator with identity matrix target.
    
    Args:
        returns: T x N matrix of returns
        demean: Whether to demean the returns
        
    Returns:
        Shrinkage estimator, shrinkage intensity
    """
    returns, T, N = preprocess_returns(returns, demean)
    
    # Sample covariance
    S = np.cov(returns, rowvar=False, ddof=1)
    
    # Identity target
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    
    # Frobenius norm
    d2 = np.sum((S - F) ** 2)
    
    # Estimate pi (sum of asymptotic variances)
    Y = returns ** 2
    phi_mat = (Y.T @ Y) / T - S ** 2
    pi = np.sum(phi_mat)
    
    # Estimate rho (asymptotic covariance)
    theta_mat = (returns ** 3).T @ returns / T
    rho = np.sum(np.diag(phi_mat))
    
    # Compute optimal shrinkage intensity
    shrinkage = max(0, min(1, (pi - rho) / (d2 * T)))
    
    # Compute estimator
    sigma = shrinkage * F + (1 - shrinkage) * S
    
    return sigma, shrinkage
  

def constant_correlation_target(returns: np.ndarray) -> np.ndarray:
    """
    Compute constant correlation target matrix.
    
    Args:
        returns: T x N matrix of returns
        
    Returns:
        Target matrix
    """
    # Sample covariance and standard deviations
    S = np.cov(returns, rowvar=False, ddof=1)
    s = np.sqrt(np.diag(S))
    s = s.reshape(-1,1)
    
    # Average correlation
    R = S / (s @ s.T)
    r_bar = (np.sum(R) - R.shape[0]) / (R.shape[0] * (R.shape[0] - 1))
    
    # Construct target
    F = r_bar * (s @ s.T)
    np.fill_diagonal(F, np.diag(S))
    
    return F

def linear_shrinkage_constant_correlation(returns: np.ndarray,
                                        demean: bool = True) -> Tuple[np.ndarray, float]:
    """
    Compute linear shrinkage estimator with constant correlation target.
    
    Args:
        returns: T x N matrix of returns
        demean: Whether to demean the returns
        
    Returns:
        Shrinkage estimator, shrinkage intensity 
    """
    returns, T, N = preprocess_returns(returns, demean)
    
    # Sample covariance
    S = np.cov(returns, rowvar=False, ddof=1)
    
    # Constant correlation target
    F = constant_correlation_target(returns)
    
    # Frobenius norm
    d2 = np.sum((S - F) ** 2)
    
    # Estimate pi
    Y = returns ** 2
    phi_mat = (Y.T @ Y) / T - S ** 2
    pi = np.sum(phi_mat)
    
    # Estimate rho with asymptotic covariance formula
    theta_mat = (returns ** 3).T @ returns / T
    s = np.sqrt(np.diag(S)).reshape(-1,1)
    rho = np.sum(np.diag(phi_mat)) + np.mean(theta_mat) * np.sum(F - np.diag(np.diag(F)))
    
    # Compute shrinkage intensity with scaling
    shrinkage = max(0, min(1, (pi - rho) / (d2 * T)))
    
    # Compute estimator
    sigma = shrinkage * F + (1 - shrinkage) * S
    
    return sigma, shrinkage

def single_factor_target(returns: np.ndarray,
                        market_returns: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute single factor (market) model target matrix.
    
    Args:
        returns: T x N matrix of returns
        market_returns: Optional T x 1 market returns (if None, equal-weighted portfolio used)
        
    Returns:
        Target matrix
    """
    T, N = returns.shape
    
    if market_returns is None:
        market_returns = returns.mean(axis=1)
        
    # Estimate betas and residual variances
    X = np.column_stack([np.ones(T), market_returns])
    betas = np.linalg.lstsq(X, returns, rcond=None)[0][1]
    residuals = returns - market_returns.reshape(-1,1) @ betas.reshape(1,-1)
    D = np.diag(np.var(residuals, axis=0, ddof=1))
    
    # Market variance
    var_market = np.var(market_returns, ddof=1)
    
    # Construct target
    F = var_market * np.outer(betas, betas) + D
    
    return F

def linear_shrinkage_single_factor(returns: np.ndarray,
                                 market_returns: Optional[np.ndarray] = None,
                                 demean: bool = True) -> Tuple[np.ndarray, float]:
    """
    Compute linear shrinkage estimator with single factor target.
    """
    returns, T, N = preprocess_returns(returns, demean)
    
    if market_returns is not None and demean:
        market_returns = market_returns - market_returns.mean()
    
    # Sample covariance  
    S = np.cov(returns, rowvar=False, ddof=1)
    
    # Single factor target
    F = single_factor_target(returns, market_returns)
    
    # Frobenius norm
    d2 = np.sum((S - F) ** 2)
    
    # Estimate pi (sum of asymptotic variances)
    Y = returns ** 2
    phi_mat = (Y.T @ Y) / T - S ** 2 
    pi = np.sum(phi_mat)
    
    # Estimate rho (asymptotic covariance)
    theta_mat = (returns ** 3).T @ returns / T
    if market_returns is None:
        market_returns = returns.mean(axis=1)
    betas = np.linalg.lstsq(np.column_stack([np.ones(T), market_returns]), returns, rcond=None)[0][1]
    var_market = np.var(market_returns, ddof=1)

    rho = np.sum(np.diag(phi_mat)) + \
          var_market * np.mean(theta_mat) * np.sum(F - np.diag(np.diag(F)))
                                   
    # Compute optimal shrinkage intensity
    shrinkage = max(0, min(1, (pi - rho) / (d2 * T)))
    
    # Compute estimator
    sigma = shrinkage * F + (1 - shrinkage) * S
    
    return sigma, shrinkage

def nonlinear_analytical_shrinkage(returns: np.ndarray,
                                 demean: bool = True) -> np.ndarray:
    """
    Compute nonlinear shrinkage following Ledoit-Wolf 2020 paper.
    """
    # Initial setup
    returns, T, N = preprocess_returns(returns, demean)
    
    # Sample covariance with proper bias correction
    S = np.cov(returns, rowvar=False, ddof=1)  # Important: ddof=1
    
    # Get eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure positivity
    
    # Compute concentration ratio c = N/T
    c = N / T
    
    # Define kernel parameters
    # Silverman's rule for bandwidth
    h = 0.9 * min(np.std(eigenvalues), stats.iqr(eigenvalues)/1.34) * N**(-0.2)
    
    # Set up grid for density estimation
    grid_min = max(0, min(eigenvalues) - 4*h)
    grid_max = max(eigenvalues) + 4*h
    grid = np.linspace(grid_min, grid_max, min(512, N))
    
    # Compute density estimate with proper scaling
    kde = stats.gaussian_kde(eigenvalues, bw_method=h)
    density = kde(grid) 
    
    # Compute Hilbert transform  
    def hilbert_transform(x):
        # Handle edge cases
        if x <= grid_min or x >= grid_max:
            return 0
        # Principal value computation
        indices = grid != x
        return np.mean(density[indices] / (grid[indices] - x))
        
    # Compute optimal nonlinear shrinkage
    d = np.zeros(N)
    for i in range(N):
        if eigenvalues[i] == 0 and c > 1:
            # Handle zero eigenvalues case
            d[i] = 1 / (np.pi * (c-1) * hilbert_transform(0))
        else:
            # Main formula from paper
            ht = hilbert_transform(eigenvalues[i])
            d[i] = eigenvalues[i] / (
                (np.pi * c * eigenvalues[i] * density[i])**2 + 
                (1 - c - np.pi * c * eigenvalues[i] * ht)**2
            )
    
    # Reconstruct with numerical stability check
    d = np.maximum(d, np.min(eigenvalues[eigenvalues > 0])/100)  # Floor for stability
    sigma = eigenvectors @ np.diag(d) @ eigenvectors.T
    
    # Symmetry
    sigma = (sigma + sigma.T) / 2
    
    return sigma

def shrinkage_estimation(returns: np.ndarray,
                        method: str = 'nonlinear',
                        market_returns: Optional[np.ndarray] = None,
                        demean: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Main function to compute covariance matrix shrinkage estimators.
    
    Args:
        returns: T x N matrix of returns
        method: Shrinkage method - one of:
               'identity' - linear shrinkage to identity matrix
               'const_corr' - linear shrinkage to constant correlation target
               'single_factor' - linear shrinkage to single factor model target
               'nonlinear' - nonlinear analytical shrinkage
        market_returns: Optional market returns for single factor method
        demean: Whether to demean the returns
        
    Returns:
        Shrinkage estimator and shrinkage intensity (for linear methods)
        or just shrinkage estimator (for nonlinear method)
    """
    method = method.lower()
    valid_methods = ['identity', 'const_corr', 'single_factor', 'nonlinear']
    
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")
        
    if method == 'identity':
        return linear_shrinkage_identity(returns, demean)
    
    elif method == 'const_corr':
        return linear_shrinkage_constant_correlation(returns, demean)
    
    elif method == 'single_factor':
        return linear_shrinkage_single_factor(returns, market_returns, demean)
    
    else:  # nonlinear
        return nonlinear_analytical_shrinkage(returns, demean)
