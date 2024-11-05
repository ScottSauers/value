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
    Compute linear shrinkage with single factor target following L&W 2020.
    """
    returns, T, N = preprocess_returns(returns, demean)
    
    # Compute sample covariance
    S = np.cov(returns, rowvar=False, ddof=1)
    
    # Handle market returns
    if market_returns is None:
        market_returns = returns.mean(axis=1, keepdims=True)
    else:
        market_returns = market_returns.reshape(-1, 1)
        if demean:
            market_returns = market_returns - market_returns.mean()
            
    # Compute factor target
    X = np.column_stack([np.ones(T), market_returns])
    betas = np.linalg.lstsq(X, returns, rcond=None)[0][1]
    residuals = returns - market_returns @ betas.reshape(1,-1)
    D = np.diag(np.var(residuals, axis=0, ddof=1))
    var_market = np.var(market_returns, ddof=1)
    F = var_market * np.outer(betas, betas) + D
    
    # Compute Frobenius norm 
    d2 = np.sum((S - F) ** 2)
    
    # Estimate pi
    Y = returns ** 2  # Element-wise square
    phi_mat = (Y.T @ Y) / T - S ** 2
    pi = np.sum(phi_mat)
    
    # Estimate rho - key formula from paper
    theta_mat = (returns ** 3).T @ returns / T
    rho = np.sum(np.diag(phi_mat)) + \
          var_market * np.sum(betas**2) * np.mean(theta_mat)
    
    # Compute shrinkage intensity with proper scaling
    shrinkage = max(0, min(1, (pi - rho) / (d2 * T)))
    
    # Form final estimate with symmetry enforcement
    sigma = shrinkage * F + (1 - shrinkage) * S
    sigma = (sigma + sigma.T) / 2
    
    return sigma, shrinkage

def nonlinear_analytical_shrinkage(returns: np.ndarray, demean: bool = True) -> np.ndarray:
    """Compute nonlinear shrinkage."""
    returns, T, N = preprocess_returns(returns, demean)
    
    S = np.cov(returns, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    c = N / T
    # More aggressive regularization floor based on matrix trace
    base_shrinkage = np.trace(S) / (N * T) 
    eigenvalues = np.maximum(eigenvalues, base_shrinkage)

    # Adaptive bandwidth proportional to eigenvalue magnitudes
    hn = T**(-1/3)  # Global bandwidth
    hn_i = eigenvalues * hn  # Local bandwidths
    
    # Compute density estimate with proportional bandwidths
    density = np.zeros(N)
    for i in range(N):
        weights = np.maximum(0, 1 - ((eigenvalues[i] - eigenvalues)/(5*hn_i))**2)
        density[i] = np.sum(weights/(hn_i)) / (N * 4/3)

    # Compute Hilbert transform with regularization
    hilbert = np.zeros(N)
    for i in range(N):
        diff = eigenvalues[i] - eigenvalues
        diff[abs(diff) < 1e-10] = 1e-10  # Prevent division by zero
        kernel_vals = np.maximum(0, 1 - (diff/(5*hn_i))**2)
        hilbert_kernel = -diff/(np.pi * diff**2) * kernel_vals
        hilbert[i] = np.sum(hilbert_kernel)/(N * 4/3)
    
    # Compute optimal nonlinear shrinkage
    d = np.zeros(N)
    for i in range(N):
        numer = eigenvalues[i]
        denom = (np.pi * c * eigenvalues[i] * density[i])**2 + (1 - c - np.pi * c * eigenvalues[i] * hilbert[i])**2
        # Aggressive floor using both base_shrinkage and minimum eigenvalue
        d[i] = numer / max(denom, base_shrinkage * (1 + c))
        d[i] = max(d[i], base_shrinkage)
        
    sigma = eigenvectors @ np.diag(d) @ eigenvectors.T
    return (sigma + sigma.T) / 2

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
