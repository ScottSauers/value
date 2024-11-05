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
    """Compute nonlinear shrinkage for stock return covariance matrices."""
    returns, T, N = preprocess_returns(returns, demean)
    
    S = np.cov(returns, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    c = N / T
    base_shrinkage = np.trace(S) / (N * T)
    min_var = base_shrinkage * (1 + c)  # Minimum variance scaled by concentration

    # Separate treatment for diagonal vs off-diagonal
    d = np.zeros(N)
    for i in range(N):
        # For diagonal elements, use direct variance estimate with floor
        var_est = max(eigenvalues[i], min_var)
        
        # Adjust for estimation error based on concentration
        d[i] = var_est / (1 + c)
    
        # Additional safeguard against very small values
        d[i] = max(d[i], min_var)

    sigma = eigenvectors @ np.diag(d) @ eigenvectors.T
    return (sigma + sigma.T) / 2

def rscm_shrinkage(returns: np.ndarray):
    """
    Compute the Ell3-RSCM estimator.

    Input:
    X: n x p data matrix (rows are observations).

    Output:
    RSCM: p x p regularized covariance matrix estimate.
    """

    X = returns

    n, p = X.shape

    if n < 4:
        raise ValueError("Sample size n must be at least 4.")

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute sample covariance matrix S
    S = np.cov(X_centered, rowvar=False, bias=False)  # Unbiased estimator

    # Compute eta and eta2
    eta = np.trace(S) / p
    eta2 = np.trace(S @ S) / p

    # Estimate elliptical kurtosis parameter kappa_hat
    ka_lb = -2 / (p + 2)  # Lower bound for kappa
    vari = np.mean(X_centered ** 2, axis=0)
    idx_nonzero_var = vari > 0
    if np.sum(idx_nonzero_var) == 0:
        raise ValueError("All variables have zero variance.")
    X_kurt = X_centered[:, idx_nonzero_var]
    vari = vari[idx_nonzero_var]
    if n <= 3:
        raise ValueError("Sample size n must be greater than 3 for kurtosis estimation.")
    kurt1n = (n - 1) / ((n - 2) * (n - 3))
    g2 = np.mean(X_kurt ** 4, axis=0) / (vari ** 2) - 3
    G2 = kurt1n * ((n + 1) * g2 + 6)
    kurtest = np.mean(G2)
    kappahat = (1 / 3) * kurtest
    if kappahat > 1e6:
        raise ValueError("Too large value for kurtosis.")
    if kappahat <= ka_lb + abs(ka_lb) / 40:
        kappahat = ka_lb + abs(ka_lb) / 40

    # Compute gamma_hat1 (Ell1 estimate)
    d = np.sqrt(np.sum(X_centered ** 2, axis=1))
    nonzero_d = d > 1e-8
    if np.sum(nonzero_d) < 2:
        raise ValueError("Not enough non-zero observations for Ell1 estimation.")
    X_nonzero = X_centered[nonzero_d, :]
    d_nonzero = d[nonzero_d]
    n_nonzero = X_nonzero.shape[0]
    X_normed = X_nonzero / d_nonzero[:, None]
    Csgn = (1 / n_nonzero) * X_normed.T @ X_normed
    trace_Csgn_sq = np.trace(Csgn @ Csgn)
    gammahat0 = (p * n_nonzero / (n_nonzero - 1)) * (trace_Csgn_sq - 1 / n_nonzero)
    m2 = np.mean(1 / (d_nonzero ** 2))
    m1 = np.mean(1 / d_nonzero)
    ratio = m2 / (m1 ** 2)
    delta = (1 / n_nonzero ** 2) * (2 - 2 * ratio + ratio ** 2)
    gammahat1 = gammahat0 - p * delta
    gammahat1 = min(p, max(1, gammahat1))

    # Compute gamma_hat2 (Ell2 estimate)
    gammahat0 = eta2 / (eta ** 2)
    a = (n / (n + kappahat)) * (n / (n - 1) + kappahat)
    numerator = (kappahat + n) * (n - 1) ** 2
    denominator = (n - 2) * (3 * kappahat * (n - 1) + n * (n + 1))
    if denominator == 0:
        raise ValueError("Denominator in computation of 'b' is zero.")
    b = numerator / denominator
    gammahat2 = b * (gammahat0 - a * (p / n))
    gammahat2 = min(p, max(1, gammahat2))

    # Compute beta_o for each estimate
    numerator1 = gammahat1 - 1
    denominator1 = numerator1 + kappahat * (2 * gammahat1 + p) / n + (gammahat1 + p) / (n - 1)
    if denominator1 == 0:
        raise ValueError("Denominator in computation of beta_o1 is zero.")
    beta_o1 = numerator1 / denominator1
    beta_o1 = np.clip(beta_o1, 0, 1)

    numerator2 = gammahat2 - 1
    denominator2 = numerator2 + kappahat * (2 * gammahat2 + p) / n + (gammahat2 + p) / (n - 1)
    if denominator2 == 0:
        raise ValueError("Denominator in computation of beta_o2 is zero.")
    beta_o2 = numerator2 / denominator2
    beta_o2 = np.clip(beta_o2, 0, 1)

    # Compute alpha_o
    alpha_o1 = (1 - beta_o1) * eta
    alpha_o2 = (1 - beta_o2) * eta

    # Compute RSCM estimators
    RSCM1 = beta_o1 * S + alpha_o1 * np.eye(p)
    RSCM2 = beta_o2 * S + alpha_o2 * np.eye(p)

    # Choose the estimator with smaller gamma_hat
    if gammahat1 <= gammahat2:
        RSCM = RSCM1
    else:
        RSCM = RSCM2

    return RSCM

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

    elif method == 'rscm_shrinkage':
        return rscm_shrinkage(returns, demean)
    
    else:  # nonlinear
        return nonlinear_analytical_shrinkage(returns, demean)
