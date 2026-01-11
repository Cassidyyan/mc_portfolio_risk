import numpy as np  
import pandas as pd 

def _validate_sim_inputs(mu: np.ndarray, cov: np.ndarray, T: int, N: int) -> int:
    # Validate mu is 1D
    if mu.ndim != 1:
        raise ValueError(f"mu must be 1D array, got shape {mu.shape}")
    
    # Validate cov is 2D square
    if cov.ndim != 2:
        raise ValueError(f"cov must be 2D array, got shape {cov.shape}")
    
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    
    # Validate dimensions match
    if cov.shape[0] != len(mu):
        raise ValueError(
            f"cov dimensions ({cov.shape[0]}) don't match mu length ({len(mu)})"
        )
    
    # Validate T and N
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    
    # Validate all values are finite
    if not np.isfinite(mu).all():
        raise ValueError("mu contains non-finite values (inf or NaN)")
    
    if not np.isfinite(cov).all():
        raise ValueError("cov contains non-finite values (inf or NaN)")
    
    return len(mu)

def _cholesky_factor_pd(cov: np.ndarray, eps_start: float = 1e-10, eps_max: float = 1e-3, growth: float = 10.0) -> np.ndarray:
    """Robustly compute Cholesky factor of covariance matrix.
    
    Attempts Cholesky decomposition with progressively increasing regularization
    to ensure the matrix is Positive Definite (all eigenvalues > 0).
    
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix, shape (k, k)
    eps_start : float, default=1e-10
        Initial regularization epsilon
    eps_max : float, default=1e-3
        Maximum regularization epsilon before giving up
    growth : float, default=10.0
        Multiplicative growth factor for epsilon
    
    Returns
    -------
    L : np.ndarray
        Lower-triangular Cholesky factor, shape (k, k) such that
        cov_reg = L @ L.T is positive definite.
    """

    k = cov.shape[0]
    eps = eps_start

    while eps <= eps_max:
        try: 
            # Regularize covariance matrix
            cov_reg = cov + eps * np.eye(k)
            # Attempt Cholesky decomposition
            L = np.linalg.cholesky(cov_reg)
            return L
        except np.linalg.LinAlgError:
            # Increase regularization and retry
            eps *= growth

    raise ValueError("Cholesky decomposition failed even after regularization")

def simulate_correlated_returns(mu: np.ndarray, cov: np.ndarray, T: int, N: int, seed: int = None) -> np.ndarray:
    """Simulate correlated asset returns using Geometric Brownian Motion (GBM)
    
    Parameters
    ----------
    mu : np.ndarray
        Mean daily return for each asset, shape (k,)
    cov : np.ndarray
        Daily covariance matrix of returns, shape (k, k)
    T : int
        Number of time steps (days) to simulate
    N : int
        Number of scenarios to simulate
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    sim_returns : np.ndarray
        Simulated returns, shape (N, T, k)
        sim_returns[n, t, i] is the return of asset i on day t in scenario n.
    """

    k = _validate_sim_inputs(mu, cov, T, N)
    # Set up random number generator
    rng = np.random.default_rng(seed)
    L = _cholesky_factor_pd(cov)

    # Initialize output array, Matrix of shape (N, T, k), tensor of simulated returns
    sim_returns = np.zeros((N, T, k), dtype=float)

    # Simulate each scenario independently
    for n in range(N):
        # Generating a random shock for each day and each asset
        Z = rng.standard_normal((T, k))
        # Applying Cholesky factor to induce correlations amongst assets
        correlated_Z = Z @ L.T  # Shape (T, k)
        # Compute returns using GBM formula: r = mu + correlated_Z
        sim_returns[n] = mu + correlated_Z

    return sim_returns