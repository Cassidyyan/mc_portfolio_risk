import numpy as np
from .utils import validate_sim_inputs, cholesky_factor_pd

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

    k = validate_sim_inputs(mu, cov, T, N)
    # Set up random number generator
    rng = np.random.default_rng(seed)
    L = cholesky_factor_pd(cov)

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