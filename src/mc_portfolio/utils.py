import numpy as np
import pandas as pd

"""
This module contains helper functions organized by type:
- Constants: Global configuration values
- Validation Functions: Input validation and error checking
- Mathematical Functions: Numerical computations and transformations
- Analysis Functions: Parameter summarization and reporting
"""

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_OBSERVATIONS = 126  # Minimum number of observations (e.g., 6 months of daily data)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

# Used in portfolio.py and simulate.py 
def validate_returns_df(returns_df: pd.DataFrame) -> None:
    """Validate the returns DataFrame for parameter estimation.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of historical returns to validate.
    
    Raises
    ------
    ValueError
        - If returns_df is None or empty
        - If returns_df contains non-numeric data or NaN values after coercion
        - If returns_df has fewer than MIN_OBSERVATIONS rows
        - If returns_df has fewer than 2 columns (assets)
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("returns_df cannot be None or empty")
    
    # Coerce to numeric and check for NaNs
    returns_df = returns_df.apply(pd.to_numeric, errors='coerce')
    if returns_df.isnull().any().any():
        raise ValueError("returns_df must contain only numeric data without NaNs")
    
    if returns_df.shape[0] < MIN_OBSERVATIONS:
        raise ValueError(
            f"returns_df must have at least {MIN_OBSERVATIONS} rows; "
            f"found {returns_df.shape[0]}"
        )
    
    if returns_df.shape[1] < 2:
        raise ValueError("returns_df must have at least 2 columns (assets)")

# Used in simulate.py
def validate_sim_inputs(mu: np.ndarray, cov: np.ndarray, T: int, N: int) -> int:
    """Validate inputs for Monte Carlo simulation.
    
    Parameters
    ----------
    mu : np.ndarray
        Mean returns, should be 1D
    cov : np.ndarray
        Covariance matrix, should be 2D square
    T : int
        Number of time steps
    N : int
        Number of scenarios
    
    Returns
    -------
    k : int
        Number of assets
    
    Raises
    ------
    ValueError
        If any validation check fails
    """
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

# Used in portfolio.py
def validate_weights(
    weights: list[float] | np.ndarray,
    k: int,
    allow_short: bool = False,
    tol: float = 1e-8,
    normalize: bool = False
) -> np.ndarray:
    """Validate and convert portfolio weights to numpy array.
    
    Parameters
    ----------
    weights : list[float] or np.ndarray
        Portfolio weights for k assets
    k : int
        Expected number of assets
    allow_short : bool, default=False
        If False, raise ValueError if any weight is negative
    tol : float, default=1e-8
        Tolerance for sum(weights) == 1 check
    normalize : bool, default=False
        If True, normalize weights to sum to 1. If False, raise error if sum != 1.
    
    Returns
    -------
    weights : np.ndarray
        Validated weights array, shape (k,)
    
    Raises
    ------
    ValueError
        If weights are invalid (wrong length, non-finite, negative when shorts not allowed,
        or sum != 1 when normalize=False)
    """
    # Convert to numpy array
    weights_arr = np.asarray(weights, dtype=float)
    
    # Check shape
    if weights_arr.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights_arr.shape}")
    
    if len(weights_arr) != k:
        raise ValueError(f"weights length ({len(weights_arr)}) doesn't match k ({k})")
    
    # Check for non-finite values
    if not np.isfinite(weights_arr).all():
        raise ValueError("weights contain non-finite values (inf or NaN)")
    
    # Check for negative weights if shorts not allowed
    if not allow_short and (weights_arr < 0).any():
        raise ValueError("Negative weights not allowed (set allow_short=True to enable)")
    
    # Check sum
    weight_sum = weights_arr.sum()
    if abs(weight_sum - 1.0) > tol:
        if normalize:
            weights_arr = weights_arr / weight_sum
        else:
            raise ValueError(
                f"weights sum to {weight_sum:.10f}, expected 1.0 (±{tol}). "
                f"Set normalize=True to auto-normalize."
            )
    
    return weights_arr

# Used in portfolio.py
def validate_sim_returns(sim_returns: np.ndarray, k: int) -> None:
    """Validate simulated returns array.
    
    Parameters
    ----------
    sim_returns : np.ndarray
        Simulated returns with shape (N, T, k)
    k : int
        Expected number of assets
    
    Raises
    ------
    ValueError
        If sim_returns has wrong shape or contains non-finite values
    """
    if sim_returns.ndim != 3:
        raise ValueError(f"sim_returns must be 3D, got shape {sim_returns.shape}")
    
    if sim_returns.shape[2] != k:
        raise ValueError(
            f"sim_returns has {sim_returns.shape[2]} assets (last dim), expected {k}"
        )
    
    if not np.isfinite(sim_returns).all():
        raise ValueError("sim_returns contains non-finite values (inf or NaN)")


# ============================================================================
# MATHEMATICAL FUNCTIONS
# ============================================================================

# Used in simulate.py
def cholesky_factor_pd(
    cov: np.ndarray,
    eps_start: float = 1e-10,
    eps_max: float = 1e-3,
    growth: float = 10.0
) -> np.ndarray:
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
    
    Raises
    ------
    ValueError
        If Cholesky decomposition fails even after regularization
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

# Used in portfolio.py
def compute_daily_volatility(cov: np.ndarray, assets: list[str]) -> pd.Series:
    """Compute daily volatility (standard deviation) for each asset from covariance matrix.
    
    This is a helper utility for debugging and validation.
    
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix, shape (k, k)
    assets : list[str]
        List of asset names, length k
    
    Returns
    -------
    pd.Series
        Daily volatility (sqrt of diagonal of cov) for each asset
    """
    # Std Dev is sqrt of variance (diagonal of covariance matrix) for individual assets
    vol_daily = np.sqrt(np.diag(cov))
    return pd.Series(vol_daily, index=assets, name='daily_volatility')


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def summarize_params(mu: np.ndarray, cov: np.ndarray, assets: list[str], periods: int = 252) -> pd.DataFrame:
    """Summarize estimated parameters with daily and annualized metrics.
    
    Produces a per-asset summary table showing daily return/volatility statistics
    and their annualized equivalents.
    
    Parameters
    ----------
    mu : np.ndarray
        Mean daily return for each asset, shape (k,) where k = number of assets.
        Units: daily returns (e.g., 0.001 = 0.1% per day)
    cov : np.ndarray
        Daily covariance matrix, shape (k, k).
        Units: daily variance/covariance
    assets : list[str]
        Asset names in exact order corresponding to mu and cov, length k.
    periods : int, default=252
        Number of trading periods per year for annualization.
        252 is standard for daily data (~ trading days per year).
    
    Returns
    -------
    pd.DataFrame
        Summary table indexed by assets (preserving input order) with columns:
        - mean_daily: daily mean return (same as mu)
        - vol_daily: daily volatility (standard deviation)
        - mean_annual: annualized mean return = mean_daily * periods
        - vol_annual: annualized volatility = vol_daily * np.sqrt(periods)
    
    Raises
    ------
    ValueError
        - If mu is not 1D
        - If len(assets) != len(mu)
        - If cov is not (k, k) where k = len(assets)
    """
    # Ensure numpy arrays for robustness
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    
    # Validate mu is 1D
    if mu.ndim != 1:
        raise ValueError(f"mu must be 1D array, got shape {mu.shape}")
    
    # Validate length consistency
    if len(mu) != len(assets):
        raise ValueError(
            f"Length mismatch: assets has {len(assets)} elements, mu has {len(mu)}"
        )
    
    # Validate cov shape
    if cov.shape != (len(assets), len(assets)):
        raise ValueError(
            f"cov shape mismatch: expected ({len(assets)}, {len(assets)}), got {cov.shape}"
        )
    
    # Extract daily volatilities (std dev) from covariance matrix
    # Ensure non-negative variances for sqrt by clipping at zero
    diag_var = np.maximum(np.diag(cov), 0.0)
    std_dev_daily = np.sqrt(diag_var)  # std dev
    
    # Annualize
    mean_annual = mu * periods
    vol_annual = std_dev_daily * np.sqrt(periods)
    
    # Build DataFrame preserving asset order
    summary_df = pd.DataFrame({
        'mean_daily': mu,
        'vol_daily': std_dev_daily,
        'mean_annual': mean_annual,
        'vol_annual': vol_annual
    }, index=assets)
    
    return summary_df
