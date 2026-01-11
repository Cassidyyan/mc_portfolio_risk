import numpy as np
import pandas as pd

# Minimum number of observations required for reliable parameter estimation (e.g. 6 months of daily data)
MIN_OBSERVATIONS = 126

def _validate_returns_df(returns_df: pd.DataFrame) -> None:
    """Validate the returns DataFrame for parameter estimation (estimate_mu_cov).
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of historical returns to validate.
    
    Raises
    ------
    ValueError
        - If returns_df is None or empty
        - If returns_df contains non-numeric data or NaN values after coercion
        - If returns_df has fewer than MIN_OBSERVATIONS rows and fewer than 2 columns (assets)
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

def estimate_mu_cov(returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Estimate mean returns and covariance matrix from historical returns of portfolio.
    
    Given a DataFrame of daily returns (rows=dates, columns=assets), compute:
    - mu: expected daily return per asset (mean of each column)
    - cov: daily covariance matrix of returns
    - assets: list of asset names in the exact order used for mu and cov
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Historical returns data with:
        - rows: dates/observations
        - columns: asset/ticker names
        - values: daily returns (Logarithmic Returns) 
        Condition: Must have at least MIN_OBSERVATIONS rows and at least 2 columns.
    
    Returns
    -------
    mu : np.ndarray
        Mean daily return for each asset, shape (k,) where k = number of assets.
        mu[i] corresponds to assets[i].
    cov : np.ndarray
        Daily covariance matrix of returns, shape (k, k).
        cov[i, j] is the covariance between assets[i] and assets[j].
        Guaranteed to be symmetric.
    assets : list[str]
        List of asset names in the exact order corresponding to mu and cov.
    
    Raises
    ------
    ValueError
        - If returns_df is None or empty
        - If returns_df contains non-numeric data or NaN values after coercion
        - If returns_df has fewer than MIN_OBSERVATIONS rows
        - If returns_df has fewer than 2 columns (assets)
        - If computed mu or cov contain non-finite values
    """

    _validate_returns_df(returns_df)

    # Preserve asset order
    assets = list(returns_df.columns)

    # Compute mean returns (expected daily return per asset) 
    # axis=0 computes mean for each column
    mu = returns_df.mean(axis=0).to_numpy(dtype=float)

    # Compute covariance matrix (daily covariance)
    # returns a 2D array where cov[i, j] is covariance between asset i and j
    cov = returns_df.cov().to_numpy(dtype=float)

    # Ensure numerical stability for mean and covariance
    if not np.isfinite(mu).all():
        raise ValueError("Computed mean returns contain non-finite values (inf or NaN)")

    if not np.isfinite(cov).all():
        raise ValueError("Computed covariance matrix contains non-finite values (inf or NaN)")
    
    # Ensure perfect symmetry of covariance matrix
    cov = 0.5 * (cov + cov.T)

    return mu, cov, assets

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
    std_dev_daily = np.sqrt(diag_var) # std dev
    
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