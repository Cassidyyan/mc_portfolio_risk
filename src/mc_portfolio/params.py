import numpy as np
import pandas as pd
from .utils import validate_returns_df, MIN_OBSERVATIONS


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

    validate_returns_df(returns_df)

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