import numpy as np
import pandas as pd


def compute_returns(prices_df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute returns from price data.
    
    Args:
        prices_df: Wide DataFrame of prices (DateTimeIndex, tickers as columns).
        method: Return calculation method:
                - "log": logarithmic returns = ln(P_t / P_{t-1})
                - "pct": percentage returns = (P_t - P_{t-1}) / P_{t-1}
    
    Returns:
        Wide DataFrame of returns (same columns as input, one fewer row).
        First row is dropped as it contains NaN from the shift operation.
        
    Raises:
        ValueError: If prices_df is empty, non-numeric, or method is invalid.
    """
    if prices_df.empty:
        raise ValueError("prices_df cannot be empty")
    
    if method not in ("log", "pct"):
        raise ValueError(f"Invalid method '{method}'. Use 'log' or 'pct'")
    
    # Validate numeric data
    if not all(prices_df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("prices_df must contain only numeric data")
    
    if method == "log":
        # Log returns: ln(P_t / P_{t-1})
        returns_df = np.log(prices_df / prices_df.shift(1))
    else:
        # Percentage returns: (P_t - P_{t-1}) / P_{t-1}
        returns_df = prices_df.pct_change()
    
    # Drop first row with NaN
    returns_df = returns_df.iloc[1:]
    
    # Validate no NaNs remain
    if returns_df.isnull().any().any():
        raise ValueError("NaN values remain in returns after computation. Check input prices.")
    
    return returns_df