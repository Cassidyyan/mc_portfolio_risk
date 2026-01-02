import numpy as np
import pandas as pd
import yfinance as yf
import os

# Loads historical price data from Yahoo Finance
def load_prices_yf(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Load historical price data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols to download.
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.
    
    Returns:
        Wide DataFrame with DateTimeIndex (rows) and tickers (columns).
        Values are adjusted closing prices.
        
    Raises:
        ValueError: If download results in empty DataFrame or too few rows.
    """
    if not tickers: 
        raise ValueError("tickers list cannot be empty")
    
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    
    if data.empty:
        raise ValueError(f"No data returned for {tickers} in date range {start} to {end}")
    
    # Handle single vs multiple ticker case
    prices_df = data['Close']
    if len(tickers) == 1 and isinstance(prices_df, pd.Series):
        prices_df = prices_df.to_frame(name=tickers[0])
    
    # Validate minimum data
    if len(prices_df) < 2:
        raise ValueError(f"Insufficient data: received {len(prices_df)} rows, need at least 2")
    
    # Ensure DateTimeIndex is sorted and data is float
    prices_df = prices_df.sort_index().astype(float)
    
    return prices_df

def load_prices_csv(folder: str, tickers: list[str] | None = None) -> pd.DataFrame:
    """Load historical price data from CSV files.
    
    Each CSV file should be named <TICKER>.csv and contain:
    - 'Date' column (will be parsed as DateTimeIndex)
    - 'Adj Close' or 'Close' column
    
    Args:
        folder: Path to folder containing CSV files.
        tickers: Optional list of tickers to load. If None, loads all CSV files in folder.
    
    Returns:
        Wide DataFrame aligned by Date with tickers as columns.
        
    Raises:
        ValueError: If folder doesn't exist, files missing, or required columns not found.
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Folder does not exist: {folder}")
    
    # If no tickers provided, load all CSV files in folder
    if tickers is None:
        tickers = [f[:-4] for f in os.listdir(folder) if f.endswith('.csv')]
        if not tickers:
            raise ValueError(f"No CSV files found in {folder}")
    
    price_data = {}
    for ticker in tickers:
        file_path = os.path.join(folder, f"{ticker}.csv")
        
        if not os.path.exists(file_path):
            raise ValueError(f"CSV file not found for ticker {ticker}: {file_path}")
        
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        
        # Look for Adj Close, fallback to Close
        if 'Adj Close' in df.columns:
            price_data[ticker] = df['Adj Close']
        elif 'Close' in df.columns:
            price_data[ticker] = df['Close']
        else:
            raise ValueError(f"CSV file for {ticker} missing 'Adj Close' or 'Close' column")
    
    prices_df = pd.DataFrame(price_data).sort_index().astype(float)
    
    return prices_df

def clean_prices(prices_df: pd.DataFrame, method: str = "drop", ffill_limit: int = 1) -> pd.DataFrame:
    """Clean price data by handling missing values.
    
    Args:
        prices_df: DataFrame of prices with DateTimeIndex.
        method: Cleaning method - "drop" removes rows with any NaNs,
                "ffill" forward-fills up to ffill_limit then drops remaining NaNs.
        ffill_limit: Maximum number of consecutive NaNs to forward-fill (only for method="ffill").
    
    Returns:
        Cleaned DataFrame with no NaNs and sorted DateTimeIndex.
        
    Raises:
        ValueError: If method is invalid.
    """
    if method not in ("drop", "ffill"):
        raise ValueError(f"Invalid method '{method}'. Use 'drop' or 'ffill'")
    
    if method == "drop":
        prices_clean = prices_df.dropna()
    else:
        prices_clean = prices_df.ffill(limit=ffill_limit).dropna()
    
    # Minor optimization: only sort if not already sorted
    return prices_clean.sort_index() if not prices_clean.index.is_monotonic_increasing else prices_clean