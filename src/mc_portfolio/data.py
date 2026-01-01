import numpy as np
import pandas as pd
import yfinance as yf
import os

# Loads historical price data from Yahoo Finance
def load_prices_yf(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Parameters:
    tickers (list[str]): List of ticker symbols.
    start_date (str): Start date for the data in 'YYYY-MM-DD' format.
    end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing adjusted closing prices for the tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    prices = data['Close']
    prices.columns.name = None  # Remove the 'Ticker' label
    return prices

def load_prices_csv(folder: str, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Parameters:
    folder (str): Path to the folder containing CSV files.
    tickers (list[str] | None): List of ticker symbols to load. If None, loads all CSV files in the folder.

    Returns:
    pd.DataFrame: DataFrame containing adjusted closing prices for the tickers.
    """

    price_data = pd.DataFrame()

    # If no tickers provided, load all CSV files in the folder
    if tickers is None:
        tickers = [f.split('.csv')[0] for f in os.listdir(folder) if f.endswith('.csv')]

    # Load each ticker's data
    for ticker in tickers:
        file_path = os.path.join(folder, f"{ticker}.csv")
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        price_data[ticker] = df['Adj Close']

    return price_data

def clean_price_data(prices_df: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    """
    Parameters:
    prices_df (pd.DataFrame): DataFrame containing price data.
    method (str): Method to handle missing values. Options are "drop" or "fill".

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if method == "drop":
        return prices_df.dropna()
    elif method == "fill": # wont be using this method for future reference
        return prices_df.fillna(method='ffill').fillna(method='bfill')
    else:
        raise ValueError("Invalid method. Use 'drop' or 'fill'.")