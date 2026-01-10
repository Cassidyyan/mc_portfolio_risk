"""Simple demo: Load 2 years of data for 3 assets and display parameter summary."""

from datetime import datetime, timedelta
from mc_portfolio.data import load_prices_yf, clean_prices
from mc_portfolio.returns import compute_returns
from mc_portfolio.params import estimate_mu_cov, summarize_params


def main():
    # Configuration
    tickers = ['NVDA', 'AAPL', 'MSFT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # ~2 years
    
    print(f"\n{'='*60}")
    print("Monte Carlo Portfolio Risk Simulator - Phase 2 Demo")
    print(f"{'='*60}\n")
    
    # Load prices from Yahoo Finance
    print(f"Loading prices for {tickers}...")
    print(f"  Date range: {start_date} to {end_date}")
    prices_df = load_prices_yf(tickers, start_date, end_date)
    print(f"✓ Loaded {len(prices_df)} rows, {len(prices_df.columns)} assets\n")
    
    # Clean prices
    print("Cleaning prices...")
    prices_clean = clean_prices(prices_df, method='drop')
    print(f"✓ Cleaned: {len(prices_clean)} rows\n")
    
    # Compute log returns
    print("Computing log returns...")
    returns_df = compute_returns(prices_clean, method='log')
    print(f"✓ Returns: {len(returns_df)} rows\n")
    
    # Estimate parameters
    print("Estimating parameters (mu, cov)...")
    mu, cov, assets = estimate_mu_cov(returns_df)
    print(f"✓ Estimated parameters for {len(assets)} assets\n")
    
    # Summarize and display
    print(f"{'='*60}")
    print("PARAMETER SUMMARY")
    print(f"{'='*60}")
    summary_df = summarize_params(mu, cov, assets, periods=252)
    print(summary_df.to_string())
    print()
    
    print(f"{'='*60}")
    print("ADDITIONAL INFO")
    print(f"{'='*60}")
    print(f"Total trading days: {len(returns_df)}")
    print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"\n✓ Complete!")


if __name__ == "__main__":
    main()
