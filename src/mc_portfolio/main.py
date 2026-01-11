"""Simple demo: Load 2 years of data for 3 assets and display parameter summary."""

import numpy as np
from datetime import datetime, timedelta
from mc_portfolio.data import load_prices_yf, clean_prices
from mc_portfolio.returns import compute_returns
from mc_portfolio.params import estimate_mu_cov, summarize_params
from mc_portfolio.simulate import simulate_correlated_returns


def main():
    # Configuration
    tickers = ['NVDA', 'AAPL', 'MSFT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # ~2 years
    
    print(f"\n{'='*60}")
    print("Monte Carlo Portfolio Risk Simulator - Demo")
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
    
    # Phase 3: Monte Carlo Simulation
    print(f"{'='*60}")
    print("MONTE CARLO SIMULATION")
    print(f"{'='*60}")
    
    # Simulation parameters
    T = 252  # 1 year of daily returns
    N = 10000  # 10,000 scenarios
    seed = 42
    
    print(f"Simulating {N} scenarios over {T} days...")
    sim_returns = simulate_correlated_returns(mu, cov, T, N, seed=seed)
    print(f"✓ Simulation complete!")
    print(f"  Output shape: {sim_returns.shape} (N={N}, T={T}, k={len(assets)})\n")
    
    # Sanity check: compute sample moments
    print("Running moment sanity check...")
    all_samples = sim_returns.reshape(-1, len(assets))
    sample_mean = all_samples.mean(axis=0)
    sample_cov = np.cov(all_samples, rowvar=False)
    
    max_mean_diff = np.abs(sample_mean - mu).max()
    max_cov_diff = np.abs(sample_cov - cov).max()
    mean_diff_pct = (max_mean_diff / np.abs(mu).max()) * 100
    
    print(f"✓ Sanity check complete!")
    print(f"  Total samples: {N * T:,}")
    print(f"  Max mean difference: {max_mean_diff:.6f}")
    print(f"  Max cov difference: {max_cov_diff:.6f}")
    print(f"  Mean diff %: {mean_diff_pct:.2f}%")
    print()
    
    print(f"{'='*60}")
    print("ADDITIONAL INFO")
    print(f"{'='*60}")
    print(f"Total trading days: {len(returns_df)}")
    print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"\n✓ Phase 1, 2 & 3 Complete!")


if __name__ == "__main__":
    main()

