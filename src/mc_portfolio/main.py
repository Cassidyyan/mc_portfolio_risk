import numpy as np
import os
from datetime import datetime, timedelta
from mc_portfolio.data import load_prices_yf, clean_prices
from mc_portfolio.returns import compute_returns
from mc_portfolio.params import estimate_mu_cov
from mc_portfolio.utils import summarize_params
from mc_portfolio.simulate import simulate_correlated_returns
from mc_portfolio.portfolio import run_portfolio_aggregation

def main():
    # ========================================================================
    # SIMULATION CONFIGURATION - MODIFY THESE SETTINGS
    # ========================================================================
    
    # Historical Data Settings
    tickers = ['NVDA', 'AAPL', 'MSFT']  # Assets to include in portfolio
    historical_years = 20  # Years of historical data for parameter estimation
    
    # Monte Carlo Simulation Settings
    simulation_years = 20  # Number of years to simulate forward
    num_scenarios = 10_000  # Number of Monte Carlo scenarios to generate
    random_seed = 10  # For reproducibility (use None for random)
    
    # Portfolio Settings
    weights = [0.4, 0.3, 0.3]  # Portfolio allocation (must sum to 1.0)
    initial_value = 1_000_000  # Starting portfolio value ($)
    allow_short_selling = False  # Allow negative weights (short positions)?
    
    # Return Calculation Method
    return_type = "log"  # "log" or "pct"
    
    # Output Settings
    save_results = True  # Save terminal returns to CSV?
    output_dir = "outputs"  # Directory for output files
    
    # ========================================================================
    # DERIVED PARAMETERS (DO NOT MODIFY)
    # ========================================================================
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*historical_years)).strftime('%Y-%m-%d')
    T = simulation_years * 252  # Convert years to trading days
    
    # ========================================================================
    # PHASE 1: DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("Monte Carlo Portfolio Risk Simulator")
    print(f"{'='*60}\n")
    
    print("CONFIGURATION:")
    print(f"  Assets: {', '.join(tickers)}")
    print(f"  Historical data: {historical_years} years ({start_date} to {end_date})")
    print(f"  Simulation horizon: {simulation_years} years ({T:,} trading days)")
    print(f"  Number of scenarios: {num_scenarios:,}")
    print(f"  Initial portfolio value: ${initial_value:,}")
    print()
    
    # Load prices from Yahoo Finance
    print(f"Loading prices for {tickers}...")
    prices_df = load_prices_yf(tickers, start_date, end_date)
    print(f"✓ Loaded {len(prices_df)} rows, {len(prices_df.columns)} assets\n")
    
    # Clean prices
    print("Cleaning prices...")
    prices_clean = clean_prices(prices_df, method='drop')
    print(f"✓ Cleaned: {len(prices_clean)} rows\n")
    
    # Compute returns
    print(f"Computing {return_type} returns...")
    returns_df = compute_returns(prices_clean, method=return_type)
    print(f"✓ Returns: {len(returns_df)} rows\n")
    
    # ========================================================================
    # PHASE 2: PARAMETER ESTIMATION
    # ========================================================================
    
    print(f"{'='*60}")
    print("PARAMETER ESTIMATION")
    print(f"{'='*60}\n")
    
    # Estimate parameters
    print("Estimating parameters (mu, cov)...")
    mu, cov, assets = estimate_mu_cov(returns_df)
    print(f"✓ Estimated parameters for {len(assets)} assets\n")
    
    # Display parameter summary
    summary_df = summarize_params(mu, cov, assets, periods=252)
    print(summary_df.to_string())
    print()
    
    # ========================================================================
    # PHASE 3: MONTE CARLO SIMULATION
    # ========================================================================
    
    print(f"{'='*60}")
    print("MONTE CARLO SIMULATION")
    print(f"{'='*60}\n")
    
    print(f"Simulating {num_scenarios:,} scenarios over {T:,} days ({simulation_years} years)...")
    sim_returns = simulate_correlated_returns(mu, cov, T, num_scenarios, seed=random_seed)
    print(f"✓ Simulation complete!")
    print(f"  Output shape: {sim_returns.shape} (N={num_scenarios}, T={T}, k={len(assets)})\n")
    
    # Sanity check: compute sample moments
    print("Running moment sanity check...")
    all_samples = sim_returns.reshape(-1, len(assets))
    sample_mean = all_samples.mean(axis=0)
    sample_cov = np.cov(all_samples, rowvar=False)
    
    max_mean_diff = np.abs(sample_mean - mu).max()
    max_cov_diff = np.abs(sample_cov - cov).max()
    mean_diff_pct = (max_mean_diff / np.abs(mu).max()) * 100
    
    print(f"✓ Sanity check complete!")
    print(f"  Total samples: {num_scenarios * T:,}")
    print(f"  Max mean difference: {max_mean_diff:.6f}")
    print(f"  Max cov difference: {max_cov_diff:.6f}")
    print(f"  Mean diff %: {mean_diff_pct:.2f}%")
    print()
    
    # ========================================================================
    # PHASE 4: PORTFOLIO AGGREGATION
    # ========================================================================
    
    print(f"{'='*60}")
    print("PORTFOLIO AGGREGATION")
    print(f"{'='*60}\n")
    
    print(f"Portfolio weights:")
    for asset, weight in zip(assets, weights):
        print(f"  {asset}: {weight*100:.1f}%")
    print(f"Initial value: ${initial_value:,}")
    print()
    
    print("Running portfolio aggregation...")
    portfolio_results = run_portfolio_aggregation(
        sim_returns=sim_returns,
        assets=assets,
        weights=weights,
        v0=initial_value,
        return_type=return_type,
        allow_short=allow_short_selling
    )
    
    print(f"✓ Portfolio aggregation complete!")
    print(f"  Portfolio returns shape: {portfolio_results['port_returns'].shape}")
    print(f"  Portfolio values shape: {portfolio_results['port_values'].shape}")
    print(f"  Terminal values shape: {portfolio_results['terminal_values'].shape}")
    print()
    
    # ========================================================================
    # RESULTS ANALYSIS
    # ======================================================================== 

    # Analyze terminal returns
    terminal_returns = portfolio_results['terminal_returns']
    annualized_returns = (1 + terminal_returns) ** (1 / simulation_years) - 1
    
    print(f"{'='*60}")
    print("TERMINAL RETURN STATISTICS")
    print(f"{'='*60}\n")
    print(f"Total Returns (over {simulation_years:.1f} years):")
    print(f"  Mean: {terminal_returns.mean()*100:.2f}%")
    print(f"  Median: {np.median(terminal_returns)*100:.2f}%")
    print(f"  Std dev: {terminal_returns.std()*100:.2f}%")
    print(f"  Best case (99th percentile): {np.percentile(terminal_returns, 99)*100:.2f}%")
    print(f"  Worst case (1st percentile): {np.percentile(terminal_returns, 1)*100:.2f}%")
    print()
    print(f"Annualized Returns:")
    print(f"  Mean: {annualized_returns.mean()*100:.2f}%")
    print(f"  Median: {np.median(annualized_returns)*100:.2f}%")
    print(f"  Std dev: {annualized_returns.std()*100:.2f}%")
    print(f"  Best case (99th percentile): {np.percentile(annualized_returns, 99)*100:.2f}%")
    print(f"  Worst case (1st percentile): {np.percentile(annualized_returns, 1)*100:.2f}%")
    print()
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    if save_results:
        print(f"Saving terminal returns to {output_dir}/terminal_returns.csv...")
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(
            f"{output_dir}/terminal_returns.csv",
            terminal_returns,
            delimiter=",",
            header="terminal_return",
            comments=""
        )
        print("✓ Saved!")
        print()
    
    # ========================================================================
    # PRINT SIMULATION SUMMARY
    # ========================================================================
    
    print(f"{'='*60}")
    print("SIMULATION SUMMARY")
    print(f"{'='*60}\n")
    print(f"Historical data used for parameter estimation:")
    print(f"  Trading days: {len(returns_df)}")
    print(f"  Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"\nMonte Carlo simulation:")
    print(f"  Future days simulated: {T:,} ({simulation_years:.1f} years)")
    print(f"  Number of scenarios: {num_scenarios:,}")
    print(f"  Total simulated returns: {num_scenarios*T:,}")
    print(f"\n✓ All phases complete!\n")


if __name__ == "__main__":
    main()