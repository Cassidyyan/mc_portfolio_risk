import numpy as np
import os
from datetime import datetime, timedelta
from mc_portfolio.data import load_prices_yf, clean_prices
from mc_portfolio.returns import compute_returns
from mc_portfolio.params import estimate_mu_cov
from mc_portfolio.utils import summarize_params
from mc_portfolio.simulate import simulate_correlated_returns
from mc_portfolio.portfolio import run_portfolio_aggregation
from mc_portfolio.risk import compute_risk_report
from mc_portfolio.config import (
    TICKERS, HISTORICAL_YEARS, SIMULATION_YEARS, NUM_SCENARIOS, RANDOM_SEED,
    WEIGHTS, INITIAL_VALUE, ALLOW_SHORT_SELLING, RETURN_TYPE,
    SAVE_RESULTS, OUTPUT_DIR
)


def main():
    # Import configuration settings from config.py
    tickers = TICKERS
    historical_years = HISTORICAL_YEARS
    simulation_years = SIMULATION_YEARS
    num_scenarios = NUM_SCENARIOS
    random_seed = RANDOM_SEED
    weights = WEIGHTS
    initial_value = INITIAL_VALUE
    allow_short_selling = ALLOW_SHORT_SELLING
    return_type = RETURN_TYPE
    save_results = SAVE_RESULTS
    output_dir = OUTPUT_DIR
    
    # ========================================================================
    # DERIVED PARAMETERS
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
    
    # Reorder weights to match assets order (assets may be sorted differently than tickers)
    # Create a mapping from ticker to weight
    ticker_to_weight = dict(zip(tickers, weights))
    weights_reordered = [ticker_to_weight[asset] for asset in assets]
    
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
    
    print(f"Portfolio weights (in order specified):")
    # Display weights in original TICKERS order for clarity
    for ticker, weight in zip(tickers, weights):
        print(f"  {ticker:6s} {weight*100:6.2f}%")
    print(f"  {'─'*16}")
    print(f"  {'Total':6s} {sum(weights)*100:6.2f}%")
    print(f"\nInitial value: ${initial_value:,}")
    print()
    
    print("Running portfolio aggregation...")
    portfolio_results = run_portfolio_aggregation(
        sim_returns=sim_returns,
        assets=assets,
        weights=weights_reordered,
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
    #  RISK METRICS
    # ========================================================================
    
    print(f"{'='*60}")
    print("RISK METRICS COMPUTATION")
    print(f"{'='*60}\n")
    
    print("Computing comprehensive risk metrics...")
    
    risk_report = compute_risk_report(
        terminal_returns=portfolio_results['terminal_returns'],
        port_values=portfolio_results['port_values'],
        alpha=0.05
    )
    
    print(f"✓ Risk metrics computed!\n")
    
    # Display basic statistics
    print("BASIC STATISTICS:")
    basic = risk_report['basic_stats']
    print(f"  Mean return: {basic['mean_return']*100:.2f}%")
    print(f"  Std deviation: {basic['std_return']*100:.2f}%")
    print(f"  Probability of loss: {basic['prob_loss']*100:.2f}%")
    print(f"  Min return: {basic['min_return']*100:.2f}%")
    print(f"  Max return: {basic['max_return']*100:.2f}%")
    print()
    
    # Display VaR and CVaR
    print("RISK MEASURES (Value at Risk):")
    var_cvar = risk_report['var_cvar']
    print(f"  VaR (5%): {var_cvar['var_alpha']*100:.2f}%")
    print(f"  CVaR (5%): {var_cvar['cvar_alpha']*100:.2f}%")
    print(f"  → 5% of scenarios have returns of {var_cvar['var_alpha']*100:.2f}% or worse")
    print(f"  → Expected return in worst 5% tail: {var_cvar['cvar_alpha']*100:.2f}%")
    print()
    
    # Display quantiles
    print("RETURN DISTRIBUTION QUANTILES:")
    quantiles = risk_report['quantiles_table']
    for label in ['1%', '5%', '10%', '50%', '90%', '95%', '99%']:
        if label in quantiles:
            print(f"  {label:>4s}: {quantiles[label]*100:>7.2f}%")
    print()
    
    # Display shape statistics if available
    if 'shape_stats' in risk_report and 'error' not in risk_report['shape_stats']:
        print("DISTRIBUTION SHAPE:")
        shape = risk_report['shape_stats']
        print(f"  Skewness: {shape['skewness']:.3f}")
        print(f"  Kurtosis: {shape['kurtosis']:.3f}")
        print(f"  Excess kurtosis: {shape['excess_kurtosis']:.3f}")
        print()
    
    # Display drawdown statistics if available
    if 'mdd_summary' in risk_report and 'error' not in risk_report['mdd_summary']:
        print("MAXIMUM DRAWDOWN STATISTICS:")
        mdd = risk_report['mdd_summary']
        print(f"  Mean MDD: {mdd['mean_mdd']*100:.2f}%")
        print(f"  Median MDD: {mdd['median_mdd']*100:.2f}%")
        print(f"  95th percentile MDD: {mdd['p95_mdd']*100:.2f}%")
        print(f"  Max MDD (worst scenario): {mdd['max_mdd']*100:.2f}%")
        print()
    
    # ========================================================================
    # ADDITIONAL ANALYSIS
    # ======================================================================== 

    terminal_values = portfolio_results['terminal_values']
    terminal_returns = portfolio_results['terminal_returns']
    annualized_returns = (1 + terminal_returns) ** (1 / simulation_years) - 1
    
    print(f"{'='*60}")
    print("PORTFOLIO VALUE & ANNUALIZED RETURNS")
    print(f"{'='*60}\n")
    
    print(f"Portfolio Value After {simulation_years:.0f} Years:")
    print(f"  Initial investment: ${initial_value:,.0f}")
    print(f"  Mean final value: ${terminal_values.mean():,.0f}")
    print(f"  Median final value: ${np.median(terminal_values):,.0f}")
    print(f"  Best case (99th percentile): ${np.percentile(terminal_values, 99):,.0f}")
    print(f"  Worst case (1st percentile): ${np.percentile(terminal_values, 1):,.0f}")
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