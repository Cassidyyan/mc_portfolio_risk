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
from mc_portfolio.plots import (
    plot_fan_chart, plot_terminal_hist, plot_terminal_cdf, plot_sample_paths,
    plot_max_drawdown_hist, plot_weights_bar, plot_corr_heatmap,
    plot_risk_contribution, plot_mean_path_vs_benchmark
)
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
    weights = np.array(WEIGHTS)
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
    
    # Compute correlation matrix from covariance
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    
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
    # PHASE 6: VISUALIZATIONS
    # ========================================================================
    
    print(f"{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    # Create outputs directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fan chart of portfolio values
    print("Creating fan chart...")
    plot_fan_chart(
        port_values=portfolio_results['port_values'],
        outpath=f"{output_dir}/fan_chart.png",
        percentiles=(5, 25, 50, 75, 95),
        title=f"Portfolio Value Fan Chart ({simulation_years}-Year Horizon)",
        xlabel="Trading Day",
        ylabel="Portfolio Value ($)"
    )
    print(f"✓ Saved to {output_dir}/fan_chart.png")
    
    # 2. Histogram of terminal returns with VaR/CVaR
    print("Creating terminal returns histogram...")
    plot_terminal_hist(
        terminal_returns=terminal_returns,
        outpath=f"{output_dir}/terminal_hist.png",
        bins=60,
        alpha=0.05,
        var_alpha=risk_report['var_cvar']['var_alpha'],
        cvar_alpha=risk_report['var_cvar']['cvar_alpha'],
        title=f"Terminal Return Distribution ({simulation_years}-Year Horizon)",
        xlabel="Terminal Return (%)",
        ylabel="Frequency"
    )
    print(f"✓ Saved to {output_dir}/terminal_hist.png")
    
    # 3. Sample simulation paths
    print("Creating sample paths plot...")
    plot_sample_paths(
        port_values=portfolio_results['port_values'],
        outpath=f"{output_dir}/sample_paths.png",
        n_paths=100,
        seed=random_seed,
        title=f"Sample Portfolio Trajectories (100 scenarios)",
        xlabel="Trading Day",
        ylabel="Portfolio Value ($)"
    )
    print(f"✓ Saved to {output_dir}/sample_paths.png")
    
    # 4. Terminal Return CDF
    print("Creating terminal return CDF plot...")
    plot_terminal_cdf(
        terminal_returns=terminal_returns,
        outpath=f"{output_dir}/terminal_cdf.png",
        alpha=0.05,
        title=f"Empirical CDF of Terminal Returns",
        xlabel="Terminal Return (%)",
        ylabel="Cumulative Probability"
    )
    print(f"✓ Saved to {output_dir}/terminal_cdf.png")
    
    # 5. Max Drawdown Distribution
    print("Creating max drawdown histogram...")
    plot_max_drawdown_hist(
        port_values=portfolio_results['port_values'],
        outpath=f"{output_dir}/max_drawdown_hist.png",
        bins=50,
        title=f"Distribution of Maximum Drawdowns",
        xlabel="Maximum Drawdown (%)",
        ylabel="Frequency"
    )
    print(f"✓ Saved to {output_dir}/max_drawdown_hist.png")
    
    # 6. Portfolio Weights Bar Chart
    print("Creating portfolio weights bar chart...")
    plot_weights_bar(
        assets=tickers,
        weights=weights,
        outpath=f"{output_dir}/weights_bar.png",
        title=f"Portfolio Weights",
        xlabel="Asset",
        ylabel="Weight"
    )
    print(f"✓ Saved to {output_dir}/weights_bar.png")
    
    # 7. Correlation Heatmap
    print("Creating correlation heatmap...")
    plot_corr_heatmap(
        assets=tickers,
        corr=corr,
        outpath=f"{output_dir}/corr_heatmap.png",
        title=f"Asset Correlation Matrix"
    )
    print(f"✓ Saved to {output_dir}/corr_heatmap.png")
    
    # 8. Risk Contribution
    print("Creating risk contribution plot...")
    plot_risk_contribution(
        assets=tickers,
        weights=weights,
        cov=cov,
        outpath=f"{output_dir}/risk_contribution.png",
        title=f"Risk Contribution by Asset",
        xlabel="Asset",
        ylabel="Contribution to Portfolio Variance (%)"
    )
    print(f"✓ Saved to {output_dir}/risk_contribution.png")
    
    # 9. Mean Portfolio Path vs SPY Benchmark
    print("Creating benchmark comparison plot...")
    try:
        # Use SPY returns from historical data to project forward
        spy_returns = returns_df['SPY'].values if 'SPY' in returns_df.columns else None
        
        if spy_returns is not None and len(spy_returns) > 0:
            # Use all available SPY returns, repeat if needed to match T
            n_available = len(spy_returns)
            if n_available < T:
                # Use recent returns and repeat them to fill the gap
                recent_spy_returns = spy_returns[-min(T, n_available):]
                # Repeat to match simulation length
                n_repeats = (T // len(recent_spy_returns)) + 1
                benchmark_returns = np.tile(recent_spy_returns, n_repeats)[:T]
            else:
                benchmark_returns = spy_returns[-T:]
            
            # Convert to price path
            benchmark_prices = initial_value * np.exp(np.cumsum(benchmark_returns))
            
            plot_mean_path_vs_benchmark(
                port_values=portfolio_results['port_values'],
                v0=initial_value,
                benchmark_prices=benchmark_prices,
                outpath=f"{output_dir}/mean_vs_benchmark.png",
                title=f"Mean Portfolio Path vs SPY Benchmark",
                xlabel="Trading Day",
                ylabel="Cumulative Return"
            )
            print(f"✓ Saved to {output_dir}/mean_vs_benchmark.png")
        else:
            print("⚠ Warning: No SPY data available for benchmark comparison")
    except Exception as e:
        print(f"⚠ Warning: Benchmark comparison skipped ({str(e)})")
    
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