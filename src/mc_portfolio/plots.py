import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ensure_outdir(filepath: str) -> None:
    """
    Ensure the parent directory of a file path exists.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def plot_fan_chart(
    port_values: np.ndarray,
    outpath: str,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
    x: np.ndarray | None = None,
    title: str | None = None,
    xlabel: str = "Day",
    ylabel: str = "Portfolio Value"
) -> None:
    """
    Create a fan chart showing percentile bands of portfolio values over time.
    
    Parameters
    ----------
    port_values : np.ndarray
        Portfolio value paths, shape (N, T)
    outpath : str
        Path to save the figure
    percentiles : tuple[int, ...], default=(5, 25, 50, 75, 95)
        Percentiles to plot as bands
    x : np.ndarray, optional
        X-axis values. If None, uses np.arange(T)
    title : str, optional
        Plot title
    xlabel : str, default="Day"
        X-axis label
    ylabel : str, default="Portfolio Value"
        Y-axis label
    """
    if port_values.ndim != 2:
        raise ValueError(f"port_values must be 2D (N, T), got shape {port_values.shape}")
    
    if not np.all(np.isfinite(port_values)):
        raise ValueError("port_values contains non-finite values")
    
    N, T = port_values.shape
    
    if x is None:
        x = np.arange(T)
    
    # Compute percentile curves across simulations at each time step
    p = np.percentile(port_values, q=percentiles, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    perc_dict = {pct: i for i, pct in enumerate(percentiles)}
    
    # Plot median line
    if 50 in perc_dict:
        ax.plot(x, p[perc_dict[50]], color='darkblue', linewidth=2, label='Median (50th)')
    
    # Shade 25-75 percentile band
    if 25 in perc_dict and 75 in perc_dict:
        ax.fill_between(x, p[perc_dict[25]], p[perc_dict[75]], 
                        alpha=0.3, color='steelblue', label='25th-75th percentile')
    
    # Shade 5-95 percentile band
    if 5 in perc_dict and 95 in perc_dict:
        ax.fill_between(x, p[perc_dict[5]], p[perc_dict[95]], 
                        alpha=0.15, color='lightblue', label='5th-95th percentile')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title or 'Portfolio Value Fan Chart', fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_terminal_hist(
    terminal_returns: np.ndarray,
    outpath: str,
    bins: int = 60,
    alpha: float = 0.05,
    var_alpha: float | None = None,
    cvar_alpha: float | None = None,
    title: str | None = None,
    xlabel: str = "Terminal Return",
    ylabel: str = "Frequency"
) -> None:
    """
    Create a histogram of terminal returns with risk metric annotations.
    
    Parameters
    ----------
    terminal_returns : np.ndarray
        Array of terminal simple returns, shape (N,)
    outpath : str
        Path to save the figure
    bins : int, default=60
        Number of histogram bins
    alpha : float, default=0.05
        Confidence level for VaR/CVaR if not provided
    var_alpha : float, optional
        Pre-computed VaR value
    cvar_alpha : float, optional
        Pre-computed CVaR value
    title : str, optional
        Plot title
    xlabel : str, default="Terminal Return"
        X-axis label
    ylabel : str, default="Frequency"
        Y-axis label
    """
    if terminal_returns.ndim != 1:
        raise ValueError(f"terminal_returns must be 1D, got shape {terminal_returns.shape}")
    
    if not np.all(np.isfinite(terminal_returns)):
        raise ValueError("terminal_returns contains non-finite values")
    
    # Compute risk metrics if not provided
    if var_alpha is None:
        var_alpha = np.quantile(terminal_returns, alpha)
    
    if cvar_alpha is None:
        tail_mask = terminal_returns <= var_alpha
        if np.any(tail_mask):
            cvar_alpha = np.mean(terminal_returns[tail_mask])
        else:
            cvar_alpha = np.min(terminal_returns)
    
    mean_return = np.mean(terminal_returns)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(terminal_returns, bins=bins, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.axvline(mean_return, color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {mean_return*100:.1f}%')
    ax.axvline(var_alpha, color='orange', linestyle='--', linewidth=2, 
               label=f'VaR({alpha*100:.0f}%): {var_alpha*100:.1f}%')
    ax.axvline(cvar_alpha, color='red', linestyle='--', linewidth=2, 
               label=f'CVaR({alpha*100:.0f}%): {cvar_alpha*100:.1f}%')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title or 'Distribution of Terminal Returns', fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_terminal_cdf(
    terminal_returns: np.ndarray,
    outpath: str,
    alpha: float = 0.05,
    title: str | None = None,
    xlabel: str = "Terminal Return",
    ylabel: str = "Cumulative Probability"
) -> None:
    """
    Plot empirical CDF of terminal returns with VaR marker.
    
    Parameters
    ----------
    terminal_returns : np.ndarray
        Array of terminal simple returns, shape (N,)
    outpath : str
        Path to save the figure
    alpha : float, default=0.05
        Confidence level for VaR marker
    title : str, optional
        Plot title
    xlabel : str, default="Terminal Return"
        X-axis label
    ylabel : str, default="Cumulative Probability"
        Y-axis label
    """
    if terminal_returns.ndim != 1:
        raise ValueError(f"terminal_returns must be 1D, got shape {terminal_returns.shape}")
    
    if not np.all(np.isfinite(terminal_returns)):
        raise ValueError("terminal_returns contains non-finite values")
    
    # Sort returns for CDF
    sorted_returns = np.sort(terminal_returns)
    cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    # Compute VaR
    var_alpha = np.quantile(terminal_returns, alpha)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sorted_returns, cdf, color='steelblue', linewidth=2)
    
    # Mark VaR on the plot
    ax.axvline(var_alpha, color='red', linestyle='--', linewidth=2, 
               label=f'VaR({alpha*100:.0f}%) = {var_alpha*100:.1f}%')
    ax.axhline(alpha, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title or 'Empirical CDF of Terminal Returns', fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_sample_paths(
    port_values: np.ndarray,
    outpath: str,
    n_paths: int = 25,
    seed: int = 67,
    x: np.ndarray | None = None,
    title: str | None = None,
    xlabel: str = "Day",
    ylabel: str = "Portfolio Value"
) -> None:
    """
    Plot a sample of simulation paths to visualize individual trajectories.
    
    Parameters
    ----------
    port_values : np.ndarray
        Portfolio value paths, shape (N, T)
    outpath : str
        Path to save the figure
    n_paths : int, default=25
        Number of paths to plot
    seed : int, default=67
        Random seed for reproducible path selection
    x : np.ndarray, optional
        X-axis values. If None, uses np.arange(T)
    title : str, optional
        Plot title
    xlabel : str, default="Day"
        X-axis label
    ylabel : str, default="Portfolio Value"
        Y-axis label
    """
    if port_values.ndim != 2:
        raise ValueError(f"port_values must be 2D (N, T), got shape {port_values.shape}")
    
    if not np.all(np.isfinite(port_values)):
        raise ValueError("port_values contains non-finite values")
    
    N, T = port_values.shape
    
    if x is None:
        x = np.arange(T)
    
    rng = np.random.default_rng(seed)
    n_to_plot = min(n_paths, N)
    selected_indices = rng.choice(N, size=n_to_plot, replace=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx in selected_indices:
        ax.plot(x, port_values[idx], alpha=0.4, linewidth=1)
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title or f'Sample Simulation Paths (n={n_to_plot})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_max_drawdown_hist(
    port_values: np.ndarray,
    outpath: str,
    bins: int = 60,
    title: str | None = None,
    xlabel: str = "Maximum Drawdown",
    ylabel: str = "Frequency"
) -> None:
    """
    Plot histogram of maximum drawdown distribution across simulations.
    
    Parameters
    ----------
    port_values : np.ndarray
        Portfolio value paths, shape (N, T)
    outpath : str
        Path to save the figure
    bins : int, default=60
        Number of histogram bins
    title : str, optional
        Plot title
    xlabel : str, default="Maximum Drawdown"
        X-axis label
    ylabel : str, default="Frequency"
        Y-axis label
    """
    if port_values.ndim != 2:
        raise ValueError(f"port_values must be 2D (N, T), got shape {port_values.shape}")
    
    if not np.all(np.isfinite(port_values)):
        raise ValueError("port_values contains non-finite values")
    
    # Compute max drawdown per simulation
    running_max = np.maximum.accumulate(port_values, axis=1)
    drawdown = (running_max - port_values) / running_max
    mdd = np.max(drawdown, axis=1)
    
    # Compute statistics
    mean_mdd = np.mean(mdd)
    median_mdd = np.median(mdd)
    p95_mdd = np.percentile(mdd, 95)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(mdd, bins=bins, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.axvline(mean_mdd, color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {mean_mdd*100:.1f}%')
    ax.axvline(median_mdd, color='blue', linestyle='--', linewidth=2, 
               label=f'Median: {median_mdd*100:.1f}%')
    ax.axvline(p95_mdd, color='red', linestyle='--', linewidth=2, 
               label=f'95th percentile: {p95_mdd*100:.1f}%')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title or 'Maximum Drawdown Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_weights_bar(
    assets: list[str],
    weights: np.ndarray,
    outpath: str,
    title: str | None = None,
    xlabel: str = "Asset",
    ylabel: str = "Weight"
) -> None:
    """
    Plot portfolio weights as a pie chart.
    
    Parameters
    ----------
    assets : list[str]
        List of asset names
    weights : np.ndarray
        Array of portfolio weights, shape (k,)
    outpath : str
        Path to save the figure
    title : str, optional
        Plot title
    xlabel : str, default="Asset"
        X-axis label (not used for pie chart, kept for compatibility)
    ylabel : str, default="Weight"
        Y-axis label (not used for pie chart, kept for compatibility)
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights.shape}")
    
    if len(assets) != len(weights):
        raise ValueError(f"Length mismatch: {len(assets)} assets vs {len(weights)} weights")
    
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contains non-finite values")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a nice color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(assets)))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        weights,
        labels=assets,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title or 'Portfolio Weights', fontsize=13, fontweight='bold', pad=20)
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_corr_heatmap(
    assets: list[str],
    cov: np.ndarray | None = None,
    corr: np.ndarray | None = None,
    outpath: str = "outputs/corr_heatmap.png",
    title: str | None = None
) -> None:
    """
    Plot correlation matrix as a heatmap.
    
    Parameters
    ----------
    assets : list[str]
        List of asset names
    cov : np.ndarray, optional
        Covariance matrix, shape (k, k). Used to compute correlation if corr not provided
    corr : np.ndarray, optional
        Correlation matrix, shape (k, k). Takes precedence over cov
    outpath : str, default="outputs/corr_heatmap.png"
        Path to save the figure
    title : str, optional
        Plot title
    """
    if corr is None and cov is None:
        raise ValueError("Either corr or cov must be provided")
    
    # Compute correlation from covariance if needed
    if corr is None:
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be square 2D matrix, got shape {cov.shape}")
        
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
    
    if corr.shape[0] != len(assets) or corr.shape[1] != len(assets):
        raise ValueError(f"Correlation matrix shape {corr.shape} doesn't match {len(assets)} assets")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(assets)))
    ax.set_yticks(np.arange(len(assets)))
    ax.set_xticklabels(assets, rotation=45, ha='right')
    ax.set_yticklabels(assets)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11)
    
    # Add text annotations
    for i in range(len(assets)):
        for j in range(len(assets)):
            text = ax.text(j, i, f'{corr[i, j]:.2f}',
                          ha='center', va='center', color='white' if abs(corr[i, j]) > 0.5 else 'black',
                          fontsize=8)
    
    ax.set_title(title or 'Asset Correlation Matrix', fontsize=13, fontweight='bold')
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_risk_contribution(
    assets: list[str],
    weights: np.ndarray,
    cov: np.ndarray,
    outpath: str,
    title: str | None = None,
    xlabel: str = "Asset",
    ylabel: str = "Risk Contribution (%)"
) -> None:
    """
    Plot variance contribution by asset.
    
    Risk contribution based on:
    - Portfolio variance: σ_p^2 = w^T Σ w
    - Marginal contribution: m = Σ w
    - Component contribution: c_i = w_i * m_i
    - Percent contribution: c_i / σ_p^2
    
    Parameters
    ----------
    assets : list[str]
        List of asset names
    weights : np.ndarray
        Array of portfolio weights, shape (k,)
    cov : np.ndarray
        Covariance matrix, shape (k, k)
    outpath : str
        Path to save the figure
    title : str, optional
        Plot title
    xlabel : str, default="Asset"
        X-axis label
    ylabel : str, default="Risk Contribution (%)"
        Y-axis label
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights.shape}")
    
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square 2D matrix, got shape {cov.shape}")
    
    if len(assets) != len(weights) or len(weights) != cov.shape[0]:
        raise ValueError(f"Dimension mismatch: {len(assets)} assets, {len(weights)} weights, {cov.shape[0]} cov dim")
    
    # Compute portfolio variance
    port_var = weights @ cov @ weights
    
    # Marginal contribution: Σ w
    marginal = cov @ weights
    
    # Component contribution: w_i * m_i
    component = weights * marginal
    
    # Percent contribution
    if port_var > 0:
        pct_contrib = (component / port_var) * 100
    else:
        pct_contrib = np.zeros_like(component)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['steelblue' if c >= 0 else 'coral' for c in pct_contrib]
    ax.bar(assets, pct_contrib, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title or 'Risk Contribution by Asset', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    if len(assets) > 10:
        plt.xticks(rotation=45, ha='right')
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mean_path_vs_benchmark(
    port_values: np.ndarray,
    v0: float,
    outpath: str,
    benchmark_prices: np.ndarray | None = None,
    benchmark_returns: np.ndarray | None = None,
    x: np.ndarray | None = None,
    title: str = "Mean Simulated Portfolio vs SPY",
    xlabel: str = "Day",
    ylabel: str = "Cumulative Return"
) -> None:
    """
    Plot average simulated portfolio path versus benchmark (e.g., SPY).
    
    Average path is defined as:
    - mean_path = mean(port_values, axis=0)  # average across simulations
    - mean_cum_return = mean_path / v0 - 1
    
    Parameters
    ----------
    port_values : np.ndarray
        Portfolio value paths, shape (N, T)
    v0 : float
        Initial portfolio value
    outpath : str
        Path to save the figure
    benchmark_prices : np.ndarray, optional
        Benchmark price series, shape (T,). If provided, computes cumulative return
    benchmark_returns : np.ndarray, optional
        Benchmark daily simple returns, shape (T,). If provided, computes cumulative return
    x : np.ndarray, optional
        X-axis values. If None, uses np.arange(T)
    title : str, default="Mean Simulated Portfolio vs SPY"
        Plot title
    xlabel : str, default="Day"
        X-axis label
    ylabel : str, default="Cumulative Return"
        Y-axis label
    """
    if port_values.ndim != 2:
        raise ValueError(f"port_values must be 2D (N, T), got shape {port_values.shape}")
    
    if not np.all(np.isfinite(port_values)):
        raise ValueError("port_values contains non-finite values")
    
    N, T = port_values.shape
    
    if x is None:
        x = np.arange(T)
    
    # Compute mean path across simulations
    mean_path = np.mean(port_values, axis=0)
    mean_cum_return = mean_path / v0 - 1
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot simulated portfolio
    ax.plot(x, mean_cum_return, color='darkblue', linewidth=2, linestyle=':', label='Simulated Portfolio (mean path)')
    
    # Plot benchmark if provided
    if benchmark_prices is not None:
        if len(benchmark_prices) != T:
            raise ValueError(f"benchmark_prices length {len(benchmark_prices)} doesn't match T={T}")
        bench_cum = benchmark_prices / benchmark_prices[0] - 1
        ax.plot(x, bench_cum, color='orange', linewidth=2, linestyle='--', label='SPY (benchmark)')
    
    elif benchmark_returns is not None:
        if len(benchmark_returns) != T:
            raise ValueError(f"benchmark_returns length {len(benchmark_returns)} doesn't match T={T}")
        bench_cum = np.cumprod(1 + benchmark_returns) - 1
        ax.plot(x, bench_cum, color='orange', linewidth=2, linestyle='--', label='SPY (benchmark)')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    
    ensure_outdir(outpath)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)