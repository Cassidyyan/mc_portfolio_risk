import numpy as np
from typing import Optional
from scipy import stats


def validate_terminal_returns(x: np.ndarray | list) -> np.ndarray:
    """
    Validate and convert terminal returns to a 1D numpy array.
    """
    arr = np.asarray(x, dtype=float)
    
    if arr.size == 0:
        raise ValueError("terminal_returns cannot be empty")
    if arr.ndim != 1:
        raise ValueError(f"terminal_returns must be 1D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("terminal_returns contains non-finite values (NaN or inf)")
    
    return arr


def compute_basic_stats(terminal_returns: np.ndarray) -> dict:
    """
    Compute basic statistical measures of terminal returns.
    
    Parameters
    ----------
    terminal_returns : np.ndarray
        Array of terminal simple returns, shape (N,)
        
    Returns
    -------
    dict
        Dictionary containing:
        - mean_return: float
        - std_return: float (sample std, ddof=1)
        - prob_loss: float (probability of negative return)
        - min_return: float
        - max_return: float
    """
    returns = validate_terminal_returns(terminal_returns)
    
    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns, ddof=1)), # sample standard deviation
        'prob_loss': float(np.mean(returns < 0)), # probability proportion of returns < 0
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns))
    }


def compute_var_cvar(terminal_returns: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Compute Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    
    Uses returns-based convention:
    - VaR_alpha = quantile(returns, alpha)
    - CVaR_alpha = mean(returns[returns <= VaR_alpha])
    
    Parameters
    ----------
    terminal_returns : np.ndarray
        Array of terminal simple returns, shape (N,)
    alpha : float, default=0.05
        Confidence level (e.g., 0.05 for 5% VaR)
        
    Returns
    -------
    dict
        Dictionary containing:
        - var_alpha: float (alpha-quantile of returns)
        - cvar_alpha: float (expected return in worst alpha tail)
        - alpha: float (echoed back)
        
    Raises
    ------
    ValueError
        If alpha not in (0, 1)
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    returns = validate_terminal_returns(terminal_returns)
    
    # Compute VaR as alpha-quantile (cutoff/threashold return at the alpha percentile)
    var_alpha = np.quantile(returns, alpha, method='linear')
    
    # Compute CVaR as mean of tail returns <= VaR
    # Handle edge case: ensure at least one element in tail
    tail_mask = returns <= var_alpha
    if not np.any(tail_mask):
        # if no returns <= VaR (should not happen), fallback to min
        cvar_alpha = np.min(returns)
    else:
        # Get the average of the tail (values <= VaR @ alpha PERCENTILE)
        cvar_alpha = np.mean(returns[tail_mask])
    
    return {
        'alpha': float(alpha),
        'var_alpha': float(var_alpha),
        'cvar_alpha': float(cvar_alpha)
    }


def compute_quantiles_table(
    terminal_returns: np.ndarray,
    qs: list[float] = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
) -> dict:
    """
    Compute quantiles table for terminal returns distribution.
    
    Parameters
    ----------
    terminal_returns : np.ndarray
        Array of terminal simple returns, shape (N,)
    qs : list[float], default=[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
        Quantile levels to compute
        
    Returns
    -------
    dict
        Dictionary mapping percentile labels (e.g., '1%', '50%') to quantile values
    """
    returns = validate_terminal_returns(terminal_returns)
    
    # Compute quantiles and cutoffs returns at specified levels qs (0 < q < 1)
    quantiles = np.quantile(returns, qs, method='linear')
    
    # Format as percentile labels
    result = {}
    for q, val in zip(qs, quantiles):
        pct = int(q * 100) if (q * 100).is_integer() else q * 100
        label = f'{pct}%'
        result[label] = float(val)
    
    return result


def compute_shape_stats(terminal_returns: np.ndarray) -> dict:
    """
    Compute distribution shape statistics (skewness and kurtosis).
    
    Uses scipy.stats for robust computation:
    - skewness: measure of asymmetry
    - kurtosis: measure of tail heaviness (raw fourth moment)
    - excess_kurtosis: kurtosis - 3 (normal distribution has excess kurtosis = 0)
    
    Parameters
    ----------
    terminal_returns : np.ndarray
        Array of terminal simple returns, shape (N,)
        
    Returns
    -------
    dict
        Dictionary containing:
        - skewness: float
        - kurtosis: float (raw fourth moment)
        - excess_kurtosis: float (kurtosis - 3)
        
    Raises
    ------
    ValueError
        If standard deviation is zero (constant returns)
    """
    returns = validate_terminal_returns(terminal_returns)
    
    # Check for zero variance
    if np.std(returns, ddof=1) == 0:
        raise ValueError("Cannot compute shape stats: standard deviation is zero (constant returns)")
    
    return {
        'skewness': float(stats.skew(returns, bias=False)),
        'kurtosis': float(stats.kurtosis(returns, fisher=False, bias=False)),
        'excess_kurtosis': float(stats.kurtosis(returns, fisher=True, bias=False))
    }


def compute_max_drawdown(port_values: np.ndarray) -> np.ndarray:
    """
    Compute maximum drawdown for each simulation path.
    
    Drawdown at time t = (running_max - value_t) / running_max
    Maximum drawdown = max(drawdown_t) over all t
    
    Parameters
    ----------
    port_values : np.ndarray
        Portfolio value paths, shape (N, T) where N is number of simulations
        and T is number of time steps
        
    Returns
    -------
    np.ndarray
        Maximum drawdown for each simulation, shape (N,)
        
    Raises
    ------
    ValueError
        If input is not 2D or contains non-positive values
    """
    if port_values.ndim != 2:
        raise ValueError(f"port_values must be 2D (N, T), got shape {port_values.shape}")
    
    if not np.all(np.isfinite(port_values)):
        raise ValueError("port_values contains non-finite values")
    
    # No negative portfolio values allowed (no margin calls)
    if np.any(port_values <= 0):
        raise ValueError("port_values must be positive (required for drawdown calculation)")
    
    # Compute running maximum along time axis (axis=1, across columns)
    running_max = np.maximum.accumulate(port_values, axis=1)
    
    # Compute drawdown at each time step (N, T)
    drawdown = (running_max - port_values) / running_max
    
    # Maximum drawdown per N simulation (N, )
    max_dd = np.max(drawdown, axis=1)
    
    return max_dd


def summarize_mdd(mdd_array: np.ndarray) -> dict:
    """
    Summarize maximum drawdown statistics.
    
    Parameters
    ----------
    mdd_array : np.ndarray
        Array of maximum drawdowns, shape (N,)
        
    Returns
    -------
    dict
        Dictionary containing:
        - mean_mdd: float
        - median_mdd: float
        - std_mdd: float
        - p95_mdd: float (95th percentile)
        - min_mdd: float
        - max_mdd: float
    """
    mdd = validate_terminal_returns(mdd_array)
    
    return {
        'mean_mdd': float(np.mean(mdd)),
        'median_mdd': float(np.median(mdd)),
        'std_mdd': float(np.std(mdd, ddof=1)),
        'p95_mdd': float(np.quantile(mdd, 0.95)),
        'min_mdd': float(np.min(mdd)),
        'max_mdd': float(np.max(mdd))
    }


def compute_risk_report(
    terminal_returns: np.ndarray,
    port_values: np.ndarray,
    alpha: float = 0.05,
    quantiles: list[float] = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
) -> dict:
    """
    Compute comprehensive risk metrics report.
    
    High-level function that combines all risk metrics into a single report. Wrapper function.
    
    Parameters
    ----------
    terminal_returns : np.ndarray
        Array of terminal simple returns, shape (N,)
    port_values : np.ndarray
        Portfolio value paths for drawdown analysis, shape (N, T)
    alpha : float, default=0.05
        Confidence level for VaR/CVaR
    quantiles : list[float], default=[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
        Quantile levels to compute
        
    Returns
    -------
    dict
        Comprehensive risk report containing:
        - basic_stats: dict (mean, std, prob_loss, min, max)
        - var_cvar: dict (VaR, CVaR at alpha level)
        - quantiles_table: dict (quantile values)
        - shape_stats: dict (skewness, kurtosis) if computable
        - mdd_summary: dict (drawdown stats)
    """
    # Validate inputs
    returns = validate_terminal_returns(terminal_returns)
    
    # Initialize report
    report = {}
    
    # Core metrics
    report['basic_stats'] = compute_basic_stats(returns)
    report['var_cvar'] = compute_var_cvar(returns, alpha=alpha)
    report['quantiles_table'] = compute_quantiles_table(returns, qs=quantiles)
    
    # Optional: shape statistics
    try:
        report['shape_stats'] = compute_shape_stats(returns)
    except ValueError as e:
        # Skip if std=0 or other issues
        report['shape_stats'] = {'error': str(e)}
    
    # Drawdown analysis
    try:
        mdd = compute_max_drawdown(port_values)
        report['mdd_summary'] = summarize_mdd(mdd)
    except (ValueError, Exception) as e:
        report['mdd_summary'] = {'error': str(e)}
    
    return report
