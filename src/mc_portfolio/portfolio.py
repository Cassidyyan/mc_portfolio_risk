import numpy as np
from .utils import validate_weights, validate_sim_returns

def compute_portfolio_returns(sim_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute portfolio daily returns as weighted sum of asset daily returns.
    
    For each scenario n and time step t, calculates:
        port_returns[n, t] = sum(weights[i] * sim_returns[n, t, i]) for all assets i
    
    Parameters
    ----------
    sim_returns : np.ndarray
        Simulated asset returns, shape (N, T, k)
        N = number of scenarios, T = time steps, k = number of assets
    weights : np.ndarray
        Portfolio weights, shape (k,)
    
    Returns
    -------
    port_returns : np.ndarray
        Portfolio returns, shape (N, T)
    """
    # Vectorized weighted sum using Einstein summation
    # 'ntk,k->nt' means: sum over k dimension
    port_returns = np.einsum('ntk,k->nt', sim_returns, weights)
    
    # Converted from tensor to matrix shape (N, T)
    return port_returns

def compute_portfolio_values(port_returns: np.ndarray, v0: float, return_type: str = "log") -> np.ndarray:
    """Convert portfolio returns to portfolio value paths.
    
    Parameters
    ----------
    port_returns : np.ndarray
        Portfolio returns, shape (N, T)
    v0 : float
        Initial portfolio value
    return_type : {"log", "pct"}, default="log"
        Type of returns:
        - "log": V(t) = V0 * exp(cumsum(r))
        - "pct": V(t) = V0 * cumprod(1 + r)
    
    Returns
    -------
    port_values : np.ndarray
        Portfolio value paths, shape (N, T)
    
    Raises
    ------
    ValueError
        If return_type is invalid
    """
    N, T = port_returns.shape
    port_values = np.zeros((N, T), dtype=float)

    if return_type == "log":
        # For log returns: cumulative sum then exponentiate since log returns are additive
        # A NxT array with entries that represent the portfolio value at time t in scenario n
        cum_log_returns = np.cumsum(port_returns, axis=1)
        port_values = v0 * np.exp(cum_log_returns)
        
    elif return_type == "pct":
        # For percentage returns: cumulative product
        # A NxT array with entries that represent the portfolio value at time t in scenario n
        cum_pct_returns = np.cumprod(1 + port_returns, axis=1)
        port_values = v0 * cum_pct_returns
        
    else:
        raise ValueError(f"Invalid return_type '{return_type}'. Use 'log' or 'pct'.")

    return port_values

def compute_terminal_values(port_values: np.ndarray) -> np.ndarray:
    """Extract terminal (final) portfolio values from value paths.
    
    Parameters
    ----------
    port_values : np.ndarray
        Portfolio value paths, shape (N, T)
    
    Returns
    -------
    terminal_values : np.ndarray
        Terminal portfolio values, shape (N,)
        terminal_values[n] = port_values[n, -1]
    """
    return port_values[:, -1]


def compute_terminal_returns(terminal_values: np.ndarray, v0: float) -> np.ndarray:
    """Compute simple returns from initial to terminal values.
    
    Parameters
    ----------
    terminal_values : np.ndarray
        Terminal portfolio values, shape (N,)
    v0 : float
        Initial portfolio value
    
    Returns
    -------
    terminal_returns : np.ndarray
        Terminal simple returns, shape (N,)
        terminal_returns[n] = (terminal_values[n] / v0) - 1
    """
    return terminal_values / v0 - 1.0
