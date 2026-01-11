import numpy as np  
import pandas as pd 

def _validate_sim_inputs(mu: np.ndarray, cov: np.ndarray, T: int, N: int) -> int:
    # Validate mu is 1D
    if mu.ndim != 1:
        raise ValueError(f"mu must be 1D array, got shape {mu.shape}")
    
    # Validate cov is 2D square
    if cov.ndim != 2:
        raise ValueError(f"cov must be 2D array, got shape {cov.shape}")
    
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    
    # Validate dimensions match
    if cov.shape[0] != len(mu):
        raise ValueError(
            f"cov dimensions ({cov.shape[0]}) don't match mu length ({len(mu)})"
        )
    
    # Validate T and N
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    
    # Validate all values are finite
    if not np.isfinite(mu).all():
        raise ValueError("mu contains non-finite values (inf or NaN)")
    
    if not np.isfinite(cov).all():
        raise ValueError("cov contains non-finite values (inf or NaN)")
    
    return len(mu)

