# ============================================================================
# HISTORICAL DATA SETTINGS
# ============================================================================

# Assets to include in portfolio
TICKERS = [
    "NVDA", "META", "SOFI", "AMZN", "SPY", "AVGO",
    "TSM", "TSLA", "PHYS", "PLTR", "GOOG", "URA",
    "UBER", "NFLX"
]

# Years of historical data for parameter estimation
HISTORICAL_YEARS = 5

# ============================================================================
# MONTE CARLO SIMULATION SETTINGS
# ============================================================================

# Number of years to simulate forward
SIMULATION_YEARS = 5

# Number of Monte Carlo scenarios to generate
NUM_SCENARIOS = 20_000

# Random seed for reproducibility (use None for random)
RANDOM_SEED = 1

# ============================================================================
# PORTFOLIO SETTINGS
# ============================================================================

# Portfolio allocation (must sum to 1.0)
# Note: Length must match number of TICKERS
WEIGHTS = [
    0.2807,  # NVDA
    0.1368,  # META
    0.1359,  # SOFI
    0.1264,  # AMZN
    0.0814,  # SPY (formerly VFV)
    0.0524,  # AVGO
    0.0506,  # TSM
    0.0446,  # TSLA
    0.0437,  # PHYS
    0.0185,  # PLTR
    0.0156,  # GOOG
    0.0061,  # URA
    0.0038,  # UBER
    0.0035   # NFLX
]


# Starting portfolio value ($)
INITIAL_VALUE = 1000

# Allow negative weights (short positions)?
ALLOW_SHORT_SELLING = False

# ============================================================================
# RETURN CALCULATION METHOD
# ============================================================================

# "log" for logarithmic returns (recommended for Monte Carlo)
# "pct" for percentage returns
RETURN_TYPE = "log"

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Save terminal returns to CSV?
SAVE_RESULTS = True

# Directory for output files
OUTPUT_DIR = "outputs"
