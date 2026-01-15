# ============================================================================
# HISTORICAL DATA SETTINGS
# ============================================================================

# Assets to include in portfolio
TICKERS = [
    "GRND",   # Grindr
    "MTCH",   # Match Group
    "BMBL",   # Bumble
    "MOMO"    # Hello Group
]

# Years of historical data for parameter estimation
HISTORICAL_YEARS = 3

# ============================================================================
# MONTE CARLO SIMULATION SETTINGS
# ============================================================================

# Number of years to simulate forward
SIMULATION_YEARS = 2

# Number of Monte Carlo scenarios to generate
NUM_SCENARIOS = 20_000

# Random seed for reproducibility (use None for random)
RANDOM_SEED = 67

# ============================================================================
# PORTFOLIO SETTINGS
# ============================================================================

# Portfolio allocation (must sum to 1.0)
# Note: Length must match number of TICKERS
WEIGHTS = [0.25] * 4  # Equal weight for 4 assets

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

# Base directory for all outputs
OUTPUT_DIR = "outputs"

# Subdirectories for organized output
CHARTS_DIR = f"{OUTPUT_DIR}/charts"
DATA_DIR = f"{OUTPUT_DIR}/data"
