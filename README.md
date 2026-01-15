# Monte Carlo Portfolio Risk Simulator

A Python-based Monte Carlo simulation framework for portfolio risk analysis. Uses historical data to estimate parameters and simulate thousands of correlated return scenarios, computing comprehensive risk metrics including VaR, CVaR, maximum drawdown, and return distributions.

## Features

- **Historical Parameter Estimation**: Automatically fetches data from Yahoo Finance and estimates mean returns and covariance matrices
- **Monte Carlo Simulation**: Generates thousands of correlated return scenarios using Cholesky decomposition
- **Portfolio Aggregation**: Supports custom portfolio weights with validation
- **Comprehensive Risk Metrics**: VaR, CVaR, drawdown statistics, return quantiles, skewness, and kurtosis
- **Rich Visualizations**: Fan charts, histograms, CDFs, correlation heatmaps, risk contribution, and benchmark comparisons
- **Organized Output**: Automatically saves charts and data to separate subdirectories

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mc_portfolio_risk

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

1. **Configure your portfolio** in `src/mc_portfolio/config.py`:
```python
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
INITIAL_VALUE = 10000
SIMULATION_YEARS = 2
NUM_SCENARIOS = 20_000
```

2. **Run the simulation**:
```bash
python src/mc_portfolio/main.py
```

3. **View results** in the `outputs/` directory:
   - `outputs/charts/` - All visualization PNG files
   - `outputs/data/` - Terminal returns CSV

## Project Structure

```
mc_portfolio_risk/
├── src/mc_portfolio/
│   ├── config.py          # Configuration settings
│   ├── main.py            # Main execution script
│   ├── data.py            # Data loading from Yahoo Finance
│   ├── returns.py         # Return calculation
│   ├── params.py          # Parameter estimation
│   ├── simulate.py        # Monte Carlo simulation engine
│   ├── portfolio.py       # Portfolio aggregation
│   ├── risk.py            # Risk metrics computation
│   ├── plots.py           # Visualization functions
│   └── utils.py           # Helper utilities
├── tests/                 # Unit tests
├── outputs/               # Generated outputs
│   ├── charts/            # Visualization PNG files
│   └── data/              # CSV data files
└── README.md
```

## Configuration Options

### Portfolio Settings
- `TICKERS`: List of stock symbols
- `WEIGHTS`: Portfolio allocation (must sum to 1.0)
- `INITIAL_VALUE`: Starting portfolio value
- `ALLOW_SHORT_SELLING`: Enable negative weights

### Simulation Settings
- `HISTORICAL_YEARS`: Years of historical data for parameter estimation
- `SIMULATION_YEARS`: Forward simulation horizon
- `NUM_SCENARIOS`: Number of Monte Carlo scenarios
- `RANDOM_SEED`: Seed for reproducibility
- `RETURN_TYPE`: "log" or "pct" returns

### Output Settings
- `SAVE_RESULTS`: Save terminal returns to CSV
- `CHARTS_DIR`: Directory for visualization outputs
- `DATA_DIR`: Directory for CSV outputs

## Mathematical Framework

### 1. Parameter Estimation

From historical returns data, we estimate using pandas built-in methods:

**Mean Returns (μ)**:
```
μᵢ = (1/n) Σₜ rᵢₜ
```
Computed as the column-wise mean of the returns DataFrame.

**Covariance Matrix (Σ)**:
```
Σᵢⱼ = (1/(n-1)) Σₜ (rᵢₜ - μᵢ)(rⱼₜ - μⱼ)
```
Computed using pandas `.cov()` method, then symmetrized: `Σ = 0.5(Σ + Σᵀ)`

**Correlation Matrix**:
```
ρᵢⱼ = Σᵢⱼ / (σᵢ σⱼ)
```
where `σᵢ = √Σᵢᵢ` is the standard deviation of asset `i`.

### 2. Monte Carlo Simulation via Cholesky Decomposition

**Step 1: Cholesky Factorization**

Decompose the covariance matrix:
```
Σ = L Lᵀ
```
where `L` is a lower triangular matrix (Cholesky factor).

**Step 2: Generate Correlated Returns**

For each scenario `n` and time step `t`:

1. Draw independent standard normal random variables: `Z ~ N(0, I)`
2. Apply Cholesky factor to induce correlation: `ε = Z Lᵀ`
3. Add mean to get correlated returns: `rₙₜ = μ + ε`

Implementation:
```python
Z = rng.standard_normal((T, k))          # Independent normal (T×k)
correlated_Z = Z @ L.T                   # Induce correlation (T×k)
sim_returns[n] = mu + correlated_Z       # Add mean
```

### 3. Portfolio Aggregation

**Portfolio Returns** (vectorized using Einstein summation):
```
rₚₜ = Σᵢ wᵢ rᵢₜ = wᵀ rₜ
```
Implemented as: `np.einsum('ntk,k->nt', sim_returns, weights)`

**Portfolio Variance**:
```
σₚ² = wᵀ Σ w
```

**Portfolio Value Evolution**:

For **logarithmic returns** (default):
```
Vₜ = V₀ exp(Σₛ₌₁ᵗ rₚₛ)
```
Log returns are additive, so we use cumulative sum then exponentiate.

For **percentage returns**:
```
Vₜ = V₀ Πₛ₌₁ᵗ (1 + rₚₛ)
```
Percentage returns require cumulative product.

**Terminal Returns**:
```
Rₜₑᵣₘᵢₙₐₗ = (Vₜ / V₀) - 1
```

### 4. Risk Metrics

**Value at Risk (VaR)**:
```
VaRα = inf{x : P(R ≤ x) ≥ α}
```
The α-quantile of the return distribution. For α = 0.05, this is the 5th percentile return.

Implementation: `np.quantile(returns, alpha, method='linear')`

**Conditional Value at Risk (CVaR)**:
```
CVaRα = E[R | R ≤ VaRα]
```
The expected return given that we're in the worst α% of scenarios (tail average).

Implementation:
```python
tail_mask = returns <= var_alpha
cvar_alpha = np.mean(returns[tail_mask])
```

**Maximum Drawdown (MDD)**:

For each time step `t`, compute:
```
Running Maxₜ = max_{s≤t} Vₛ

Drawdownₜ = (Running Maxₜ - Vₜ) / Running Maxₜ

MDD = max_{t∈[0,T]} Drawdownₜ
```

Implementation:
```python
running_max = np.maximum.accumulate(port_values, axis=1)
drawdown = (running_max - port_values) / running_max
max_dd = np.max(drawdown, axis=1)
```

**Risk Contribution**:
```
RCᵢ = wᵢ (Σw)ᵢ / (wᵀΣw)
```
The marginal contribution of asset `i` to total portfolio variance.

### 5. Statistical Moments

**Skewness**:
```
Skew = E[(R - μ)³] / σ³
```
Measures asymmetry of the return distribution. Positive skew indicates a right tail (more extreme positive returns).

**Excess Kurtosis**:
```
Kurt = E[(R - μ)⁴] / σ⁴ - 3
```
Measures tail heaviness relative to normal distribution. Positive excess kurtosis indicates heavier tails (more extreme events).

Implementation uses `scipy.stats.skew()` and `scipy.stats.kurtosis()`.

**Probability of Loss**:
```
P(Loss) = (# of scenarios with R < 0) / N
```

## Generated Visualizations

1. **Fan Chart**: Portfolio value percentile bands over time
2. **Terminal Returns Histogram**: Distribution with VaR/CVaR markers
3. **Sample Paths**: 100 randomly selected simulation trajectories
4. **Terminal Returns CDF**: Cumulative distribution function
5. **Maximum Drawdown Distribution**: Histogram of worst drawdowns
6. **Portfolio Weights**: Pie chart of asset allocation
7. **Correlation Heatmap**: Asset return correlations
8. **Risk Contribution**: Each asset's contribution to portfolio variance
9. **Mean Path vs Benchmark**: Comparison with SPY (if available)

## Risk Metrics

- **VaR (Value at Risk)**: Maximum loss at specified confidence level
- **CVaR (Conditional VaR)**: Expected loss in worst scenarios
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Return Quantiles**: Distribution percentiles (1%, 5%, 50%, 95%, 99%)
- **Shape Statistics**: Skewness and kurtosis
- **Probability of Loss**: Likelihood of negative returns

## Example Output

```
BASIC STATISTICS:
  Mean return: 15.23%
  Std deviation: 28.45%
  Probability of loss: 35.20%

RISK MEASURES (Value at Risk):
  VaR (5%): -25.67%
  CVaR (5%): -32.14%

Portfolio Value After 2 Years:
  Mean final value: $11,523
  Median final value: $10,987
  Best case (99th percentile): $18,450
  Worst case (1st percentile): $5,230
```

## Testing

```bash
pytest tests/
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.