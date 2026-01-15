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