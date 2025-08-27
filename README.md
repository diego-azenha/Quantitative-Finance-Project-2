# Mean–Variance Portfolio Optimization (Long-Only, Fixed-Weight Rebalance)

This repo implements a clean, reproducible pipeline to estimate a long-only mean–variance efficient frontier from 10-year monthly data, locate the Sharpe-max (“optimal”) and minimum-variance portfolios under fully-invested constraints, and backtest both with monthly rebalancing to fixed weights. It also generates a two-asset diversification demo by sweeping correlations from −1 to +1.

---

## 📁 Repo Structure
```
clean_data/
└─ clean_stock_prices.parquet # wide table of adjusted close prices (Date index, one column per ticker)
results/
├─ mean_variance_frontier.png # cloud + efficient frontier + optimal & min-var markers
└─ two_asset_frontier.png # correlation-sweep illustration (file name may differ in your plots.py)
main.py # end-to-end script: estimates, optimization, backtest, plots, console output
plots.py # all plotting routines (frontier cloud, styling, labels, two-asset figure)
requirements.txt # Python dependencies
README.md # this file
```


**Data expectation**: `clean_stock_prices.parquet` must contain a DateTime index (daily or higher frequency), columns named by tickers, numeric prices (no returns). The script resamples to month-end and computes monthly returns internally.

---

## 🧰 Installation

```bash
# (Recommended) Create and activate a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

Minimal packages (if you don’t use requirements.txt): numpy, pandas, scipy, matplotlib, pyarrow (or fastparquet) for Parquet I/O.
```

## How to run

```
# From the repo root (where main.py lives)
python main.py
``` 
