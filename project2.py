import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# ---------- Load stock prices data ----------

# Set the base directory and load the stock price data
base = Path(__file__).resolve().parent
data_dir = base / "clean_data"
prices = pd.read_parquet(data_dir / "clean_stock_prices.parquet").sort_index()
prices.index = pd.to_datetime(prices.index)

# ---------- Calculate monthly returns ----------

# Calculate monthly returns for the 30 stocks
returns = prices.pct_change().dropna()

# Annualized risk-free rate (4.92% per year, assuming it’s constant)
rf_rate = 0.0492

# ---------- Calculate the risk premium (10-year excess return) ----------

# Calculate annualized returns over the last 10 years (120 months)
annualized_returns = returns.mean() * 12

# Calculate the excess returns (risk premium)
excess_returns = annualized_returns - rf_rate

# ---------- Covariance matrix of returns ----------

# Covariance matrix of returns
cov_matrix = returns.cov() * 12  # Annualize the covariance matrix

# ---------- Portfolio Performance ----------

# Function to calculate portfolio performance (return and volatility)
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# ---------- Sharpe Ratio ----------

# Function to calculate the Sharpe ratio
def sharpe_ratio(weights, mean_returns, cov_matrix, rf_rate):
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(portfolio_return - rf_rate) / portfolio_volatility  # Negative for minimization

# ---------- Minimum Variance Portfolio ----------

# Function to find the Minimum Variance Portfolio
def minimum_variance_portfolio(cov_matrix):
    num_assets = len(cov_matrix)
    args = (cov_matrix,)
    weights = np.ones(num_assets) / num_assets  # Initial guess: equal weights

    def variance(weights, cov_matrix):
        return portfolio_performance(weights, np.zeros(len(weights)), cov_matrix)[1] ** 2

    result = minimize(variance, weights, args=args, method="SLSQP", bounds=[(0, 1)] * num_assets, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    return result.x

# ---------- Find the Optimal Portfolio and Minimum Variance Portfolio ----------

# Find the optimal portfolio (maximum Sharpe ratio)
num_assets = len(excess_returns)
initial_weights = np.ones(num_assets) / num_assets  # Initial guess

result = minimize(sharpe_ratio, initial_weights, args=(excess_returns, cov_matrix, rf_rate),
                  method="SLSQP", bounds=[(0, 1)] * num_assets, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

optimal_weights = result.x
optimal_return, optimal_volatility = portfolio_performance(optimal_weights, excess_returns, cov_matrix)

# Find the Minimum Variance Portfolio
mv_weights = minimum_variance_portfolio(cov_matrix)
mv_return, mv_volatility = portfolio_performance(mv_weights, excess_returns, cov_matrix)

# ---------- Performance Stats ----------

# Performance stats: Sharpe Ratio for both portfolios
optimal_sharpe_ratio = (optimal_return - rf_rate) / optimal_volatility
mv_sharpe_ratio = (mv_return - rf_rate) / mv_volatility

print("Optimal Portfolio Performance:")
print(f"Return: {optimal_return:.4f}, Volatility: {optimal_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

print("Minimum Variance Portfolio Performance:")
print(f"Return: {mv_return:.4f}, Volatility: {mv_volatility:.4f}")
print(f"Sharpe Ratio: {mv_sharpe_ratio:.4f}")

# ---------- Mean-Variance Frontier ----------

# Generate the mean-variance frontier
portfolio_returns = []
portfolio_volatilities = []
portfolio_weights = []

for i in range(10000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # Ensure the sum of weights is 1
    portfolio_return, portfolio_volatility = portfolio_performance(weights, excess_returns, cov_matrix)
    portfolio_returns.append(portfolio_return)
    portfolio_volatilities.append(portfolio_volatility)
    portfolio_weights.append(weights)

# Plot the mean-variance frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatilities, portfolio_returns, c=(np.array(portfolio_returns) - rf_rate) / np.array(portfolio_volatilities), cmap='viridis', marker='o')
plt.colorbar(label="Sharpe Ratio")
plt.title("Mean-Variance Frontier")
plt.xlabel("Portfolio Volatility (Risk)")
plt.ylabel("Portfolio Return")
plt.grid(True)
plt.show()

# ---------- Varying Correlation with Two Assets ----------

# Select two assets
asset1 = 0
asset2 = 1
cov_matrix_2assets = returns.iloc[:, [asset1, asset2]].cov() * 12

# Vary correlation and plot the mean-variance frontier
correlations = np.linspace(-1, 1, 5)
plt.figure(figsize=(10, 6))

for corr in correlations:
    cov_matrix_2assets.iloc[0, 1] = cov_matrix_2assets.iloc[1, 0] = corr * np.sqrt(cov_matrix_2assets.iloc[0, 0] * cov_matrix_2assets.iloc[1, 1])
    portfolio_returns = []
    portfolio_volatilities = []
    
    for i in range(10000):
        w1 = np.random.random()
        w2 = 1 - w1
        weights = np.array([w1, w2])
        portfolio_return, portfolio_volatility = portfolio_performance(weights, excess_returns.iloc[[asset1, asset2]], cov_matrix_2assets)
        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_volatility)
    
    plt.scatter(portfolio_volatilities, portfolio_returns, label=f"Corr={corr:.2f}")

plt.title("Mean-Variance Frontier with Two Assets (Varying Correlation)")
plt.xlabel("Portfolio Volatility")
plt.ylabel("Portfolio Return")
plt.legend()
plt.grid(True)
plt.show()
