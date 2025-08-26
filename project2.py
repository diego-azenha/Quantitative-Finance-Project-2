# main.py

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from plots import plot_mean_variance_frontier, plot_two_asset_frontier

# ---- Silence ONLY the legacy 'M' alias warning (defensive) ----
warnings.filterwarnings(
    "ignore",
    message=".*'M' is deprecated and will be removed.*",
    category=FutureWarning,
)

# ---------- Load stock prices ----------
base = Path(__file__).resolve().parent
data_dir = base / "clean_data"
prices = pd.read_parquet(data_dir / "clean_stock_prices.parquet").sort_index()
prices.index = pd.to_datetime(prices.index)

# ---------- Monthly returns (use MonthEnd to avoid deprecation) ----------
m_prices = prices.resample("ME").last()
m_rets   = m_prices.pct_change().dropna(how="any")

# ---------- Annualized inputs ----------
rf_rate = 0.0492
mu_annual    = (m_rets.mean() * 12).values
Sigma_annual = (m_rets.cov()  * 12).values
tickers = m_rets.columns.to_list()
n = len(tickers)

# ---------- Core portfolio functions ----------
def port_perf(w, mu, Sigma, rf):
    r = float(w @ mu)
    v = float(np.sqrt(w @ Sigma @ w))
    sr = (r - rf) / v if v > 0 else np.nan
    return r, v, sr

def neg_sharpe(w, mu, Sigma, rf):
    r, v, _ = port_perf(w, mu, Sigma, rf)
    return - (r - rf) / v

def min_var_weights(Sigma):
    n_ = Sigma.shape[0]
    x0 = np.full(n_, 1/n_)
    bnds = [(0.0, 1.0)] * n_
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    obj = lambda w: w @ Sigma @ w
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x

# ---------- Optimal (Sharpe-max) and Minimum-Variance portfolios ----------
x0 = np.full(n, 1/n)
bnds = [(0.0, 1.0)] * n
cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]

res_opt = minimize(neg_sharpe, x0,
                   args=(mu_annual, Sigma_annual, rf_rate),
                   method="SLSQP", bounds=bnds, constraints=cons)
if not res_opt.success:
    raise RuntimeError(res_opt.message)
w_opt = res_opt.x

w_mv  = min_var_weights(Sigma_annual)

opt_ret, opt_vol, opt_sr = port_perf(w_opt, mu_annual, Sigma_annual, rf_rate)
mv_ret,  mv_vol,  mv_sr  = port_perf(w_mv,  mu_annual, Sigma_annual, rf_rate)

# ---------- Realized perf with monthly rebalancing to FIXED weights ----------
def backtest_fixed_weights(monthly_returns: pd.DataFrame, w: np.ndarray):
    pr = monthly_returns.dot(w).dropna()
    ann_ret = (1 + pr.mean())**12 - 1
    ann_vol = pr.std() * np.sqrt(12)
    sharpe  = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + pr).cumprod()
    return cum, ann_ret, ann_vol, sharpe

cum_opt, ann_opt, vol_opt, sr_opt = backtest_fixed_weights(m_rets, w_opt)
cum_mv,  ann_mv,  vol_mv,  sr_mv  = backtest_fixed_weights(m_rets, w_mv)

# ---------- Helper: sample long-only weights on simplex ----------
def sample_cloud(mu, Sigma, N=100_000, alpha=1.0, rng=None):
    """
    Draw N long-only fully-invested portfolios from Dirichlet(alpha).
    alpha=1   => uniform interior cloud
    alpha<1   => sparse, corner-hugging cloud (touches frontier)
    """
    rng = np.random.default_rng(None if rng is None else rng)
    W = rng.dirichlet(alpha=np.full(n, alpha), size=N)     # (N, n)
    R = W @ mu
    V = np.sqrt(np.einsum('ij,jk,ik->i', W, Sigma, W))
    return V.tolist(), R.tolist()

# ---------- Clouds: interior + edge (sparse) ----------
V_in, R_in = sample_cloud(mu_annual, Sigma_annual, N=150_000, alpha=1.0)
V_edge, R_edge = sample_cloud(mu_annual, Sigma_annual, N=150_000, alpha=0.15)

# ---------- True efficient frontier (same constraints) ----------
def solve_min_var_for_target(mu, Sigma, target_ret):
    n_ = len(mu)
    x0 = np.full(n_, 1/n_)
    bnds = [(0.0, 1.0)] * n_
    cons = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},
        {'type': 'eq', 'fun': lambda w, t=target_ret: float(w @ mu) - t},
    ]
    obj = lambda w: w @ Sigma @ w
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons)
    return res

mu_min, mu_max = float(np.min(mu_annual)), float(np.max(mu_annual))
targets = np.linspace(mu_min*1.02, mu_max*0.98, 100)
ef_vols, ef_rets = [], []
for t in targets:
    res = solve_min_var_for_target(mu_annual, Sigma_annual, t)
    if res.success:
        v = np.sqrt(res.x @ Sigma_annual @ res.x)
        ef_vols.append(float(v))
        ef_rets.append(float(t))

# ---------- Plotting (now with two clouds) ----------
plot_mean_variance_frontier(
    cloud_interior=(V_in, R_in),
    cloud_edge=(V_edge, R_edge),
    rf_rate=rf_rate,
    opt_point=(opt_vol, opt_ret),
    mv_point=(mv_vol, mv_ret),
    ef_curve=(ef_vols, ef_rets),
    filename="mean_variance_frontier.png",
)

# ---------- Two-asset frontier (unchanged) ----------
asset1, asset2 = 0, 1
Sigma2_base = (m_rets.iloc[:, [asset1, asset2]].cov() * 12)
mu2 = (m_rets.mean() * 12).iloc[[asset1, asset2]].values
correlations = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

vols_list, rets_list = [], []
s1, s2 = Sigma2_base.iloc[0,0], Sigma2_base.iloc[1,1]
for corr in correlations:
    Sigma2 = Sigma2_base.copy()
    Sigma2.iloc[0,1] = Sigma2.iloc[1,0] = corr * np.sqrt(s1 * s2)
    v_list, r_list = [], []
    for _ in range(10_000):
        w1 = np.random.random(); w = np.array([w1, 1 - w1])
        r, v, _ = port_perf(w, mu2, Sigma2.values, rf_rate)
        r_list.append(r); v_list.append(v)
    vols_list.append(v_list); rets_list.append(r_list)

plot_two_asset_frontier(vols_list, rets_list, correlations)

# ---------- Pretty console output ----------
def print_results(title, ann_ret, ann_vol, sharpe):
    print(f"{title:<45}")
    print(f"  Annual Return    : {ann_ret:>7.2%}")
    print(f"  Annual Volatility: {ann_vol:>7.2%}")
    print(f"  Sharpe Ratio     : {sharpe:>7.2f}")
    print("-" * 55)

print("\n" + "=" * 55)
print(" STATIC OPTIMIZATION (10Y ESTIMATES, FIXED WEIGHTS) ")
print("=" * 55)
print_results("Optimal Portfolio (Sharpe-max)", opt_ret, opt_vol, opt_sr)
print_results("Minimum Variance Portfolio",     mv_ret,  mv_vol,  mv_sr)

print("\n" + "=" * 55)
print(" BACKTEST RESULTS (MONTHLY REBALANCE TO FIXED WEIGHTS) ")
print("=" * 55)
print_results("Backtested Optimal Portfolio",    ann_opt, vol_opt, sr_opt)
print_results("Backtested Minimum Variance",     ann_mv,  vol_mv,  sr_mv)
