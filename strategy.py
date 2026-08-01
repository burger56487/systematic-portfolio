# strategy.py — data, signals, allocation schemes, backtest (with costs), metrics
import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# NOTE: This is a fixed list of current large-cap survivors — it carries
# survivorship bias (see README). A true point-in-time universe requires
# paid historical constituent data.
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","WMT",
    "PG","MA","HD","BAC","XOM","CVX","KO","PEP","ABBV","MRK",
    "COST","MCD","CSCO","ADBE","CRM","NKE","INTC","T","VZ","DIS",
]

def get_data(start, end, tickers=None):
    """Cache keyed by (start, end, universe) so results are reproducible."""
    tickers = tickers or TICKERS
    os.makedirs("data", exist_ok=True)
    key = f"data/px_{start}_{end}_{len(tickers)}.csv"
    if os.path.exists(key):
        return pd.read_csv(key, index_col=0, parse_dates=True)
    px = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(axis=1, how="any")
    px.to_csv(key)
    return px

# ---------------- Signals ----------------
def momentum(prices, lookback=60, skip=5):
    return prices.shift(skip) / prices.shift(lookback) - 1

def low_volatility(returns, window=60):
    return -returns.rolling(window).std()

def zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

def composite_signal(prices, returns, mom_lb=60, mom_skip=5, vol_w=60):
    return (zscore(momentum(prices, mom_lb, mom_skip)) +
            zscore(low_volatility(returns, vol_w))) / 2

# ---------------- Allocation schemes ----------------
def equal_weight(selected, cov=None):
    return pd.Series(1.0 / len(selected), index=selected)

def inverse_volatility(selected, cov):
    """Inverse-volatility weighting (NOT true risk parity)."""
    vol = np.sqrt(np.diag(cov.loc[selected, selected].values))
    inv = 1.0 / np.where(vol == 0, np.nan, vol)
    inv = np.nan_to_num(inv)
    if inv.sum() == 0:
        inv = np.ones(len(selected))
    return pd.Series(inv / inv.sum(), index=selected)

def min_variance(selected, cov):
    """Long-only minimum-variance via constrained optimisation (w>=0, sum=1)."""
    S = cov.loc[selected, selected].values
    n = len(selected)
    def pvar(w): return float(w @ S @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n
    res = minimize(pvar, w0, method="SLSQP", bounds=bnds, constraints=cons)
    w = res.x if res.success else w0
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    return pd.Series(w, index=selected)

# ---------------- Backtest (with costs + turnover) ----------------
def backtest(signal, returns, scheme_fn, top_pct=0.3, rebalance_days=5,
             cov_window=60, cost_bps=0.0):
    held = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    last_w = None
    for i, date in enumerate(returns.index):
        if i % rebalance_days == 0:
            s = signal.loc[date].dropna() if date in signal.index else pd.Series(dtype=float)
            if len(s) >= 10:
                k = max(int(len(s) * top_pct), 1)
                picks = s.nlargest(k).index
                hist = returns.iloc[max(0, i - cov_window):i][picks]
                cov = hist.cov()
                w = scheme_fn(picks, cov)
                full = pd.Series(0.0, index=returns.columns)
                full[picks] = w
                last_w = full
        if last_w is not None:
            held.loc[date] = last_w
    gross = (held.shift(1) * returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    net = gross - turnover * (cost_bps / 1e4)
    return net.dropna(), turnover

# ---------------- Metrics ----------------
def metrics(ret, rf=0.0, turnover=None):
    ret = ret.dropna()
    if len(ret) == 0:
        return {}
    cum = (1 + ret).cumprod()
    years = len(ret) / 252
    cagr = cum.iloc[-1] ** (1 / years) - 1 if years > 0 and cum.iloc[-1] > 0 else np.nan
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252 - rf) / vol if vol > 0 else 0.0
    dn = ret[ret < rf / 252] - rf / 252
    dd_dev = np.sqrt((dn ** 2).mean()) * np.sqrt(252) if len(dn) else 0.0
    sortino = (ret.mean() * 252 - rf) / dd_dev if dd_dev > 0 else 0.0
    mdd = (cum / cum.cummax() - 1).min()
    calmar = cagr / abs(mdd) if (mdd < 0 and not np.isnan(cagr)) else np.nan
    out = {"CAGR %": cagr * 100, "Vol %": vol * 100, "Sharpe": sharpe,
           "Sortino": sortino, "MaxDD %": mdd * 100, "Calmar": calmar}
    if turnover is not None:
        out["Avg Turnover"] = float(turnover.mean())
    return out
