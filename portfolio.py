# portfolio.py
import pandas as pd

def build_long_only(signal, top_pct=0.3):
    """Each day, long the top_pct stocks by signal, equal-weighted."""
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date in signal.index:
        s = signal.loc[date].dropna()
        if len(s) < 10:
            continue
        n = max(int(len(s) * top_pct), 1)
        picks = s.nlargest(n).index
        weights.loc[date, picks] = 1.0 / n
    return weights

def backtest_portfolio(weights, returns, rebalance_days=5):
    """Rebalance every `rebalance_days` trading days; hold in between."""
    w = weights.reindex(returns.index).fillna(0.0)
    held = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    last_w = None
    for i, date in enumerate(returns.index):
        if i % rebalance_days == 0:
            row = w.loc[date]
            if row.abs().sum() > 0:      # only update if there are valid picks
                last_w = row
        if last_w is not None:
            held.loc[date] = last_w
    port_ret = (held.shift(1) * returns).sum(axis=1)
    return port_ret

def backtest_with_scheme(signal, returns, scheme_fn, top_pct=0.3,
                         rebalance_days=5, cov_window=60):
    """Select top stocks by signal, weight them with scheme_fn, rebalance periodically."""
    held = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    last_w = None
    dates = returns.index
    for i, date in enumerate(dates):
        if i % rebalance_days == 0:
            s = signal.loc[date].dropna() if date in signal.index else pd.Series(dtype=float)
            if len(s) >= 10:
                n = max(int(len(s) * top_pct), 1)
                picks = s.nlargest(n).index
                hist = returns.iloc[max(0, i - cov_window):i][picks]
                cov = hist.cov()
                w = scheme_fn(picks, cov)
                full = pd.Series(0.0, index=returns.columns)
                full[picks] = w
                last_w = full
        if last_w is not None:
            held.loc[date] = last_w
    port_ret = (held.shift(1) * returns).sum(axis=1)
    return port_ret, held          # <-- 改这里:多返回 held

