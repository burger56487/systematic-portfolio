import numpy as np
import pandas as pd

def performance(port_ret, bench_ret=None):
    port_ret = port_ret.dropna()
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + port_ret).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    out = {
        "Annual Return": ann_ret,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }
    if bench_ret is not None:
        b = bench_ret.reindex(port_ret.index).dropna()
        p = port_ret.reindex(b.index)
        # beta / alpha vs 基准
        cov = np.cov(p, b)[0, 1]
        beta = cov / np.var(b)
        alpha = (p.mean() - beta * b.mean()) * 252
        out["Beta vs SPY"] = beta
        out["Alpha vs SPY"] = alpha
    return out
