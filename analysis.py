# analysis.py
import numpy as np
import pandas as pd

def rolling_sharpe(port_ret, window=126):
    """Rolling annualised Sharpe over a trailing window (126 days ≈ 6 months)."""
    roll_mean = port_ret.rolling(window).mean() * 252
    roll_vol = port_ret.rolling(window).std() * np.sqrt(252)
    return (roll_mean / roll_vol).dropna()

def attribution(held, returns, top_n=10):
    """Total return contribution of each stock over the whole period."""
    contrib = (held.shift(1) * returns).sum(axis=0)   # sum over time, per stock
    contrib = contrib.sort_values(ascending=False)
    top = contrib.head(top_n)
    bottom = contrib.tail(top_n)
    return contrib, top, bottom
