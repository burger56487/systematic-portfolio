# optimize.py
import numpy as np
import pandas as pd

def equal_weight(selected, cov=None):
    """Equal weight across selected assets."""
    n = len(selected)
    return pd.Series(1.0 / n, index=selected)

def min_variance(selected, cov):
    """Minimum-variance weights (long-only, normalized)."""
    sub = cov.loc[selected, selected].values
    try:
        inv = np.linalg.pinv(sub)
        ones = np.ones(len(selected))
        w = inv @ ones
        w = np.clip(w, 0, None)          # long-only
        if w.sum() == 0:
            w = ones
        w = w / w.sum()
    except Exception:
        w = np.ones(len(selected)) / len(selected)
    return pd.Series(w, index=selected)

def risk_parity(selected, cov):
    """Simple risk-parity: weight inversely proportional to volatility."""
    vol = np.sqrt(np.diag(cov.loc[selected, selected].values))
    inv_vol = 1.0 / np.where(vol == 0, np.nan, vol)
    inv_vol = np.nan_to_num(inv_vol)
    if inv_vol.sum() == 0:
        inv_vol = np.ones(len(selected))
    w = inv_vol / inv_vol.sum()
    return pd.Series(w, index=selected)
