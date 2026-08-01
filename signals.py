import pandas as pd

def momentum(prices, lookback=120, skip=5):
    return prices.shift(skip) / prices.shift(lookback) - 1

def low_volatility(returns, window=60):
    return -returns.rolling(window).std()

def zscore(df):
    """按横截面(每天所有股票)标准化,方便合成不同量纲的因子"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

def composite_signal(prices, returns):
    """把动量和低波动标准化后相加,得到综合打分"""
    mom = zscore(momentum(prices))
    lowvol = zscore(low_volatility(returns))
    score = (mom + lowvol) / 2
    return score
