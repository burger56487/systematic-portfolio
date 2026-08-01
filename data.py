import os
import yfinance as yf
import pandas as pd

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","V","JNJ","WMT",
    "PG","MA","HD","BAC","XOM","CVX","KO","PEP","ABBV","MRK",
    "COST","MCD","CSCO","ADBE","CRM","NKE","INTC","T","VZ","DIS",
]
BENCHMARK = "SPY"

def get_data(start="2019-01-01", end="2024-01-01"):
    os.makedirs("data", exist_ok=True)
    cache = "data/prices.csv"
    bench_cache = "data/benchmark.csv"
    if os.path.exists(cache) and os.path.exists(bench_cache):
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        bench = pd.read_csv(bench_cache, index_col=0, parse_dates=True)["SPY"]
    else:
        prices = yf.download(TICKERS, start=start, end=end)["Close"].dropna(axis=1, how="any")
        bench = yf.download(BENCHMARK, start=start, end=end)["Close"]
        # bench 可能是 DataFrame,转成 Series
        if isinstance(bench, pd.DataFrame):
            bench = bench.iloc[:, 0]
        bench.name = "SPY"
        prices.to_csv(cache)
        bench.to_frame("SPY").to_csv(bench_cache)
    return prices, bench
