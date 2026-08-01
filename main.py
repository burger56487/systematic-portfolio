# main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import get_data
from signals import composite_signal
from portfolio import backtest_with_scheme
from optimize import equal_weight, min_variance, risk_parity
from metrics import performance
from analysis import rolling_sharpe, attribution


def main():
    # ---- 1. Load data ----
    prices, bench = get_data()
    returns = prices.pct_change().dropna()
    bench_ret = bench.pct_change().reindex(returns.index).dropna()
    print(f"Loaded {prices.shape[1]} stocks + SPY, {prices.shape[0]} days\n")

    # ---- 2. Composite signal ----
    signal = composite_signal(prices, returns).reindex(returns.index)

    # ---- 3. Compare allocation schemes (Phase 2) ----
    schemes = {
        "Equal Weight": equal_weight,
        "Min Variance": min_variance,
        "Risk Parity":  risk_parity,
    }
    os.makedirs("results", exist_ok=True)
    rows = {}
    scheme_returns = {}
    scheme_held = {}

    plt.figure(figsize=(10, 6))
    for name, fn in schemes.items():
        pr, held = backtest_with_scheme(signal, returns, fn, top_pct=0.3, rebalance_days=5)
        scheme_returns[name] = pr
        scheme_held[name] = held
        rows[name] = {k: round(v, 3) for k, v in performance(pr, bench_ret).items()}
        cum = (1 + pr).cumprod()
        plt.plot(cum.index, cum.values, label=name)

    cum_b = (1 + bench_ret).cumprod()
    plt.plot(cum_b.index, cum_b.values, label="SPY", linestyle="--", color="black")
    rows["SPY"] = {k: round(v, 3) for k, v in performance(bench_ret).items()}
    plt.title("Portfolio Allocation Schemes vs SPY")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return (NAV)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("results/allocation_schemes.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("=== Allocation Schemes vs SPY ===")
    print(pd.DataFrame(rows).to_string())

    # ---- 4. Phase 3a: Rolling Sharpe (use Equal Weight, the best scheme) ----
    best = "Equal Weight"
    rs = rolling_sharpe(scheme_returns[best], window=126)
    rs_bench = rolling_sharpe(bench_ret, window=126)

    plt.figure(figsize=(10, 6))
    plt.plot(rs.index, rs.values, label=f"{best} Portfolio")
    plt.plot(rs_bench.index, rs_bench.values, label="SPY", linestyle="--", color="black")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("Rolling 6-Month Sharpe Ratio")
    plt.xlabel("Date"); plt.ylabel("Rolling Sharpe (annualised)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("results/rolling_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved results/rolling_sharpe.png")
    print(f"Rolling Sharpe range for {best}: {rs.min():.2f} to {rs.max():.2f}")

    # ---- 5. Phase 3b: Performance attribution (Equal Weight) ----
    contrib, top, bottom = attribution(scheme_held[best], returns, top_n=10)
    print(f"\n=== Top contributors ({best}) ===")
    print(top.round(4).to_string())
    print(f"\n=== Bottom contributors ({best}) ===")
    print(bottom.round(4).to_string())

    plt.figure(figsize=(10, 6))
    combined = pd.concat([top, bottom])
    colors = ["#5cb85c" if v >= 0 else "#d9534f" for v in combined.values]
    plt.bar(combined.index, combined.values, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Top & Bottom Return Contributors ({best})")
    plt.xlabel("Stock"); plt.ylabel("Total Return Contribution")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/attribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved results/attribution.png")


if __name__ == "__main__":
    main()
