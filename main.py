# main.py — in-sample selection, out-of-sample validation, cost sensitivity
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategy import (get_data, composite_signal, equal_weight,
                      inverse_volatility, min_variance, backtest, metrics)

pd.set_option("display.width", 120)
RF = 0.04
COST_BPS = 10          # base transaction cost assumption
SCHEMES = {"Equal Weight": equal_weight,
           "Min Variance": min_variance,
           "Inverse Vol":  inverse_volatility}

def prep(start, end):
    px = get_data(start, end)
    rets = px.pct_change().dropna()
    sig = composite_signal(px, rets).reindex(rets.index)
    return px, rets, sig

def spy_returns(start, end, index):
    spy = get_data(start, end, tickers=["SPY"])["SPY"]
    return spy.pct_change().reindex(index).dropna()

def main():
    # ================= IN-SAMPLE: 2019-2021, select scheme =================
    print("="*70); print("IN-SAMPLE 2019-2021 — scheme selection (net of 10bps)"); print("="*70)
    _, rets_is, sig_is = prep("2019-01-01", "2022-01-01")
    is_rows = {}
    for name, fn in SCHEMES.items():
        net, tno = backtest(sig_is, rets_is, fn, cost_bps=COST_BPS)
        is_rows[name] = metrics(net, RF, tno)
    is_df = pd.DataFrame(is_rows).T
    print(is_df.round(3).to_string())
    best = is_df["Sharpe"].idxmax()
    print(f"\n>> Selected (max in-sample Sharpe): {best}")

    # ================= OUT-OF-SAMPLE: 2022-2023, frozen =================
    print("\n"+"="*70); print(f"OUT-OF-SAMPLE 2022-2023 — frozen scheme: {best}"); print("="*70)
    _, rets_oos, sig_oos = prep("2022-01-01", "2024-01-01")
    net_oos, tno_oos = backtest(sig_oos, rets_oos, SCHEMES[best], cost_bps=COST_BPS)
    m_oos = metrics(net_oos, RF, tno_oos)
    print(pd.Series(m_oos).round(3).to_string())

    spy_oos = spy_returns("2022-01-01", "2024-01-01", net_oos.index)
    m_spy = metrics(spy_oos, RF)
    print("\nSPY (out-of-sample):")
    print(pd.Series(m_spy).round(3).to_string())

    # OOS equity curve chart
    os.makedirs("charts", exist_ok=True)
    pc = (1+net_oos).cumprod(); pc = pc/pc.iloc[0]*100
    bc = (1+spy_oos.reindex(net_oos.index).fillna(0)).cumprod(); bc = bc/bc.iloc[0]*100
    plt.figure(figsize=(9,5))
    plt.plot(pc.index, pc.values, label=f"{best} (net 10bps)")
    plt.plot(bc.index, bc.values, "--", label="SPY")
    plt.title("Out-of-Sample: Strategy vs SPY (2022-2023)")
    plt.ylabel("Growth of 100"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("charts/oos_vs_spy.png", dpi=150); plt.close()

    # ================= COST SENSITIVITY: full period =================
    print("\n"+"="*70); print("COST SENSITIVITY 2019-2023 — net Sharpe by cost"); print("="*70)
    _, rets_all, sig_all = prep("2019-01-01", "2024-01-01")
    cost_rows = {}
    for name, fn in SCHEMES.items():
        row = {}
        for bps in [0, 5, 10, 20]:
            net, _ = backtest(sig_all, rets_all, fn, cost_bps=bps)
            row[f"{bps}bps"] = metrics(net, RF)["Sharpe"]
        cost_rows[name] = row
    cost_df = pd.DataFrame(cost_rows).T
    print(cost_df.round(3).to_string())

    plt.figure(figsize=(9,5))
    for name in cost_df.index:
        plt.plot([0,5,10,20], cost_df.loc[name].values, marker="o", label=name)
    plt.title("Sharpe vs Transaction Cost"); plt.xlabel("Cost (bps)"); plt.ylabel("Net Sharpe")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("charts/cost_sensitivity.png", dpi=150); plt.close()

    print("\nSaved charts to charts/. Done.")

if __name__ == "__main__":
    main()
