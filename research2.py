"""
RP2 — cross-sectional momentum (X1/X2) and funding overlay (F1) drivers.

Pre-registered 2026-07-18 in the vault note "Research Program 2 — Non-Price-Timing
Signals" BEFORE the first run. Candidates and criteria are fixed there; this file
only implements them. Protocol shared with walkforward.py: daily bars, decisions
at close, fills next open ±slip, taker fee on turnover, OOS folds 2023/24/25/26*.

  X1  weekly top-2 of 8 by 28d return, eligible if return>0 and close>SMA200
  X2  same with 90d lookback — robustness read only, cannot pass for X1
  F1  H3 sleeves + step-aside when 7d-mean funding > 90th trailing percentile
      (needs data/funding/<INST>.csv from fetch_funding.py; skipped if absent)

Usage:  python3 research2.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from backtest import load_candles
from walkforward import DATA_END, FOLDS, INSTS, OOS_START, _dates, sim_bh, sim_h3
from xsmom import select_targets

FUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "funding")


def load_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for i in INSTS:
        df = load_candles(i, "1D")
        idx = _dates(df)
        opens[i] = pd.Series(df["open"].to_numpy(), index=idx)
        closes[i] = pd.Series(df["close"].to_numpy(), index=idx)
    return pd.DataFrame(opens).sort_index(), pd.DataFrame(closes).sort_index()


# ── Simulators ────────────────────────────────────────────────────────────────────

def sim_xs(O: pd.DataFrame, C: pd.DataFrame, lookback: int, topk: int = 2,
           fee_bp: float = 10.0, slip_bp: float = 5.0) -> pd.Series:
    """Weekly-rebalanced cross-sectional momentum portfolio, equity starts 1.0."""
    fee, slip = fee_bp / 1e4, slip_bp / 1e4
    sma200 = C.rolling(200).mean()
    mom = C / C.shift(lookback) - 1.0
    dates = C.index
    cash, units = 1.0, dict.fromkeys(INSTS, 0.0)
    target: dict | None = None
    anchor = dates[0].toordinal()
    eq = np.empty(len(dates))

    for t in range(len(dates)):
        if target is not None:                        # execute at today's open
            po = O.iloc[t]
            V = cash + sum(units[i] * po[i] for i in INSTS
                           if units[i] > 0 and np.isfinite(po[i]))
            for i in INSTS:
                price = po[i]
                if not np.isfinite(price):
                    continue
                delta = target.get(i, 0.0) * V - units[i] * price
                if abs(delta) < V * 1e-6:
                    continue
                u = abs(delta) / price
                if delta > 0:
                    cash -= u * price * (1 + slip) * (1 + fee)
                    units[i] += u
                else:
                    cash += u * price * (1 - slip) * (1 - fee)
                    units[i] -= u
            target = None

        pc = C.iloc[t]
        eq[t] = cash + sum(units[i] * pc[i] for i in INSTS
                           if units[i] > 0 and np.isfinite(pc[i]))

        if (dates[t].toordinal() - anchor) % 7 == 0:  # decision at this close
            m, s2 = mom.iloc[t], sma200.iloc[t]
            target = select_targets({i: m[i] for i in INSTS},
                                    {i: pc[i] for i in INSTS},
                                    {i: s2[i] for i in INSTS}, topk)
    return pd.Series(eq, index=dates)


def load_funding(inst: str) -> pd.Series | None:
    """Daily-resampled funding rate for one instrument (sum of the day's rates)."""
    path = os.path.join(FUND_DIR, f"{inst}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return pd.Series(df["rate"].to_numpy(), index=ts).resample("1D").sum(min_count=1)


def sim_f1(df: pd.DataFrame, funding_daily: pd.Series, fee_bp: float = 10.0,
           slip_bp: float = 5.0, pctile: float = 0.90) -> pd.Series:
    """H3 sleeve + funding-crowdedness step-aside. Equity starts 1.0.

    Crowded(t) = 7d mean funding > its `pctile` over the trailing 365d
    (min 180d of history; pass-through before that). Decisions at close,
    fills next open.
    """
    fee, slip = fee_bp / 1e4, slip_bp / 1e4
    o, cl = df["open"].to_numpy(), df["close"].to_numpy()
    s200 = df["close"].rolling(200).mean().to_numpy()
    dates = _dates(df)

    f7 = funding_daily.rolling(7, min_periods=4).mean()
    thresh = f7.rolling(365, min_periods=180).quantile(pctile)
    crowded_s = (f7 > thresh) & thresh.notna()
    # Map to this asset's bar dates: funding day D (UTC) is known by the 16:00 UTC
    # daily close of D, since Binance pays 00/08/16 UTC.
    crowded = crowded_s.reindex(dates.normalize(), fill_value=False).to_numpy()

    n = len(df)
    cash, units, pending = 1.0, 0.0, None
    eq = np.empty(n)
    for i in range(n):
        if pending == "buy":
            fill = o[i] * (1 + slip)
            units = cash / (fill * (1 + fee))
            cash, pending = 0.0, None
        elif pending == "sell":
            cash = o[i] * (1 - slip) * units * (1 - fee)
            units, pending = 0.0, None
        want = bool(np.isfinite(s200[i]) and cl[i] > s200[i] and not crowded[i])
        if want and units == 0.0:
            pending = "buy"
        elif not want and units > 0.0:
            pending = "sell"
        eq[i] = cash + units * cl[i]
    return pd.Series(eq, index=dates)


# ── Metrics & report ──────────────────────────────────────────────────────────────

def wmetrics(curve: pd.Series, start: str, end: str) -> dict:
    r = curve.loc[start:end].pct_change().dropna()
    if len(r) < 30:
        return {}
    c = (1 + r).cumprod()
    sd = r.std()
    return {"ret": float(c.iloc[-1] - 1), "rets": r,
            "sharpe": float(r.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
            "maxDD": float(((c - c.cummax()) / c.cummax()).min())}


def sleeves_curve(simfn, extra: dict | None = None) -> pd.Series:
    """Equal-weight portfolio curve from per-asset sleeve simulators."""
    rets = []
    for i in INSTS:
        df = load_candles(i, "1D")
        eqs = simfn(df, 10.0, 5.0) if extra is None else simfn(df, extra[i])
        eqs = eqs[1] if isinstance(eqs, tuple) else eqs
        rets.append(eqs.pct_change().dropna())
    port = pd.concat(rets, axis=1).mean(axis=1, skipna=True).dropna()
    return (1 + port).cumprod()


def report(name: str, curve: pd.Series, h3: pd.Series, bh: pd.Series,
           is_candidate: bool = True) -> None:
    p, p3, pb = (wmetrics(c, OOS_START, DATA_END) for c in (curve, h3, bh))
    act = (p["rets"] - p3["rets"]).dropna()
    t = float(act.mean() / act.std() * np.sqrt(len(act))) if act.std() > 0 else 0.0
    folds_won = 0
    cells = []
    for fname, s, e in FOLDS:
        m, m3 = wmetrics(curve, s, e), wmetrics(h3, s, e)
        won = bool(m and m3 and m["ret"] > m3["ret"])
        folds_won += won
        cells.append(f"{m.get('ret', 0):+8.1%}{'*' if won else ' '}")
    print(f"\n{name}: pooled OOS ret={p['ret']:+.1%}  sharpe={p['sharpe']:.2f}  "
          f"maxDD={p['maxDD']:.1%}  active-t vs H3={t:+.1f}")
    print(f"  folds (* = beat H3): {'  '.join(cells)}")
    if not is_candidate:
        return
    checks = [
        ("Sharpe > H3 and ret > H3", p["sharpe"] > p3["sharpe"] and p["ret"] > p3["ret"],
         f"{p['sharpe']:.2f}/{p3['sharpe']:.2f}, {p['ret']:+.1%}/{p3['ret']:+.1%}"),
        ("active-return t ≥ 2", t >= 2, f"t={t:+.1f}"),
        ("beats H3 in ≥ 2/4 folds", folds_won >= 2, f"{folds_won}/4"),
        ("maxDD < B&H", p["maxDD"] > pb["maxDD"], f"{p['maxDD']:.1%} vs {pb['maxDD']:.1%}"),
    ]
    n_pass = sum(ok for _, ok, _ in checks)
    print(f"  criteria: {n_pass}/{len(checks)} passed")
    for label, ok, detail in checks:
        print(f"    {'✓' if ok else '✗'} {label:26} {detail}")


def funding_quintile_diag() -> None:
    """Descriptive: forward 7d spot return by funding-7d-mean quintile, pooled."""
    rows = []
    for i in INSTS:
        fd = load_funding(i)
        if fd is None:
            continue
        df = load_candles(i, "1D")
        cl = pd.Series(df["close"].to_numpy(), index=_dates(df).normalize())
        f7 = fd.rolling(7, min_periods=4).mean().reindex(cl.index)
        fwd = cl.shift(-7) / cl - 1
        q = f7.rank(pct=True)
        rows.append(pd.DataFrame({"q": q, "fwd": fwd}).dropna())
    if not rows:
        return
    d = pd.concat(rows)
    d["bucket"] = pd.cut(d["q"], [0, .2, .4, .6, .8, 1.0], labels=list("12345"))
    g = d.groupby("bucket", observed=True)["fwd"].agg(["count", "mean", "median"])
    print("\nDiagnostic — fwd 7d spot return by funding-percentile quintile (pooled):")
    for b, r in g.iterrows():
        print(f"    Q{b}  n={r['count']:<6.0f} mean={r['mean']:+7.2%}  median={r['median']:+7.2%}")


def main() -> None:
    O, C = load_matrix()
    h3 = sleeves_curve(sim_h3)
    bh = sleeves_curve(sim_bh)
    p3, pb = wmetrics(h3, OOS_START, DATA_END), wmetrics(bh, OOS_START, DATA_END)
    print(f"benchmarks pooled OOS:  H3 ret={p3['ret']:+.1%} sharpe={p3['sharpe']:.2f} "
          f"maxDD={p3['maxDD']:.1%}   B&H ret={pb['ret']:+.1%} sharpe={pb['sharpe']:.2f} "
          f"maxDD={pb['maxDD']:.1%}")

    report("X1 (28d, primary)", sim_xs(O, C, 28), h3, bh)
    report("X1 maker-sensitivity", sim_xs(O, C, 28, fee_bp=8.0, slip_bp=0.0), h3, bh,
           is_candidate=False)
    report("X2 (90d, robustness — cannot pass for X1)", sim_xs(O, C, 90), h3, bh)

    funding = {i: load_funding(i) for i in INSTS}
    have = [i for i, f in funding.items() if f is not None]
    if have:
        rets = []
        for i in have:
            eqs = sim_f1(load_candles(i, "1D"), funding[i])
            rets.append(eqs.pct_change().dropna())
        port = pd.concat(rets, axis=1).mean(axis=1, skipna=True).dropna()
        f1 = (1 + port).cumprod()
        # Fair H3 comparator on the same asset subset
        h3s = []
        for i in have:
            eqs = sim_h3(load_candles(i, "1D"), 10.0, 5.0)[1]
            h3s.append(eqs.pct_change().dropna())
        h3_sub = (1 + pd.concat(h3s, axis=1).mean(axis=1, skipna=True).dropna()).cumprod()
        report(f"F1 (funding overlay, {len(have)} assets)", f1, h3_sub, bh)
        funding_quintile_diag()
    else:
        print("\nF1 skipped — no funding data in data/funding/ (run fetch_funding.py)")


if __name__ == "__main__":
    main()
