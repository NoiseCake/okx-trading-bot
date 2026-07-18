"""
Phase 1 walk-forward driver — pre-registered daily-timeframe hypothesis test.

Candidates (fixed 2026-07-06 in `To-Do — Next Session.md`, ambiguities resolved
2026-07-18 BEFORE the first run; single parameter set each, no grids):

  H1  Existing CombinedStrategy on 1D bars, as-is signals/regime/macro gates
      (no 4H-style confirmation — no timeframe above 1D, and the event-coincidence
      check is the documented Finding-17 design flaw). Exits geometry-scaled:
      1.5×ATR stop, TP ladder at 1R/2R/3R (30/40/30), trailing after TP3 at
      1×ATR(entry) distance. Stops/TPs intraday on daily high/low, stop-first.
  H2  Trend filter + breakout: long when close > SMA200 and SMA200 > SMA200[20d ago]
      and close > prior 20-day high (Donchian). Exit when close < max(SMA100,
      highest-close-since-entry − 3×ATR14). Indicator exits at close, filled next
      open; no intraday stop.
  H3  Benchmark: long when close > SMA200, flat below. Any candidate must beat
      this on OOS Sharpe AND total return, else complexity is unjustified.
  B   Buy & hold (context reference line only).

Protocol: signals at daily close, fills at next daily open ±5 bp slip, 10 bp
taker fee per side (primary). Maker sensitivity = 8 bp / 0 slip, same fills
(optimistic — assumes the limit fills). Zero-cost = gross signal diagnostic.
Each asset is an equal-weight sleeve, fully invested while in a trade. Folds =
OOS calendar years 2023 / 2024 / 2025 / 2026-partial (nothing is tuned, so the
3y in-sample windows are context only). Trades belong to the fold of their entry.

Success criteria (all required, fixed in advance): pooled OOS n ≥ 50;
OOS avgR > 0 with t ≥ 2; PF ≥ 1.3; portfolio maxDD < buy & hold's; beats H3 on
Sharpe AND total return OOS; all under taker costs.

Usage:  python3 walkforward.py [--insts BTC-USDT ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest import StratCfg, compute_signals, daily_context, gate_allows, load_candles
from strategy import atr, sma

INSTS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "DOGE-USDT",
         "LTC-USDT", "LINK-USDT", "SOL-USDT", "AVAX-USDT"]

OOS_START = "2023-01-01"
DATA_END = "2026-07-06"
FOLDS = [("2023", "2023-01-01", "2023-12-31"),
         ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"),
         ("2026*", "2026-01-01", DATA_END)]

COSTS = {"taker": (10.0, 5.0), "maker": (8.0, 0.0), "zero": (0.0, 0.0)}

ATR_MULT = 1.5          # H1 stop distance, as live
TP_TIERS = ((1.0, 0.30), (2.0, 0.40), (3.0, 0.30))   # R-multiples of stop distance
TRAIL_ATR = 1.0         # H1 trailing distance after TP3, ×ATR(entry)
DON_N = 20              # H2 Donchian lookback
SLOPE_N = 20            # H2 SMA200 rising lookback
CHAND_ATR = 3.0         # H2 chandelier multiple


def _dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(df["close_ms"], unit="ms", utc=True))


# ── Per-asset sleeve simulators (equity starts at 1.0) ───────────────────────────

def sim_h1(df: pd.DataFrame, fee_bp: float, slip_bp: float):
    sc = StratCfg()
    s = compute_signals(df, sc)
    d = daily_context(df)
    fee, slip = fee_bp / 1e4, slip_bp / 1e4
    o = df["open"].to_numpy()
    hi, lo, cl = s["high"].to_numpy(), s["low"].to_numpy(), s["close"].to_numpy()
    sig, regime = s["signal"].to_numpy(), s["regime"].to_numpy()
    rsi_a, atr_a = s["rsi"].to_numpy(), s["atr"].to_numpy()

    n = len(df)
    cash, pos, pending = 1.0, None, None
    trades, eq = [], np.empty(n)

    def close_out(p, level, i, reason):
        nonlocal cash, pos
        fill = level * (1 - slip)
        proceeds = fill * p["units"]
        cash += proceeds * (1 - fee)
        p["fees"] += proceeds * fee
        p["realized"] += (fill - p["entry"]) * p["units"]
        net = p["realized"] - p["fees"]
        trades.append({"entry_i": p["entry_i"], "exit_i": i, "reason": reason,
                       "r": net / p["risk"], "net": net, "eq_at_entry": p["eq0"]})
        pos = None

    for i in range(n):
        if pending is not None:                       # fill at today's open
            fill = o[i] * (1 + slip)
            units = cash / (fill * (1 + fee))
            dist = pending["atr"] * ATR_MULT
            pos = {"entry": fill, "units": units, "ounits": units,
                   "stop": fill - dist, "dist": dist, "atr0": pending["atr"],
                   "tps": [[fill + r * dist, f, False] for r, f in TP_TIERS],
                   "trail_on": False, "trail_hi": 0.0, "trail_stop": 0.0,
                   "risk": units * dist, "fees": units * fill * fee,
                   "realized": 0.0, "entry_i": i, "eq0": cash}
            cash = 0.0
            pending = None

        p = pos
        if p is not None:                             # manage intraday, stop first
            if lo[i] <= p["stop"]:
                close_out(p, p["stop"], i, "STOP")
            else:
                if p["trail_on"] and lo[i] <= p["trail_stop"]:
                    close_out(p, p["trail_stop"], i, "TRAIL")
                if pos is not None:
                    for k, tp in enumerate(p["tps"]):
                        if tp[2] or hi[i] < tp[0]:
                            continue
                        tp[2] = True
                        if k == len(p["tps"]) - 1:    # TP3 → trail the last tranche
                            p["trail_on"] = True
                            p["trail_hi"] = tp[0]
                            p["trail_stop"] = tp[0] - TRAIL_ATR * p["atr0"]
                        else:
                            csize = p["ounits"] * tp[1]
                            fill = tp[0] * (1 - slip)
                            cash += fill * csize * (1 - fee)
                            p["fees"] += fill * csize * fee
                            p["realized"] += (fill - p["entry"]) * csize
                            p["units"] -= csize
                    if p["trail_on"]:
                        if lo[i] <= p["trail_stop"]:
                            close_out(p, p["trail_stop"], i, "TRAIL")
                        elif hi[i] > p["trail_hi"]:
                            p["trail_hi"] = hi[i]
                            p["trail_stop"] = max(p["trail_stop"],
                                                  hi[i] - TRAIL_ATR * p["atr0"])

        if pos is None and pending is None and sig[i] == "buy":
            gate = sc.gate_ranging if regime[i] == "ranging" else sc.gate_trending
            if (np.isfinite(atr_a[i]) and atr_a[i] > 0
                    and gate_allows(gate, d.iloc[i], cl[i], rsi_a[i])):
                pending = {"atr": atr_a[i]}

        eq[i] = cash + (pos["units"] * cl[i] if pos is not None else 0.0)

    if pos is not None:
        close_out(pos, cl[n - 1], n - 1, "EOD")
        eq[n - 1] = cash
    return trades, pd.Series(eq, index=_dates(df))


def sim_h2(df: pd.DataFrame, fee_bp: float, slip_bp: float):
    fee, slip = fee_bp / 1e4, slip_bp / 1e4
    o, hi, cl = df["open"].to_numpy(), df["high"].to_numpy(), df["close"].to_numpy()
    close_s = df["close"]
    s200 = sma(close_s, 200).to_numpy()
    s100 = sma(close_s, 100).to_numpy()
    a14 = atr(df, 14).to_numpy()
    don = pd.Series(hi).rolling(DON_N).max().shift(1).to_numpy()
    s200_past = pd.Series(s200).shift(SLOPE_N).to_numpy()

    n = len(df)
    cash, pos, pending = 1.0, None, None    # pending: ("buy", risk_dist) | ("sell",)
    trades, eq = [], np.empty(n)

    for i in range(n):
        if pending is not None:
            if pending[0] == "buy":
                fill = o[i] * (1 + slip)
                units = cash / (fill * (1 + fee))
                pos = {"entry": fill, "units": units, "hh": fill,
                       "risk": units * pending[1], "fees": units * fill * fee,
                       "entry_i": i, "eq0": cash}
                cash = 0.0
            else:
                fill = o[i] * (1 - slip)
                proceeds = fill * pos["units"]
                cash = proceeds * (1 - fee)
                net = (fill - pos["entry"]) * pos["units"] - pos["fees"] - proceeds * fee
                trades.append({"entry_i": pos["entry_i"], "exit_i": i, "reason": "EXIT",
                               "r": net / pos["risk"], "net": net, "eq_at_entry": pos["eq0"]})
                pos = None
            pending = None

        if pos is not None:
            pos["hh"] = max(pos["hh"], cl[i])
            exit_lvl = max(s100[i], pos["hh"] - CHAND_ATR * a14[i])
            if np.isfinite(exit_lvl) and cl[i] < exit_lvl:
                pending = ("sell",)
        elif (np.isfinite(s200[i]) and np.isfinite(s200_past[i]) and np.isfinite(don[i])
                and np.isfinite(a14[i]) and np.isfinite(s100[i])
                and cl[i] > s200[i] and s200[i] > s200_past[i] and cl[i] > don[i]):
            risk_dist = cl[i] - max(s100[i], cl[i] - CHAND_ATR * a14[i])
            if risk_dist > 0:
                pending = ("buy", risk_dist)

        eq[i] = cash + (pos["units"] * cl[i] if pos is not None else 0.0)

    if pos is not None:
        fill = cl[n - 1]
        net = (fill - pos["entry"]) * pos["units"] - pos["fees"] - fill * pos["units"] * fee
        trades.append({"entry_i": pos["entry_i"], "exit_i": n - 1, "reason": "EOD",
                       "r": net / pos["risk"], "net": net, "eq_at_entry": pos["eq0"]})
        eq[n - 1] = cash + fill * pos["units"] * (1 - fee)
    return trades, pd.Series(eq, index=_dates(df))


def sim_h3(df: pd.DataFrame, fee_bp: float, slip_bp: float):
    fee, slip = fee_bp / 1e4, slip_bp / 1e4
    o, cl = df["open"].to_numpy(), df["close"].to_numpy()
    s200 = sma(df["close"], 200).to_numpy()
    n = len(df)
    cash, units, pending = 1.0, 0.0, None
    trades, eq = [], np.empty(n)
    entry_i = -1

    for i in range(n):
        if pending == "buy":
            fill = o[i] * (1 + slip)
            units = cash / (fill * (1 + fee))
            cash, entry_i, pending = 0.0, i, None
        elif pending == "sell":
            fill = o[i] * (1 - slip)
            cash = fill * units * (1 - fee)
            trades.append({"entry_i": entry_i, "exit_i": i, "reason": "EXIT",
                           "r": np.nan, "net": np.nan, "eq_at_entry": np.nan})
            units, pending = 0.0, None
        want = bool(np.isfinite(s200[i]) and cl[i] > s200[i])
        if want and units == 0.0 and pending is None:
            pending = "buy"
        elif not want and units > 0.0 and pending is None:
            pending = "sell"
        eq[i] = cash + units * cl[i]
    return trades, pd.Series(eq, index=_dates(df))


def sim_bh(df: pd.DataFrame, fee_bp: float, slip_bp: float):
    cl = df["close"].to_numpy()
    return [], pd.Series(cl / cl[0], index=_dates(df))


SIMS = {"H1": sim_h1, "H2": sim_h2, "H3": sim_h3, "B&H": sim_bh}


# ── Metrics ───────────────────────────────────────────────────────────────────────

def trade_stats(rs: list[float]) -> dict:
    rs = [r for r in rs if np.isfinite(r)]
    n = len(rs)
    if n == 0:
        return {"n": 0}
    a = np.array(rs)
    wins, losses = a[a > 0], a[a <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    sd = a.std(ddof=1) if n > 1 else 0.0
    return {"n": n, "win": float((a > 0).mean()), "avgR": float(a.mean()),
            "t": float(a.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
            "PF": float(pf)}


def window_metrics(curves: dict[str, pd.Series], start: str, end: str) -> dict:
    """Equal-weight portfolio of per-asset sleeve curves over [start, end]."""
    rets = []
    for eqs in curves.values():
        w = eqs.loc[start:end]
        if len(w) > 30:
            rets.append(w.pct_change().dropna())
    if not rets:
        return {}
    port = pd.concat(rets, axis=1).mean(axis=1, skipna=True).dropna()
    curve = (1 + port).cumprod()
    peak = curve.cummax()
    sd = port.std()
    return {"ret": float(curve.iloc[-1] - 1),
            "sharpe": float(port.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
            "maxDD": float(((curve - peak) / peak).min())}


def fold_of(date: pd.Timestamp) -> str | None:
    for name, s, e in FOLDS:
        if pd.Timestamp(s, tz="UTC") <= date <= pd.Timestamp(e, tz="UTC") + pd.Timedelta(days=1):
            return name
    return None


# ── Driver ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insts", nargs="+", default=INSTS)
    args = ap.parse_args()

    data = {i: load_candles(i, "1D") for i in args.insts}
    results: dict = {}     # (strat, cost) → {"trades": [...], "curves": {inst: eq}}
    for strat, fn in SIMS.items():
        for cost, (fee, slip) in COSTS.items():
            if cost != "taker" and strat in ("H3", "B&H"):
                continue
            all_tr, curves = [], {}
            for inst, df in data.items():
                trs, eqs = fn(df, fee, slip)
                dates = _dates(df)
                for t in trs:
                    t["inst"], t["date"] = inst, dates[t["entry_i"]]
                    t["hold"] = t["exit_i"] - t["entry_i"]
                all_tr.extend(trs)
                curves[inst] = eqs
            results[(strat, cost)] = {"trades": all_tr, "curves": curves}

    line = "─" * 78
    print(line)
    print("PHASE 1 — pre-registered daily hypothesis test  "
          f"({len(args.insts)} assets, data → {DATA_END})")
    print(line)

    for strat in ("H1", "H2"):
        tr = results[(strat, "taker")]["trades"]
        oos = [t for t in tr if fold_of(t["date"])]
        is_ = [t for t in tr if not fold_of(t["date"])]
        print(f"\n{strat} — taker costs (primary)")
        st = trade_stats([t["r"] for t in oos])
        si = trade_stats([t["r"] for t in is_])
        if si.get("n"):
            print(f"  IS  (≤2022): n={si['n']:<4} win={si['win']:.0%} "
                  f"avgR={si['avgR']:+.2f} t={si['t']:+.1f} PF={si['PF']:.2f}")
        if st.get("n"):
            print(f"  OOS (2023→): n={st['n']:<4} win={st['win']:.0%} "
                  f"avgR={st['avgR']:+.2f} t={st['t']:+.1f} PF={st['PF']:.2f}")
        else:
            print("  OOS (2023→): 0 trades")
        for name, s, e in FOLDS:
            fs = trade_stats([t["r"] for t in oos if fold_of(t["date"]) == name])
            if fs.get("n"):
                print(f"    {name:5} n={fs['n']:<3} win={fs['win']:.0%} "
                      f"avgR={fs['avgR']:+.2f} PF={fs['PF']:.2f}")
        by_inst = {}
        for t in oos:
            by_inst.setdefault(t["inst"], []).append(t["r"])
        row = "  ".join(f"{i.split('-')[0]}:{trade_stats(rs)['n']}/{trade_stats(rs)['avgR']:+.2f}"
                        for i, rs in sorted(by_inst.items()))
        print(f"    per-asset n/avgR: {row}")
        for cost in ("maker", "zero"):
            co = [t for t in results[(strat, cost)]["trades"] if fold_of(t["date"])]
            cs = trade_stats([t["r"] for t in co])
            if cs.get("n"):
                print(f"  OOS {cost:5}: n={cs['n']:<4} avgR={cs['avgR']:+.2f} "
                      f"t={cs['t']:+.1f} PF={cs['PF']:.2f}")

    print(f"\n{line}\nPortfolio (equal-weight sleeves), OOS {OOS_START} → {DATA_END}")
    header = f"  {'':6}" + "".join(f"{n:>10}" for n, _, _ in FOLDS) + f"{'pooled':>12}"
    print(header)
    pooled_m, fold_m = {}, {}
    for strat in ("H1", "H2", "H3", "B&H"):
        curves = results[(strat, "taker")]["curves"]
        pooled_m[strat] = window_metrics(curves, OOS_START, DATA_END)
        cells = []
        for name, s, e in FOLDS:
            m = window_metrics(curves, s, e)
            fold_m[(strat, name)] = m
            cells.append(f"{m.get('ret', 0):+9.1%}" if m else f"{'—':>10}")
        p = pooled_m[strat]
        print(f"  {strat:6}" + "".join(f"{c:>10}" for c in cells)
              + f"  ret={p.get('ret', 0):+7.1%}")
        print(f"  {'':6}" + " " * (10 * len(FOLDS))
              + f"  sharpe={p.get('sharpe', 0):.2f}  maxDD={p.get('maxDD', 0):.1%}")

    print(f"\n{line}\nSUCCESS CRITERIA (pooled OOS, taker costs — all required)")
    for strat in ("H1", "H2"):
        st = trade_stats([t["r"] for t in results[(strat, "taker")]["trades"]
                          if fold_of(t["date"])])
        p, h3, bh = pooled_m[strat], pooled_m["H3"], pooled_m["B&H"]
        checks = [
            ("n ≥ 50", st.get("n", 0) >= 50, f"n={st.get('n', 0)}"),
            ("avgR > 0, t ≥ 2", st.get("avgR", 0) > 0 and st.get("t", 0) >= 2,
             f"avgR={st.get('avgR', 0):+.2f}, t={st.get('t', 0):+.1f}"),
            ("PF ≥ 1.3", st.get("PF", 0) >= 1.3, f"PF={st.get('PF', 0):.2f}"),
            ("maxDD < B&H", p.get("maxDD", -9) > bh.get("maxDD", 0),
             f"{p.get('maxDD', 0):.1%} vs {bh.get('maxDD', 0):.1%}"),
            ("Sharpe > H3", p.get("sharpe", 0) > h3.get("sharpe", 0),
             f"{p.get('sharpe', 0):.2f} vs {h3.get('sharpe', 0):.2f}"),
            ("return > H3", p.get("ret", 0) > h3.get("ret", 0),
             f"{p.get('ret', 0):+.1%} vs {h3.get('ret', 0):+.1%}"),
        ]
        n_pass = sum(ok for _, ok, _ in checks)
        print(f"\n  {strat}: {n_pass}/{len(checks)} passed")
        for name, ok, detail in checks:
            print(f"    {'✓' if ok else '✗'} {name:18} {detail}")


if __name__ == "__main__":
    main()
