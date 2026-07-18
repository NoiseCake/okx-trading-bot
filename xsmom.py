"""
X1 cross-sectional momentum — pure selection logic, shared verbatim by the
backtest (research2.py) and the live bot (bot.py XsmomBot).

Registered configuration (RP2 vault note, frozen 2026-07-18): 8-asset basket,
score = 28d close-to-close return, eligible if score > 0 and close > SMA200,
hold top-2 eligible at 50/50. Rebalance decisions at every 7th daily close from
the backtest anchor — which lands on Thursday 16:00 UTC closes.

Nothing in this module may touch the network or the exchange: it is the single
source of truth for "what should the portfolio be, given these closes", so the
paper trade's selection-parity criterion is checkable by construction.
"""

from __future__ import annotations

from datetime import date

import numpy as np

UNIVERSE = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "DOGE-USDT",
            "LTC-USDT", "LINK-USDT", "SOL-USDT", "AVAX-USDT"]

LOOKBACK = 28      # momentum formation, daily closes
SMA_N = 200        # eligibility trend filter
TOP_K = 2

# First close date of the backtest matrix (BTC 1D, 2019-05-02 16:00 UTC — a
# Thursday). Live must keep the same modular schedule for parity.
ANCHOR_ORD = date(2019, 5, 2).toordinal()


def is_rebalance_date(d: date) -> bool:
    """True if the daily close dated `d` (UTC date of the 16:00 UTC close) is a
    scheduled rebalance decision point."""
    return (d.toordinal() - ANCHOR_ORD) % 7 == 0


def latest_due_date(d: date) -> date:
    """Most recent scheduled rebalance date on or before `d` (for catch-up
    after downtime)."""
    return date.fromordinal(d.toordinal() - (d.toordinal() - ANCHOR_ORD) % 7)


def select_targets(mom: dict[str, float], close: dict[str, float],
                   sma200: dict[str, float], topk: int = TOP_K) -> dict[str, float]:
    """Target weights from per-asset momentum score, latest close, and SMA200.

    Assets with any non-finite input are ineligible (insufficient history).
    Ties break by instrument id, descending — identical in backtest and live.
    """
    elig = sorted(
        ((m, i) for i, m in mom.items()
         if np.isfinite(m) and np.isfinite(close.get(i, np.nan))
         and np.isfinite(sma200.get(i, np.nan))
         and m > 0 and close[i] > sma200[i]),
        reverse=True)
    return {i: 1.0 / topk for _, i in elig[:topk]}


def compute_inputs(closes: dict[str, "np.ndarray | list[float]"]) -> tuple[dict, dict, dict]:
    """(mom, close, sma200) per asset from trailing close arrays (oldest→newest).

    Needs SMA_N closes for eligibility and LOOKBACK+1 for the score; shorter
    series yield NaN → ineligible, same as the backtest warm-up.
    """
    mom, close, sma = {}, {}, {}
    for inst, arr in closes.items():
        a = np.asarray(arr, dtype=float)
        close[inst] = a[-1] if len(a) else np.nan
        mom[inst] = a[-1] / a[-1 - LOOKBACK] - 1.0 if len(a) > LOOKBACK else np.nan
        sma[inst] = float(a[-SMA_N:].mean()) if len(a) >= SMA_N else np.nan
    return mom, close, sma
