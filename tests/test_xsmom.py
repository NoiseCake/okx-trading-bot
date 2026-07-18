"""Unit tests for the shared X1 selection logic (xsmom.py) plus a data-dependent
regression pin against the registered backtest result."""

import os
import sys
from datetime import date

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xsmom import (ANCHOR_ORD, LOOKBACK, SMA_N, compute_inputs,
                   is_rebalance_date, latest_due_date, select_targets)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ── Anchor / schedule ─────────────────────────────────────────────────────────────

def test_anchor_is_thursday():
    assert date.fromordinal(ANCHOR_ORD) == date(2019, 5, 2)
    assert date.fromordinal(ANCHOR_ORD).weekday() == 3   # Thursday


def test_rebalance_dates_are_weekly_thursdays():
    assert is_rebalance_date(date(2026, 7, 2))       # a Thursday
    assert not is_rebalance_date(date(2026, 7, 3))   # Friday
    assert is_rebalance_date(date(2026, 7, 9))       # next Thursday


def test_latest_due_date():
    assert latest_due_date(date(2026, 7, 2)) == date(2026, 7, 2)    # due day itself
    assert latest_due_date(date(2026, 7, 5)) == date(2026, 7, 2)    # Sunday → prior Thu
    assert latest_due_date(date(2026, 7, 8)) == date(2026, 7, 2)    # Wednesday → prior Thu


# ── Selection rules ───────────────────────────────────────────────────────────────

def test_select_top2_by_momentum():
    mom = {"A": 0.30, "B": 0.20, "C": 0.10}
    close = {"A": 100, "B": 100, "C": 100}
    sma = {"A": 90, "B": 90, "C": 90}
    assert select_targets(mom, close, sma) == {"A": 0.5, "B": 0.5}


def test_negative_momentum_ineligible():
    mom = {"A": 0.30, "B": -0.01, "C": 0.10}
    close = {"A": 100, "B": 100, "C": 100}
    sma = {"A": 90, "B": 90, "C": 90}
    assert select_targets(mom, close, sma) == {"A": 0.5, "C": 0.5}


def test_below_sma200_ineligible():
    mom = {"A": 0.30, "B": 0.20}
    close = {"A": 100, "B": 100}
    sma = {"A": 90, "B": 110}                       # B below its SMA200
    assert select_targets(mom, close, sma) == {"A": 0.5}


def test_nan_inputs_ineligible_and_empty_ok():
    mom = {"A": np.nan, "B": 0.2}
    close = {"A": 100, "B": 100}
    sma = {"A": 90, "B": np.nan}
    assert select_targets(mom, close, sma) == {}


def test_tie_breaks_by_inst_id_desc():
    mom = {"A-USDT": 0.20, "B-USDT": 0.20, "C-USDT": 0.20}
    close = {k: 100 for k in mom}
    sma = {k: 90 for k in mom}
    assert set(select_targets(mom, close, sma)) == {"C-USDT", "B-USDT"}


def test_compute_inputs_warmup_and_values():
    n = SMA_N + 10
    arr = np.linspace(100, 200, n)
    mom, close, sma = compute_inputs({"A": arr, "B": arr[: LOOKBACK]})
    assert close["A"] == arr[-1]
    assert mom["A"] == pytest.approx(arr[-1] / arr[-1 - LOOKBACK] - 1)
    assert sma["A"] == pytest.approx(arr[-SMA_N:].mean())
    assert np.isnan(mom["B"]) and np.isnan(sma["B"])   # short history → ineligible


# ── Regression pin vs the registered backtest (needs local candle data) ──────────

@pytest.mark.skipif(not os.path.exists(os.path.join(DATA_DIR, "BTC-USDT_1D.csv")),
                    reason="candle data not fetched")
def test_x1_backtest_regression():
    """The X1 sim must keep reproducing the registered pooled-OOS result
    (+219.3% at taker costs). Guards both research2.py and xsmom.py refactors."""
    from research2 import load_matrix, sim_xs, wmetrics
    from walkforward import DATA_END, OOS_START

    O, C = load_matrix()
    m = wmetrics(sim_xs(O, C, 28), OOS_START, DATA_END)
    assert m["ret"] == pytest.approx(2.193, abs=0.01)
    assert m["sharpe"] == pytest.approx(0.85, abs=0.01)
