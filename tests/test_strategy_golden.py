"""
Golden-master regression test for CombinedStrategy.

`golden_signals.json` freezes the (signal, regime) output of the strategy over a
deterministic synthetic candle corpus. It was generated from the behaviour that
a side-by-side test proved identical to the pre-`evaluate()`-refactor code, so it
encodes the original, trusted behaviour. Any future change that alters a trade
decision or a regime label will fail this test.

Regenerate intentionally (only when a behaviour change is *expected* and reviewed):
    python tests/test_strategy_golden.py --regen
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest import mock

import strategy
from strategy import CombinedStrategy

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_signals.json")

# Deterministic corpus definition — must stay in lockstep with golden_signals.json.
KINDS = ["choppy", "range", "trend_down", "trend_up", "walk"]
LENGTHS = [20, 49, 50, 60, 100, 168, 200]
REPEATS = 25
SEED = 20260625


def make_df(rng: np.random.Generator, n: int, kind: str) -> pd.DataFrame:
    """Build a synthetic OHLCV frame of a given character."""
    if kind == "trend_up":
        close = 100 + np.cumsum(rng.normal(0.4, 1.0, n))
    elif kind == "trend_down":
        close = 100 + np.cumsum(rng.normal(-0.4, 1.0, n))
    elif kind == "range":
        close = 100 + 5 * np.sin(np.linspace(0, rng.uniform(4, 12), n)) + rng.normal(0, 0.6, n)
    elif kind == "choppy":
        close = 100 + np.cumsum(rng.normal(0, 2.5, n))
    else:  # walk
        close = 100 + np.cumsum(rng.normal(0, 1.0, n))
    close = np.maximum(close, 1.0)
    spread = np.abs(rng.normal(0, 0.8, n))
    high = close + spread
    low = np.maximum(close - spread, 0.5)
    open_ = close + rng.normal(0, 0.5, n)
    vol = rng.uniform(80, 120, n)
    for idx in rng.choice(n, size=max(1, n // 20), replace=False):
        vol[idx] *= rng.uniform(1.3, 3.0)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"ts": ts, "open": open_, "high": high,
                         "low": low, "close": close, "vol": vol})


def build_corpus():
    """Deterministic list of (kind, n, DataFrame). Identical every run given SEED."""
    rng = np.random.default_rng(SEED)
    corpus = []
    for kind in KINDS:
        for n in LENGTHS:
            for _ in range(REPEATS):
                corpus.append((kind, n, make_df(rng, n, kind)))
    return corpus


def _evaluate_corpus():
    strat = CombinedStrategy()
    return [{"signal": d.signal, "regime": d.regime}
            for _, _, df in build_corpus() for d in (strat.evaluate(df),)]


def test_golden_master_signals_and_regimes():
    with open(GOLDEN) as f:
        golden = json.load(f)
    results = _evaluate_corpus()
    assert len(results) == len(golden), "corpus size drifted from golden file"
    for i, (got, exp) in enumerate(zip(results, golden)):
        assert got["signal"] == exp["signal"], f"case {i}: signal {got['signal']} != golden {exp['signal']}"
        assert got["regime"] == exp["regime"], f"case {i}: regime {got['regime']} != golden {exp['regime']}"


def test_corpus_actually_exercises_non_hold():
    # Guards against the suite degenerating into "hold == hold".
    results = _evaluate_corpus()
    non_hold = sum(1 for r in results if r["signal"] != "hold")
    assert non_hold > 0, "corpus produced no buy/sell signals — not a meaningful test"


def test_signal_wrapper_matches_evaluate():
    strat = CombinedStrategy()
    for _, _, df in build_corpus():
        assert strat.signal(df) == strat.evaluate(df).signal


def test_evaluate_computes_each_indicator_once():
    """The whole point of the refactor: indicators are computed in one place.
    adx/rsi/hurst exactly once; atr twice (once directly, once inside adx())."""
    strat = CombinedStrategy()
    df = make_df(np.random.default_rng(1), 200, "trend_up")
    with mock.patch.object(strategy, "adx", wraps=strategy.adx) as m_adx, \
         mock.patch.object(strategy, "atr", wraps=strategy.atr) as m_atr, \
         mock.patch.object(strategy, "rsi", wraps=strategy.rsi) as m_rsi, \
         mock.patch.object(strategy, "hurst_exponent", wraps=strategy.hurst_exponent) as m_h:
        strat.evaluate(df)
    assert m_adx.call_count == 1
    assert m_rsi.call_count == 1
    assert m_h.call_count == 1
    assert m_atr.call_count == 2   # 1 direct + 1 inside adx()


def _regen():
    with open(GOLDEN, "w") as f:
        json.dump(_evaluate_corpus(), f, indent=0)
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    if "--regen" in sys.argv:
        _regen()
    else:
        print("pass --regen to regenerate the golden file")
