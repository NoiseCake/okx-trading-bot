"""Unit tests for the pure technical-indicator functions in strategy.py."""
import numpy as np
import pandas as pd
import pytest

from strategy import (
    sma, ema, rsi, atr, adx, bollinger_bands,
    hurst_exponent, volume_confirmation, parse_candles,
)


def test_sma_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)   # mean(1,2,3)
    assert out.iloc[4] == pytest.approx(4.0)   # mean(3,4,5)


def test_ema_recursive_values():
    # span=3 → alpha=0.5, adjust=False: e0=1, e1=1.5, e2=2.25
    out = ema(pd.Series([1, 2, 3], dtype=float), 3)
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(1.5)
    assert out.iloc[2] == pytest.approx(2.25)


def test_rsi_bounded_and_high_in_uptrend():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0.8, 0.5, 300)))   # strong up-drift, some pullbacks
    r = rsi(close).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] > 50   # net-up series should read above the midline


def test_atr_constant_range_no_gap():
    n = 50
    close = pd.Series(np.full(n, 100.0))
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    out = atr(df, 14)
    assert out.iloc[-1] == pytest.approx(2.0, abs=1e-6)   # TR == high-low == 2 every bar
    assert (out.dropna() > 0).all()


def test_adx_structure_and_bounds():
    rng = np.random.default_rng(1)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    a, plus_di, minus_di = adx(df)
    assert len(a) == len(df)
    for s in (a, plus_di, minus_di):
        v = s.dropna()
        assert (v >= 0).all() and (v <= 100).all()


def test_bollinger_band_ordering():
    rng = np.random.default_rng(2)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
    upper, mid, lower = bollinger_bands(s)
    idx = upper.dropna().index
    assert (upper.loc[idx] >= mid.loc[idx]).all()
    assert (mid.loc[idx] >= lower.loc[idx]).all()


def test_hurst_short_series_is_neutral():
    assert hurst_exponent(pd.Series(range(5)), max_lag=20) == 0.5


def test_hurst_bounded():
    rng = np.random.default_rng(3)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))
    assert 0.0 <= hurst_exponent(s) <= 1.0


def test_volume_confirmation_spike_vs_flat():
    spike = pd.DataFrame({"vol": [100.0] * 20 + [200.0]})
    flat = pd.DataFrame({"vol": [100.0] * 21})
    assert volume_confirmation(spike) is True
    assert volume_confirmation(flat) is False


def test_parse_candles_drops_unconfirmed_and_sorts():
    raw = [
        # newest-first, as OKX returns; the second is still forming (confirm="0")
        ["1700003600000", "11", "13", "10", "12", "110", "0", "0", "0"],
        ["1700000000000", "10", "12", "9", "11", "100", "0", "0", "1"],
    ]
    df = parse_candles(raw)
    assert len(df) == 1
    assert df["close"].iloc[0] == pytest.approx(11.0)
    assert df["ts"].is_monotonic_increasing
