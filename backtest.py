"""
Backtest harness for the OKX spot bot.

Replicates the live decision pipeline (bot.py + strategy.py + risk.py) bar-by-bar
over historical OKX candles, with fee and slippage modelling. This is the
prerequisite for any parameter change: nothing ships unless it survives this
harness out-of-sample.

Faithfulness to live (and the deliberate approximations):

  Decision timing   Signals are evaluated on each *confirmed* 1H bar close T,
                    exactly as live does ~48 min after the close (no new 1H/4H/1D
                    bar confirms in between, so the information set is identical).
  Entry fill        At the close of the *next* 1H bar (live fills mid-way through
                    that bar; using its close is the conservative choice for
                    mean-reversion entries) plus slippage. Stops/TPs are rebuilt
                    from the fill, size from the signal-bar close — same as live.
  Stop/TP/trailing  Live monitors 1m wicks; on 1H bars we use the bar's high/low
                    with pessimistic ordering: stop before TP when both are
                    touched in one bar, and the trailing stop is checked against
                    the level carried *into* the bar before ratcheting on its
                    high. Trailing activation anchors at the TP3 price, not the
                    bar high.
  Fees/slippage     Taker fee (default 10 bp) on every fill's notional, both
                    sides. Slippage (default 5 bp) applied adversely per fill.
  Equity            Single shared USDT cash pool across instruments (live sizes
                    from available USDT). The daily-loss and cross-instrument
                    breakers replicate live's accounting quirks verbatim (only
                    the final tranche's PnL is counted, partials are not).
  SMA200 filter     Daily bars are UTC+8-aligned (OKX default "1D", close 16:00
                    UTC) — the same series the live bot fetches. The last daily
                    bar with close <= T is used, matching parse_candles dropping
                    the forming candle.
  4H confirmation   The 4H signal from the last 4H bar with close <= T.

Usage:
    python3 backtest.py fetch                    # download candle history into data/
    python3 backtest.py run                      # baseline backtest, live config
    python3 backtest.py run --start 2024-01-01 --end 2026-07-01
    python3 backtest.py validate --db /path/to/trades.db   # cross-check vs live signals
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import requests

from strategy import CombinedStrategy, adx, atr, bollinger_bands, ema, rsi, sma

# ── Data fetching ─────────────────────────────────────────────────────────────────

OKX_HISTORY_URL = "https://www.okx.com/api/v5/market/history-candles"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

INSTRUMENTS = ["BTC-USDT", "ETH-USDT"]

# 1D starts earlier so SMA200 has warmed up before the first 1H decision.
FETCH_SPEC = [("1H", "2020-01-01"), ("4H", "2020-01-01"), ("1D", "2019-05-01")]

BAR_MS = {"1H": 3_600_000, "4H": 14_400_000, "1D": 86_400_000}

_COLUMNS = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]


def _get_with_retry(params: dict, retries: int = 5) -> list:
    for attempt in range(retries):
        try:
            resp = requests.get(OKX_HISTORY_URL, params=params, timeout=20)
            j = resp.json()
            if j.get("code") == "0":
                return j["data"]
            raise RuntimeError(f"OKX error code={j.get('code')} msg={j.get('msg')}")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def fetch_history(inst_id: str, bar: str, start: str) -> pd.DataFrame:
    """Page backwards through history-candles until `start` (UTC date) is reached."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list = []
    after: int | None = None
    while True:
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after is not None:
            params["after"] = str(after)
        data = _get_with_retry(params)
        if not data:
            break
        rows.extend(data)
        oldest = int(data[-1][0])
        if oldest <= start_ms:
            break
        after = oldest
        time.sleep(0.12)  # stay well under the 20 req / 2 s public limit

    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["ts"] = df["ts"].astype(np.int64)
    for c in ("open", "high", "low", "close", "vol", "volCcyQuote"):
        df[c] = df[c].astype(float)
    df = df[df["ts"] >= start_ms]
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "vol", "volCcyQuote"]]


def _data_path(inst_id: str, bar: str) -> str:
    return os.path.join(DATA_DIR, f"{inst_id}_{bar}.csv")


def cmd_fetch(insts: list[str] | None = None, start_override: str | None = None) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for inst in insts or INSTRUMENTS:
        for bar, start in FETCH_SPEC:
            start = start_override or start
            t0 = time.time()
            df = fetch_history(inst, bar, start)
            gaps = int((df["ts"].diff().dropna() != BAR_MS[bar]).sum())
            df.to_csv(_data_path(inst, bar), index=False)
            first = pd.Timestamp(df["ts"].iloc[0], unit="ms", tz="UTC")
            last = pd.Timestamp(df["ts"].iloc[-1], unit="ms", tz="UTC")
            print(
                f"{inst} {bar:3} {len(df):6d} bars  {first:%Y-%m-%d} → {last:%Y-%m-%d %H:%M}"
                f"  gaps={gaps}  ({time.time() - t0:.0f}s)",
                flush=True,
            )


def load_candles(inst_id: str, bar: str) -> pd.DataFrame:
    """Load a cached candle series. Adds close_ms (bar close timestamp)."""
    df = pd.read_csv(_data_path(inst_id, bar))
    df["close_ms"] = df["ts"] + BAR_MS[bar]
    return df


# ── Strategy configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StratCfg:
    """Every knob of the signal pipeline. Defaults == live behaviour exactly."""

    # Regime detection
    adx_trend: float = 25.0
    adx_range: float = 20.0
    adx_hyst: float = 0.0          # >0 → enter trend at adx_trend+hyst, leave at adx_trend-hyst
                                   #      enter range at adx_range-hyst, leave at adx_range+hyst
    hurst_on: bool = True
    hurst_window: int = 168
    hurst_low: float = 0.45        # trending requires H >= hurst_low
    hurst_high: float = 0.55       # ranging  requires H <= hurst_high

    # Sub-strategy parameters
    ema_fast: int = 9
    ema_slow: int = 21
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    atr_period: int = 14
    vol_threshold: float = 1.2     # <=0 disables volume confirmation
    vol_col: str = "vol"           # "volCcyQuote" = quote-volume variant
    trend_rsi_buy_block: float = 70.0
    trend_rsi_sell_block: float = 30.0
    range_rsi_buy: float = 40.0
    range_rsi_sell: float = 60.0

    # Macro (daily SMA200) gates — names resolved in gate_allows()
    gate_trending: str = "sma200_level"   # live: price >= SMA200 (NaN passes)
    gate_ranging: str = "sma200_slope4"   # live: SMA200 >= SMA200 4 daily bars ago (NaN blocks)

    # 4H multi-timeframe confirmation, per entry regime. "signal" (live) demands
    # the 4H *signal* also be "buy" — a 1-bar event, so it filters out almost
    # everything. State-based options check persistent 4H conditions instead:
    #   "signal"        4H CombinedStrategy signal == buy (live behaviour)
    #   "signal_fresh"  as "signal", but only enforced when a 4H bar closed at
    #                   decision time (stale 4H auto-passes)
    #   "ema_align"     4H EMA(fast) >= EMA(slow)
    #   "rsi_lt_45"     4H RSI below 45 (dip visible on 4H too); any threshold
    #   "none"          no 4H check
    #   "or(a|b)"       any of the above
    mtf_trending: str = "signal"
    mtf_ranging: str = "signal"


@dataclass(frozen=True)
class SimCfg:
    """Execution / risk knobs. Defaults == live behaviour exactly."""

    start_equity: float = 95_000.0
    fee_bp: float = 10.0           # taker, per side
    slip_bp: float = 5.0           # adverse, per market fill
    # "taker" (live): market order fills at next bar close + slippage.
    # "maker_limit": post-only limit at the signal-bar close; fills during the
    # next bar only if its low trades through the limit (else the entry is
    # missed), at the limit price with maker fee and no slippage.
    entry_style: str = "taker"
    maker_fee_bp: float = 8.0      # OKX spot VIP0 maker
    risk_pct: float = 0.015
    notional_cap: float = 0.02     # fraction of equity
    atr_mult: float = 1.5
    trail_pct: float = 0.007
    tp_mode: str = "pct"           # "pct" (live: +1/2/3%) or "atr" (multiples of stop distance)
    tp_tiers_pct: tuple = ((0.01, 0.30), (0.02, 0.40), (0.03, 0.30))
    tp_tiers_atr: tuple = ((1.0, 0.30), (2.0, 0.40), (3.0, 0.30))   # R-multiples of stop distance
    max_daily_loss_pct: float = 0.03
    max_consecutive_losses: int = 3
    consec_reset_daily: bool = True   # False = breaker persists across days once tripped
    cooldown_bars: int = 0            # bars to wait after a stop-out before re-entering
    baseline_vol: dict = field(default_factory=lambda: {"BTC-USDT": 0.008, "ETH-USDT": 0.011})
    min_lot: dict = field(default_factory=lambda: {"BTC-USDT": 0.00001, "ETH-USDT": 0.001})


# ── Vectorized signal computation ─────────────────────────────────────────────────

def hurst_rolling(close: pd.Series, window: int = 168, min_lag: int = 2, max_lag: int = 20) -> np.ndarray:
    """Rolling Hurst exponent, exactly equal to strategy.hurst_exponent applied to
    each trailing `window`-bar slice (same lags, same sample-std, same masked
    log-log least-squares slope), vectorized across all bars."""
    n = len(close)
    lags = np.arange(min_lag, max_lag)
    log_lags = np.log(lags.astype(float))

    # tau[i, t] = std of close.diff(lag_i) over the trailing `window`-bar slice ending at t
    tau = np.full((len(lags), n), np.nan)
    for i, lag in enumerate(lags):
        d = close.diff(lag)
        tau[i] = d.rolling(window - lag).std().to_numpy()   # ddof=1, same as .std() live

    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(tau > 0, np.log(tau), np.nan)

    valid = np.isfinite(y)
    cnt = valid.sum(axis=0).astype(float)
    x = np.where(valid, log_lags[:, None], 0.0)
    yv = np.where(valid, y, 0.0)
    sx, sy = x.sum(axis=0), yv.sum(axis=0)
    sxx, sxy = (x * x).sum(axis=0), (x * yv).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (sxy - sx * sy / cnt) / (sxx - sx * sx / cnt)

    h = np.clip(slope, 0.0, 1.0)
    h = np.where(np.isfinite(h) & (cnt >= 3), h, 0.5)
    h[: window - 1] = 0.5   # incomplete window — live never trades here (warm-up)
    return h


def compute_regime(adx_arr: np.ndarray, cfg: StratCfg) -> np.ndarray:
    """Regime labels. With adx_hyst == 0 this is exactly live's _regime()."""
    if cfg.adx_hyst <= 0:
        regime = np.where(adx_arr > cfg.adx_trend, "trending",
                 np.where(adx_arr < cfg.adx_range, "ranging", "transitional"))
        return regime.astype(object)

    # Hysteresis: harder to enter a regime, easier to stay in it.
    t_in, t_out = cfg.adx_trend + cfg.adx_hyst, cfg.adx_trend - cfg.adx_hyst
    r_in, r_out = cfg.adx_range - cfg.adx_hyst, cfg.adx_range + cfg.adx_hyst
    regime = np.empty(len(adx_arr), dtype=object)
    prev = "transitional"
    for i, a in enumerate(adx_arr):
        if np.isnan(a):
            cur = "transitional"
        elif prev == "trending":
            cur = "trending" if a > t_out else ("ranging" if a < r_in else "transitional")
        elif prev == "ranging":
            cur = "ranging" if a < r_out else ("trending" if a > t_in else "transitional")
        else:
            cur = "trending" if a > t_in else ("ranging" if a < r_in else "transitional")
        regime[i] = cur
        prev = cur
    return regime


def compute_signals(df: pd.DataFrame, cfg: StratCfg) -> pd.DataFrame:
    """Vectorized replica of CombinedStrategy._decide() for every bar.

    Uses the very same indicator functions from strategy.py on the full series.
    The only divergence from live is EWM warm-up (live sees a 199-bar window),
    which decays to < 1e-6 by bar ~200 — the simulation skips warm-up anyway,
    and `validate` quantifies the residual against live-logged signals.
    """
    close = df["close"]

    adx_line, plus_di, minus_di = adx(df)
    atr_s = atr(df, cfg.atr_period)
    rsi_s = rsi(close, cfg.rsi_period)
    h = hurst_rolling(close, cfg.hurst_window)

    ema_f, ema_s = ema(close, cfg.ema_fast), ema(close, cfg.ema_slow)
    diff = ema_f - ema_s
    prev_diff = diff.shift(1)
    cross_buy = (prev_diff < 0) & (diff >= 0)     # NaN comparisons → False, same as live warm-up hold
    cross_sell = (prev_diff > 0) & (diff <= 0)

    bb_upper, _, bb_lower = bollinger_bands(close, cfg.bb_period, cfg.bb_std)
    prev_upper, prev_lower = bb_upper.shift(1), bb_lower.shift(1)

    if cfg.vol_threshold > 0:
        avg_vol = df[cfg.vol_col].rolling(20).mean()
        vol_ok = (avg_vol > 0) & (df[cfg.vol_col] / avg_vol > cfg.vol_threshold)
        vol_ok = vol_ok.fillna(False).to_numpy()
    else:
        vol_ok = np.ones(len(df), dtype=bool)

    adx_arr = adx_line.to_numpy()
    regime = compute_regime(adx_arr, cfg)
    trending = regime == "trending"
    ranging = regime == "ranging"

    rsi_arr = rsi_s.to_numpy()
    rsi_ok = np.isfinite(rsi_arr)

    # +DI/-DI agreement — live blocks only when the comparison is True (NaN passes)
    di_block_buy = (plus_di <= minus_di).fillna(False).to_numpy()
    di_block_sell = (minus_di <= plus_di).fillna(False).to_numpy()

    hurst_trend_ok = (h >= cfg.hurst_low) if cfg.hurst_on else np.ones(len(df), bool)
    hurst_range_ok = (h <= cfg.hurst_high) if cfg.hurst_on else np.ones(len(df), bool)

    buy_trend = (trending & hurst_trend_ok & cross_buy.to_numpy()
                 & ~di_block_buy & vol_ok & rsi_ok & ~(rsi_arr > cfg.trend_rsi_buy_block))
    sell_trend = (trending & hurst_trend_ok & cross_sell.to_numpy()
                  & ~di_block_sell & vol_ok & rsi_ok & ~(rsi_arr < cfg.trend_rsi_sell_block))

    bands_ok = np.isfinite(prev_upper.to_numpy()) & np.isfinite(prev_lower.to_numpy()) & rsi_ok
    buy_range = (ranging & hurst_range_ok & bands_ok
                 & (close.to_numpy() <= prev_lower.to_numpy()) & (rsi_arr < cfg.range_rsi_buy))
    sell_range = (ranging & hurst_range_ok & bands_ok
                  & (close.to_numpy() >= prev_upper.to_numpy()) & (rsi_arr > cfg.range_rsi_sell))

    signal = np.where(buy_trend | buy_range, "buy",
             np.where(sell_trend | sell_range, "sell", "hold")).astype(object)
    signal[:50] = "hold"                      # live: len(df) < 50 → hold
    signal[~np.isfinite(adx_arr)] = "hold"    # live: NaN ADX → hold

    return pd.DataFrame({
        "ts": df["ts"], "close_ms": df["close_ms"],
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "signal": signal, "regime": regime,
        "adx": adx_arr, "hurst": h, "rsi": rsi_arr, "atr": atr_s.to_numpy(),
        "ema_align": (diff >= 0).to_numpy(),   # fast EMA at/above slow — trend-state flag
    })


# ── Daily context & macro gates ───────────────────────────────────────────────────

def daily_context(df_1d: pd.DataFrame) -> pd.DataFrame:
    """Per-daily-bar columns used by the macro gates."""
    out = pd.DataFrame({"close_ms": df_1d["close_ms"], "close": df_1d["close"]})
    out["sma200"] = sma(df_1d["close"], 200)
    out["sma100"] = sma(df_1d["close"], 100)
    out["sma50"] = sma(df_1d["close"], 50)
    for w in (2, 4, 9, 19):
        out[f"sma200_s{w}"] = out["sma200"].shift(w)
        out[f"sma50_s{w}"] = out["sma50"].shift(w)
        out[f"sma100_s{w}"] = out["sma100"].shift(w)
    return out


def gate_allows(gate: str, drow: pd.Series | None, price: float, rsi_val: float) -> bool:
    """Evaluate a named macro gate. `drow` is the last completed daily-context row
    (None if no daily bar is available yet).

    NaN semantics follow live: level gates fail open, slope gates fail closed.
    """
    if gate == "none":
        return True
    if drow is None:
        # Live with <200 daily bars: level gate passes (NaN skip), slope gate blocks.
        return gate.endswith("_level")

    if gate == "sma200_level":                      # live trending gate
        s = drow["sma200"]
        return bool(pd.isna(s) or price >= s)

    if gate.startswith("sma") and "_slope" in gate:  # e.g. sma200_slope4, sma50_slope2
        ma, slope = gate.split("_slope")
        cur, past = drow[ma], drow[f"{ma}_s{slope}"]
        return bool(pd.notna(cur) and pd.notna(past) and cur >= past)

    if gate.startswith("ext_gt_"):                   # price extended ≥ X% below SMA200
        x = float(gate.removeprefix("ext_gt_")) / 100.0
        s = drow["sma200"]
        return bool(pd.notna(s) and price <= s * (1 - x))

    if gate.startswith("rsi_lt_"):                   # deep 1H oversold rescue
        x = float(gate.removeprefix("rsi_lt_"))
        return bool(np.isfinite(rsi_val) and rsi_val < x)

    if gate.startswith("or(") and gate.endswith(")"):
        return any(gate_allows(p, drow, price, rsi_val) for p in gate[3:-1].split("|"))

    if gate.startswith("and(") and gate.endswith(")"):
        return all(gate_allows(p, drow, price, rsi_val) for p in gate[4:-1].split("|"))

    raise ValueError(f"Unknown gate: {gate}")


def mtf_allows(check: str, s4: dict, k4: int, fresh: bool) -> bool:
    """Evaluate a named 4H confirmation check against the last completed 4H bar."""
    if check == "none":
        return True
    if check == "signal":
        return s4["signal"][k4] == "buy"
    if check == "signal_fresh":
        return s4["signal"][k4] == "buy" if fresh else True
    if check == "ema_align":
        return bool(s4["ema_align"][k4])
    if check.startswith("rsi_lt_"):
        x = float(check.removeprefix("rsi_lt_"))
        return bool(np.isfinite(s4["rsi"][k4]) and s4["rsi"][k4] < x)
    if check.startswith("or(") and check.endswith(")"):
        return any(mtf_allows(p, s4, k4, fresh) for p in check[3:-1].split("|"))
    if check.startswith("and(") and check.endswith(")"):
        return all(mtf_allows(p, s4, k4, fresh) for p in check[4:-1].split("|"))
    raise ValueError(f"Unknown mtf check: {check}")


# ── Portfolio simulation ──────────────────────────────────────────────────────────

@dataclass
class _Pos:
    inst: str
    entry: float
    stop: float
    size: float
    original_size: float
    tps: list                      # [{"price","fraction","hit"}]
    entry_bar_ms: int
    regime: str
    risk_usdt: float               # original_size × (entry − stop): 1R in dollars
    fees: float = 0.0
    realized: float = 0.0          # net of nothing — gross tranche PnL vs entry
    trailing_active: bool = False
    trailing_high: float = 0.0
    trailing_stop: float = 0.0


@dataclass
class _InstState:
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    trade_date: str = ""
    cooldown_until: int = -1       # bar index; entries blocked while i <= cooldown_until


def simulate(data: dict, scfg: SimCfg, stcfg: StratCfg, start: str | None = None,
             end: str | None = None) -> dict:
    """Run the portfolio simulation.

    `data` maps inst_id → {"1H": df, "4H": df, "1D": df} (raw candles).
    Returns a dict with summary metrics, the trade list, and the equity curve.
    """
    insts = list(data.keys())
    sig, sig4h, dctx, arr = {}, {}, {}, {}
    for inst in insts:
        sig[inst] = compute_signals(data[inst]["1H"], stcfg)
        # Plain numpy views for the hot loop — .iloc per bar is ~50x slower.
        arr[inst] = {c: sig[inst][c].to_numpy()
                     for c in ("high", "low", "close", "signal", "regime", "rsi", "atr")}
        s4 = compute_signals(data[inst]["4H"], stcfg)
        sig4h[inst] = (s4["close_ms"].to_numpy(),
                       {c: s4[c].to_numpy() for c in ("signal", "ema_align", "rsi")})
        d = daily_context(data[inst]["1D"])
        dctx[inst] = (d["close_ms"].to_numpy(), d)

    # Master timeline = union of 1H bar close times.
    all_ms = sorted(set().union(*[set(sig[i]["close_ms"]) for i in insts]))
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if start else 0
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else 1 << 62
    # Row lookup per instrument: close_ms → integer position
    idx = {i: dict(zip(sig[i]["close_ms"].to_numpy(), range(len(sig[i])))) for i in insts}

    cash = scfg.start_equity
    pos: dict[str, _Pos | None] = {i: None for i in insts}
    pending: dict[str, dict | None] = {i: None for i in insts}
    state = {i: _InstState() for i in insts}
    closed_by_day: dict[str, float] = {}   # UTC date → Σ final-tranche pnl_usdt (live DB semantics)
    trades: list[dict] = []
    equity_curve: list[tuple[int, float]] = []
    # Where do buy signals die? Keyed by the first gate that rejects each one.
    funnel = {k: 0 for k in ("buy_signals", "busy", "atr", "cooldown", "breaker",
                             "macro_gate", "mtf", "size", "entered")}

    fee = scfg.fee_bp / 10_000.0
    slip = scfg.slip_bp / 10_000.0

    # Live rounds stop/TP prices to 2 decimals — fine at BTC/ETH scale (<0.01 bp
    # error) but disastrous for sub-dollar assets (DOGE at $0.12 → 4% error).
    # 8 significant digits keeps live parity on majors and stays exact on the rest.
    def rpx(x: float) -> float:
        return float(f"{x:.8g}")

    def tp_ladder(entry_fill: float, stop_dist: float) -> list[dict]:
        if scfg.tp_mode == "pct":
            return [{"price": rpx(entry_fill * (1 + p)), "fraction": f, "hit": False}
                    for p, f in scfg.tp_tiers_pct]
        return [{"price": rpx(entry_fill + r * stop_dist), "fraction": f, "hit": False}
                for r, f in scfg.tp_tiers_atr]

    def sell_fill(level: float) -> float:
        return level * (1 - slip)

    def record_close(inst: str, p: _Pos, level: float, bar_ms: int, reason: str) -> None:
        """Close the remaining position at `level` and replicate live accounting."""
        nonlocal cash
        fill = sell_fill(level)
        proceeds = fill * p.size
        fee_usdt = proceeds * fee
        cash += proceeds - fee_usdt
        p.fees += fee_usdt
        p.realized += (fill - p.entry) * p.size

        # Live daily-breaker semantics: only the final tranche's PnL is recorded.
        pnl_pct = (fill - p.entry) / p.entry
        pnl_usdt = (fill - p.entry) * p.size
        st = state[inst]
        prev_equity = max(cash - pnl_usdt, 1e-9)
        st.daily_pnl_pct += pnl_usdt / prev_equity
        st.consecutive_losses = st.consecutive_losses + 1 if pnl_pct < 0 else 0
        day = str(pd.Timestamp(bar_ms, unit="ms", tz="UTC").date())
        closed_by_day[day] = closed_by_day.get(day, 0.0) + pnl_usdt

        net = p.realized - p.fees
        trades.append({
            "inst": inst, "regime": p.regime,
            "opened": str(pd.Timestamp(p.entry_bar_ms, unit="ms", tz="UTC")),
            "closed": str(pd.Timestamp(bar_ms, unit="ms", tz="UTC")),
            "entry": p.entry, "exit": fill, "reason": reason,
            "hours": (bar_ms - p.entry_bar_ms) / 3_600_000,
            "pnl_usdt": net, "fees": p.fees,
            "r_multiple": net / p.risk_usdt if p.risk_usdt > 0 else 0.0,
            "tp_hits": sum(t["hit"] for t in p.tps),
        })
        pos[inst] = None

    for i_ms, bar_ms in enumerate(all_ms):
        if not (start_ms <= bar_ms <= end_ms):
            continue
        day = str(pd.Timestamp(bar_ms, unit="ms", tz="UTC").date())

        for inst in insts:
            j = idx[inst].get(bar_ms)
            if j is None:
                continue
            a = arr[inst]
            st = state[inst]

            # Daily reset (live: first tick after UTC midnight)
            if st.trade_date != day:
                st.daily_pnl_pct = 0.0
                if scfg.consec_reset_daily:
                    st.consecutive_losses = 0
                st.trade_date = day

            # ── 1. Monitor an open position over this bar's range ────────────────
            p = pos[inst]
            if p is not None:
                wl, wh = a["low"][j], a["high"][j]
                if wl <= p.stop:
                    # Pessimistic: if stop and TP are both inside the bar, stop first.
                    record_close(inst, p, p.stop, bar_ms, "STOP LOSS")
                    if scfg.cooldown_bars > 0:
                        st.cooldown_until = j + scfg.cooldown_bars
                else:
                    exited = False
                    if p.trailing_active and wl <= p.trailing_stop:
                        record_close(inst, p, p.trailing_stop, bar_ms, "TRAILING STOP")
                        exited = True
                    if not exited:
                        for k, tp in enumerate(p.tps):
                            if tp["hit"] or wh < tp["price"]:
                                continue
                            tp["hit"] = True
                            if k == len(p.tps) - 1:
                                # Trailing activates anchored at the TP3 level (pessimistic:
                                # live anchors at the 1m wick high at touch time).
                                p.trailing_active = True
                                p.trailing_high = tp["price"]
                                p.trailing_stop = rpx(tp["price"] * (1 - scfg.trail_pct))
                            else:
                                close_size = round(p.original_size * tp["fraction"], 8)
                                fill = sell_fill(tp["price"])
                                proceeds = fill * close_size
                                fee_usdt = proceeds * fee
                                cash += proceeds - fee_usdt
                                p.fees += fee_usdt
                                p.realized += (fill - p.entry) * close_size
                                p.size = round(p.size - close_size, 8)
                        # Same-bar trailing exit after activation needs a real drop below
                        # the fresh anchor; then ratchet on the bar high for later bars.
                        if p.trailing_active:
                            if wl <= p.trailing_stop:
                                record_close(inst, p, p.trailing_stop, bar_ms, "TRAILING STOP")
                            elif wh > p.trailing_high:
                                p.trailing_high = wh
                                p.trailing_stop = max(p.trailing_stop,
                                                      rpx(wh * (1 - scfg.trail_pct)))

            # ── 2. Fill a pending entry during/at the close of this bar ──────────
            q = pending[inst]
            if q is not None:
                if scfg.entry_style == "maker_limit":
                    if a["low"][j] > q["limit"]:
                        pending[inst] = None      # limit never touched — entry missed
                        q = None
                    else:
                        fill, entry_fee = q["limit"], scfg.maker_fee_bp / 10_000.0
                else:
                    fill, entry_fee = a["close"][j] * (1 + slip), fee
                if q is not None:
                    notional = fill * q["size"]
                    fee_usdt = notional * entry_fee
                    cash -= notional + fee_usdt
                    stop_dist = q["atr"] * scfg.atr_mult
                    stop = rpx(fill - stop_dist)
                    pos[inst] = _Pos(
                        inst=inst, entry=fill, stop=stop, size=q["size"],
                        original_size=q["size"], tps=tp_ladder(fill, stop_dist),
                        entry_bar_ms=bar_ms, regime=q["regime"],
                        risk_usdt=q["size"] * stop_dist, fees=fee_usdt,
                    )
                    pending[inst] = None

            # ── 3. Strategy decision at this bar's close ──────────────────────────
            if a["signal"][j] != "buy":
                continue
            funnel["buy_signals"] += 1
            if pos[inst] is not None or pending[inst] is not None:
                funnel["busy"] += 1
                continue
            sig_close, sig_atr, sig_rsi = a["close"][j], a["atr"][j], a["rsi"][j]
            sig_regime = a["regime"][j]
            if not np.isfinite(sig_atr) or sig_atr <= 0:
                funnel["atr"] += 1
                continue
            if j <= st.cooldown_until:
                funnel["cooldown"] += 1
                continue

            # Circuit breakers (per-instrument, then cross-instrument)
            if (st.daily_pnl_pct <= -scfg.max_daily_loss_pct
                    or st.consecutive_losses >= scfg.max_consecutive_losses
                    or (cash > 0 and closed_by_day.get(day, 0.0) / cash <= -scfg.max_daily_loss_pct)):
                funnel["breaker"] += 1
                continue

            # Macro gate (regime-aware, same routing as live)
            d_ms, d_df = dctx[inst]
            di = np.searchsorted(d_ms, bar_ms, side="right") - 1
            drow = d_df.iloc[di] if di >= 0 else None
            gate = stcfg.gate_ranging if sig_regime == "ranging" else stcfg.gate_trending
            if not gate_allows(gate, drow, sig_close, sig_rsi):
                funnel["macro_gate"] += 1
                continue

            # 4H confirmation (per-regime check)
            mtf_check = stcfg.mtf_ranging if sig_regime == "ranging" else stcfg.mtf_trending
            if mtf_check != "none":
                ms4, s4 = sig4h[inst]
                k4 = np.searchsorted(ms4, bar_ms, side="right") - 1
                if k4 < 0 or not mtf_allows(mtf_check, s4, k4, fresh=(ms4[k4] == bar_ms)):
                    funnel["mtf"] += 1
                    continue

            # Sizing — from the signal-bar close, exactly like live (fill differs)
            equity = cash
            if equity <= 0:
                funnel["size"] += 1
                continue
            stop_for_size = rpx(sig_close - sig_atr * scfg.atr_mult)
            stop_distance = abs(sig_close - stop_for_size)
            if stop_distance < 1e-9:
                funnel["size"] += 1
                continue
            base = equity * scfg.risk_pct / stop_distance
            dvol = sig_atr / sig_close
            bvol = scfg.baseline_vol.get(inst, 0.010)
            base *= min(bvol / max(dvol, bvol * 0.5), 1.5)
            size = round(min(base, equity * scfg.notional_cap / sig_close), 6)
            if size <= 0 or size < scfg.min_lot.get(inst, 0.0):
                funnel["size"] += 1
                continue

            funnel["entered"] += 1
            pending[inst] = {"size": size, "atr": sig_atr, "regime": sig_regime,
                             "limit": sig_close}

        # Mark-to-market equity at bar close (positions valued at 1H close)
        eq = cash
        for inst in insts:
            p = pos[inst]
            if p is not None:
                jj = idx[inst].get(bar_ms)
                eq += p.size * (arr[inst]["close"][jj] if jj is not None else p.entry)
        equity_curve.append((bar_ms, eq))

    res = _metrics(trades, equity_curve, scfg)
    res["funnel"] = funnel
    return res


def _metrics(trades: list[dict], curve: list[tuple[int, float]], scfg: SimCfg) -> dict:
    eq = pd.Series({pd.Timestamp(ms, unit="ms", tz="UTC"): v for ms, v in curve})
    out: dict = {"trades": pd.DataFrame(trades), "equity_curve": eq}
    n = len(trades)
    out["n_trades"] = n
    if n:
        t = out["trades"]
        wins = t["pnl_usdt"] > 0
        out["win_rate"] = float(wins.mean())
        out["avg_r"] = float(t["r_multiple"].mean())
        out["expectancy_usdt"] = float(t["pnl_usdt"].mean())
        out["total_pnl"] = float(t["pnl_usdt"].sum())
        out["total_fees"] = float(t["fees"].sum())
        gross_win = t.loc[wins, "pnl_usdt"].sum()
        gross_loss = -t.loc[~wins, "pnl_usdt"].sum()
        out["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
        out["avg_hours"] = float(t["hours"].mean())
        out["by_regime"] = (
            t.groupby("regime")["pnl_usdt"].agg(["count", "sum", "mean"]).to_dict("index"))
        out["by_reason"] = (
            t.groupby("reason")["pnl_usdt"].agg(["count", "sum", "mean"]).to_dict("index"))
    if len(eq) > 1:
        out["final_equity"] = float(eq.iloc[-1])
        out["return_pct"] = float(eq.iloc[-1] / scfg.start_equity - 1)
        peak = eq.cummax()
        out["max_dd_pct"] = float(((eq - peak) / peak).min())
        daily = eq.resample("1D").last().dropna().pct_change().dropna()
        sd = daily.std()
        out["sharpe"] = float(daily.mean() / sd * np.sqrt(365)) if sd and sd > 0 else 0.0
    return out


def summarize(res: dict, label: str = "") -> str:
    fn = res.get("funnel", {})
    funnel_str = "  ".join(f"{k}={v}" for k, v in fn.items() if v)
    if res["n_trades"] == 0:
        return f"{label}\n  0 trades   [{funnel_str}]"
    lines = [
        f"{label}",
        f"  [{funnel_str}]",
        f"  trades={res['n_trades']}  win={res['win_rate']:.1%}  avgR={res['avg_r']:+.2f}"
        f"  PF={res['profit_factor']:.2f}  expectancy={res['expectancy_usdt']:+.2f} USDT",
        f"  totalPnL={res['total_pnl']:+.0f} USDT  fees={res['total_fees']:.0f}"
        f"  return={res.get('return_pct', 0):+.2%}  maxDD={res.get('max_dd_pct', 0):.2%}"
        f"  sharpe={res.get('sharpe', 0):.2f}  avg_hold={res['avg_hours']:.0f}h",
    ]
    for regime, s in res.get("by_regime", {}).items():
        lines.append(f"    {regime:12s} n={s['count']:<4.0f} pnl={s['sum']:+9.1f} avg={s['mean']:+7.2f}")
    for reason, s in res.get("by_reason", {}).items():
        lines.append(f"    {reason:14s} n={s['count']:<4.0f} pnl={s['sum']:+9.1f} avg={s['mean']:+7.2f}")
    return "\n".join(lines)


def load_all(insts: list[str] | None = None) -> dict:
    insts = insts or INSTRUMENTS
    return {i: {b: load_candles(i, b) for b, _ in FETCH_SPEC} for i in insts}


# ── Validation against the live signals table ─────────────────────────────────────

def cmd_validate(db_path: str, sample: int = 0) -> None:
    """Replay every live-logged signal through the real CombinedStrategy on a
    reconstructed 199-bar window (what live actually saw) and compare."""
    import sqlite3

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT ts, inst_id, signal, regime, adx, hurst, rsi, atr, price FROM signals ORDER BY id")]
    con.close()
    if sample:
        rows = rows[:: max(1, len(rows) // sample)]

    strat = CombinedStrategy()
    data = {i: load_candles(i, "1H") for i in INSTRUMENTS}
    n = mismatch_sig = mismatch_reg = price_missing = 0
    diffs = {"adx": [], "hurst": [], "rsi": [], "atr": []}
    examples = []

    for r in rows:
        inst = r["inst_id"]
        df = data[inst]
        ts_ms = int(pd.Timestamp(r["ts"]).tz_convert("UTC").timestamp() * 1000) \
            if "+" in r["ts"] else int(pd.Timestamp(r["ts"], tz="UTC").timestamp() * 1000)
        # Last confirmed bar at call time, then a 199-bar trailing window
        end = np.searchsorted(df["close_ms"].to_numpy(), ts_ms, side="right")
        if end < 60:
            continue
        win = df.iloc[max(0, end - 199): end].reset_index(drop=True)
        if abs(win["close"].iloc[-1] - r["price"]) > 1e-6:
            price_missing += 1
            continue   # candle revision/mismatch — don't count indicator diffs
        n += 1
        dec = strat.evaluate(win)
        if dec.signal != r["signal"]:
            mismatch_sig += 1
            if len(examples) < 10:
                examples.append((r["ts"], inst, r["signal"], dec.signal))
        if dec.regime != r["regime"]:
            mismatch_reg += 1
        for k, live_v, ours in (("adx", r["adx"], dec.adx), ("hurst", r["hurst"], dec.hurst),
                                ("rsi", r["rsi"], dec.rsi), ("atr", r["atr"], dec.atr)):
            if live_v is not None and np.isfinite(ours):
                diffs[k].append(abs(ours - live_v))

    print(f"validated rows          : {n}  (skipped {price_missing} with close-price mismatch)")
    print(f"signal mismatches       : {mismatch_sig}  ({mismatch_sig / max(n,1):.3%})")
    print(f"regime mismatches       : {mismatch_reg}  ({mismatch_reg / max(n,1):.3%})")
    for k, v in diffs.items():
        if v:
            print(f"max |Δ{k:5s}|           : {max(v):.6g}")
    for e in examples:
        print("  mismatch:", e)


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fetch")
    fp.add_argument("--insts", nargs="+", default=None, help="instrument ids (default: INSTRUMENTS)")
    fp.add_argument("--start", default=None, help="override start date for all bars, e.g. 2020-01-01")
    runp = sub.add_parser("run")
    runp.add_argument("--start", default=None)
    runp.add_argument("--end", default=None)
    runp.add_argument("--out", default=None, help="write per-trade CSV here")
    valp = sub.add_parser("validate")
    valp.add_argument("--db", required=True)
    valp.add_argument("--sample", type=int, default=0, help="validate every Nth row only")
    args = ap.parse_args()

    if args.cmd == "fetch":
        cmd_fetch(args.insts, args.start)
    elif args.cmd == "validate":
        cmd_validate(args.db, args.sample)
    elif args.cmd == "run":
        data = load_all()
        res = simulate(data, SimCfg(), StratCfg(), start=args.start, end=args.end)
        print(summarize(res, "baseline (live config)"))
        if args.out and res["n_trades"]:
            res["trades"].to_csv(args.out, index=False)
            print(f"trades written to {args.out}")


if __name__ == "__main__":
    main()
