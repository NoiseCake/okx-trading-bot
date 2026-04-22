import pandas as pd
import numpy as np


def parse_candles(raw: list) -> pd.DataFrame:
    """Convert raw OKX candlestick response to a DataFrame."""
    df = pd.DataFrame(
        raw,
        columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"],
    )
    df = df[df["confirm"] == "1"].copy()
    df[["open", "high", "low", "close", "vol"]] = df[["open", "high", "low", "close", "vol"]].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Indicators ────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's exponential smoothing (alpha = 1/period), not SMA."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder's exponential smoothing — more reactive to recent vol spikes."""
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (upper, mid, lower).
    Uses population std (ddof=0) to match the canonical Bollinger Band definition.
    """
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index using Wilder's smoothing.
    Returns (ADX, +DI, -DI).

    ADX > 25 → trending regime (use trend-following strategies)
    ADX < 20 → ranging regime  (use mean-reversion strategies)
    """
    high, low = df["high"], df["low"]
    prev_high, prev_low = high.shift(1), low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=df.index, dtype=float)

    atr_val = atr(df, period)
    safe_atr = atr_val.replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / safe_atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / safe_atr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_line = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx_line, plus_di, minus_di


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Estimate the Hurst exponent via rescaled range analysis.

    H > 0.55 → persistent / trending price process
    H < 0.45 → anti-persistent / mean-reverting price process
    0.45 ≤ H ≤ 0.55 → random walk, no reliable edge for either regime

    Applied on a rolling window (caller decides length).
    """
    if len(series) < max_lag * 2:
        return 0.5

    lags = list(range(min_lag, max_lag))
    tau = []
    for lag in lags:
        diff = series.diff(lag).dropna()
        std = diff.std()
        tau.append(std if std > 0 else np.nan)

    tau = np.array(tau)
    valid = ~np.isnan(tau)
    if valid.sum() < 3:
        return 0.5

    try:
        poly = np.polyfit(np.log(np.array(lags)[valid]), np.log(tau[valid]), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def volume_confirmation(df: pd.DataFrame, period: int = 20, threshold: float = 1.2) -> bool:
    """
    Returns True if current bar's volume is above threshold × rolling average.
    Provides a signal dimension independent from all price-derived indicators.
    """
    avg_vol = df["vol"].rolling(period).mean()
    if pd.isna(avg_vol.iloc[-1]) or avg_vol.iloc[-1] == 0:
        return False
    return bool(df["vol"].iloc[-1] / avg_vol.iloc[-1] > threshold)


# ── Strategies ────────────────────────────────────────────────────────────────

class SMACrossStrategy:
    """BUY when fast SMA crosses above slow SMA, SELL when it crosses below."""

    def __init__(self, fast: int = 9, slow: int = 21) -> None:
        self.fast = fast
        self.slow = slow

    def signal(self, df: pd.DataFrame) -> str:
        if len(df) < self.slow + 1:
            return "hold"
        fast_sma = sma(df["close"], self.fast)
        slow_sma = sma(df["close"], self.slow)
        prev_diff = fast_sma.iloc[-2] - slow_sma.iloc[-2]
        curr_diff = fast_sma.iloc[-1] - slow_sma.iloc[-1]
        if prev_diff < 0 and curr_diff >= 0:
            return "buy"
        if prev_diff > 0 and curr_diff <= 0:
            return "sell"
        return "hold"


class EMACrossStrategy:
    """BUY when fast EMA crosses above slow EMA, SELL when it crosses below."""

    def __init__(self, fast: int = 9, slow: int = 21) -> None:
        self.fast = fast
        self.slow = slow

    def signal(self, df: pd.DataFrame) -> str:
        if len(df) < self.slow + 1:
            return "hold"
        fast_ema = ema(df["close"], self.fast)
        slow_ema = ema(df["close"], self.slow)
        prev_diff = fast_ema.iloc[-2] - slow_ema.iloc[-2]
        curr_diff = fast_ema.iloc[-1] - slow_ema.iloc[-1]
        if prev_diff < 0 and curr_diff >= 0:
            return "buy"
        if prev_diff > 0 and curr_diff <= 0:
            return "sell"
        return "hold"


class CombinedStrategy:
    """
    Regime-aware strategy routing between two sub-strategies based on ADX + Hurst exponent.

    TRENDING regime (ADX > 25, Hurst ≥ 0.45):
        Primary signal : EMA crossover (9/21)
        Confirmation   : +DI/-DI directional agreement
        Filter         : Volume above 20-bar average (independent signal dimension)
        Guard          : RSI not at extreme overbought/oversold on entry

    RANGING regime (ADX < 20, Hurst ≤ 0.55):
        Signal         : Price breaks prior bar's BB lower/upper (avoids self-reference)
        Confirmation   : RSI oversold (<40) for buy, overbought (>60) for sell

    TRANSITIONAL (20 ≤ ADX ≤ 25):
        No trade — regime is ambiguous and both sub-strategies perform poorly here.
    """

    TREND_THRESHOLD = 25.0
    RANGE_THRESHOLD = 20.0
    HURST_WINDOW = 168  # one week of 1H bars

    def __init__(self) -> None:
        self._ema_cross = EMACrossStrategy(fast=9, slow=21)

    def signal(self, df: pd.DataFrame) -> str:
        # Need enough bars for ADX warmup (2× period) plus BB lookback
        if len(df) < 50:
            return "hold"

        close = df["close"]

        # ── Regime detection ─────────────────────────────────────────────────
        adx_line, plus_di, minus_di = adx(df)
        curr_adx = adx_line.iloc[-1]
        if pd.isna(curr_adx):
            return "hold"

        # Hurst on a rolling one-week window
        window = close.iloc[-self.HURST_WINDOW:] if len(close) >= self.HURST_WINDOW else close
        h = hurst_exponent(window)

        # ── Trending regime ───────────────────────────────────────────────────
        if curr_adx > self.TREND_THRESHOLD:
            # Hurst gate: suppress if price process is mean-reverting or random walk
            if h < 0.45:
                return "hold"

            ema_signal = self._ema_cross.signal(df)
            if ema_signal == "hold":
                return "hold"

            # +DI/-DI must agree with the crossover direction
            curr_plus_di = plus_di.iloc[-1]
            curr_minus_di = minus_di.iloc[-1]
            if ema_signal == "buy" and curr_plus_di <= curr_minus_di:
                return "hold"
            if ema_signal == "sell" and curr_minus_di <= curr_plus_di:
                return "hold"

            # Volume must be above average (independent confirmation)
            if not volume_confirmation(df):
                return "hold"

            # RSI guard: avoid entering into already-exhausted momentum
            curr_rsi = rsi(close).iloc[-1]
            if pd.isna(curr_rsi):
                return "hold"
            if ema_signal == "buy" and curr_rsi > 70:
                return "hold"
            if ema_signal == "sell" and curr_rsi < 30:
                return "hold"

            return ema_signal

        # ── Ranging regime ────────────────────────────────────────────────────
        elif curr_adx < self.RANGE_THRESHOLD:
            # Hurst gate: suppress if price process is actually trending
            if h > 0.55:
                return "hold"

            # Use prior bar's BB to avoid self-referencing (close_T in band computed from close_T)
            bb_upper, _, bb_lower = bollinger_bands(close)
            prev_upper = bb_upper.iloc[-2]
            prev_lower = bb_lower.iloc[-2]
            curr_price = close.iloc[-1]

            curr_rsi = rsi(close).iloc[-1]
            if any(pd.isna(v) for v in [curr_rsi, prev_upper, prev_lower]):
                return "hold"

            if curr_price <= prev_lower and curr_rsi < 40:
                return "buy"
            if curr_price >= prev_upper and curr_rsi > 60:
                return "sell"

            return "hold"

        # ── Transitional regime — no trade ───────────────────────────────────
        return "hold"
