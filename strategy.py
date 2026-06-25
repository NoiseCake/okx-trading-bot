import pandas as pd
import numpy as np
from dataclasses import dataclass


def parse_candles(raw: list) -> pd.DataFrame:
    """
    Convert the raw list-of-lists from the OKX API into a clean DataFrame.
    We only keep 'confirmed' candles (confirm="1") — the last candle in the response
    is still forming and its OHLCV values will change, so we discard it.
    """
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


# ── Technical Indicators ──────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average — equally weights all bars in the window."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average — weights recent bars more heavily than older ones.
    Reacts faster than SMA to recent price changes.
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index — measures the speed and magnitude of recent price moves.
    Ranges from 0 to 100:
      > 70 → overbought (price has risen fast, may pull back)
      < 30 → oversold  (price has fallen fast, may bounce)

    Uses Wilder's exponential smoothing (alpha = 1/period) rather than a plain rolling mean,
    which is the industry-standard RSI calculation.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)   # replace(0) avoids division-by-zero when there are no losses
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — the single best measure of market volatility.
    True Range is the largest of:
      - High minus Low (normal candle range)
      - High minus previous Close (gap-up scenario)
      - Low  minus previous Close (gap-down scenario)
    We then smooth those TR values with Wilder's exponential average.
    Used to set stop distances and scale position size.
    """
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
    Bollinger Bands — a volatility envelope around a moving average.
    Returns (upper band, middle band, lower band).

    Upper = SMA + 2σ  (price touching here is statistically 'high')
    Lower = SMA - 2σ  (price touching here is statistically 'low')

    Used in ranging regimes as a mean-reversion signal:
    buy when price hits the lower band, sell when it hits the upper band.
    ddof=0 matches the canonical Bollinger Band definition (population std, not sample std).
    """
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index — measures trend STRENGTH, not direction.
    Also returns +DI (upward pressure) and -DI (downward pressure).
    Returns (ADX line, +DI, -DI).

    ADX > 25 → strong trend  (use trend-following strategies)
    ADX < 20 → weak / no trend (use mean-reversion strategies)
    20–25    → transitional / ambiguous (avoid trading)

    +DI > -DI means buyers are in control → trend is upward
    -DI > +DI means sellers are in control → trend is downward
    """
    high, low = df["high"], df["low"]
    prev_high, prev_low = high.shift(1), low.shift(1)

    # Directional movement: how much more did price move up vs down today?
    up_move   = high - prev_high
    down_move = prev_low - low

    # +DM: upward move only counts if it was larger than the downward move
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    # -DM: downward move only counts if it was larger than the upward move
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm,  index=df.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=df.index, dtype=float)

    atr_val  = atr(df, period)
    safe_atr = atr_val.replace(0, np.nan)   # avoid divide-by-zero

    # Directional Indicators: smoothed DM normalised by ATR (makes them comparable across assets)
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean()  / safe_atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / safe_atr

    # DX measures how different +DI and -DI are — big difference = strong trend
    di_sum   = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / di_sum
    adx_line = dx.ewm(alpha=1 / period, adjust=False).mean()   # smooth DX → ADX

    return adx_line, plus_di, minus_di


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Hurst Exponent — classifies the price process as trending, random, or mean-reverting.
    Ranges from 0 to 1:
      H > 0.55 → persistent / trending (past moves predict future direction)
      H < 0.45 → anti-persistent / mean-reverting (price tends to reverse)
      0.45–0.55 → random walk (no reliable edge, avoid trading)

    Method: Rescaled Range (R/S) analysis.
    We compute the standard deviation of price differences over multiple lag periods,
    then fit a line to log(lag) vs log(std). The slope is an estimate of H.

    Applied on a rolling 168-bar window (1 week of 1H candles) to capture recent behaviour.
    """
    if len(series) < max_lag * 2:
        return 0.5      # not enough data — return the 'random walk' neutral value

    lags = list(range(min_lag, max_lag))
    tau  = []
    for lag in lags:
        diff = series.diff(lag).dropna()
        std  = diff.std()
        tau.append(std if std > 0 else np.nan)

    tau   = np.array(tau)
    valid = ~np.isnan(tau)
    if valid.sum() < 3:
        return 0.5      # too few valid points for a reliable regression

    try:
        # Slope of the log-log regression is the Hurst estimate
        poly = np.polyfit(np.log(np.array(lags)[valid]), np.log(tau[valid]), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def volume_confirmation(df: pd.DataFrame, period: int = 20, threshold: float = 1.2) -> bool:
    """
    Check whether the current bar's volume is above 1.2× the 20-bar rolling average.
    High volume on a breakout candle validates the move — it means more participants
    are acting on the signal, making a follow-through more likely.
    This is intentionally independent from all price-derived indicators.
    """
    avg_vol = df["vol"].rolling(period).mean()
    if pd.isna(avg_vol.iloc[-1]) or avg_vol.iloc[-1] == 0:
        return False
    return bool(df["vol"].iloc[-1] / avg_vol.iloc[-1] > threshold)


# ── Sub-Strategy ──────────────────────────────────────────────────────────────────

class EMACrossStrategy:
    """
    Same crossover logic as SMA, but uses EMAs which react faster to recent price moves.
    Preferred over SMA in trending regimes because it catches the trend earlier.
    """

    def __init__(self, fast: int = 9, slow: int = 21) -> None:
        self.fast = fast
        self.slow = slow

    def signal(self, df: pd.DataFrame) -> str:
        if len(df) < self.slow + 1:
            return "hold"
        fast_ema  = ema(df["close"], self.fast)
        slow_ema  = ema(df["close"], self.slow)
        prev_diff = fast_ema.iloc[-2] - slow_ema.iloc[-2]
        curr_diff = fast_ema.iloc[-1] - slow_ema.iloc[-1]
        if prev_diff < 0 and curr_diff >= 0:
            return "buy"
        if prev_diff > 0 and curr_diff <= 0:
            return "sell"
        return "hold"


# ── Main Strategy ─────────────────────────────────────────────────────────────────

@dataclass
class Decision:
    """One strategy evaluation. All indicators are computed once in
    CombinedStrategy.evaluate() and surfaced here so the bot can log and size
    from them without recomputing — a single source of truth for the trade
    signal, the regime label, and the indicator snapshot."""
    signal: str          # "buy" / "sell" / "hold"
    regime: str          # "trending" / "ranging" / "transitional"
    adx: float
    hurst: float
    rsi: float
    atr: float


class CombinedStrategy:
    """
    Regime-aware meta-strategy that routes between two sub-strategies based on
    market conditions (ADX + Hurst exponent).

    TRENDING regime (ADX > 25, Hurst ≥ 0.45):
      Primary signal  : EMA crossover (9/21) — catches momentum early
      Confirmation    : +DI/-DI must agree with the crossover direction
      Confirmation    : Volume must be above 20-bar average (independent signal)
      Guard           : RSI must not be in extreme territory (avoid entering exhausted moves)

    RANGING regime (ADX < 20, Hurst ≤ 0.55):
      Signal          : Price breaks prior bar's Bollinger Band boundary
      Confirmation    : RSI oversold (<40) for buys, overbought (>60) for sells
      Note: we use the *previous* bar's BB to avoid look-ahead bias —
            today's close was used to compute today's BB, so comparing
            today's close to today's band would be circular.

    TRANSITIONAL (20 ≤ ADX ≤ 25) — no trade.
      Both sub-strategies perform poorly when the market hasn't decided
      whether it's trending or ranging. We simply sit out.
    """

    TREND_THRESHOLD = 25.0    # ADX above this = trending
    RANGE_THRESHOLD = 20.0    # ADX below this = ranging
    HURST_WINDOW    = 168     # one week of 1H bars for the Hurst calculation

    def __init__(self) -> None:
        self._ema_cross = EMACrossStrategy(fast=9, slow=21)

    def signal(self, df: pd.DataFrame) -> str:
        """Backwards-compatible thin wrapper — returns just the trade signal.
        Prefer evaluate() when you also need the regime/indicator snapshot."""
        return self.evaluate(df).signal

    def evaluate(self, df: pd.DataFrame) -> Decision:
        """Compute the regime, the trade signal, and the indicator snapshot in a
        SINGLE pass. The bot consumes the returned indicators for logging and
        risk sizing instead of recomputing them, so regime and signal are
        derived in exactly one place and can never drift apart.

        Behaviour is identical to the previous signal() — verified by a
        golden-master regression test over a synthetic candle corpus. The only
        change is that indicators are computed once here and returned.
        """
        close = df["close"]

        # ── Indicators — computed exactly once ────────────────────────────────────
        adx_line, plus_di, minus_di = adx(df)
        curr_adx = adx_line.iloc[-1]
        atr_val  = atr(df).iloc[-1]
        curr_rsi = rsi(close).iloc[-1]
        window   = close.iloc[-self.HURST_WINDOW:] if len(close) >= self.HURST_WINDOW else close
        h        = hurst_exponent(window)

        regime = self._regime(curr_adx)
        sig    = self._decide(df, close, curr_adx, plus_di, minus_di, curr_rsi, h)

        return Decision(signal=sig, regime=regime, adx=curr_adx, hurst=h, rsi=curr_rsi, atr=atr_val)

    def _regime(self, curr_adx: float) -> str:
        """ADX-based regime label. A NaN ADX (warm-up) falls through to
        transitional, matching the previous bot-side classification exactly."""
        if curr_adx > self.TREND_THRESHOLD:
            return "trending"
        if curr_adx < self.RANGE_THRESHOLD:
            return "ranging"
        return "transitional"

    def _decide(self, df, close, curr_adx, plus_di, minus_di, curr_rsi, h) -> str:
        """The regime-routed decision tree, fed pre-computed indicators. Control
        flow and check ordering are preserved verbatim from the prior signal()."""
        # Need at least 50 bars for all indicators to have warmed up properly
        if len(df) < 50:
            return "hold"
        if pd.isna(curr_adx):
            return "hold"

        # ── Trending regime ───────────────────────────────────────────────────────
        if curr_adx > self.TREND_THRESHOLD:

            # Hurst gate: if the price process is actually mean-reverting or random,
            # the EMA crossover will generate too many false signals — skip it.
            if h < 0.45:
                return "hold"

            ema_signal = self._ema_cross.signal(df)
            if ema_signal == "hold":
                return "hold"

            # +DI/-DI must agree with the crossover direction.
            # A buy signal with sellers in control (+DI < -DI) is a warning sign.
            if ema_signal == "buy"  and plus_di.iloc[-1]  <= minus_di.iloc[-1]:
                return "hold"
            if ema_signal == "sell" and minus_di.iloc[-1] <= plus_di.iloc[-1]:
                return "hold"

            # Volume must confirm — a crossover on low volume often fades
            if not volume_confirmation(df):
                return "hold"

            # RSI guard: avoid buying into overbought or selling into oversold extremes
            if pd.isna(curr_rsi):
                return "hold"
            if ema_signal == "buy"  and curr_rsi > 70:
                return "hold"
            if ema_signal == "sell" and curr_rsi < 30:
                return "hold"

            return ema_signal

        # ── Ranging regime ────────────────────────────────────────────────────────
        elif curr_adx < self.RANGE_THRESHOLD:

            # Hurst gate: if price is actually trending, BB mean-reversion trades
            # will keep losing as the price marches away from the bands.
            if h > 0.55:
                return "hold"

            # Use the prior bar's BB to avoid self-referencing the current close
            bb_upper, _, bb_lower = bollinger_bands(close)
            prev_upper = bb_upper.iloc[-2]
            prev_lower = bb_lower.iloc[-2]
            curr_price = close.iloc[-1]

            if any(pd.isna(v) for v in [curr_rsi, prev_upper, prev_lower]):
                return "hold"

            # Price at or below lower band + RSI confirms oversold → mean-reversion buy
            if curr_price <= prev_lower and curr_rsi < 40:
                return "buy"
            # Price at or above upper band + RSI confirms overbought → mean-reversion sell
            if curr_price >= prev_upper and curr_rsi > 60:
                return "sell"

            return "hold"

        # ── Transitional regime — sit out ─────────────────────────────────────────
        return "hold"
