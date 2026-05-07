import os
import sys
import requests
import pandas as pd
import numpy as np

# ── Credentials ───────────────────────────────────────────────────────────────────
# These are loaded from Railway env vars (or key.env locally).
# quant_report.py deliberately re-reads credentials from the environment
# rather than importing from config.py so it can be run as a standalone script.
OKX_API_KEY    = os.environ["OKX_API_KEY"]
OKX_SECRET_KEY = os.environ["OKX_SECRET_KEY"]
OKX_PASSPHRASE = os.environ["OKX_PASSPHRASE"]
OKX_FLAG       = os.environ.get("OKX_FLAG", "1")   # "1" = paper, "0" = live

TG_TOKEN = os.environ["TG_TOKEN"]    # Telegram bot token
TG_CHAT  = os.environ["TG_CHAT"]     # Telegram chat/channel ID to send the report to


# ── Telegram helper ───────────────────────────────────────────────────────────────

def send_telegram(text: str) -> None:
    """
    Send a message to the configured Telegram chat.
    Splits into 4000-character chunks because Telegram's API has a message length limit.
    """
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for i in range(0, len(text), 4000):
        requests.post(
            url,
            json={"chat_id": TG_CHAT, "text": text[i:i+4000], "parse_mode": "Markdown"},
            timeout=10,
        )


# ── Indicator helpers (duplicated from strategy.py to keep this file standalone) ──

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — used in RSI, ATR, and ADX calculations (alpha = 1/period)."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """RSI value for the most recent bar."""
    delta = close.diff()
    gain  = wilder(delta.clip(lower=0), period)
    loss  = wilder((-delta).clip(lower=0), period)
    rs    = gain / loss.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR value for the most recent bar."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(wilder(tr, period).iloc[-1])


def compute_adx(df: pd.DataFrame, period: int = 14):
    """Returns (ADX, +DI, -DI) for the most recent bar."""
    up   = df["high"].diff()
    down = -df["low"].diff()
    pdm  = up.where((up > down) & (up > 0), 0.0)
    ndm  = down.where((down > up) & (down > 0), 0.0)
    tr   = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_s = wilder(tr, period)
    pdi   = 100 * wilder(pdm, period) / atr_s
    ndi   = 100 * wilder(ndm, period) / atr_s
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx   = wilder(dx, period)
    return float(adx.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])


def compute_hurst(close: pd.Series, n: int = 100) -> float:
    """
    Simplified R/S Hurst estimate over the last n bars.
    Returns a value near 0.5 when there's not enough data or variance is zero.
    """
    prices  = close.iloc[-n:].values
    returns = np.diff(np.log(prices))
    mean    = returns.mean()
    cumdev  = np.cumsum(returns - mean)
    r       = cumdev.max() - cumdev.min()
    s       = returns.std()
    if s == 0:
        return 0.5
    return float(np.log(r / s) / np.log(n / 2))


def classify_regime(adx_val: float, hurst_val: float) -> str:
    """Map ADX + Hurst into one of three regime labels used throughout the bot."""
    if adx_val > 25 and hurst_val >= 0.45:
        return "trending"
    if adx_val < 20 and hurst_val <= 0.55:
        return "ranging"
    return "transitional"


# ── Data fetching ─────────────────────────────────────────────────────────────────

def fetch_candles() -> pd.DataFrame:
    """Fetch the last 200 confirmed 1H candles for BTC-USDT from OKX."""
    import okx.MarketData as md
    api  = md.MarketAPI(flag=OKX_FLAG)
    resp = api.get_candlesticks("BTC-USDT", bar="1H", limit="200")
    raw  = resp.get("data", [])
    df   = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
    df   = df[df["confirm"] == "1"]
    for col in ["open","high","low","close","vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    return df.sort_values("ts").reset_index(drop=True)


def fetch_trade_history():
    """
    Pull all filled orders from OKX and the current account equity.
    Paginates using the OKX cursor-based API (passes the last ordId as 'after').
    Returns (list of order dicts, equity float).
    """
    import okx.Trade as tr_api
    import okx.Account as acc_api
    trade = tr_api.TradeAPI(OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, False, OKX_FLAG)
    acct  = acc_api.AccountAPI(OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, False, OKX_FLAG)

    orders = []
    after  = ""
    while True:
        params = {"instType": "SPOT", "instId": "BTC-USDT", "state": "filled", "limit": "100"}
        if after:
            params["after"] = after
        resp  = trade.get_orders_history(**params)
        batch = resp.get("data", [])
        if not batch:
            break
        orders.extend(batch)
        after = batch[-1]["ordId"]    # cursor for the next page
        if len(batch) < 100:
            break   # last page — fewer results than the max means we've reached the end

    # Fetch equity separately from the account API
    bal_resp = acct.get_account_balance()
    equity   = 0.0
    try:
        for detail in bal_resp["data"][0]["details"]:
            if detail["ccy"] == "USDT":
                equity = float(detail["eq"])
    except Exception:
        pass

    return orders, equity


# ── Trade statistics ──────────────────────────────────────────────────────────────

def compute_trade_stats(orders: list) -> dict:
    """
    Match buy and sell orders into round-trips and compute performance metrics.
    Uses a simple FIFO queue: each sell is matched against the earliest unmatched buy.
    Returns a dict with win rate, profit factor, average win/loss, etc.
    """
    buys = {}   # maps ordId → (entry_price, size) for unmatched buys
    pnls = []   # list of {"pnl_pct": float, "pnl_usdt": float} for each closed trade

    for o in sorted(orders, key=lambda x: int(x.get("cTime", 0))):
        side = o.get("side", "")
        px   = float(o.get("avgPx") or 0)
        sz   = float(o.get("accFillSz") or 0)
        if px == 0 or sz == 0:
            continue
        if side == "buy":
            buys[o["ordId"]] = (px, sz)
        elif side == "sell" and buys:
            # Match against the oldest unmatched buy (FIFO)
            entry_id, (entry_px, _) = next(iter(buys.items()))
            buys.pop(entry_id)
            pnl_pct  = (px - entry_px) / entry_px * 100
            pnl_usdt = (px - entry_px) * sz
            pnls.append({"pnl_pct": pnl_pct, "pnl_usdt": pnl_usdt})

    # Return zeroed-out stats if there are no completed round-trips yet
    if not pnls:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
                "biggest_win": 0, "biggest_loss": 0, "total_pnl": 0}

    wins   = [p for p in pnls if p["pnl_pct"] > 0]
    losses = [p for p in pnls if p["pnl_pct"] <= 0]

    avg_win    = np.mean([p["pnl_pct"] for p in wins])   if wins   else 0
    avg_loss   = np.mean([p["pnl_pct"] for p in losses]) if losses else 0
    gross_win  = sum(p["pnl_usdt"] for p in wins)
    gross_loss = abs(sum(p["pnl_usdt"] for p in losses))

    return {
        "total":         len(pnls),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      len(wins) / len(pnls) * 100,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        # Profit factor > 1.0 means gross wins exceed gross losses (the strategy is net positive)
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "biggest_win":   max((p["pnl_pct"] for p in wins),   default=0),
        "biggest_loss":  min((p["pnl_pct"] for p in losses), default=0),
        "total_pnl":     sum(p["pnl_usdt"] for p in pnls),
    }


# ── Signal summary ────────────────────────────────────────────────────────────────

def get_signal(df: pd.DataFrame, regime: str, adx_pdi: float, adx_ndi: float) -> tuple[str, str]:
    """
    Replicate the bot's signal logic to show the current recommendation in the report.
    Returns (signal_label, human_readable_reason).
    """
    fast = ema(df["close"], 9)
    slow = ema(df["close"], 21)
    rsi  = compute_rsi(df["close"])

    if regime == "trending":
        cross_up   = fast.iloc[-2] < slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        cross_down = fast.iloc[-2] > slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]
        vol_avg    = df["vol"].rolling(20).mean().iloc[-1]
        vol_ok     = df["vol"].iloc[-1] > 1.2 * vol_avg

        if cross_up and adx_pdi > adx_ndi and vol_ok and rsi < 70:
            return "BUY",  f"EMA cross up, +DI>{adx_pdi:.1f} > -DI{adx_ndi:.1f}, vol confirmed, RSI={rsi:.1f}"
        if cross_down and adx_ndi > adx_pdi and vol_ok and rsi > 30:
            return "SELL", f"EMA cross down, -DI>{adx_ndi:.1f} > +DI{adx_pdi:.1f}, vol confirmed, RSI={rsi:.1f}"
        return "HOLD", f"No crossover or conditions not met (RSI={rsi:.1f})"

    elif regime == "ranging":
        mid      = df["close"].rolling(20).mean()
        std      = df["close"].rolling(20).std(ddof=0)
        bb_lower = (mid - 2 * std).iloc[-2]
        bb_upper = (mid + 2 * std).iloc[-2]
        price    = df["close"].iloc[-1]

        if price <= bb_lower and rsi < 40:
            return "BUY",  f"Price {price:.0f} <= BB lower {bb_lower:.0f}, RSI={rsi:.1f}"
        if price >= bb_upper and rsi > 60:
            return "SELL", f"Price {price:.0f} >= BB upper {bb_upper:.0f}, RSI={rsi:.1f}"
        return "HOLD", f"Price within bands (RSI={rsi:.1f})"

    return "HOLD", "Transitional regime — no trade"


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    print("Fetching candles...")
    df = fetch_candles()

    # Compute all market indicators
    adx_val, pdi, ndi = compute_adx(df)
    hurst_val  = compute_hurst(df["close"])
    rsi_val    = compute_rsi(df["close"])
    atr_val    = compute_atr(df)
    regime     = classify_regime(adx_val, hurst_val)
    signal, reason = get_signal(df, regime, pdi, ndi)
    price      = df["close"].iloc[-1]

    print("Fetching trade history...")
    try:
        orders, equity = fetch_trade_history()
        stats = compute_trade_stats(orders)
    except Exception as e:
        print(f"Trade history error: {e}")
        # If the OKX API fails, still send the market section of the report
        stats  = {"total": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                  "profit_factor": 0, "biggest_win": 0, "biggest_loss": 0, "total_pnl": 0}
        equity = 0.0

    # ── Quant Critique ─────────────────────────────────────────────────────────────
    # Automatic commentary flagging anything that warrants attention
    critique_lines = []
    if stats["total"] == 0:
        critique_lines.append("No closed trades yet — bot is still in early operation.")
    else:
        if stats["win_rate"] < 40:
            critique_lines.append(f"Win rate {stats['win_rate']:.1f}% is below 40% — review entry conditions.")
        if stats["profit_factor"] < 1.0:
            critique_lines.append(f"Profit factor {stats['profit_factor']:.2f} < 1.0 — strategy is net negative.")
        if abs(stats["avg_loss"]) > stats["avg_win"] * 1.5:
            critique_lines.append("Average loss is >1.5x average win — consider tighter stops or wider TPs.")
        if stats["profit_factor"] > 1.5:
            critique_lines.append(f"Profit factor {stats['profit_factor']:.2f} looks healthy.")

    # Market condition commentary
    if regime == "transitional":
        critique_lines.append("Market is in transitional regime — bot is holding, which is correct behavior.")
    if adx_val < 15:
        critique_lines.append(f"ADX={adx_val:.1f} is very low — choppy market, expect more HOLD signals.")
    if hurst_val < 0.45:
        critique_lines.append(
            f"Hurst={hurst_val:.2f} indicates mean-reverting conditions — "
            f"BB strategy should be active if ADX confirms."
        )

    critique = "\n".join(critique_lines) if critique_lines else "No major concerns at this time."

    # Static action items (updated manually as the bot matures)
    backlog = []
    if stats["total"] < 10:
        backlog.append("1. Accumulate more trades before drawing statistical conclusions — too few data points.")
    else:
        backlog.append("1. Run walk-forward parameter validation — all indicator periods are unvalidated defaults.")
    backlog.append("2. Add max daily loss limit and drawdown circuit breaker.")
    backlog.append("3. Implement backtesting harness — required before walk-forward validation in item 1 is possible.")

    # ── Build and send the report ──────────────────────────────────────────────────
    report = f"""*OKX Bot Quant Report*
_Generated automatically every 12h_

*Performance Summary*
Account Equity: {equity:.2f} USDT
Total Closed Trades: {stats['total']}
Win Rate: {stats['win_rate']:.1f}%
Avg Win: +{stats['avg_win']:.2f}%  |  Avg Loss: {stats['avg_loss']:.2f}%
Profit Factor: {stats['profit_factor']:.2f}
Biggest Win: +{stats['biggest_win']:.2f}%  |  Biggest Loss: {stats['biggest_loss']:.2f}%
Total PnL: {stats['total_pnl']:+.2f} USDT

*Current Market State*
BTC Price: {price:,.2f} USDT
Regime: {regime.upper()}
ADX: {adx_val:.1f}  |  +DI: {pdi:.1f}  |  -DI: {ndi:.1f}
Hurst Exponent: {hurst_val:.3f}
RSI(14): {rsi_val:.1f}
ATR(14): {atr_val:.2f}
Current Signal: *{signal}*
Reason: {reason}

*Quant Critique*
{critique}

*Top 3 Action Items*
{chr(10).join(backlog)}
"""

    print(report)
    print("Sending to Telegram...")
    send_telegram(report)
    print("Done.")


if __name__ == "__main__":
    main()
