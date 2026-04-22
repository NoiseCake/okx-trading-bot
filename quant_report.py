import os
import sys
import requests
import pandas as pd
import numpy as np

# ── Credentials ───────────────────────────────────────────────────────────────
OKX_API_KEY    = "2ce5ef04-9dd9-44b9-9c33-90fdb7877639"
OKX_SECRET_KEY = "DBEFDB6903D26B3ABCECC10CE976EDA5"
OKX_PASSPHRASE = "passpoorGH1234@"
OKX_FLAG       = "1"   # demo account

TG_TOKEN  = "8740666003:AAFiURpcTW4MFfRajrXjPFbTJGr2ueo__qc"
TG_CHAT   = "1688179650"

# ── Helpers ───────────────────────────────────────────────────────────────────
def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for i in range(0, len(text), 4000):
        requests.post(url, json={"chat_id": TG_CHAT, "text": text[i:i+4000], "parse_mode": "Markdown"}, timeout=10)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = wilder(delta.clip(lower=0), period)
    loss  = wilder((-delta).clip(lower=0), period)
    rs    = gain / loss.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(wilder(tr, period).iloc[-1])


def compute_adx(df: pd.DataFrame, period: int = 14):
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
    prices = close.iloc[-n:].values
    returns = np.diff(np.log(prices))
    mean = returns.mean()
    cumdev = np.cumsum(returns - mean)
    r = cumdev.max() - cumdev.min()
    s = returns.std()
    if s == 0:
        return 0.5
    return float(np.log(r / s) / np.log(n / 2))


def classify_regime(adx_val: float, hurst_val: float) -> str:
    if adx_val > 25 and hurst_val >= 0.45:
        return "trending"
    if adx_val < 20 and hurst_val <= 0.55:
        return "ranging"
    return "transitional"


# ── OKX Market Data (public, no auth needed) ──────────────────────────────────
def fetch_candles() -> pd.DataFrame:
    import okx.MarketData as md
    api = md.MarketAPI(flag=OKX_FLAG)
    resp = api.get_candlesticks("BTC-USDT", bar="1H", limit="200")
    raw  = resp.get("data", [])
    df   = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
    df   = df[df["confirm"] == "1"]
    for col in ["open","high","low","close","vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    return df.sort_values("ts").reset_index(drop=True)


# ── OKX Trade History ─────────────────────────────────────────────────────────
def fetch_trade_history():
    import okx.Trade as tr_api
    import okx.Account as acc_api
    trade  = tr_api.TradeAPI(OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, False, OKX_FLAG)
    acct   = acc_api.AccountAPI(OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, False, OKX_FLAG)

    # Fetch filled orders (paginated)
    orders = []
    after  = ""
    while True:
        params = {"instType": "SPOT", "instId": "BTC-USDT", "state": "filled", "limit": "100"}
        if after:
            params["after"] = after
        resp = trade.get_orders_history(**params)
        batch = resp.get("data", [])
        if not batch:
            break
        orders.extend(batch)
        after = batch[-1]["ordId"]
        if len(batch) < 100:
            break

    # Fetch balance
    bal_resp = acct.get_account_balance()
    equity = 0.0
    try:
        for detail in bal_resp["data"][0]["details"]:
            if detail["ccy"] == "USDT":
                equity = float(detail["eq"])
    except Exception:
        pass

    return orders, equity


def compute_trade_stats(orders: list) -> dict:
    buys  = {}
    pnls  = []

    for o in sorted(orders, key=lambda x: int(x.get("cTime", 0))):
        side  = o.get("side", "")
        px    = float(o.get("avgPx") or 0)
        sz    = float(o.get("accFillSz") or 0)
        if px == 0 or sz == 0:
            continue
        if side == "buy":
            buys[o["ordId"]] = (px, sz)
        elif side == "sell" and buys:
            entry_id, (entry_px, _) = next(iter(buys.items()))
            buys.pop(entry_id)
            pnl_pct  = (px - entry_px) / entry_px * 100
            pnl_usdt = (px - entry_px) * sz
            pnls.append({"pnl_pct": pnl_pct, "pnl_usdt": pnl_usdt})

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
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "biggest_win":   max((p["pnl_pct"] for p in wins),   default=0),
        "biggest_loss":  min((p["pnl_pct"] for p in losses), default=0),
        "total_pnl":     sum(p["pnl_usdt"] for p in pnls),
    }


# ── Current Signal ─────────────────────────────────────────────────────────────
def get_signal(df: pd.DataFrame, regime: str, adx_pdi: float, adx_ndi: float) -> tuple[str, str]:
    fast = ema(df["close"], 9)
    slow = ema(df["close"], 21)
    rsi  = compute_rsi(df["close"])

    if regime == "trending":
        cross_up   = fast.iloc[-2] < slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        cross_down = fast.iloc[-2] > slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]
        vol_avg    = df["vol"].rolling(20).mean().iloc[-1]
        vol_ok     = df["vol"].iloc[-1] > 1.2 * vol_avg

        if cross_up and adx_pdi > adx_ndi and vol_ok and rsi < 70:
            return "BUY", f"EMA cross up, +DI>{adx_pdi:.1f} > -DI{adx_ndi:.1f}, vol confirmed, RSI={rsi:.1f}"
        if cross_down and adx_ndi > adx_pdi and vol_ok and rsi > 30:
            return "SELL", f"EMA cross down, -DI>{adx_ndi:.1f} > +DI{adx_pdi:.1f}, vol confirmed, RSI={rsi:.1f}"
        return "HOLD", f"No crossover or conditions not met (RSI={rsi:.1f})"

    elif regime == "ranging":
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std(ddof=0)
        bb_lower = (mid - 2 * std).iloc[-2]
        bb_upper = (mid + 2 * std).iloc[-2]
        price = df["close"].iloc[-1]

        if price <= bb_lower and rsi < 40:
            return "BUY", f"Price {price:.0f} <= BB lower {bb_lower:.0f}, RSI={rsi:.1f}"
        if price >= bb_upper and rsi > 60:
            return "SELL", f"Price {price:.0f} >= BB upper {bb_upper:.0f}, RSI={rsi:.1f}"
        return "HOLD", f"Price within bands (RSI={rsi:.1f})"

    return "HOLD", "Transitional regime — no trade"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Fetching candles...")
    df = fetch_candles()

    adx_val, pdi, ndi = compute_adx(df)
    hurst_val = compute_hurst(df["close"])
    rsi_val   = compute_rsi(df["close"])
    atr_val   = compute_atr(df)
    regime    = classify_regime(adx_val, hurst_val)
    signal, reason = get_signal(df, regime, pdi, ndi)
    price     = df["close"].iloc[-1]

    print("Fetching trade history...")
    try:
        orders, equity = fetch_trade_history()
        stats = compute_trade_stats(orders)
    except Exception as e:
        print(f"Trade history error: {e}")
        stats  = {"total": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                  "profit_factor": 0, "biggest_win": 0, "biggest_loss": 0, "total_pnl": 0}
        equity = 0.0

    # ── Quant Critique ─────────────────────────────────────────────────────────
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
    if regime == "transitional":
        critique_lines.append("Market is in transitional regime — bot is holding, which is correct behavior.")
    if adx_val < 15:
        critique_lines.append(f"ADX={adx_val:.1f} is very low — choppy market, expect more HOLD signals.")
    if hurst_val < 0.45:
        critique_lines.append(f"Hurst={hurst_val:.2f} indicates mean-reverting conditions — BB strategy should be active if ADX confirms.")

    critique = "\n".join(critique_lines) if critique_lines else "No major concerns at this time."

    # Top 3 action items based on current state
    backlog = []
    if stats["total"] < 10:
        backlog.append("1. Accumulate more trades before drawing statistical conclusions — too few data points.")
    else:
        backlog.append("1. Run walk-forward parameter validation — all indicator periods are unvalidated defaults.")
    backlog.append("2. Add max daily loss limit and drawdown circuit breaker.")
    backlog.append("3. Validate position state on startup by querying OKX open positions.")

    # ── Build Report ───────────────────────────────────────────────────────────
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
