import time
import schedule
import pandas as pd
from loguru import logger

from client import OKXClient
from strategy import CombinedStrategy, parse_candles, atr, adx, hurst_exponent, rsi, sma
from risk import RiskManager
from state import BotState
from trade_log import init_db, log_signal, log_trade_open, log_trade_close, log_balance

# ── Constants ─────────────────────────────────────────────────────────────────────

# Instruments to trade simultaneously — each gets its own TradingBot instance.
INSTRUMENTS = ["BTC-USDT", "ETH-USDT"]

BAR = "1H"   # candle timeframe — all strategy logic is built for 1H bars

# Minimum spot lot sizes on OKX. Orders below these are rejected outright.
_MIN_LOT = {
    "BTC-USDT": 0.00001,
    "ETH-USDT": 0.001,
}

STRATEGY_INTERVAL_MIN = 60    # run the strategy check once per hour (aligned to candle close)
MONITOR_INTERVAL_SEC  = 60    # check stop/TP prices every minute
BALANCE_LOG_INTERVAL  = 6     # log a balance snapshot every N monitor ticks (~6 minutes)


def _fmt_size(size: float) -> str:
    """Format a position size as a plain decimal string (no scientific notation).
    str(4e-06) → '4e-06' which OKX rejects; this gives '0.000004' instead."""
    return f"{size:.8f}".rstrip("0").rstrip(".")


def _run_quant_report() -> None:
    """Trigger the 12-hour performance report and send it to Telegram. Errors are non-fatal."""
    try:
        import quant_report
        quant_report.main()
    except Exception as e:
        logger.error(f"Quant report failed: {e}")


# ── Bot Class ─────────────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self, inst_id: str, bar: str = BAR) -> None:
        self.inst_id  = inst_id
        self.bar      = bar
        self.min_lot  = _MIN_LOT.get(inst_id, 0.00001)
        self.client   = OKXClient()
        self.strategy = CombinedStrategy()
        self.risk     = RiskManager()
        self.state    = BotState.load(inst_id)
        self.state.reset_daily_if_needed()
        self._reconcile_state()           # sync against live OKX balance before trading
        self.state.save()
        self._monitor_tick = 0

    # ── Helpers ───────────────────────────────────────────────────────────────────

    def _reconcile_state(self) -> None:
        """
        Compare the saved state against the live OKX balance on every startup.
        Catches three failure modes from Railway container restarts:

          1. State says in_position but OKX has no balance → closed externally; clear state.
          2. State says in_position but sizes differ        → partial-close desync; correct size.
          3. State says flat but OKX has balance            → unknown manual position; warn only.
        """
        base_ccy = self.inst_id.split("-")[0]   # "BTC" or "ETH"
        try:
            balance = self.client.get_asset_balance(base_ccy)
        except Exception as e:
            logger.warning(f"[{self.inst_id}] Reconciliation skipped — could not fetch {base_ccy} balance: {e}")
            return

        if self.state.in_position:
            if balance < self.min_lot:
                logger.warning(
                    f"[{self.inst_id}] Reconciliation: state is in_position but {base_ccy} "
                    f"balance is {balance:.8f} — closed externally. Clearing state."
                )
                self.state.clear_position()
            elif abs(balance - self.state.position_size) > self.min_lot:
                logger.warning(
                    f"[{self.inst_id}] Reconciliation: position_size mismatch — "
                    f"state={self.state.position_size} OKX={balance:.8f}. Correcting."
                )
                self.state.position_size = round(balance, 8)
            else:
                logger.info(f"[{self.inst_id}] Reconciliation: position confirmed — {balance:.8f} {base_ccy}")
        else:
            if balance >= self.min_lot:
                # Adopt the existing balance as a bot-managed long position.
                # Use current price as entry proxy and set ATR-based stop/TPs from there.
                try:
                    raw   = self.client.get_candlesticks(self.inst_id, bar=self.bar, limit=200)
                    df    = parse_candles(raw)
                    price = df["close"].iloc[-1]
                    atr_v = atr(df).iloc[-1]
                except Exception as e:
                    logger.warning(
                        f"[{self.inst_id}] Reconciliation: found {balance:.8f} {base_ccy} "
                        f"but could not fetch price/ATR to adopt position: {e}"
                    )
                    return

                stop = self.risk.stop_price(price, atr_v, "buy")
                tps  = self.risk.take_profit_levels(price, "buy")

                self.state.in_position            = True
                self.state.side                   = "buy"
                self.state.entry_price            = price
                self.state.stop_price             = stop
                self.state.position_size          = round(balance, 8)
                self.state.original_position_size = round(balance, 8)
                self.state.take_profits           = tps
                self.state.save()
                logger.info(
                    f"[{self.inst_id}] Reconciliation: adopted {balance:.8f} {base_ccy} "
                    f"as managed long — entry≈{price:.2f}, stop={stop:.2f}"
                )
            else:
                logger.info(f"[{self.inst_id}] Reconciliation: flat state confirmed")

    def _equity(self) -> float:
        """Fetch available USDT balance from OKX. Returns 0.0 on failure (non-fatal)."""
        try:
            balance = self.client.get_balance()
            for detail in balance.get("details", []):
                if detail.get("ccy") == "USDT":
                    return float(detail.get("availEq") or detail.get("availBal", 0))
        except Exception as e:
            logger.error(f"[{self.inst_id}] Failed to fetch equity: {e}")
        return 0.0

    def _record_close(self, pnl_pct: float) -> None:
        """Update daily PnL and consecutive-loss counter after every trade close."""
        self.state.daily_pnl_pct += pnl_pct
        if pnl_pct < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def _close_position(self, price: float, reason: str) -> None:
        """
        Submit a market order to exit the entire remaining position.
        Logs the trade to the database, updates daily counters, and clears state.
        If the order fails we log the error but don't crash — the monitor will retry next tick.
        """
        exit_side = "sell" if self.state.side == "buy" else "buy"

        # Guard: if tracked size is dust/zero (e.g. state desync after a Railway restart),
        # clear state without placing an order rather than sending a bad request to OKX.
        if self.state.position_size < self.min_lot:
            logger.warning(
                f"[{self.inst_id}] position_size {self.state.position_size} is below minimum — "
                f"clearing state without order ({reason})"
            )
            self.state.clear_position()
            self.state.save()
            return

        try:
            self.client.place_market_order(self.inst_id, exit_side, _fmt_size(self.state.position_size))
        except Exception as e:
            logger.error(f"[{self.inst_id}] Close order failed ({reason}): {e}")
            return

        if self.state.side == "buy":
            pnl_pct = (price - self.state.entry_price) / self.state.entry_price
        else:
            pnl_pct = (self.state.entry_price - price) / self.state.entry_price

        logger.info(f"[{self.inst_id}] Position closed [{reason}] at {price:.2f} | PnL: {pnl_pct:+.2%}")

        if self.state.trade_id:
            log_trade_close(
                trade_id=self.state.trade_id,
                exit_price=price,
                close_size=self.state.position_size,
                entry_price=self.state.entry_price,
                side=self.state.side,
                close_reason=reason,
            )

        self._record_close(pnl_pct)
        self.state.clear_position()
        self.state.save()

    def _partial_close(self, fraction: float, price: float, label: str) -> None:
        """
        Sell a fraction of the original position size (e.g. 50% at TP1).
        We track position_size so subsequent partials and the final trailing stop
        only touch the remaining quantity.
        """
        close_size = round(self.state.original_position_size * fraction, 8)
        exit_side  = "sell" if self.state.side == "buy" else "buy"
        try:
            self.client.place_market_order(self.inst_id, exit_side, _fmt_size(close_size))
        except Exception as e:
            logger.error(f"[{self.inst_id}] Partial close failed ({label}): {e}")
            return
        self.state.position_size = round(self.state.position_size - close_size, 8)
        self.state.save()   # persist immediately so a Railway restart can't desync position_size
        logger.info(
            f"[{self.inst_id}] {label} hit at {price:.2f} | Closed {fraction:.0%} ({close_size}) "
            f"| Remaining: {self.state.position_size}"
        )

    # ── Position Monitor (runs every minute) ──────────────────────────────────────

    def monitor_position(self) -> None:
        """
        Check live price every minute and handle:
          - Daily reset (midnight UTC)
          - Periodic balance snapshot
          - Stop-loss exit
          - Take-profit levels (partial closes + trailing activation)
          - Trailing stop exit
        """
        self._monitor_tick += 1

        self.state.reset_daily_if_needed()

        # Log a balance snapshot every ~6 minutes (not every tick to reduce DB writes)
        if self._monitor_tick % BALANCE_LOG_INTERVAL == 0:
            equity = self._equity()
            if equity > 0:
                log_balance(equity)

        if not self.state.in_position:
            return

        try:
            ticker = self.client.get_ticker(self.inst_id)
            price  = float(ticker["last"])
        except Exception as e:
            logger.error(f"[{self.inst_id}] Ticker fetch failed: {e}")
            return

        # Fetch the current 1m candle to catch intra-minute wicks that may have
        # touched a stop or TP level since the last poll. Falls back to ticker
        # price alone if the fetch fails.
        wick_low = wick_high = price
        try:
            raw_1m = self.client.get_candlesticks(self.inst_id, bar="1m", limit=2)
            if raw_1m:
                c = raw_1m[0]                          # most recent candle (may still be forming)
                wick_low  = min(price, float(c[3]))    # c[3] = low
                wick_high = max(price, float(c[2]))    # c[2] = high
        except Exception:
            pass

        is_long = self.state.side == "buy"

        # ── Stop loss ─────────────────────────────────────────────────────────────
        check_price = wick_low if is_long else wick_high
        if (is_long and check_price <= self.state.stop_price) or \
           (not is_long and check_price >= self.state.stop_price):
            self._close_position(price, "STOP LOSS")
            return

        # ── Take-profit levels ────────────────────────────────────────────────────
        for i, tp in enumerate(self.state.take_profits):
            if tp["hit"]:
                continue

            hit = wick_high >= tp["price"] if is_long else wick_low <= tp["price"]
            if not hit:
                continue

            tp["hit"] = True

            if i == len(self.state.take_profits) - 1:
                # Final TP — activate trailing stop anchored to the wick extreme
                activation_price = wick_high if is_long else wick_low
                self.state.trailing_active = True
                self.state.trailing_high   = activation_price
                self.state.trailing_stop   = round(activation_price * (1 - self.risk.trail_pct), 2)
                logger.info(f"[{self.inst_id}] TP{i+1} reached — trailing stop at {self.state.trailing_stop:.2f}")
            else:
                self._partial_close(tp["fraction"], price, f"TP{i+1}")

        # ── Trailing stop ─────────────────────────────────────────────────────────
        if self.state.trailing_active:
            ratchet_price = wick_high if is_long else wick_low
            self.risk.update_trailing_stop(ratchet_price, self.state)
            trail_check = wick_low if is_long else wick_high
            if (is_long and trail_check <= self.state.trailing_stop) or \
               (not is_long and trail_check >= self.state.trailing_stop):
                self._close_position(price, "TRAILING STOP")
                return

        self.state.save()

    # ── Strategy Check (runs every 60 minutes) ────────────────────────────────────

    def run_strategy(self) -> None:
        """
        Called once per candle close (every hour). Evaluates the strategy and
        opens a new position if conditions are met.

        Skipped if:
          - We're already in a position (one trade at a time per instrument)
          - The circuit breaker is active (daily loss or consecutive loss limit hit)
        """
        self.state.reset_daily_if_needed()

        if self.state.in_position:
            logger.info(f"[{self.inst_id}] Already in position — skipping entry check")
            return

        if self.risk.circuit_breaker_triggered(self.state):
            return

        logger.info(f"[{self.inst_id}] Strategy check — {self.bar}")
        try:
            raw = self.client.get_candlesticks(self.inst_id, bar=self.bar, limit=200)
            df  = parse_candles(raw)

            # Daily candles for the macro trend filter (SMA200 on 1D bars ≈ 200-day average)
            raw_daily   = self.client.get_candlesticks(self.inst_id, bar="1D", limit=250)
            df_daily    = parse_candles(raw_daily)
            curr_sma200 = sma(df_daily["close"], 200).iloc[-1]
            sma200_str  = f"{curr_sma200:.2f}" if not pd.isna(curr_sma200) else "N/A"

            # ── Compute indicators for logging and signal generation ───────────────
            atr_val            = atr(df).iloc[-1]
            adx_line, _, _     = adx(df)
            curr_adx           = adx_line.iloc[-1]
            window             = df["close"].iloc[-168:] if len(df) >= 168 else df["close"]
            curr_hurst         = hurst_exponent(window)
            curr_rsi           = rsi(df["close"]).iloc[-1]
            price              = df["close"].iloc[-1]

            if curr_adx > CombinedStrategy.TREND_THRESHOLD:
                regime = "trending"
            elif curr_adx < CombinedStrategy.RANGE_THRESHOLD:
                regime = "ranging"
            else:
                regime = "transitional"

            signal = self.strategy.signal(df)
            logger.info(
                f"[{self.inst_id}] Signal: {signal.upper()} | Regime: {regime} | ADX: {curr_adx:.1f} "
                f"| H: {curr_hurst:.2f} | RSI: {curr_rsi:.1f} | Price: {price:.2f} "
                f"| SMA200d: {sma200_str}"
            )

            log_signal(self.inst_id, signal, regime, curr_adx, curr_hurst, curr_rsi, atr_val, price)

            if signal != "buy":
                return

            if pd.isna(atr_val) or atr_val <= 0:
                logger.warning(f"[{self.inst_id}] ATR unavailable — skipping trade")
                return

            # ── Macro trend filter: daily SMA200 ─────────────────────────────────
            if not pd.isna(curr_sma200) and price < curr_sma200:
                logger.info(f"[{self.inst_id}] Buy filtered — price {price:.2f} below daily SMA200 {curr_sma200:.2f}")
                return

            # ── 4H multi-timeframe confirmation ───────────────────────────────────
            # Fetched lazily — only when 1H signal is actionable and passed SMA200
            try:
                raw_4h    = self.client.get_candlesticks(self.inst_id, bar="4H", limit=200)
                df_4h     = parse_candles(raw_4h)
                signal_4h = self.strategy.signal(df_4h)
            except Exception as e:
                logger.warning(f"[{self.inst_id}] 4H fetch failed — skipping MTF check: {e}")
                signal_4h = signal   # fail open

            if signal_4h != signal:
                logger.info(f"[{self.inst_id}] 1H {signal.upper()} not confirmed on 4H ({signal_4h.upper()}) — skipping")
                return
            logger.info(f"[{self.inst_id}] 4H confirmed: {signal_4h.upper()}")

            equity = self._equity()
            if equity <= 0:
                logger.error(f"[{self.inst_id}] Could not retrieve equity — skipping trade")
                return

            # ── Calculate entry parameters ────────────────────────────────────────
            stop = self.risk.stop_price(price, atr_val, signal)
            size = self.risk.position_size(equity, price, stop, atr_val)

            if size <= 0:
                logger.warning(f"[{self.inst_id}] Position size is zero — skipping")
                return

            # ── Place the entry order ─────────────────────────────────────────────
            self.client.place_market_order(self.inst_id, signal, _fmt_size(size))

            trade_id = log_trade_open(
                inst_id=self.inst_id,
                side=signal,
                entry_price=price,
                size=size,
                regime=regime,
                adx_val=curr_adx,
                hurst_val=curr_hurst,
                rsi_val=curr_rsi,
                atr_val=atr_val,
            )

            # ── Update state so the monitor knows what to watch ───────────────────
            self.state.in_position            = True
            self.state.side                   = signal
            self.state.entry_price            = price
            self.state.stop_price             = stop
            self.state.position_size          = size
            self.state.original_position_size = size
            self.state.take_profits           = self.risk.take_profit_levels(price, signal)
            self.state.trailing_active        = False
            self.state.trailing_high          = price
            self.state.trailing_stop          = 0.0
            self.state.trade_id               = trade_id
            self.state.entry_regime           = regime
            self.state.entry_adx              = curr_adx
            self.state.entry_hurst            = curr_hurst
            self.state.entry_rsi              = curr_rsi
            self.state.entry_atr_val          = atr_val
            self.state.trades_today           += 1
            self.state.save()

            logger.info(f"[{self.inst_id}] Entered {signal.upper()} #{trade_id} | Size: {size} | Stop: {stop:.2f}")
            for j, tp in enumerate(self.state.take_profits):
                logger.info(f"[{self.inst_id}]   TP{j+1}: {tp['price']:.2f} ({tp['fraction']:.0%})")

        except Exception as e:
            logger.error(f"[{self.inst_id}] Strategy error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Create a TradingBot for each instrument in INSTRUMENTS, wire up the shared
    scheduler, and run the event loop forever.

    Tasks per instrument:
      - run_strategy    : every 60 minutes (1H candle close)
      - monitor_position: every 60 seconds (stop/TP check)

    Shared task:
      - quant report    : every 12 hours (Telegram summary)
    """
    init_db()

    bots = [TradingBot(inst_id) for inst_id in INSTRUMENTS]
    logger.info(f"Bot started — instruments: {INSTRUMENTS}")

    for bot in bots:
        bot.run_strategy()   # run immediately on startup, don't wait for the first interval
        schedule.every(STRATEGY_INTERVAL_MIN).minutes.do(bot.run_strategy)
        schedule.every(MONITOR_INTERVAL_SEC).seconds.do(bot.monitor_position)

    schedule.every(12).hours.do(_run_quant_report)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
