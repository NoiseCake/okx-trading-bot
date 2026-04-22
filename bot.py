import time
import schedule
import pandas as pd
from loguru import logger

from client import OKXClient
from strategy import CombinedStrategy, parse_candles, atr, adx, hurst_exponent, rsi
from risk import RiskManager
from state import BotState
from trade_log import init_db, log_signal, log_trade_open, log_trade_close, log_balance

INST_ID = "BTC-USDT"
BAR = "1H"
STRATEGY_INTERVAL_MIN = 60   # align with 1H candle close
MONITOR_INTERVAL_SEC = 60    # check stop/TP every minute
BALANCE_LOG_INTERVAL = 6     # log balance every N monitor ticks (~6 min)

_monitor_tick = 0


class TradingBot:
    def __init__(self) -> None:
        init_db()
        self.client = OKXClient()
        self.strategy = CombinedStrategy()
        self.risk = RiskManager()
        self.state = BotState.load()
        self.state.reset_daily_if_needed()
        self.state.save()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _equity(self) -> float:
        try:
            balance = self.client.get_balance()
            for detail in balance.get("details", []):
                if detail.get("ccy") == "USDT":
                    return float(detail.get("availEq") or detail.get("availBal", 0))
        except Exception as e:
            logger.error(f"Failed to fetch equity: {e}")
        return 0.0

    def _record_close(self, pnl_pct: float) -> None:
        self.state.daily_pnl_pct += pnl_pct
        if pnl_pct < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def _close_position(self, price: float, reason: str) -> None:
        exit_side = "sell" if self.state.side == "buy" else "buy"
        try:
            self.client.place_market_order(INST_ID, exit_side, str(self.state.position_size))
        except Exception as e:
            logger.error(f"Close order failed ({reason}): {e}")
            return

        if self.state.side == "buy":
            pnl_pct = (price - self.state.entry_price) / self.state.entry_price
        else:
            pnl_pct = (self.state.entry_price - price) / self.state.entry_price

        logger.info(f"Position closed [{reason}] at {price:.2f} | PnL: {pnl_pct:+.2%}")

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
        close_size = round(self.state.original_position_size * fraction, 6)
        exit_side = "sell" if self.state.side == "buy" else "buy"
        try:
            self.client.place_market_order(INST_ID, exit_side, str(close_size))
        except Exception as e:
            logger.error(f"Partial close failed ({label}): {e}")
            return
        self.state.position_size = round(self.state.position_size - close_size, 6)
        logger.info(f"{label} hit at {price:.2f} | Closed {fraction:.0%} ({close_size} BTC) | Remaining: {self.state.position_size}")

    # ── Position Monitor (runs every minute) ─────────────────────────────────

    def monitor_position(self) -> None:
        global _monitor_tick
        _monitor_tick += 1

        self.state.reset_daily_if_needed()

        # Periodic balance snapshot
        if _monitor_tick % BALANCE_LOG_INTERVAL == 0:
            equity = self._equity()
            if equity > 0:
                log_balance(equity)

        if not self.state.in_position:
            return

        try:
            ticker = self.client.get_ticker(INST_ID)
            price = float(ticker["last"])
        except Exception as e:
            logger.error(f"Ticker fetch failed: {e}")
            return

        is_long = self.state.side == "buy"

        # ── Stop loss ────────────────────────────────────────────────────────
        stop_hit = price <= self.state.stop_price if is_long else price >= self.state.stop_price
        if stop_hit:
            self._close_position(price, "STOP LOSS")
            return

        # ── Take profit levels ───────────────────────────────────────────────
        for i, tp in enumerate(self.state.take_profits):
            if tp["hit"]:
                continue
            hit = price >= tp["price"] if is_long else price <= tp["price"]
            if not hit:
                continue
            tp["hit"] = True
            if i == len(self.state.take_profits) - 1:
                self.state.trailing_active = True
                self.state.trailing_high = price
                self.state.trailing_stop = round(price * (1 - self.risk.trail_pct), 2)
                logger.info(f"TP{i+1} reached at {price:.2f} — trailing stop at {self.state.trailing_stop:.2f}")
            else:
                self._partial_close(tp["fraction"], price, f"TP{i+1}")

        # ── Trailing stop ────────────────────────────────────────────────────
        if self.state.trailing_active:
            self.risk.update_trailing_stop(price, self.state)
            trail_hit = price <= self.state.trailing_stop if is_long else price >= self.state.trailing_stop
            if trail_hit:
                self._close_position(price, "TRAILING STOP")
                return

        self.state.save()

    # ── Strategy Check (runs every 60 min) ────────────────────────────────────

    def run_strategy(self) -> None:
        self.state.reset_daily_if_needed()

        if self.state.in_position:
            logger.info("Already in position — skipping entry check")
            return

        if self.risk.circuit_breaker_triggered(self.state):
            return

        logger.info(f"Strategy check — {INST_ID} {BAR}")
        try:
            raw = self.client.get_candlesticks(INST_ID, bar=BAR, limit=200)
            df = parse_candles(raw)

            # ── Compute context for logging ───────────────────────────────────
            atr_val = atr(df).iloc[-1]
            adx_line, _, _ = adx(df)
            curr_adx = adx_line.iloc[-1]
            window = df["close"].iloc[-168:] if len(df) >= 168 else df["close"]
            curr_hurst = hurst_exponent(window)
            curr_rsi = rsi(df["close"]).iloc[-1]
            price = df["close"].iloc[-1]

            if curr_adx > CombinedStrategy.TREND_THRESHOLD:
                regime = "trending"
            elif curr_adx < CombinedStrategy.RANGE_THRESHOLD:
                regime = "ranging"
            else:
                regime = "transitional"

            signal = self.strategy.signal(df)
            logger.info(
                f"Signal: {signal.upper()} | Regime: {regime} | ADX: {curr_adx:.1f} "
                f"| H: {curr_hurst:.2f} | RSI: {curr_rsi:.1f} | Price: {price:.2f}"
            )

            log_signal(signal, regime, curr_adx, curr_hurst, curr_rsi, atr_val, price)

            if signal not in ("buy", "sell"):
                return

            if pd.isna(atr_val) or atr_val <= 0:
                logger.warning("ATR unavailable — skipping trade")
                return

            equity = self._equity()
            if equity <= 0:
                logger.error("Could not retrieve equity — skipping trade")
                return

            stop = self.risk.stop_price(price, atr_val, signal)
            size = self.risk.position_size(equity, price, stop, atr_val)

            if size <= 0:
                logger.warning("Position size is zero — skipping")
                return

            self.client.place_market_order(INST_ID, signal, str(size))

            trade_id = log_trade_open(
                side=signal,
                entry_price=price,
                size=size,
                regime=regime,
                adx_val=curr_adx,
                hurst_val=curr_hurst,
                rsi_val=curr_rsi,
                atr_val=atr_val,
            )

            self.state.in_position = True
            self.state.side = signal
            self.state.entry_price = price
            self.state.stop_price = stop
            self.state.position_size = size
            self.state.original_position_size = size
            self.state.take_profits = self.risk.take_profit_levels(price, signal)
            self.state.trailing_active = False
            self.state.trailing_high = price
            self.state.trailing_stop = 0.0
            self.state.trade_id = trade_id
            self.state.entry_regime = regime
            self.state.entry_adx = curr_adx
            self.state.entry_hurst = curr_hurst
            self.state.entry_rsi = curr_rsi
            self.state.entry_atr_val = atr_val
            self.state.trades_today += 1
            self.state.save()

            logger.info(f"Entered {signal.upper()} #{trade_id} | Size: {size} BTC | Stop: {stop:.2f}")
            for j, tp in enumerate(self.state.take_profits):
                logger.info(f"  TP{j+1}: {tp['price']:.2f} ({tp['fraction']:.0%})")

        except Exception as e:
            logger.error(f"Strategy error: {e}")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info(f"Bot started — {INST_ID} | Risk: {self.risk.risk_pct:.1%}/trade")
        self.run_strategy()
        schedule.every(STRATEGY_INTERVAL_MIN).minutes.do(self.run_strategy)
        schedule.every(MONITOR_INTERVAL_SEC).seconds.do(self.monitor_position)
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    TradingBot().start()
