from __future__ import annotations
from loguru import logger


class RiskManager:
    """
    Handles all position sizing and risk controls.

    Parameters
    ----------
    risk_pct               : fraction of equity to risk per trade (default 1.5%)
    atr_multiplier         : stop distance = ATR × multiplier (default 1.5×)
    trail_pct              : trailing stop distance from highest price (default 0.7%)
    max_daily_loss_pct     : halt trading when daily loss exceeds this (default 3%)
    max_consecutive_losses : halt trading after N losses in a row (default 3)
    """

    def __init__(
        self,
        risk_pct: float = 0.015,
        atr_multiplier: float = 1.5,
        trail_pct: float = 0.007,
        max_daily_loss_pct: float = 0.03,
        max_consecutive_losses: int = 3,
    ) -> None:
        self.risk_pct = risk_pct
        self.atr_multiplier = atr_multiplier
        self.trail_pct = trail_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses

    def stop_price(self, entry: float, atr_val: float, side: str) -> float:
        distance = atr_val * self.atr_multiplier
        return round(entry - distance if side == "buy" else entry + distance, 2)

    def position_size(self, equity: float, entry: float, stop: float, atr_val: float = None) -> float:
        """
        Volatility-scaled position sizing.

        Base size = (equity × risk_pct) / stop_distance.
        Then scaled inversely with current volatility relative to a BTC 1H baseline (~0.8% ATR/price).
        High-vol environments reduce size; low-vol environments scale up, capped at 1.5×.
        Hard cap: never exceed 2% of equity in notional value.
        """
        risk_amount = equity * self.risk_pct
        stop_distance = abs(entry - stop)
        if stop_distance < 1e-9:
            return 0.0

        base_size = risk_amount / stop_distance

        if atr_val is not None and entry > 0:
            daily_vol = atr_val / entry          # ATR as fraction of price
            baseline_vol = 0.008                  # BTC 1H long-run average ~0.8%
            vol_scalar = baseline_vol / max(daily_vol, baseline_vol * 0.5)
            vol_scalar = min(vol_scalar, 1.5)     # cap upscaling in ultra-low-vol regimes
            base_size *= vol_scalar

        max_size = (equity * 0.02) / entry        # never exceed 2% notional
        return round(min(base_size, max_size), 6)

    def take_profit_levels(self, entry: float, side: str) -> list[dict]:
        """
        Returns three TP levels as fractions of the ORIGINAL position size:
          TP1 (+1%)  → close 50%
          TP2 (+2%)  → close 30%
          TP3 (+3%)  → activate trailing stop on remaining 20% (no immediate close)
        """
        tiers = [(0.01, 0.50), (0.02, 0.30), (0.03, 0.20)]
        levels = []
        for pct, fraction in tiers:
            price = entry * (1 + pct) if side == "buy" else entry * (1 - pct)
            levels.append({"price": round(price, 2), "fraction": fraction, "hit": False})
        return levels

    def update_trailing_stop(self, current_price: float, state) -> None:
        """Ratchets the trailing stop up as price makes new highs."""
        if not state.trailing_active:
            return
        if current_price > state.trailing_high:
            state.trailing_high = current_price
            new_stop = round(current_price * (1 - self.trail_pct), 2)
            if new_stop > state.trailing_stop:
                state.trailing_stop = new_stop
                logger.info(f"Trailing stop ratcheted → {state.trailing_stop:.2f}")

    def circuit_breaker_triggered(self, state) -> bool:
        if state.daily_pnl_pct <= -self.max_daily_loss_pct:
            logger.warning(
                f"Circuit breaker: daily loss {state.daily_pnl_pct:.2%} "
                f"≥ limit {self.max_daily_loss_pct:.2%} — halting"
            )
            return True
        if state.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(
                f"Circuit breaker: {state.consecutive_losses} consecutive losses — halting"
            )
            return True
        return False
