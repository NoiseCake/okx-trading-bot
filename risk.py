from __future__ import annotations
from loguru import logger


class RiskManager:
    """
    Central place for all risk decisions: how big to size a trade, where to put the stop,
    when to scale out at take-profit levels, and when to stop trading altogether.

    Default parameters (all overridable via constructor):
      risk_pct               = 1.5% of equity risked per trade
      atr_multiplier         = stop placed 1.5× ATR away from entry
      trail_pct              = trailing stop sits 0.7% below the running high
      max_daily_loss_pct     = halt all trading if daily loss exceeds 3%
      max_consecutive_losses = halt after 3 losses in a row (avoid revenge trading)
    """

    def __init__(
        self,
        risk_pct: float = 0.015,
        atr_multiplier: float = 1.5,
        trail_pct: float = 0.007,
        max_daily_loss_pct: float = 0.03,
        max_consecutive_losses: int = 3,
        baseline_vol: float = 0.008,
    ) -> None:
        self.risk_pct               = risk_pct
        self.atr_multiplier         = atr_multiplier
        self.trail_pct              = trail_pct
        self.max_daily_loss_pct     = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        # 1H ATR/price baseline used by the volatility scalar in position_size().
        # Set per-instrument by the caller — BTC ≈ 0.008, ETH ≈ 0.011. Default
        # value is BTC's, kept for backwards compatibility with existing callers.
        self.baseline_vol           = baseline_vol

    # ── Stop loss ─────────────────────────────────────────────────────────────────

    def stop_price(self, entry: float, atr_val: float, side: str) -> float:
        """
        Place the stop 1.5 ATRs away from the entry price.
        ATR (Average True Range) measures typical candle-to-candle volatility,
        so the stop automatically widens in choppy markets and tightens in calm ones.
        Long:  stop = entry - (ATR × 1.5)
        Short: stop = entry + (ATR × 1.5)
        """
        distance = atr_val * self.atr_multiplier
        return round(entry - distance if side == "buy" else entry + distance, 2)

    # ── Position sizing ───────────────────────────────────────────────────────────

    def position_size(self, equity: float, entry: float, stop: float, atr_val: float = None) -> float:
        """
        Volatility-scaled position sizing — two layers:

        Layer 1 — Fixed-risk base size:
          We know our max dollar risk (equity × risk_pct) and our stop distance.
          Size = risk_amount / stop_distance
          This guarantees we lose at most risk_pct of equity if the stop is hit.

        Layer 2 — ATR volatility scalar:
          If current volatility is higher than the BTC 1H long-run average (0.8%),
          we shrink size proportionally — high vol = wider stops = bigger potential loss.
          If vol is below average we can scale up, but capped at 1.5× to avoid oversizing.

        Hard cap: never risk more than 2% of equity in notional value regardless of the above.
        """
        risk_amount   = equity * self.risk_pct
        stop_distance = abs(entry - stop)
        if stop_distance < 1e-9:
            return 0.0          # degenerate case: entry == stop, can't size

        base_size = risk_amount / stop_distance

        if atr_val is not None and entry > 0:
            daily_vol    = atr_val / entry                  # ATR as a fraction of current price
            vol_scalar   = self.baseline_vol / max(daily_vol, self.baseline_vol * 0.5)
            vol_scalar   = min(vol_scalar, 1.5)             # cap upscaling in ultra-low-vol regimes
            base_size   *= vol_scalar

        max_size = (equity * 0.02) / entry           # 2% notional hard cap
        return round(min(base_size, max_size), 6)

    # ── Take-profit ladder ────────────────────────────────────────────────────────

    def take_profit_levels(self, entry: float, side: str) -> list[dict]:
        """
        Split the exit into three tranches to lock in gains progressively:
          TP1 (+1%)  → sell 30% of position — takes a slice of risk off
          TP2 (+2%)  → sell another 40%     — secures the bulk of the profit
          TP3 (+3%)  → activate trailing stop on the remaining 30%
                       (no immediate close — let the winner run until the trail is hit)

        Fractions chosen so >50% of the position survives past TP1, allowing
        winners to clear ≥1R given typical BTC 1H ATR ≈ 0.8% (stop ≈ 1.2%, TP1 = +1%).

        The 'hit' flag is set to True once that level is reached so we don't double-count.
        """
        tiers = [(0.01, 0.30), (0.02, 0.40), (0.03, 0.30)]
        levels = []
        for pct, fraction in tiers:
            price = entry * (1 + pct) if side == "buy" else entry * (1 - pct)
            levels.append({"price": round(price, 2), "fraction": fraction, "hit": False})
        return levels

    # ── Trailing stop ─────────────────────────────────────────────────────────────

    def update_trailing_stop(self, current_price: float, state) -> None:
        """
        Ratchet the trailing stop upward as price makes new highs.
        The stop is always trail_pct (0.7%) below the highest price seen.
        We never move the stop down — 'ratchet' means one-directional (upward only).
        """
        if not state.trailing_active:
            return
        if current_price > state.trailing_high:
            state.trailing_high = current_price
            new_stop = round(current_price * (1 - self.trail_pct), 2)
            # Only move the stop up, never down
            if new_stop > state.trailing_stop:
                state.trailing_stop = new_stop
                logger.info(f"Trailing stop ratcheted → {state.trailing_stop:.2f}")

    # ── Circuit breaker ───────────────────────────────────────────────────────────

    def circuit_breaker_triggered(self, state) -> bool:
        """
        Two independent kill-switches that pause trading for the rest of the day:

        1. Daily loss limit: if we've lost ≥ 3% of equity today, stop — bad days
           tend to keep being bad and larger positions compound the damage.

        2. Consecutive loss streak: 3 losses in a row suggests something is wrong
           with current market conditions or our signals. Pause and reassess.
        """
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
