from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone

# DATA_DIR points to a Railway persistent volume in production so state survives restarts.
# Locally it defaults to the current directory.
DATA_DIR = os.getenv("DATA_DIR", ".")


def _state_file(inst_id: str) -> str:
    """Derive the per-instrument state file path, e.g. bot_state_BTC_USDT.json."""
    return os.path.join(DATA_DIR, f"bot_state_{inst_id.replace('-', '_')}.json")


@dataclass
class BotState:
    """
    All runtime state the bot needs to survive a restart.
    Saved to a per-instrument JSON file after every position change.
    """

    # ── Instrument ───────────────────────────────────────────────────────────────
    inst_id: str = "BTC-USDT"

    # ── Position tracking ────────────────────────────────────────────────────────
    in_position: bool  = False        # True while we hold an open trade
    side: str          = ""           # "buy" (long) or "sell" (short)
    entry_price: float = 0.0          # price at which we entered the trade
    stop_price: float  = 0.0          # price that triggers the stop-loss exit
    position_size: float          = 0.0  # current quantity (shrinks after partial closes)
    original_position_size: float = 0.0  # full size at entry (used to calculate partial-close fractions)

    # ── Take-profit ladder ───────────────────────────────────────────────────────
    take_profits: list = field(default_factory=list)

    # ── Trailing stop ────────────────────────────────────────────────────────────
    trailing_active: bool  = False
    trailing_high: float   = 0.0
    trailing_stop: float   = 0.0

    # ── Trade metadata ───────────────────────────────────────────────────────────
    trade_id: int        = 0
    entry_regime: str    = ""
    entry_adx: float     = 0.0
    entry_hurst: float   = 0.0
    entry_rsi: float     = 0.0
    entry_atr_val: float = 0.0

    # ── Daily risk counters (reset each UTC day) ─────────────────────────────────
    daily_pnl_pct: float    = 0.0
    consecutive_losses: int = 0
    trades_today: int       = 0
    trade_date: str         = ""

    # ── Daily reset ──────────────────────────────────────────────────────────────

    def reset_daily_if_needed(self) -> None:
        """Zero out daily counters when the UTC calendar date has changed.

        Uses UTC to stay consistent with trade_log's UTC timestamps and the
        cross-instrument breaker (daily_realized_pnl_usdt) — otherwise the two
        could disagree on which 'day' it is if the host TZ isn't UTC.
        """
        today = str(datetime.now(timezone.utc).date())
        if self.trade_date != today:
            self.daily_pnl_pct        = 0.0
            self.trades_today         = 0
            self.consecutive_losses   = 0
            self.trade_date           = today

    # ── Position cleanup ─────────────────────────────────────────────────────────

    def clear_position(self) -> None:
        """Reset all position fields back to defaults after a trade is closed."""
        self.in_position            = False
        self.side                   = ""
        self.entry_price            = 0.0
        self.stop_price             = 0.0
        self.position_size          = 0.0
        self.original_position_size = 0.0
        self.take_profits           = []
        self.trailing_active        = False
        self.trailing_high          = 0.0
        self.trailing_stop          = 0.0
        self.trade_id               = 0
        self.entry_regime           = ""
        self.entry_adx              = 0.0
        self.entry_hurst            = 0.0
        self.entry_rsi              = 0.0
        self.entry_atr_val          = 0.0

    # ── Persistence ──────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write the full state to the instrument's JSON file atomically.

        Writes to a sibling .tmp path then os.replace() onto the real file so
        a SIGKILL mid-write can't leave a zero-byte JSON. The previous version
        (open + truncate + write) could yield an empty file that load() then
        silently fell back to default state — bot would think it's flat while
        OKX still held the position.
        """
        path = _state_file(self.inst_id)
        tmp  = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, inst_id: str = "BTC-USDT") -> BotState:
        """
        Read state from disk. Falls back to a fresh default state if the file is
        missing or corrupted.

        Backwards compatibility: if the new per-instrument file doesn't exist for
        BTC-USDT, falls back to the legacy bot_state.json so existing Railway
        deployments don't lose their state on the first deploy after this change.
        """
        path = _state_file(inst_id)
        if not os.path.exists(path) and inst_id == "BTC-USDT":
            legacy = os.path.join(DATA_DIR, "bot_state.json")
            if os.path.exists(legacy):
                path = legacy

        if not os.path.exists(path):
            return cls(inst_id=inst_id)
        try:
            with open(path) as f:
                data = json.load(f)
            valid = {f.name for f in fields(cls)}
            state = cls(**{k: v for k, v in data.items() if k in valid})
            state.inst_id = inst_id   # always authoritative from the parameter
            return state
        except Exception:
            return cls(inst_id=inst_id)


@dataclass
class XsmomState:
    """Runtime state for the X1 weekly-rebalance mode. One file for the whole
    portfolio (the strategy is portfolio-level, unlike the per-instrument 1H
    bot). Only two facts must survive a restart: which rebalance was last
    completed, and what it targeted."""

    last_rebalance: str = ""                       # ISO date of the acted-on close
    targets: dict = field(default_factory=dict)    # inst_id -> weight at that close

    _FILE = "bot_state_xsmom.json"

    def save(self) -> None:
        path = os.path.join(DATA_DIR, self._FILE)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"last_rebalance": self.last_rebalance, "targets": self.targets},
                      f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls) -> XsmomState:
        path = os.path.join(DATA_DIR, cls._FILE)
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(last_rebalance=str(data.get("last_rebalance", "")),
                       targets=dict(data.get("targets", {})))
        except Exception:
            return cls()
