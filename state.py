from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, fields, asdict
from datetime import date

STATE_FILE = "bot_state.json"


@dataclass
class BotState:
    in_position: bool = False
    side: str = ""                    # "buy" or "sell"
    entry_price: float = 0.0
    stop_price: float = 0.0
    position_size: float = 0.0
    original_position_size: float = 0.0
    take_profits: list = field(default_factory=list)
    trailing_active: bool = False
    trailing_high: float = 0.0
    trailing_stop: float = 0.0
    trade_id: int = 0
    entry_regime: str = ""
    entry_adx: float = 0.0
    entry_hurst: float = 0.0
    entry_rsi: float = 0.0
    entry_atr_val: float = 0.0
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    trade_date: str = ""

    def reset_daily_if_needed(self) -> None:
        today = str(date.today())
        if self.trade_date != today:
            self.daily_pnl_pct = 0.0
            self.trades_today = 0
            self.trade_date = today

    def clear_position(self) -> None:
        self.in_position = False
        self.side = ""
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.position_size = 0.0
        self.original_position_size = 0.0
        self.take_profits = []
        self.trailing_active = False
        self.trailing_high = 0.0
        self.trailing_stop = 0.0
        self.trade_id = 0
        self.entry_regime = ""
        self.entry_adx = 0.0
        self.entry_hurst = 0.0
        self.entry_rsi = 0.0
        self.entry_atr_val = 0.0

    def save(self) -> None:
        with open(STATE_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> BotState:
        if not os.path.exists(STATE_FILE):
            return cls()
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            valid = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in valid})
        except Exception:
            return cls()
