"""
SQLite trade logger.

Three tables:
  trades            — one row per trade, filled in at open and updated at close
  signals           — every strategy evaluation (buy / sell / hold), including non-trades
  balance_snapshots — periodic equity readings used to track overall account growth
"""

from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from loguru import logger

# Use DATA_DIR so Railway can point this at a persistent volume (survives deploys/restarts).
# Locally defaults to the current working directory.
DATA_DIR = os.getenv("DATA_DIR", ".")
DB_FILE  = os.path.join(DATA_DIR, "trades.db")


@contextmanager
def _conn():
    """
    Context manager that opens a SQLite connection, commits on success,
    and always closes the connection — even if an exception is raised.
    """
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """
    Create all tables if they don't already exist. Safe to call on every startup.
    Also runs a one-time migration to add the inst_id column to tables created
    before multi-instrument support was added.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                inst_id         TEXT NOT NULL DEFAULT 'BTC-USDT',
                opened_at       TEXT,       -- UTC ISO timestamp of entry
                closed_at       TEXT,       -- UTC ISO timestamp of exit (NULL while open)
                side            TEXT,       -- 'buy' or 'sell'
                entry_price     REAL,
                exit_price      REAL,
                original_size   REAL,       -- full position size at entry
                close_size      REAL,       -- how much was closed (may differ after partials)
                pnl_pct         REAL,       -- profit/loss as a decimal fraction (e.g. 0.015 = +1.5%)
                pnl_usdt        REAL,       -- raw dollar profit/loss
                close_reason    TEXT,       -- 'STOP LOSS', 'TRAILING STOP', 'TP1' … etc.
                regime          TEXT,       -- market regime at entry
                adx             REAL,
                hurst           REAL,
                entry_rsi       REAL,
                entry_atr       REAL
            );

            CREATE TABLE IF NOT EXISTS signals (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                inst_id TEXT NOT NULL DEFAULT 'BTC-USDT',
                ts      TEXT,       -- UTC timestamp
                signal  TEXT,       -- 'buy', 'sell', or 'hold'
                regime  TEXT,
                adx     REAL,
                hurst   REAL,
                rsi     REAL,
                atr     REAL,
                price   REAL
            );

            CREATE TABLE IF NOT EXISTS balance_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT,
                equity_usdt REAL    -- total USDT value of account at snapshot time
            );
        """)

    # Migration: add inst_id to tables that existed before multi-instrument support
    with _conn() as con:
        for table in ("trades", "signals"):
            try:
                con.execute(f"ALTER TABLE {table} ADD COLUMN inst_id TEXT NOT NULL DEFAULT 'BTC-USDT'")
            except Exception:
                pass    # column already exists

    logger.info(f"Trade database ready: {DB_FILE}")


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Signals ───────────────────────────────────────────────────────────────────────

def log_signal(
    inst_id: str,
    signal: str,
    regime: str,
    adx_val: float,
    hurst_val: float,
    rsi_val: float,
    atr_val: float,
    price: float,
) -> None:
    """Record every strategy evaluation, including holds."""
    with _conn() as con:
        con.execute(
            "INSERT INTO signals (inst_id,ts,signal,regime,adx,hurst,rsi,atr,price) VALUES (?,?,?,?,?,?,?,?,?)",
            (inst_id, _now(), signal, regime, adx_val, hurst_val, rsi_val, atr_val, price),
        )


# ── Trades ────────────────────────────────────────────────────────────────────────

def log_trade_open(
    inst_id: str,
    side: str,
    entry_price: float,
    size: float,
    regime: str,
    adx_val: float,
    hurst_val: float,
    rsi_val: float,
    atr_val: float,
) -> int:
    """
    Insert a new row when a trade is opened. Returns the auto-generated trade ID
    so the bot can store it in state and update the same row when the trade closes.
    """
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO trades
               (inst_id, opened_at, side, entry_price, original_size, regime, adx, hurst, entry_rsi, entry_atr)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (inst_id, _now(), side, entry_price, size, regime, adx_val, hurst_val, rsi_val, atr_val),
        )
        return cur.lastrowid


def log_trade_close(
    trade_id: int,
    exit_price: float,
    close_size: float,
    entry_price: float,
    side: str,
    close_reason: str,
) -> None:
    """
    Fill in the exit fields on the existing trade row.
    PnL is calculated here so it's always consistent with how the bot measured it.
    """
    if side == "buy":
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    pnl_usdt = pnl_pct * entry_price * close_size

    with _conn() as con:
        con.execute(
            """UPDATE trades
               SET closed_at=?, exit_price=?, close_size=?, pnl_pct=?, pnl_usdt=?, close_reason=?
               WHERE id=?""",
            (_now(), exit_price, close_size, pnl_pct, pnl_usdt, close_reason, trade_id),
        )
    logger.info(f"Trade #{trade_id} logged — PnL: {pnl_pct:+.2%} ({pnl_usdt:+.2f} USDT)")


def daily_realized_pnl_usdt() -> float:
    """
    Sum pnl_usdt across every trade closed today (UTC). Used by the
    cross-instrument circuit breaker so a synchronised drawdown on BTC and
    ETH can't bypass the per-instrument daily-loss limit by 2×.

    Only realised PnL is counted; open positions that are deeply underwater
    will not trigger the breaker until they actually close.
    """
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _conn() as con:
        row = con.execute(
            "SELECT COALESCE(SUM(pnl_usdt), 0) FROM trades "
            "WHERE closed_at IS NOT NULL AND substr(closed_at, 1, 10) = ?",
            (today_utc,),
        ).fetchone()
    return float(row[0])


def find_last_open_trade(inst_id: str) -> dict | None:
    """
    Return the most recent unclosed trade row for this instrument, or None.

    Used by reconciliation on bot startup to recover the original entry price
    and ATR after a Railway restart, so stops/TPs can be rebuilt from the
    real entry rather than fabricated from current price.
    """
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM trades WHERE inst_id=? AND closed_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (inst_id,),
        ).fetchone()
    return dict(row) if row else None


# ── Balance ───────────────────────────────────────────────────────────────────────

def log_balance(equity_usdt: float) -> None:
    """
    Snapshot the current account equity. Called every ~6 minutes by the monitor loop.
    These rows are used in the quant report to show equity curve over time.
    """
    with _conn() as con:
        con.execute(
            "INSERT INTO balance_snapshots (ts, equity_usdt) VALUES (?,?)",
            (_now(), equity_usdt),
        )
