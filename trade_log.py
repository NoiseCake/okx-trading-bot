"""
SQLite trade logger.

Tables
------
trades           : one row per trade, updated on close
signals          : every strategy check result (including holds)
balance_snapshots: periodic equity snapshots
"""

from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from loguru import logger

# Respect DATA_DIR env var so Railway can point to a persistent volume
DATA_DIR = os.getenv("DATA_DIR", ".")
DB_FILE = os.path.join(DATA_DIR, "trades.db")


@contextmanager
def _conn():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                opened_at       TEXT,
                closed_at       TEXT,
                side            TEXT,
                entry_price     REAL,
                exit_price      REAL,
                original_size   REAL,
                close_size      REAL,
                pnl_pct         REAL,
                pnl_usdt        REAL,
                close_reason    TEXT,
                regime          TEXT,
                adx             REAL,
                hurst           REAL,
                entry_rsi       REAL,
                entry_atr       REAL
            );

            CREATE TABLE IF NOT EXISTS signals (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT,
                signal  TEXT,
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
                equity_usdt REAL
            );
        """)
    logger.info(f"Trade database ready: {DB_FILE}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Signals ───────────────────────────────────────────────────────────────────

def log_signal(
    signal: str,
    regime: str,
    adx_val: float,
    hurst_val: float,
    rsi_val: float,
    atr_val: float,
    price: float,
) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO signals (ts,signal,regime,adx,hurst,rsi,atr,price) VALUES (?,?,?,?,?,?,?,?)",
            (_now(), signal, regime, adx_val, hurst_val, rsi_val, atr_val, price),
        )


# ── Trades ────────────────────────────────────────────────────────────────────

def log_trade_open(
    side: str,
    entry_price: float,
    size: float,
    regime: str,
    adx_val: float,
    hurst_val: float,
    rsi_val: float,
    atr_val: float,
) -> int:
    """Returns the trade id to be stored in BotState for later update."""
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO trades
               (opened_at, side, entry_price, original_size, regime, adx, hurst, entry_rsi, entry_atr)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (_now(), side, entry_price, size, regime, adx_val, hurst_val, rsi_val, atr_val),
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


# ── Balance ───────────────────────────────────────────────────────────────────

def log_balance(equity_usdt: float) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO balance_snapshots (ts, equity_usdt) VALUES (?,?)",
            (_now(), equity_usdt),
        )
