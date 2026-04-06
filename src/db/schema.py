"""
SQLAlchemy table definitions for the Council trading bot.

Tables
------
cycles          One row per council run (context snapshot + agent outputs + signal).
trades          One row per order placed (entry → exit lifecycle).
portfolio_state Singleton row tracking cash and peak portfolio value.
reflections     One row per post-trade reflection summary.

Usage
-----
    from src.db.schema import get_engine, create_all

    engine = get_engine()          # default: ./council.db
    create_all(engine)             # idempotent — safe to call every startup
"""

import os

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy import MetaData

metadata = MetaData()

from sqlalchemy import Table

cycles = Table(
    "cycles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("asset", String(16), nullable=False),
    Column("signal", String(8), nullable=False),       # BUY | SELL | HOLD
    Column("conviction", String(8), nullable=False),   # high | medium | low
    Column("vetoed", Boolean, nullable=False),
    Column("context_json", Text, nullable=False),
    Column("council_outputs_json", Text, nullable=False),
    Column("deliberation_json", Text, nullable=False),
)

trades = Table(
    "trades",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cycle_id", Integer, nullable=False),       # FK → cycles.id
    Column("symbol", String(16), nullable=False),
    Column("side", String(4), nullable=False),         # BUY | SELL
    Column("entry_price", Float, nullable=False),
    Column("entry_time", DateTime, nullable=False),
    Column("position_size_usd", Float, nullable=False),
    Column("position_size_btc", Float, nullable=False),
    Column("stop_loss", Float, nullable=False),
    Column("take_profit", Float, nullable=True),
    Column("exchange_order_id", String(64), nullable=True),
    Column("sl_order_id", String(64), nullable=True),
    # Filled in when the position is closed
    Column("exit_price", Float, nullable=True),
    Column("exit_time", DateTime, nullable=True),
    Column("pnl_usd", Float, nullable=True),
    Column("pnl_pct", Float, nullable=True),
    Column("status", String(8), nullable=False, default="open"),  # open | closed | stopped
)

portfolio_state = Table(
    "portfolio_state",
    metadata,
    # Always a single row with id=1 (upsert pattern).
    Column("id", Integer, primary_key=True),
    Column("cash_usd", Float, nullable=False),
    Column("peak_value", Float, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

reflections = Table(
    "reflections",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("trade_id", Integer, nullable=False),       # FK → trades.id
    Column("summary", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

_DEFAULT_DB_PATH = os.getenv("COUNCIL_DB_PATH", "council.db")


def get_engine(db_path: str = _DEFAULT_DB_PATH):
    """Return a SQLAlchemy engine.  Use ``db_path=':memory:'`` for tests."""
    return create_engine(f"sqlite:///{db_path}", future=True)


def create_all(engine) -> None:
    """Create all tables if they do not exist (idempotent)."""
    metadata.create_all(engine)
