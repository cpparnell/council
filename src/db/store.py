"""
Database CRUD operations for the Council trading bot.

All functions accept a SQLAlchemy ``Engine`` so callers can pass an
in-memory engine during tests without touching the real database.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from src.db.schema import cycles, portfolio_state, reflections, trades, weekly_summaries
from src.models import CouncilOutputs, DeliberationOutput, MarketContext

INITIAL_CAPITAL = float(__import__("os").getenv("INITIAL_CAPITAL", "10000"))


# ------------------------------------------------------------------ cycles


def log_cycle(
    engine: Engine,
    ctx: MarketContext,
    outputs: CouncilOutputs,
    deliberation: DeliberationOutput,
) -> int:
    """Insert a cycle row and return its auto-assigned id."""
    with engine.begin() as conn:
        result = conn.execute(
            cycles.insert().values(
                timestamp=ctx.timestamp,
                asset=ctx.asset,
                signal=deliberation.final_signal,
                conviction=deliberation.conviction,
                vetoed=outputs.risk.veto,
                context_json=ctx.model_dump_json(),
                council_outputs_json=outputs.model_dump_json(),
                deliberation_json=deliberation.model_dump_json(),
            )
        )
        return result.inserted_primary_key[0]


# ------------------------------------------------------------------ trades


def open_trade(
    engine: Engine,
    cycle_id: int,
    symbol: str,
    side: str,
    entry_price: float,
    position_size_usd: float,
    stop_loss: float,
    take_profit: float | None = None,
    exchange_order_id: str | None = None,
    sl_order_id: str | None = None,
) -> int:
    """Insert an open trade row and return its id."""
    position_size_btc = position_size_usd / entry_price
    with engine.begin() as conn:
        result = conn.execute(
            trades.insert().values(
                cycle_id=cycle_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc),
                position_size_usd=position_size_usd,
                position_size_btc=position_size_btc,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exchange_order_id=exchange_order_id,
                sl_order_id=sl_order_id,
                status="open",
            )
        )
        return result.inserted_primary_key[0]


def close_trade(
    engine: Engine,
    trade_id: int,
    exit_price: float,
    status: str = "closed",  # "closed" | "stopped"
) -> dict:
    """Mark a trade as closed, compute PnL, and return the updated row as a dict."""
    with engine.begin() as conn:
        row = conn.execute(
            select(trades).where(trades.c.id == trade_id)
        ).mappings().one()

        pnl_usd = (exit_price - row["entry_price"]) * row["position_size_btc"]
        if row["side"] == "SELL":
            pnl_usd = -pnl_usd
        pnl_pct = pnl_usd / row["position_size_usd"] * 100

        conn.execute(
            update(trades)
            .where(trades.c.id == trade_id)
            .values(
                exit_price=exit_price,
                exit_time=datetime.now(timezone.utc),
                pnl_usd=round(pnl_usd, 4),
                pnl_pct=round(pnl_pct, 4),
                status=status,
            )
        )

        # Re-fetch and return the completed row
        updated = conn.execute(
            select(trades).where(trades.c.id == trade_id)
        ).mappings().one()
        return dict(updated)


def get_open_trade(engine: Engine, symbol: str = "BTC/USD") -> dict | None:
    """Return the currently open trade for *symbol*, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            select(trades)
            .where(trades.c.symbol == symbol, trades.c.status == "open")
            .order_by(trades.c.entry_time.desc())
            .limit(1)
        ).mappings().first()
        return dict(row) if row else None


def get_trade(engine: Engine, trade_id: int) -> dict | None:
    """Return a trade row by id, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            select(trades).where(trades.c.id == trade_id)
        ).mappings().first()
        return dict(row) if row else None


# -------------------------------------------------------- portfolio_state


def get_portfolio_state(engine: Engine) -> dict:
    """Return the singleton portfolio state row, creating it if absent."""
    with engine.connect() as conn:
        row = conn.execute(
            select(portfolio_state).where(portfolio_state.c.id == 1)
        ).mappings().first()
        if row:
            return dict(row)

    # First run — seed with initial capital
    _seed_portfolio(engine)
    with engine.connect() as conn:
        row = conn.execute(
            select(portfolio_state).where(portfolio_state.c.id == 1)
        ).mappings().one()
        return dict(row)


def _seed_portfolio(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            portfolio_state.insert().values(
                id=1,
                cash_usd=INITIAL_CAPITAL,
                peak_value=INITIAL_CAPITAL,
                updated_at=datetime.now(timezone.utc),
            )
        )


def update_portfolio_state(
    engine: Engine,
    cash_usd: float,
    peak_value: float,
) -> None:
    """Upsert the singleton portfolio state row."""
    with engine.begin() as conn:
        exists = conn.execute(
            select(portfolio_state).where(portfolio_state.c.id == 1)
        ).first()

        now = datetime.now(timezone.utc)
        if exists:
            conn.execute(
                update(portfolio_state)
                .where(portfolio_state.c.id == 1)
                .values(cash_usd=cash_usd, peak_value=peak_value, updated_at=now)
            )
        else:
            conn.execute(
                portfolio_state.insert().values(
                    id=1, cash_usd=cash_usd, peak_value=peak_value, updated_at=now
                )
            )


# ----------------------------------------------------------- reflections


def save_reflection(engine: Engine, trade_id: int, summary: str) -> int:
    """Insert a reflection row and return its id."""
    with engine.begin() as conn:
        result = conn.execute(
            reflections.insert().values(
                trade_id=trade_id,
                summary=summary,
                created_at=datetime.now(timezone.utc),
            )
        )
        return result.inserted_primary_key[0]


def get_reflection(engine: Engine, trade_id: int) -> dict | None:
    """Return the reflection for a trade, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            select(reflections).where(reflections.c.trade_id == trade_id)
        ).mappings().first()
        return dict(row) if row else None


# ---------------------------------------------------- windowed queries


def get_closed_trades_in_window(
    engine: Engine,
    since: datetime,
    until: datetime,
) -> list[dict]:
    """Return all closed/stopped trades with exit_time in [since, until)."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(trades)
            .where(
                trades.c.status.in_(["closed", "stopped"]),
                trades.c.exit_time >= since,
                trades.c.exit_time < until,
            )
            .order_by(trades.c.exit_time.asc())
        ).mappings().all()
        return [dict(r) for r in rows]


def get_reflections_for_trades(engine: Engine, trade_ids: list[int]) -> dict[int, str]:
    """Return a {trade_id: summary} dict for the given trade ids."""
    if not trade_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            select(reflections).where(reflections.c.trade_id.in_(trade_ids))
        ).mappings().all()
        return {r["trade_id"]: r["summary"] for r in rows}


def get_cycles_in_window(
    engine: Engine,
    since: datetime,
    until: datetime,
) -> list[dict]:
    """Return lightweight cycle rows (id, timestamp, signal, conviction, vetoed) in window."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                cycles.c.id,
                cycles.c.timestamp,
                cycles.c.signal,
                cycles.c.conviction,
                cycles.c.vetoed,
                cycles.c.council_outputs_json,
            )
            .where(
                cycles.c.timestamp >= since,
                cycles.c.timestamp < until,
            )
            .order_by(cycles.c.timestamp.asc())
        ).mappings().all()
        return [dict(r) for r in rows]


# ---------------------------------------------- weekly_summaries


def save_weekly_summary(
    engine: Engine,
    window_start: datetime,
    window_end: datetime,
    summary: str,
    metrics: dict,
) -> int:
    """Insert a weekly summary row and return its id."""
    with engine.begin() as conn:
        result = conn.execute(
            weekly_summaries.insert().values(
                window_start=window_start,
                window_end=window_end,
                summary=summary,
                metrics_json=json.dumps(metrics),
                created_at=datetime.now(timezone.utc),
            )
        )
        return result.inserted_primary_key[0]


def get_weekly_summaries(engine: Engine, limit: int = 10) -> list[dict]:
    """Return the most recent weekly summaries, newest first."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(weekly_summaries)
            .order_by(weekly_summaries.c.created_at.desc())
            .limit(limit)
        ).mappings().all()
        return [dict(r) for r in rows]
