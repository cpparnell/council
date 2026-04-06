"""
Council — main cycle entry point.

One-shot usage (run once, then exit):
    python -m src.main

Scheduled usage (daily at 00:05 UTC + weekly summary Sundays 08:00 UTC):
    python -m src.main --schedule

Weekly summary only:
    python -m src.main --weekly

Environment variables
---------------------
ANTHROPIC_API_KEY         Required — used by all LLM agents.
BINANCE_TESTNET_API_KEY   Required for order execution in paper mode.
BINANCE_TESTNET_SECRET    Required for order execution in paper mode.
BINANCE_API_KEY           Required for live trading (TRADING_MODE=live).
BINANCE_SECRET            Required for live trading (TRADING_MODE=live).
COUNCIL_DB_PATH           Path to SQLite file (default: ./council.db).
INITIAL_CAPITAL           Starting paper capital in USD (default: 10000).
TRADING_MODE              "paper" (default, Binance testnet) or "live" (mainnet).
COUNCIL_LIVE_CONFIRMED    Must be "1" when TRADING_MODE=live — safety gate.
DRY_RUN                   If "1", skips order execution entirely (logs only).
"""

import argparse
import asyncio
import logging
import os
import sys

import anthropic

from src.agents.reflection import run_reflection
from src.agents.runner import run_council
from src.data.assembler import assemble_context
from src.data.price import get_exchange
from src.db.schema import create_all, get_engine
from src.db.store import save_reflection
from src.execution.router import reconcile_open_position, route_signal
from src.execution.state import build_portfolio_dict
from src.models import CouncilOutputs, DeliberationOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("council.main")

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
SYMBOL = "BTC/USDT"

# ── Live trading safety gate ───────────────────────────────────────────────
if TRADING_MODE == "live":
    if os.getenv("COUNCIL_LIVE_CONFIRMED") != "1":
        logging.getLogger("council.main").error(
            "TRADING_MODE=live requires COUNCIL_LIVE_CONFIRMED=1. "
            "Set this env var only when you intend to trade real funds."
        )
        sys.exit(1)
    logging.getLogger("council.main").warning(
        "⚠️  LIVE TRADING MODE ACTIVE — real funds at risk"
    )


def _get_exchange():
    """Return a ccxt exchange configured for the current TRADING_MODE."""
    use_sandbox = TRADING_MODE != "live"
    return get_exchange(sandbox=use_sandbox)


async def _do_weekly() -> None:
    from src.reporting.weekly import run_weekly_summary
    engine = get_engine()
    create_all(engine)
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    summary = await run_weekly_summary(engine, client)
    print(summary)


async def run_cycle() -> None:
    """Execute one full trading cycle end-to-end."""
    engine = get_engine()
    create_all(engine)

    exchange = _get_exchange()
    anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Reconcile: check if an open position has been stopped out ──────────
    if not DRY_RUN:
        stopped_trade = reconcile_open_position(engine, exchange)
        if stopped_trade:
            logger.info(
                "Stop-out detected during reconciliation: trade_id=%d pnl=%.2f USD",
                stopped_trade["id"], stopped_trade["pnl_usd"],
            )
            await _run_reflection_for_trade(engine, stopped_trade, anthropic_client)

    # ── Assemble context with live portfolio state ─────────────────────────
    portfolio = build_portfolio_dict(engine, SYMBOL)
    logger.info(
        "Portfolio: cash=%.2f USD btc_pos=%.2f USD drawdown=%.1f%%",
        portfolio["cash_usd"],
        portfolio["btc_position_usd"],
        portfolio["current_drawdown_pct"] * 100,
    )

    try:
        ctx = await assemble_context(exchange=exchange, portfolio=portfolio)
    except Exception as exc:
        logger.error("Context assembly failed: %s — defaulting to HOLD", exc)
        return

    # ── Run the council ───────────────────────────────────────────────────
    logger.info("Running council for %s @ %.2f USD", ctx.asset, ctx.price.current)
    try:
        result = await run_council(ctx, client=anthropic_client)
    except Exception as exc:
        logger.error("Council failed: %s — defaulting to HOLD", exc)
        return

    logger.info(
        "Council decision: %s (conviction=%s, vetoed=%s)",
        result.signal, result.deliberation.conviction, result.vetoed,
    )

    # ── Route signal → orders ────────────────────────────────────────────
    if DRY_RUN:
        logger.info("[DRY RUN] Would route signal=%s — skipping order execution", result.signal)
        from src.db.store import log_cycle
        log_cycle(engine, ctx, result.outputs, result.deliberation)
        return

    route = route_signal(engine, exchange, ctx, result)
    logger.info("Route result: action=%s cycle_id=%d trade_id=%s error=%s",
                route.action, route.cycle_id, route.trade_id, route.error)

    # ── Reflection for closed trades ──────────────────────────────────────
    if route.closed_trade:
        await _run_reflection_for_trade(engine, route.closed_trade, anthropic_client)


async def _run_reflection_for_trade(
    engine,
    closed_trade: dict,
    client: anthropic.AsyncAnthropic,
) -> None:
    """Fetch the council outputs for a closed trade and run the reflection agent."""
    from src.db.schema import cycles as cycles_table
    from sqlalchemy import select

    trade_id = closed_trade["id"]
    cycle_id = closed_trade.get("cycle_id")
    if not cycle_id:
        logger.warning("No cycle_id on trade %d; skipping reflection", trade_id)
        return

    with engine.connect() as conn:
        cycle_row = conn.execute(
            select(cycles_table).where(cycles_table.c.id == cycle_id)
        ).mappings().first()

    if not cycle_row:
        logger.warning("Cycle %d not found; skipping reflection", cycle_id)
        return

    outputs = CouncilOutputs.model_validate_json(cycle_row["council_outputs_json"])
    deliberation = DeliberationOutput.model_validate_json(cycle_row["deliberation_json"])

    try:
        summary = await run_reflection(outputs, deliberation, closed_trade, client)
        reflection_id = save_reflection(engine, trade_id, summary)
        logger.info("Reflection saved: reflection_id=%d", reflection_id)
        logger.info("Reflection summary:\n%s", summary)
    except Exception as exc:
        logger.warning("Reflection failed for trade %d: %s", trade_id, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Council BTC trading bot")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on a daily schedule (00:05 UTC) + weekly summary (Sunday 08:00 UTC)",
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Run the weekly summary report and exit",
    )
    args = parser.parse_args()

    if args.weekly:
        asyncio.run(_do_weekly())
        return

    if args.schedule:
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("APScheduler not installed. Run: pip install apscheduler")
            sys.exit(1)

        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(
            lambda: asyncio.run(run_cycle()),
            CronTrigger(hour=0, minute=5),
            id="council_daily",
            name="Council daily cycle",
            misfire_grace_time=300,
        )
        scheduler.add_job(
            lambda: asyncio.run(_do_weekly()),
            CronTrigger(day_of_week="sun", hour=8, minute=0),
            id="council_weekly",
            name="Council weekly summary",
            misfire_grace_time=3600,
        )
        logger.info(
            "Scheduler started — daily cycle 00:05 UTC, weekly summary Sunday 08:00 UTC. "
            "Ctrl+C to stop."
        )
        try:
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped.")
    else:
        asyncio.run(run_cycle())


if __name__ == "__main__":
    main()
