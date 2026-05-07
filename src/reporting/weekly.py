"""
Weekly summary agent — synthesises one week of trading data into an
actionable performance review.

Usage
-----
    # As a module (manual or cron)
    python -m src.reporting.weekly

    # Via main.py
    python -m src.main --weekly
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone

import anthropic

from src.agents.base import AgentError, load_prompt
from src.db.schema import create_all, get_engine
from src.db.store import (
    get_closed_trades_in_window,
    get_cycles_in_window,
    get_reflections_for_trades,
    save_weekly_summary,
)
from src.reporting.metrics import compute_all_metrics

MODEL = "claude-sonnet-4-6"
PROMPT_FILE = "personal/weekly_summary_v1.txt"


def _build_weekly_user_message(
    trades: list[dict],
    reflections: dict[int, str],
    cycle_rows: list[dict],
    metrics: dict,
) -> str:
    lines: list[str] = []

    # ── 1. Metrics snapshot ────────────────────────────────────────────────
    lines.append("## Performance Metrics")
    lines.append(f"Period: {metrics.get('period_start', 'N/A')} → {metrics.get('period_end', 'N/A')}")
    lines.append(f"Trades closed: {metrics['trade_count']}")

    sharpe = metrics.get("sharpe_ratio")
    lines.append(f"Sharpe ratio: {sharpe:.4f}" if sharpe is not None else "Sharpe ratio: insufficient data (< 2 trades)")
    lines.append(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")

    exp = metrics["expectancy"]
    lines.append(
        f"Expectancy: {exp['expectancy_usd']:+.2f} USD/trade "
        f"(win rate {exp['win_rate']:.0%}, avg win {exp['avg_win_usd']:.2f} USD, "
        f"avg loss {exp['avg_loss_usd']:.2f} USD)"
    )

    cons = metrics["consistency"]
    lines.append(
        f"Council consistency: {cons['unanimous_pct']:.0%} unanimous cycles, "
        f"avg confidence spread {cons['avg_confidence_spread']:.1f} pts, "
        f"veto rate {cons['veto_rate']:.0%} ({cons['total_cycles']} total cycles)"
    )

    targets = []
    targets.append(f"  Sharpe ≥ 1.5: {'✓' if metrics['meets_sharpe_target'] else '✗'}")
    targets.append(f"  Drawdown < 20%: {'✓' if metrics['meets_drawdown_target'] else '✗'}")
    targets.append(f"  Positive expectancy: {'✓' if metrics['meets_expectancy_target'] else '✗'}")
    lines.append("Targets met:\n" + "\n".join(targets))

    # ── 2. Cycle summary ───────────────────────────────────────────────────
    lines.append("\n## Cycle Summary")
    hold_count = sum(1 for c in cycle_rows if c["signal"] == "HOLD")
    buy_count = sum(1 for c in cycle_rows if c["signal"] == "BUY")
    sell_count = sum(1 for c in cycle_rows if c["signal"] == "SELL")
    lines.append(
        f"Signals this period: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count} "
        f"of {cons['total_cycles']} cycles"
    )

    # ── 3. Trade log with reflections ──────────────────────────────────────
    lines.append("\n## Trade Log with Reflections")
    if not trades:
        lines.append("No trades closed in this period.")
    else:
        for t in trades:
            tid = t["id"]
            pnl = t.get("pnl_usd", 0.0) or 0.0
            pct = t.get("pnl_pct", 0.0) or 0.0
            lines.append(
                f"\n### Trade #{tid} — {t['side']} {t['symbol']}\n"
                f"Entry: {t['entry_price']:,.2f} USD  |  "
                f"Exit: {t.get('exit_price', 'N/A'):,.2f} USD  |  "
                f"Stop: {t['stop_loss']:,.2f} USD\n"
                f"Status: {t['status']}  |  "
                f"PnL: {pnl:+,.2f} USD ({pct:+.2f}%)"
            )
            ref = reflections.get(tid)
            if ref:
                lines.append(f"\nReflection:\n{ref}")
            else:
                lines.append("\nReflection: (none recorded)")

    lines.append("\nPlease write your weekly summary now.")
    return "\n".join(lines)


async def _persist_weekly_summary(
    engine,
    summary: str,
    metrics: dict,
    window_start: datetime,
    window_end: datetime,
) -> int:
    return save_weekly_summary(engine, window_start, window_end, summary, metrics)


async def run_weekly_summary(
    engine,
    client: anthropic.AsyncAnthropic,
    weeks_back: int = 1,
) -> str:
    """Generate and persist a weekly performance summary.

    Args:
        engine:     Council SQLite DB engine.
        client:     AsyncAnthropic client.
        weeks_back: How many weeks to look back (default 1 = last 7 days).

    Returns:
        Plain-text summary (400–600 words).

    Raises:
        AgentError: If the LLM call fails.
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(weeks=weeks_back)
    window_end = now

    trades = get_closed_trades_in_window(engine, window_start, window_end)
    trade_ids = [t["id"] for t in trades]
    reflections = get_reflections_for_trades(engine, trade_ids)
    cycle_rows = get_cycles_in_window(engine, window_start, window_end)
    metrics = compute_all_metrics(engine, since=window_start)

    system = load_prompt(PROMPT_FILE)
    user = _build_weekly_user_message(trades, reflections, cycle_rows, metrics)

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIError as exc:
        raise AgentError(f"Weekly summary agent API error: {exc}") from exc

    summary = response.content[0].text.strip()
    await _persist_weekly_summary(engine, summary, metrics, window_start, window_end)
    return summary


async def main() -> None:
    engine = get_engine()
    create_all(engine)
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    summary = await run_weekly_summary(engine, client)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
