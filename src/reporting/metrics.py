"""
Performance metrics for the Council paper-trading review.

All functions derive their data from the SQLite DB via the store layer.
No LLM calls — pure arithmetic.

Success targets (from v0_spec.md):
    Sharpe ratio  ≥ 1.5
    Max drawdown  < 20%
    Expectancy    positive  (avg_win × win_rate > avg_loss × loss_rate)
    Consistency   agents do not wildly disagree on obvious conditions
"""

import json
import math
import os
from datetime import datetime, timezone

from sqlalchemy.engine import Engine

from src.db.store import get_closed_trades_in_window, get_cycles_in_window

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))

SHARPE_TARGET = 1.5
DRAWDOWN_TARGET = 0.20   # fraction


# ------------------------------------------------------------------ helpers


def _all_closed_trades(engine: Engine, since: datetime | None = None) -> list[dict]:
    """Return all closed/stopped trades, optionally filtered by exit_time."""
    if since is None:
        since = datetime(2000, 1, 1, tzinfo=timezone.utc)
    until = datetime(2100, 1, 1, tzinfo=timezone.utc)
    return get_closed_trades_in_window(engine, since, until)


# ------------------------------------------------------------------ metrics


def compute_sharpe_ratio(trades: list[dict], risk_free_rate: float = 0.0) -> float | None:
    """Annualised Sharpe ratio computed from per-trade PnL percentages.

    Uses each trade's pnl_pct as a return observation.  Annualises with
    sqrt(252) (daily cadence).  Returns None when fewer than 2 trades are
    present (standard deviation is undefined).

    Target: ≥ 1.5
    """
    returns = [t["pnl_pct"] for t in trades if t.get("pnl_pct") is not None]
    if len(returns) < 2:
        return None

    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)

    if std == 0:
        return None

    return round((mean - risk_free_rate) / std * math.sqrt(252), 4)


def compute_max_drawdown(engine: Engine, since: datetime | None = None) -> float:
    """Maximum historical drawdown as a fraction of the running equity peak.

    Walks the closed-trade equity curve chronologically starting from
    INITIAL_CAPITAL.  Returns 0.0 when there are no closed trades.

    Target: < 0.20
    """
    trades = _all_closed_trades(engine, since)
    if not trades:
        return 0.0

    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0

    for t in trades:
        equity += t.get("pnl_usd", 0.0) or 0.0
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    return round(max_dd, 6)


def compute_expectancy(trades: list[dict]) -> dict:
    """Expected PnL per trade in USD.

    Expectancy = avg_win_usd × win_rate − avg_loss_usd × loss_rate

    Returns a dict of zeros / False when there are no closed trades.
    """
    if not trades:
        return {
            "win_rate": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "expectancy_usd": 0.0,
            "positive": False,
        }

    wins = [t["pnl_usd"] for t in trades if (t.get("pnl_usd") or 0.0) > 0]
    losses = [abs(t["pnl_usd"]) for t in trades if (t.get("pnl_usd") or 0.0) <= 0]

    total = len(trades)
    win_rate = len(wins) / total
    loss_rate = len(losses) / total

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    expectancy = avg_win * win_rate - avg_loss * loss_rate

    return {
        "win_rate": round(win_rate, 4),
        "avg_win_usd": round(avg_win, 4),
        "avg_loss_usd": round(avg_loss, 4),
        "expectancy_usd": round(expectancy, 4),
        "positive": expectancy > 0,
    }


def compute_council_consistency(engine: Engine, since: datetime | None = None) -> dict:
    """Measure how often the three directional agents agree.

    Parses council_outputs_json for each cycle in the window.  A cycle is
    "unanimous" when the technical, sentiment, and fundamental agents all
    return the same direction.

    Returns:
        total_cycles        int
        unanimous_pct       float  (fraction, not %)
        avg_confidence_spread float  (max agent confidence − min per cycle, averaged)
        veto_rate           float  (fraction of cycles with risk manager veto)

    Consistency concern thresholds (from spec spirit):
        unanimous_pct < 0.40  → agents disagree frequently
        avg_confidence_spread > 30 → agents are far apart in confidence
    """
    if since is None:
        since = datetime(2000, 1, 1, tzinfo=timezone.utc)
    until = datetime(2100, 1, 1, tzinfo=timezone.utc)
    cycle_rows = get_cycles_in_window(engine, since, until)

    if not cycle_rows:
        return {
            "total_cycles": 0,
            "unanimous_pct": 0.0,
            "avg_confidence_spread": 0.0,
            "veto_rate": 0.0,
        }

    unanimous_count = 0
    veto_count = 0
    spreads: list[float] = []

    for row in cycle_rows:
        if row.get("vetoed"):
            veto_count += 1

        try:
            outputs = json.loads(row["council_outputs_json"])
        except (json.JSONDecodeError, KeyError):
            continue

        directions = [
            outputs.get("technical", {}).get("direction"),
            outputs.get("sentiment", {}).get("direction"),
            outputs.get("fundamental", {}).get("direction"),
        ]
        confidences = [
            outputs.get("technical", {}).get("confidence"),
            outputs.get("sentiment", {}).get("confidence"),
            outputs.get("fundamental", {}).get("confidence"),
        ]

        # Unanimous if all three directions are the same non-None value
        valid_dirs = [d for d in directions if d]
        if len(valid_dirs) == 3 and len(set(valid_dirs)) == 1:
            unanimous_count += 1

        valid_confs = [c for c in confidences if c is not None]
        if len(valid_confs) >= 2:
            spreads.append(max(valid_confs) - min(valid_confs))

    total = len(cycle_rows)
    return {
        "total_cycles": total,
        "unanimous_pct": round(unanimous_count / total, 4),
        "avg_confidence_spread": round(sum(spreads) / len(spreads), 2) if spreads else 0.0,
        "veto_rate": round(veto_count / total, 4),
    }


def compute_all_metrics(engine: Engine, since: datetime | None = None) -> dict:
    """Run all four metric groups and return a single dict for logging and reporting.

    Args:
        engine: Council SQLite DB engine.
        since:  Optional lower bound for the reporting window (UTC datetime).
                When None, all historical data is used.

    Returns dict with keys:
        period_start, period_end, trade_count,
        sharpe_ratio, max_drawdown_pct,
        expectancy (sub-dict), consistency (sub-dict),
        meets_sharpe_target, meets_drawdown_target, meets_expectancy_target
    """
    trades = _all_closed_trades(engine, since)

    sharpe = compute_sharpe_ratio(trades)
    max_dd = compute_max_drawdown(engine, since)
    expectancy = compute_expectancy(trades)
    consistency = compute_council_consistency(engine, since)

    period_start = trades[0]["exit_time"].isoformat() if trades else None
    period_end = trades[-1]["exit_time"].isoformat() if trades else None

    return {
        "period_start": period_start,
        "period_end": period_end,
        "trade_count": len(trades),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": round(max_dd * 100, 4),
        "expectancy": expectancy,
        "consistency": consistency,
        "meets_sharpe_target": (sharpe is not None and sharpe >= SHARPE_TARGET),
        "meets_drawdown_target": max_dd < DRAWDOWN_TARGET,
        "meets_expectancy_target": expectancy["positive"],
    }
