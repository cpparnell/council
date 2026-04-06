"""
Signal router — translates a CouncilResult into exchange orders and DB updates.

Logic (mirrors the spec exactly):

BUY signal (no open position):
    1. Calculate position size from risk manager + portfolio constraints.
    2. Place market buy order.
    3. Place stop-loss order.
    4. Log trade to DB.
    5. Deduct cost from cash, update peak if new high.

SELL signal (open position):
    1. Close position at market.
    2. Record PnL, mark trade closed.
    3. Return cash to portfolio state.
    4. Queue reflection (caller handles async reflection call).

HOLD:
    1. Log cycle only — no order placed.
    2. Existing stop-loss remains active on the exchange.

Reconciliation (runs at the start of every cycle):
    - If an open trade's current price has breached its stop, mark it stopped.
"""

import logging
from dataclasses import dataclass

import ccxt
from sqlalchemy.engine import Engine

from src.agents.runner import CouncilResult
from src.db.store import (
    close_trade,
    get_open_trade,
    get_portfolio_state,
    log_cycle,
    open_trade,
    update_portfolio_state,
)
from src.execution.orders import (
    OrderError,
    cancel_order,
    fetch_current_price,
    place_market_buy,
    place_market_sell,
    place_stop_loss,
)
from src.models import MarketContext

logger = logging.getLogger(__name__)

# Hard risk constants (enforced in code — not delegatable to LLMs)
MAX_POSITION_SIZE_PCT = 0.20
MAX_TRADE_RISK_PCT = 0.02
MIN_RISK_REWARD = 1.5

SYMBOL = "BTC/USDT"


@dataclass
class RouteResult:
    """Summary of what the router did this cycle."""

    cycle_id: int
    action: str          # "buy" | "sell" | "hold" | "hold_no_position"
    trade_id: int | None = None
    closed_trade: dict | None = None  # set when a position was closed
    error: str | None = None          # set if an order failed


def reconcile_open_position(engine: Engine, exchange: ccxt.Exchange) -> dict | None:
    """Check if an open position's stop has been breached; if so, close it.

    Returns the closed trade dict if a stop-out occurred, otherwise None.
    """
    trade = get_open_trade(engine, SYMBOL)
    if not trade:
        return None

    try:
        current_price = fetch_current_price(exchange, SYMBOL)
    except OrderError as exc:
        logger.warning("Could not fetch price for reconciliation: %s", exc)
        return None

    if current_price <= trade["stop_loss"]:
        logger.warning(
            "Stop-loss breached (price %.2f ≤ stop %.2f); closing trade #%d",
            current_price, trade["stop_loss"], trade["id"],
        )
        closed = close_trade(engine, trade["id"], exit_price=current_price, status="stopped")

        # Restore cash and update peak
        state = get_portfolio_state(engine)
        new_cash = state["cash_usd"] + closed["position_size_usd"] + closed["pnl_usd"]
        new_peak = max(state["peak_value"], new_cash)
        update_portfolio_state(engine, cash_usd=new_cash, peak_value=new_peak)

        return closed

    return None


def route_signal(
    engine: Engine,
    exchange: ccxt.Exchange,
    ctx: MarketContext,
    result: CouncilResult,
) -> RouteResult:
    """Execute the council's decision against the exchange and update the DB.

    Args:
        engine:   Council SQLite DB engine.
        exchange: Authenticated ccxt exchange instance.
        ctx:      The MarketContext for this cycle (used for portfolio data).
        result:   The council's CouncilResult for this cycle.

    Returns:
        RouteResult describing what happened.
    """
    signal = result.signal
    risk = result.outputs.risk
    delib = result.deliberation

    # ── 1. Log the cycle regardless of signal ──────────────────────────────
    cycle_id = log_cycle(engine, ctx, result.outputs, delib)

    open_pos = get_open_trade(engine, SYMBOL)
    state = get_portfolio_state(engine)

    # ── 2. BUY ─────────────────────────────────────────────────────────────
    if signal == "BUY" and open_pos is None:
        portfolio_value = state["cash_usd"]  # cash only (no open position)

        # Position size: smaller of risk-manager recommendation and hard cap
        agent_pct = min(risk.approved_position_size_pct / 100, MAX_POSITION_SIZE_PCT)

        # Also enforce max-trade-risk rule: size = (portfolio × 2%) / (entry - stop)
        stop_distance = ctx.price.current - risk.recommended_stop_loss
        if stop_distance <= 0:
            logger.warning("Invalid stop distance (%.2f); defaulting to HOLD", stop_distance)
            return RouteResult(cycle_id=cycle_id, action="hold", error="invalid_stop_distance")

        risk_capped_size = (portfolio_value * MAX_TRADE_RISK_PCT) / stop_distance
        position_size_usd = min(portfolio_value * agent_pct, risk_capped_size)
        position_size_usd = min(position_size_usd, portfolio_value)  # never exceed cash
        amount_btc = position_size_usd / ctx.price.current

        # Enforce R:R
        rr = risk.risk_reward_ratio
        if rr < MIN_RISK_REWARD:
            logger.info("R:R %.2f below minimum %.2f; holding", rr, MIN_RISK_REWARD)
            return RouteResult(cycle_id=cycle_id, action="hold", error="insufficient_rr")

        try:
            buy_order = place_market_buy(exchange, SYMBOL, amount_btc)
            actual_entry = float(buy_order.get("average") or buy_order.get("price") or ctx.price.current)

            sl_order = place_stop_loss(exchange, SYMBOL, amount_btc, risk.recommended_stop_loss)

            trade_id = open_trade(
                engine,
                cycle_id=cycle_id,
                symbol=SYMBOL,
                side="BUY",
                entry_price=actual_entry,
                position_size_usd=position_size_usd,
                stop_loss=risk.recommended_stop_loss,
                take_profit=risk.recommended_take_profit,
                exchange_order_id=buy_order.get("id"),
                sl_order_id=sl_order.get("id"),
            )

            new_cash = state["cash_usd"] - position_size_usd
            update_portfolio_state(engine, cash_usd=new_cash, peak_value=state["peak_value"])

            logger.info(
                "BUY executed: trade_id=%d size=%.2f USD entry=%.2f SL=%.2f",
                trade_id, position_size_usd, actual_entry, risk.recommended_stop_loss,
            )
            return RouteResult(cycle_id=cycle_id, action="buy", trade_id=trade_id)

        except OrderError as exc:
            logger.error("BUY order failed: %s", exc)
            return RouteResult(cycle_id=cycle_id, action="hold", error=str(exc))

    # ── 3. SELL (close long position) ──────────────────────────────────────
    elif signal == "SELL" and open_pos is not None:
        amount_btc = open_pos["position_size_btc"]
        try:
            # Cancel the standing stop-loss before placing a market sell
            if open_pos.get("sl_order_id"):
                cancel_order(exchange, open_pos["sl_order_id"], SYMBOL)

            sell_order = place_market_sell(exchange, SYMBOL, amount_btc)
            exit_price = float(sell_order.get("average") or sell_order.get("price") or ctx.price.current)

            closed = close_trade(engine, open_pos["id"], exit_price=exit_price, status="closed")

            new_cash = state["cash_usd"] + open_pos["position_size_usd"] + closed["pnl_usd"]
            new_peak = max(state["peak_value"], new_cash)
            update_portfolio_state(engine, cash_usd=new_cash, peak_value=new_peak)

            logger.info(
                "SELL executed: trade_id=%d exit=%.2f pnl=%.2f USD (%.2f%%)",
                open_pos["id"], exit_price, closed["pnl_usd"], closed["pnl_pct"],
            )
            return RouteResult(
                cycle_id=cycle_id,
                action="sell",
                trade_id=open_pos["id"],
                closed_trade=closed,
            )

        except OrderError as exc:
            logger.error("SELL order failed: %s", exc)
            return RouteResult(cycle_id=cycle_id, action="hold", error=str(exc))

    # ── 4. HOLD ────────────────────────────────────────────────────────────
    else:
        reason = (
            "already_in_position" if signal == "BUY" and open_pos
            else "no_position_to_close" if signal == "SELL" and not open_pos
            else "hold_signal"
        )
        logger.info("HOLD this cycle (%s)", reason)
        return RouteResult(cycle_id=cycle_id, action="hold")
