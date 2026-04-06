"""
Portfolio state helpers — bridges the DB and the MarketContext assembler.

``build_portfolio_dict`` loads the current DB state (cash, open position,
peak value) and returns a dict compatible with ``PortfolioData`` so that
``assemble_context`` gets accurate portfolio data on every cycle.
"""

from sqlalchemy.engine import Engine

from src.db.store import get_open_trade, get_portfolio_state


def build_portfolio_dict(engine: Engine, symbol: str = "BTC/USDT") -> dict:
    """Build a PortfolioData-compatible dict from the current DB state.

    Args:
        engine: SQLAlchemy engine pointing at the council DB.
        symbol: Trading symbol to check for an open position.

    Returns:
        dict with keys: btc_position_usd, cash_usd, current_drawdown_pct,
        peak_portfolio_value.
    """
    state = get_portfolio_state(engine)
    open_trade = get_open_trade(engine, symbol)

    cash_usd = state["cash_usd"]
    peak_value = state["peak_value"]
    btc_position_usd = open_trade["position_size_usd"] if open_trade else 0.0

    total_value = cash_usd + btc_position_usd
    drawdown = max(0.0, (peak_value - total_value) / peak_value) if peak_value > 0 else 0.0

    return {
        "btc_position_usd": btc_position_usd,
        "cash_usd": cash_usd,
        "current_drawdown_pct": drawdown,
        "peak_portfolio_value": peak_value,
    }
