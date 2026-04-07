"""
Order placement against Kraken.

All public functions wrap exchange calls with retry logic (max 3 attempts,
exponential back-off) as required by the spec.  On final failure they raise
``OrderError`` — the router logs and alerts; it does NOT retry trade logic.
"""

import logging
import time

import ccxt

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds; doubles each attempt


class OrderError(Exception):
    """Raised when an exchange order fails after all retries."""


def _retry(fn, *args, **kwargs):
    """Call *fn* with retries.  Returns the result or raises OrderError."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except ccxt.NetworkError as exc:
            last_exc = exc
            wait = BACKOFF_BASE ** attempt
            logger.warning("Network error on attempt %d/%d: %s — retrying in %.1fs",
                           attempt + 1, MAX_RETRIES, exc, wait)
            time.sleep(wait)
        except ccxt.ExchangeError as exc:
            # Exchange errors (e.g. insufficient balance) are not retryable.
            raise OrderError(f"Exchange error: {exc}") from exc

    raise OrderError(f"Order failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc


def place_market_buy(
    exchange: ccxt.Exchange,
    symbol: str,
    amount_btc: float,
) -> dict:
    """Place a market buy order.

    Args:
        exchange: Authenticated ccxt exchange instance.
        symbol:   Trading pair, e.g. "BTC/USD".
        amount_btc: Quantity to buy in BTC.

    Returns:
        The raw ccxt order dict.

    Raises:
        OrderError: If the order fails after all retries.
    """
    logger.info("Placing market BUY %.6f %s", amount_btc, symbol)
    order = _retry(exchange.create_market_buy_order, symbol, amount_btc)
    logger.info("Market BUY filled: order_id=%s", order.get("id"))
    return order


def place_market_sell(
    exchange: ccxt.Exchange,
    symbol: str,
    amount_btc: float,
) -> dict:
    """Place a market sell order to close a long position.

    Args:
        exchange:   Authenticated ccxt exchange instance.
        symbol:     Trading pair.
        amount_btc: Quantity to sell in BTC.

    Returns:
        The raw ccxt order dict.

    Raises:
        OrderError: If the order fails after all retries.
    """
    logger.info("Placing market SELL %.6f %s", amount_btc, symbol)
    order = _retry(exchange.create_market_sell_order, symbol, amount_btc)
    logger.info("Market SELL filled: order_id=%s", order.get("id"))
    return order


def place_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    amount_btc: float,
    stop_price: float,
) -> dict:
    """Place a stop-loss sell order.

    Uses a stop-market order (``stopPrice`` param).  Falls back to a plain
    stop-limit order if the exchange does not support stop-market.

    Args:
        exchange:    Authenticated ccxt exchange instance.
        symbol:      Trading pair.
        amount_btc:  Quantity to sell when stop triggers.
        stop_price:  Trigger price.

    Returns:
        The raw ccxt order dict.

    Raises:
        OrderError: If the order fails after all retries.
    """
    logger.info("Placing stop-loss SELL %.6f %s @ %.2f", amount_btc, symbol, stop_price)
    params = {"stopPrice": stop_price}
    try:
        order = _retry(
            exchange.create_order,
            symbol,
            "stop_market",
            "sell",
            amount_btc,
            None,
            params,
        )
    except OrderError:
        # Fallback: stop-limit with limit 0.5% below stop price
        limit_price = round(stop_price * 0.995, 2)
        logger.warning("stop_market unsupported; falling back to stop-limit @ %.2f", limit_price)
        order = _retry(
            exchange.create_order,
            symbol,
            "stop_limit",
            "sell",
            amount_btc,
            limit_price,
            params,
        )

    logger.info("Stop-loss placed: order_id=%s", order.get("id"))
    return order


def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str) -> None:
    """Cancel an open order (best-effort; ignores already-filled/cancelled errors)."""
    try:
        _retry(exchange.cancel_order, order_id, symbol)
        logger.info("Cancelled order %s", order_id)
    except OrderError as exc:
        logger.warning("Could not cancel order %s: %s", order_id, exc)


def fetch_current_price(exchange: ccxt.Exchange, symbol: str) -> float:
    """Return the latest mid-price for reconciliation and PnL calculation."""
    ticker = _retry(exchange.fetch_ticker, symbol)
    return float(ticker["last"])
