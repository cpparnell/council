"""
Context object assembler — Phase 1 entry point.

Fetches all data sources in parallel, computes indicators, validates the
result, and returns a fully-populated MarketContext ready for the council.

Usage:
    import asyncio
    from council.data.assembler import assemble_context

    ctx = asyncio.run(assemble_context())
    print(ctx.model_dump_json(indent=2))
"""

import asyncio
from datetime import datetime, timezone

import ccxt

from src.data.indicators import compute_indicators
from src.data.news import fetch_news
from src.data.onchain import fetch_onchain
from src.data.price import build_price_data, fetch_ohlcv, get_exchange
from src.data.sentiment import fetch_sentiment
from src.models import (
    IndicatorData,
    MarketContext,
    NewsItem,
    OnchainData,
    PortfolioData,
    PriceData,
    SentimentData,
)
from src.validation import ValidationError, validate_context


async def assemble_context(
    exchange: ccxt.Exchange | None = None,
    portfolio: dict | None = None,
) -> MarketContext:
    """Assemble the full context object for one council cycle.

    Args:
        exchange: Pre-built ccxt exchange instance.  Defaults to a
                  Binance testnet exchange (sandbox=True).
        portfolio: Current portfolio state dict matching PortfolioData schema.
                   Defaults to an empty $10 000 paper portfolio.

    Returns:
        A validated MarketContext.

    Raises:
        ValidationError: if Tier-0 checks fail (stale data, insufficient
                         news, out-of-range values).
        HardRuleViolation: if hard risk rules are breached (drawdown halt).
    """
    if portfolio is None:
        portfolio = {
            "btc_position_usd": 0.0,
            "cash_usd": 10_000.0,
            "current_drawdown_pct": 0.0,
            "peak_portfolio_value": 10_000.0,
        }

    resolved_exchange = exchange or get_exchange(sandbox=True)

    # --- Fetch price data synchronously (ccxt is sync) --------------------
    df = fetch_ohlcv(resolved_exchange)
    price_data = build_price_data(df)
    indicator_data = compute_indicators(df)

    # --- Fetch async feeds in parallel ------------------------------------
    news_items_raw, sentiment_raw, onchain_raw = await asyncio.gather(
        fetch_news(),
        fetch_sentiment(),
        fetch_onchain(),
    )

    ctx = MarketContext(
        timestamp=datetime.now(timezone.utc),
        asset="BTC/USD",
        price=PriceData(**price_data),
        indicators=IndicatorData(**indicator_data),
        news=[NewsItem(**item) for item in news_items_raw],
        sentiment=SentimentData(**sentiment_raw),
        onchain=OnchainData(**onchain_raw),
        portfolio=PortfolioData(**portfolio),
    )

    validate_context(ctx)
    return ctx
