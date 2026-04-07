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
    timestamp: datetime | None = None,
    news_override: list[dict] | None = None,
    sentiment_override: dict | None = None,
    onchain_override: dict | None = None,
    skip_freshness: bool = False,
) -> MarketContext:
    """Assemble the full context object for one council cycle.

    Args:
        exchange: Pre-built ccxt exchange instance.  Defaults to a
                  Kraken exchange instance.
        portfolio: Current portfolio state dict matching PortfolioData schema.
                   Defaults to an empty $10 000 paper portfolio.
        timestamp: Context timestamp. Defaults to datetime.now(utc). Pass a
                   historical datetime when assembling backtest contexts.
        news_override: If provided, skip live CryptoPanic fetch and use this
                       list of news dicts instead.
        sentiment_override: If provided, skip live Alternative.me fetch and use
                            this sentiment dict instead.
        onchain_override: If provided, skip live Glassnode stub and use this
                          on-chain dict instead.
        skip_freshness: When True, the Tier-0 price-freshness check is skipped.
                        Required for historical backtest contexts whose timestamps
                        are in the past.

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

    # --- Fetch async feeds in parallel (or use overrides) -----------------
    async def _identity(value):
        return value

    fetched_news, fetched_sentiment, fetched_onchain = await asyncio.gather(
        fetch_news() if news_override is None else _identity(news_override),
        fetch_sentiment() if sentiment_override is None else _identity(sentiment_override),
        fetch_onchain() if onchain_override is None else _identity(onchain_override),
    )
    news_items_raw = fetched_news
    sentiment_raw = fetched_sentiment
    onchain_raw = fetched_onchain

    ctx = MarketContext(
        timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
        asset="BTC/USD",
        price=PriceData(**price_data),
        indicators=IndicatorData(**indicator_data),
        news=[NewsItem(**item) for item in news_items_raw],
        sentiment=SentimentData(**sentiment_raw),
        onchain=OnchainData(**onchain_raw),
        portfolio=PortfolioData(**portfolio),
    )

    validate_context(ctx, skip_freshness=skip_freshness)
    return ctx
