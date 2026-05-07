"""
Unit tests for council/data/assembler.py.

All external I/O (ccxt exchange, httpx fetchers) is mocked so these tests
run offline and deterministically.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models import MarketContext
from src.validation import ValidationError


# ----------------------------------------------------------------- Fixtures

def _fake_ohlcv_df(n: int = 250) -> pd.DataFrame:
    """Return a valid synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(0)
    closes = np.linspace(30_000, 60_000, n) + rng.normal(0, 100, n)
    opens = closes - rng.uniform(0, 100, n)
    highs = closes + rng.uniform(50, 300, n)
    lows = np.minimum(opens, closes) - rng.uniform(50, 300, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 2_000.0},
        index=idx,
    )


_FAKE_NEWS = [
    {"headline": f"Headline {i}", "source": "CryptoPanic", "published_at": "2026-04-05T00:00:00Z"}
    for i in range(15)
]

_FAKE_SENTIMENT = {
    "fear_greed_index": 62,
    "reddit_sentiment": "bullish",
    "social_volume_vs_avg": 1.2,
}

_FAKE_ONCHAIN = {
    "exchange_net_flow_btc": -4_200.0,
    "whale_transactions_24h": 183,
    "sopr": 1.04,
}

_MOCK_EXCHANGE = MagicMock()


# --------------------------------------------------------------- Happy path

class TestAssemblerHappyPath:
    @pytest.fixture(autouse=True)
    def _patch_all(self):
        df = _fake_ohlcv_df()
        with (
            patch("src.data.assembler.get_exchange", return_value=_MOCK_EXCHANGE),
            patch("src.data.assembler.fetch_ohlcv", return_value=df),
            patch("src.data.assembler.fetch_news", new=AsyncMock(return_value=_FAKE_NEWS)),
            patch("src.data.assembler.fetch_sentiment", new=AsyncMock(return_value=_FAKE_SENTIMENT)),
            patch("src.data.assembler.fetch_onchain", new=AsyncMock(return_value=_FAKE_ONCHAIN)),
        ):
            yield

    async def test_returns_market_context(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        assert isinstance(ctx, MarketContext)

    async def test_asset_is_btc_usd(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        assert ctx.asset == "BTC/USD"

    async def test_timestamp_is_recent(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        age = (datetime.now(timezone.utc) - ctx.timestamp).total_seconds()
        assert age < 5, "Context timestamp should be set at assembly time"

    async def test_price_fields_populated(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        assert ctx.price.current > 0
        assert ctx.price.volume_24h_usd > 0

    async def test_indicator_fields_populated(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        assert 0 <= ctx.indicators.rsi_14 <= 100
        assert ctx.indicators.atr_14 > 0
        assert ctx.indicators.regime in {"trending", "ranging", "high_volatility"}

    async def test_news_items_populated(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        assert len(ctx.news) == 15
        assert ctx.news[0].headline == "Headline 0"

    async def test_sentiment_populated(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        assert ctx.sentiment.fear_greed_index == 62
        assert ctx.sentiment.reddit_sentiment == "bullish"

    async def test_onchain_populated(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        assert ctx.onchain.exchange_net_flow_btc == -4_200.0

    async def test_default_portfolio_applied(self):
        from src.data.assembler import assemble_context
        ctx = await assemble_context()
        assert ctx.portfolio.cash_usd == 10_000.0
        assert ctx.portfolio.btc_position_usd == 0.0

    async def test_custom_portfolio_respected(self):
        from src.data.assembler import assemble_context
        portfolio = {
            "btc_position_usd": 5_000.0,
            "cash_usd": 5_000.0,
            "current_drawdown_pct": 0.05,
            "peak_portfolio_value": 11_000.0,
        }
        ctx = await assemble_context(portfolio=portfolio)
        assert ctx.portfolio.btc_position_usd == 5_000.0
        assert ctx.portfolio.cash_usd == 5_000.0

    async def test_provided_exchange_is_used(self):
        from src.data.assembler import assemble_context
        mock_ex = MagicMock()
        ctx = await assemble_context(exchange=mock_ex)
        assert isinstance(ctx, MarketContext)


# ----------------------------------------------------- Validation integration

class TestAssemblerValidation:
    async def test_insufficient_news_raises_validation_error(self):
        """Assembler should propagate ValidationError when news count is low."""
        from src.data.assembler import assemble_context
        df = _fake_ohlcv_df()
        with (
            patch("src.data.assembler.get_exchange", return_value=_MOCK_EXCHANGE),
            patch("src.data.assembler.fetch_ohlcv", return_value=df),
            patch("src.data.assembler.fetch_news", new=AsyncMock(return_value=[])),
            patch("src.data.assembler.fetch_sentiment", new=AsyncMock(return_value=_FAKE_SENTIMENT)),
            patch("src.data.assembler.fetch_onchain", new=AsyncMock(return_value=_FAKE_ONCHAIN)),
        ):
            with pytest.raises(ValidationError, match="Insufficient"):
                await assemble_context()


class TestAssemblerOverrides:
    """Tests for the backtest override parameters added in v1."""

    @pytest.fixture(autouse=True)
    def _patch_price(self):
        df = _fake_ohlcv_df()
        with (
            patch("src.data.assembler.get_exchange", return_value=_MOCK_EXCHANGE),
            patch("src.data.assembler.fetch_ohlcv", return_value=df),
        ):
            yield

    async def test_news_override_skips_live_fetch(self):
        from src.data.assembler import assemble_context
        custom_news = [
            {"headline": f"Custom {i}", "source": "stub", "published_at": "2024-01-01T00:00:00Z"}
            for i in range(15)
        ]
        with (
            patch("src.data.assembler.fetch_news", new=AsyncMock(side_effect=Exception("should not be called"))),
            patch("src.data.assembler.fetch_sentiment", new=AsyncMock(return_value=_FAKE_SENTIMENT)),
            patch("src.data.assembler.fetch_onchain", new=AsyncMock(return_value=_FAKE_ONCHAIN)),
        ):
            ctx = await assemble_context(news_override=custom_news, skip_freshness=True)
        assert ctx.news[0].headline == "Custom 0"

    async def test_sentiment_override_skips_live_fetch(self):
        from src.data.assembler import assemble_context
        custom_sentiment = {"fear_greed_index": 25, "reddit_sentiment": "bearish", "social_volume_vs_avg": 0.8}
        with (
            patch("src.data.assembler.fetch_news", new=AsyncMock(return_value=_FAKE_NEWS)),
            patch("src.data.assembler.fetch_sentiment", new=AsyncMock(side_effect=Exception("should not be called"))),
            patch("src.data.assembler.fetch_onchain", new=AsyncMock(return_value=_FAKE_ONCHAIN)),
        ):
            ctx = await assemble_context(sentiment_override=custom_sentiment, skip_freshness=True)
        assert ctx.sentiment.fear_greed_index == 25
        assert ctx.sentiment.reddit_sentiment == "bearish"

    async def test_onchain_override_skips_live_fetch(self):
        from src.data.assembler import assemble_context
        custom_onchain = {"exchange_net_flow_btc": 100.0, "whale_transactions_24h": 50, "sopr": 0.99}
        with (
            patch("src.data.assembler.fetch_news", new=AsyncMock(return_value=_FAKE_NEWS)),
            patch("src.data.assembler.fetch_sentiment", new=AsyncMock(return_value=_FAKE_SENTIMENT)),
            patch("src.data.assembler.fetch_onchain", new=AsyncMock(side_effect=Exception("should not be called"))),
        ):
            ctx = await assemble_context(onchain_override=custom_onchain, skip_freshness=True)
        assert ctx.onchain.exchange_net_flow_btc == 100.0

    async def test_timestamp_override_used_in_context(self):
        from src.data.assembler import assemble_context
        historical_ts = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        with (
            patch("src.data.assembler.fetch_news", new=AsyncMock(return_value=_FAKE_NEWS)),
            patch("src.data.assembler.fetch_sentiment", new=AsyncMock(return_value=_FAKE_SENTIMENT)),
            patch("src.data.assembler.fetch_onchain", new=AsyncMock(return_value=_FAKE_ONCHAIN)),
        ):
            ctx = await assemble_context(timestamp=historical_ts, skip_freshness=True)
        assert ctx.timestamp == historical_ts

    async def test_skip_freshness_allows_old_timestamp(self):
        from src.data.assembler import assemble_context
        old_ts = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        with (
            patch("src.data.assembler.fetch_news", new=AsyncMock(return_value=_FAKE_NEWS)),
            patch("src.data.assembler.fetch_sentiment", new=AsyncMock(return_value=_FAKE_SENTIMENT)),
            patch("src.data.assembler.fetch_onchain", new=AsyncMock(return_value=_FAKE_ONCHAIN)),
        ):
            ctx = await assemble_context(timestamp=old_ts, skip_freshness=True)
        assert isinstance(ctx, MarketContext)
