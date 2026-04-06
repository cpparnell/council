"""
Unit tests for council/validation.py.

Tests each validator in isolation using synthetic MarketContext objects.
No network calls — all data is constructed inline.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.models import (
    IndicatorData,
    MarketContext,
    NewsItem,
    OnchainData,
    PortfolioData,
    PriceData,
    SentimentData,
)
from src.validation import (
    HardRuleViolation,
    ValidationError,
    check_drawdown_halt,
    check_volatility_halt,
    validate_context,
    validate_news_count,
    validate_numeric_fields,
    validate_price_freshness,
)


# ----------------------------------------------------------------- Factory

def _news(n: int = 15) -> list[NewsItem]:
    return [
        NewsItem(headline=f"Headline {i}", source="CryptoPanic", published_at="2026-04-05T00:00:00Z")
        for i in range(n)
    ]


def _make_ctx(
    *,
    timestamp: datetime | None = None,
    price: PriceData | None = None,
    indicators: IndicatorData | None = None,
    news: list[NewsItem] | None = None,
    sentiment: SentimentData | None = None,
    onchain: OnchainData | None = None,
    portfolio: PortfolioData | None = None,
) -> MarketContext:
    return MarketContext(
        timestamp=timestamp or datetime.now(timezone.utc),
        asset="BTC/USD",
        price=price or PriceData(
            current=83_500,
            open_24h=81_200,
            high_24h=84_100,
            low_24h=80_900,
            change_pct_24h=2.8,
            volume_24h_usd=1_850_000_000,
            volume_vs_7d_avg=1.4,
        ),
        indicators=indicators or IndicatorData(
            rsi_14=58.2,
            macd_signal="bullish_cross",
            bb_position="mid",
            ema_20=81_400,
            ema_50=78_200,
            ema_200=69_800,
            atr_14=2_100,
            atr_30_avg=1_800,
            regime="trending",
        ),
        news=news if news is not None else _news(),
        sentiment=sentiment or SentimentData(
            fear_greed_index=62,
            reddit_sentiment="bullish",
            social_volume_vs_avg=1.2,
        ),
        onchain=onchain or OnchainData(
            exchange_net_flow_btc=-4_200,
            whale_transactions_24h=183,
            sopr=1.04,
        ),
        portfolio=portfolio or PortfolioData(
            btc_position_usd=0,
            cash_usd=10_000,
            current_drawdown_pct=0.0,
            peak_portfolio_value=10_000,
        ),
    )


# -------------------------------------------------- validate_price_freshness

class TestPriceFreshness:
    def test_fresh_context_passes(self):
        validate_price_freshness(_make_ctx())  # must not raise

    def test_stale_context_fails(self):
        stale_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
        with pytest.raises(ValidationError, match="stale"):
            validate_price_freshness(_make_ctx(timestamp=stale_ts))

    def test_boundary_just_within_limit_passes(self):
        ts = datetime.now(timezone.utc) - timedelta(seconds=299)
        validate_price_freshness(_make_ctx(timestamp=ts), max_age_minutes=5)

    def test_boundary_just_over_limit_fails(self):
        ts = datetime.now(timezone.utc) - timedelta(seconds=301)
        with pytest.raises(ValidationError):
            validate_price_freshness(_make_ctx(timestamp=ts), max_age_minutes=5)

    def test_custom_max_age_respected(self):
        ts = datetime.now(timezone.utc) - timedelta(minutes=3)
        with pytest.raises(ValidationError):
            validate_price_freshness(_make_ctx(timestamp=ts), max_age_minutes=2)

    def test_naive_timestamp_treated_as_utc(self):
        # Naive datetime (no tzinfo) should not crash
        naive = datetime.utcnow()
        validate_price_freshness(_make_ctx(timestamp=naive))


# ------------------------------------------------------- validate_news_count

class TestNewsCount:
    def test_sufficient_news_passes(self):
        validate_news_count(_make_ctx())  # 15 items — must not raise

    def test_exactly_10_passes(self):
        validate_news_count(_make_ctx(news=_news(10)))

    def test_nine_items_fails(self):
        with pytest.raises(ValidationError, match="Insufficient"):
            validate_news_count(_make_ctx(news=_news(9)))

    def test_zero_items_fails(self):
        with pytest.raises(ValidationError):
            validate_news_count(_make_ctx(news=[]))

    def test_custom_min_respected(self):
        validate_news_count(_make_ctx(news=_news(5)), min_items=5)
        with pytest.raises(ValidationError):
            validate_news_count(_make_ctx(news=_news(4)), min_items=5)


# ----------------------------------------------------- validate_numeric_fields

class TestNumericFields:
    def test_valid_context_passes(self):
        validate_numeric_fields(_make_ctx())

    def test_price_too_low_fails(self):
        price = PriceData(
            current=500,           # below 1 000 threshold
            open_24h=490, high_24h=510, low_24h=480,
            change_pct_24h=2.0, volume_24h_usd=1e9, volume_vs_7d_avg=1.0,
        )
        with pytest.raises(ValidationError, match="plausible range"):
            validate_numeric_fields(_make_ctx(price=price))

    def test_price_too_high_fails(self):
        price = PriceData(
            current=2_000_000,     # above 1 000 000 threshold
            open_24h=1_900_000, high_24h=2_100_000, low_24h=1_800_000,
            change_pct_24h=5.0, volume_24h_usd=1e9, volume_vs_7d_avg=1.0,
        )
        with pytest.raises(ValidationError, match="plausible range"):
            validate_numeric_fields(_make_ctx(price=price))

    def test_zero_volume_fails(self):
        price = PriceData(
            current=83_500, open_24h=81_200, high_24h=84_100, low_24h=80_900,
            change_pct_24h=2.8, volume_24h_usd=0.0, volume_vs_7d_avg=1.0,
        )
        with pytest.raises(ValidationError, match="volume"):
            validate_numeric_fields(_make_ctx(price=price))

    def test_rsi_above_100_fails(self):
        ind = IndicatorData(
            rsi_14=150,
            macd_signal="bullish", bb_position="mid",
            ema_20=81_400, ema_50=78_200, ema_200=69_800,
            atr_14=2_100, atr_30_avg=1_800, regime="trending",
        )
        with pytest.raises(ValidationError, match="RSI"):
            validate_numeric_fields(_make_ctx(indicators=ind))

    def test_rsi_below_0_fails(self):
        ind = IndicatorData(
            rsi_14=-5,
            macd_signal="bullish", bb_position="mid",
            ema_20=81_400, ema_50=78_200, ema_200=69_800,
            atr_14=2_100, atr_30_avg=1_800, regime="trending",
        )
        with pytest.raises(ValidationError, match="RSI"):
            validate_numeric_fields(_make_ctx(indicators=ind))

    def test_zero_atr_fails(self):
        ind = IndicatorData(
            rsi_14=58, macd_signal="bullish", bb_position="mid",
            ema_20=81_400, ema_50=78_200, ema_200=69_800,
            atr_14=0.0, atr_30_avg=1_800, regime="trending",
        )
        with pytest.raises(ValidationError, match="ATR"):
            validate_numeric_fields(_make_ctx(indicators=ind))


# ------------------------------------------------------ check_drawdown_halt

class TestDrawdownHalt:
    def test_no_drawdown_passes(self):
        check_drawdown_halt(_make_ctx())

    def test_below_threshold_passes(self):
        portfolio = PortfolioData(
            btc_position_usd=0, cash_usd=8_600,
            current_drawdown_pct=0.14,   # 14% — under 15% limit
            peak_portfolio_value=10_000,
        )
        check_drawdown_halt(_make_ctx(portfolio=portfolio))

    def test_at_threshold_passes(self):
        # 15% exactly is NOT over the threshold (strict >)
        portfolio = PortfolioData(
            btc_position_usd=0, cash_usd=8_500,
            current_drawdown_pct=0.15,
            peak_portfolio_value=10_000,
        )
        check_drawdown_halt(_make_ctx(portfolio=portfolio))

    def test_above_threshold_raises(self):
        portfolio = PortfolioData(
            btc_position_usd=0, cash_usd=8_400,
            current_drawdown_pct=0.16,   # 16% — exceeds 15% limit
            peak_portfolio_value=10_000,
        )
        with pytest.raises(HardRuleViolation, match="Drawdown halt"):
            check_drawdown_halt(_make_ctx(portfolio=portfolio))

    def test_severe_drawdown_raises(self):
        portfolio = PortfolioData(
            btc_position_usd=0, cash_usd=5_000,
            current_drawdown_pct=0.50,
            peak_portfolio_value=10_000,
        )
        with pytest.raises(HardRuleViolation):
            check_drawdown_halt(_make_ctx(portfolio=portfolio))


# ----------------------------------------------------- check_volatility_halt

class TestVolatilityHalt:
    def _ind(self, atr_14: float, atr_30_avg: float) -> IndicatorData:
        return IndicatorData(
            rsi_14=58, macd_signal="bullish", bb_position="mid",
            ema_20=81_400, ema_50=78_200, ema_200=69_800,
            atr_14=atr_14, atr_30_avg=atr_30_avg, regime="trending",
        )

    def test_normal_volatility_passes(self):
        # ATR-14 = 2 100, avg = 1 800 → ratio ≈ 1.17 (below 2×)
        check_volatility_halt(_make_ctx())

    def test_exactly_2x_passes(self):
        # 2× is not strictly greater than 2×
        ind = self._ind(atr_14=3_600, atr_30_avg=1_800)
        check_volatility_halt(_make_ctx(indicators=ind))

    def test_above_2x_raises(self):
        ind = self._ind(atr_14=3_601, atr_30_avg=1_800)
        with pytest.raises(HardRuleViolation, match="Volatility halt"):
            check_volatility_halt(_make_ctx(indicators=ind))

    def test_zero_avg_skips_check(self):
        # If atr_30_avg == 0 (stub data), the check must be skipped gracefully
        ind = self._ind(atr_14=99_999, atr_30_avg=0.0)
        check_volatility_halt(_make_ctx(indicators=ind))  # must not raise


# ------------------------------------------------------- validate_context (integration)

class TestValidateContext:
    def test_valid_context_passes_all_checks(self):
        validate_context(_make_ctx())

    def test_stale_context_fails(self):
        stale_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
        with pytest.raises(ValidationError):
            validate_context(_make_ctx(timestamp=stale_ts))

    def test_drawdown_halt_propagates(self):
        portfolio = PortfolioData(
            btc_position_usd=0, cash_usd=8_000,
            current_drawdown_pct=0.20,
            peak_portfolio_value=10_000,
        )
        with pytest.raises(HardRuleViolation):
            validate_context(_make_ctx(portfolio=portfolio))
