"""Tests for src/db/schema.py and src/db/store.py using an in-memory SQLite DB."""

from datetime import datetime, timezone

import pytest

from src.db.schema import create_all, get_engine
from src.db.store import (
    close_trade,
    get_open_trade,
    get_portfolio_state,
    get_reflection,
    get_trade,
    log_cycle,
    open_trade,
    save_reflection,
    update_portfolio_state,
)
from src.models import (
    AgentWeights,
    CouncilOutputs,
    DeliberationOutput,
    FundamentalAnalystOutput,
    IndicatorData,
    MarketContext,
    NewsItem,
    OnchainData,
    PortfolioData,
    PriceData,
    RiskManagerOutput,
    SentimentAnalystOutput,
    SentimentData,
    TechnicalAnalystOutput,
)


@pytest.fixture()
def engine():
    eng = get_engine(":memory:")
    create_all(eng)
    return eng


# ─── Fixtures: minimal valid models ──────────────────────────────────────────

@pytest.fixture()
def sample_ctx():
    return MarketContext(
        timestamp=datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc),
        asset="BTC/USD",
        price=PriceData(
            current=83500.0,
            open_24h=81200.0,
            high_24h=84100.0,
            low_24h=80900.0,
            change_pct_24h=2.8,
            volume_24h_usd=1_850_000_000.0,
            volume_vs_7d_avg=1.4,
        ),
        indicators=IndicatorData(
            rsi_14=58.2,
            macd_signal="bullish_cross",
            bb_position="mid",
            ema_20=81400.0,
            ema_50=78200.0,
            ema_200=69800.0,
            atr_14=2100.0,
            atr_30_avg=1800.0,
            regime="trending",
        ),
        news=[NewsItem(headline="BTC rallies", source="CryptoPanic", published_at="2026-04-05T00:00:00Z")],
        sentiment=SentimentData(fear_greed_index=62, reddit_sentiment="bullish", social_volume_vs_avg=1.2),
        onchain=OnchainData(exchange_net_flow_btc=-4200.0, whale_transactions_24h=183, sopr=1.04),
        portfolio=PortfolioData(
            btc_position_usd=0.0,
            cash_usd=10_000.0,
            current_drawdown_pct=0.0,
            peak_portfolio_value=10_000.0,
        ),
    )


@pytest.fixture()
def sample_outputs():
    return CouncilOutputs(
        technical=TechnicalAnalystOutput(
            direction="BUY", confidence=72, timeframe="24h",
            key_signals=["RSI momentum"], invalidation_level=80000.0,
        ),
        sentiment=SentimentAnalystOutput(
            direction="BUY", confidence=65, overall_sentiment="bullish",
            dominant_narrative="ETF inflows", high_impact_events=[],
            sentiment_vs_price_divergence=False,
        ),
        fundamental=FundamentalAnalystOutput(
            direction="BUY", confidence=68, onchain_bias="accumulation",
            macro_bias="risk-on", key_factors=["Exchange outflows"],
        ),
        risk=RiskManagerOutput(
            veto=False, veto_reason=None, approved_position_size_pct=15.0,
            recommended_stop_loss=80200.0, recommended_take_profit=88000.0,
            risk_reward_ratio=2.1, notes="",
        ),
    )


@pytest.fixture()
def sample_deliberation():
    return DeliberationOutput(
        final_signal="BUY",
        conviction="high",
        consensus_summary="All agents bullish",
        key_disagreements=[],
        agent_weights_applied=AgentWeights(technical=0.3, sentiment=0.25, fundamental=0.3, risk=0.15),
    )


# ─── cycles ──────────────────────────────────────────────────────────────────

class TestLogCycle:
    def test_returns_positive_int(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        assert isinstance(cycle_id, int)
        assert cycle_id > 0

    def test_stores_signal_and_conviction(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        from sqlalchemy import select
        from src.db.schema import cycles as cycles_table

        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        with engine.connect() as conn:
            row = conn.execute(
                select(cycles_table).where(cycles_table.c.id == cycle_id)
            ).mappings().one()
        assert row["signal"] == "BUY"
        assert row["conviction"] == "high"
        assert row["vetoed"] is False

    def test_increments_on_second_call(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        id1 = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        id2 = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        assert id2 == id1 + 1


# ─── trades ──────────────────────────────────────────────────────────────────

class TestOpenTrade:
    def test_creates_open_trade(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        trade_id = open_trade(
            engine, cycle_id=cycle_id, symbol="BTC/USD", side="BUY",
            entry_price=83500.0, position_size_usd=1500.0,
            stop_loss=80200.0, take_profit=88000.0,
        )
        assert isinstance(trade_id, int)

        row = get_trade(engine, trade_id)
        assert row["status"] == "open"
        assert row["side"] == "BUY"
        assert row["entry_price"] == 83500.0

    def test_position_size_btc_computed(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        trade_id = open_trade(
            engine, cycle_id=cycle_id, symbol="BTC/USD", side="BUY",
            entry_price=80000.0, position_size_usd=8000.0,
            stop_loss=75000.0,
        )
        row = get_trade(engine, trade_id)
        assert abs(row["position_size_btc"] - 0.1) < 1e-9


class TestCloseTrade:
    def _open(self, engine, cycle_id):
        return open_trade(
            engine, cycle_id=cycle_id, symbol="BTC/USD", side="BUY",
            entry_price=80000.0, position_size_usd=8000.0,
            stop_loss=75000.0, take_profit=90000.0,
        )

    def test_close_profitable_trade(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        trade_id = self._open(engine, cycle_id)

        closed = close_trade(engine, trade_id, exit_price=88000.0)
        assert closed["status"] == "closed"
        assert closed["pnl_usd"] > 0
        assert closed["exit_price"] == 88000.0

    def test_close_losing_trade(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        trade_id = self._open(engine, cycle_id)

        closed = close_trade(engine, trade_id, exit_price=75000.0, status="stopped")
        assert closed["status"] == "stopped"
        assert closed["pnl_usd"] < 0

    def test_pnl_formula(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        trade_id = self._open(engine, cycle_id)

        # entry=80000, size_btc=0.1, exit=82000 → pnl = (82000-80000)*0.1 = 200 USD
        closed = close_trade(engine, trade_id, exit_price=82000.0)
        assert abs(closed["pnl_usd"] - 200.0) < 0.01


class TestGetOpenTrade:
    def test_returns_none_when_no_open(self, engine):
        assert get_open_trade(engine) is None

    def test_returns_open_trade(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        trade_id = open_trade(
            engine, cycle_id=cycle_id, symbol="BTC/USD", side="BUY",
            entry_price=80000.0, position_size_usd=8000.0, stop_loss=75000.0,
        )
        found = get_open_trade(engine)
        assert found is not None
        assert found["id"] == trade_id

    def test_returns_none_after_close(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        trade_id = open_trade(
            engine, cycle_id=cycle_id, symbol="BTC/USD", side="BUY",
            entry_price=80000.0, position_size_usd=8000.0, stop_loss=75000.0,
        )
        close_trade(engine, trade_id, exit_price=85000.0)
        assert get_open_trade(engine) is None


# ─── portfolio_state ──────────────────────────────────────────────────────────

class TestPortfolioState:
    def test_seeds_on_first_call(self, engine):
        state = get_portfolio_state(engine)
        assert state["id"] == 1
        assert state["cash_usd"] > 0

    def test_update_persists(self, engine):
        update_portfolio_state(engine, cash_usd=9500.0, peak_value=10000.0)
        state = get_portfolio_state(engine)
        assert state["cash_usd"] == 9500.0
        assert state["peak_value"] == 10000.0

    def test_upsert_is_idempotent(self, engine):
        update_portfolio_state(engine, cash_usd=9000.0, peak_value=10000.0)
        update_portfolio_state(engine, cash_usd=9500.0, peak_value=10500.0)
        state = get_portfolio_state(engine)
        assert state["cash_usd"] == 9500.0
        assert state["peak_value"] == 10500.0


# ─── reflections ─────────────────────────────────────────────────────────────

class TestReflections:
    def test_save_and_retrieve(self, engine, sample_ctx, sample_outputs, sample_deliberation):
        cycle_id = log_cycle(engine, sample_ctx, sample_outputs, sample_deliberation)
        trade_id = open_trade(
            engine, cycle_id=cycle_id, symbol="BTC/USD", side="BUY",
            entry_price=80000.0, position_size_usd=8000.0, stop_loss=75000.0,
        )
        ref_id = save_reflection(engine, trade_id, "Great trade, technical analyst was most accurate.")
        assert ref_id > 0

        ref = get_reflection(engine, trade_id)
        assert ref is not None
        assert "technical analyst" in ref["summary"]

    def test_get_reflection_returns_none_when_absent(self, engine):
        assert get_reflection(engine, trade_id=999) is None
