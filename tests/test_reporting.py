"""Tests for src/reporting/metrics.py and src/reporting/weekly.py."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.schema import create_all, get_engine
from src.db.store import (
    log_cycle,
    open_trade,
    close_trade,
    save_reflection,
    update_portfolio_state,
)
from src.reporting.metrics import (
    compute_all_metrics,
    compute_council_consistency,
    compute_expectancy,
    compute_max_drawdown,
    compute_sharpe_ratio,
)
from src.reporting.weekly import _build_weekly_user_message, run_weekly_summary
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


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def engine():
    eng = get_engine(":memory:")
    create_all(eng)
    return eng


def _make_ctx(cash: float = 10000.0) -> MarketContext:
    return MarketContext(
        timestamp=datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc),
        asset="BTC/USD",
        price=PriceData(
            current=83500.0, open_24h=81200.0, high_24h=84100.0,
            low_24h=80900.0, change_pct_24h=2.8,
            volume_24h_usd=1_850_000_000.0, volume_vs_7d_avg=1.4,
        ),
        indicators=IndicatorData(
            rsi_14=58.2, macd_signal="bullish_cross", bb_position="mid",
            ema_20=81400.0, ema_50=78200.0, ema_200=69800.0,
            atr_14=2100.0, atr_30_avg=1800.0, regime="trending",
        ),
        news=[NewsItem(headline="Test", source="src", published_at="2026-04-05T00:00:00Z")],
        sentiment=SentimentData(fear_greed_index=62, reddit_sentiment="bullish", social_volume_vs_avg=1.2),
        onchain=OnchainData(exchange_net_flow_btc=-4200.0, whale_transactions_24h=183, sopr=1.04),
        portfolio=PortfolioData(
            btc_position_usd=0.0, cash_usd=cash,
            current_drawdown_pct=0.0, peak_portfolio_value=cash,
        ),
    )


def _make_outputs(direction: str = "BUY", veto: bool = False) -> CouncilOutputs:
    return CouncilOutputs(
        technical=TechnicalAnalystOutput(
            direction=direction, confidence=72, timeframe="24h",
            key_signals=["RSI momentum"], invalidation_level=80000.0,
        ),
        sentiment=SentimentAnalystOutput(
            direction=direction, confidence=65, overall_sentiment="bullish",
            dominant_narrative="ETF inflows", high_impact_events=[],
            sentiment_vs_price_divergence=False,
        ),
        fundamental=FundamentalAnalystOutput(
            direction=direction, confidence=68, onchain_bias="accumulation",
            macro_bias="risk-on", key_factors=["Exchange outflows"],
        ),
        risk=RiskManagerOutput(
            veto=veto, veto_reason="drawdown" if veto else None,
            approved_position_size_pct=15.0,
            recommended_stop_loss=80200.0, recommended_take_profit=88000.0,
            risk_reward_ratio=2.1, notes="",
        ),
    )


def _make_deliberation(signal: str = "BUY", veto: bool = False) -> DeliberationOutput:
    return DeliberationOutput(
        final_signal=signal,
        conviction="high",
        consensus_summary="All agents agree",
        key_disagreements=[],
        agent_weights_applied=AgentWeights(technical=0.3, sentiment=0.25, fundamental=0.3, risk=0.15),
    )


def _seed_closed_trade(engine, pnl_usd: float, entry: float = 80000.0, offset_days: int = 0) -> int:
    """Insert a cycle + closed trade with the given PnL. Returns trade_id."""
    ctx = _make_ctx()
    outputs = _make_outputs()
    delib = _make_deliberation()
    cycle_id = log_cycle(engine, ctx, outputs, delib)
    trade_id = open_trade(
        engine, cycle_id=cycle_id, symbol="BTC/USDT", side="BUY",
        entry_price=entry, position_size_usd=1000.0, stop_loss=75000.0,
    )
    # exit price to produce the desired pnl: pnl = (exit - entry) * btc
    btc = 1000.0 / entry
    exit_price = entry + pnl_usd / btc
    closed = close_trade(engine, trade_id, exit_price=exit_price)
    # Backdate exit_time so windowed queries work
    if offset_days:
        from sqlalchemy import update
        from src.db.schema import trades as trades_table
        new_time = datetime(2026, 4, 5, tzinfo=timezone.utc) - timedelta(days=offset_days)
        with engine.begin() as conn:
            conn.execute(
                update(trades_table)
                .where(trades_table.c.id == trade_id)
                .values(exit_time=new_time)
            )
    return trade_id


def _mock_llm_client(text: str = "Weekly summary text.") -> MagicMock:
    content = MagicMock()
    content.text = text
    msg = MagicMock()
    msg.content = [content]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=msg)
    return client


# ═══════════════════════════════════════════════════════════════════
# metrics.py
# ═══════════════════════════════════════════════════════════════════

class TestComputeSharpeRatio:
    def test_returns_none_with_no_trades(self):
        assert compute_sharpe_ratio([]) is None

    def test_returns_none_with_one_trade(self):
        assert compute_sharpe_ratio([{"pnl_pct": 5.0}]) is None

    def test_positive_sharpe_on_uniform_gains(self):
        # All identical returns → std = 0 → None (not computable)
        trades = [{"pnl_pct": 2.0}] * 5
        assert compute_sharpe_ratio(trades) is None

    def test_positive_mean_with_spread(self):
        trades = [{"pnl_pct": 3.0}, {"pnl_pct": 1.0}, {"pnl_pct": 5.0}]
        result = compute_sharpe_ratio(trades)
        assert result is not None
        assert result > 0

    def test_negative_sharpe_when_mean_negative(self):
        trades = [{"pnl_pct": -3.0}, {"pnl_pct": -1.0}, {"pnl_pct": -5.0}]
        result = compute_sharpe_ratio(trades)
        assert result is not None
        assert result < 0

    def test_formula_correctness(self):
        # mean=2, std=1 (population ddof=1 for sample std of [1,2,3])
        # annualised: 2/1 * sqrt(252) ≈ 31.75
        import math
        trades = [{"pnl_pct": 1.0}, {"pnl_pct": 2.0}, {"pnl_pct": 3.0}]
        result = compute_sharpe_ratio(trades)
        expected = 2.0 / 1.0 * math.sqrt(252)
        assert abs(result - expected) < 0.01

    def test_ignores_none_pnl_pct(self):
        trades = [{"pnl_pct": None}, {"pnl_pct": 2.0}, {"pnl_pct": 4.0}]
        result = compute_sharpe_ratio(trades)
        assert result is not None  # uses only the 2 valid values


class TestComputeMaxDrawdown:
    def test_zero_when_no_trades(self, engine):
        assert compute_max_drawdown(engine) == 0.0

    def test_zero_when_only_wins(self, engine):
        _seed_closed_trade(engine, pnl_usd=200.0)
        _seed_closed_trade(engine, pnl_usd=300.0)
        assert compute_max_drawdown(engine) == 0.0

    def test_nonzero_after_loss(self, engine):
        _seed_closed_trade(engine, pnl_usd=500.0)
        _seed_closed_trade(engine, pnl_usd=-2000.0)
        dd = compute_max_drawdown(engine)
        assert dd > 0

    def test_drawdown_formula(self, engine):
        # start=10000, +500 → peak=10500, -2000 → trough=8500
        # dd = (10500 - 8500) / 10500 = 2000/10500 ≈ 0.190476
        _seed_closed_trade(engine, pnl_usd=500.0)
        _seed_closed_trade(engine, pnl_usd=-2000.0)
        dd = compute_max_drawdown(engine)
        expected = 2000.0 / 10500.0
        assert abs(dd - expected) < 0.001

    def test_resets_after_recovery(self, engine):
        # loss then bigger recovery → drawdown from loss phase only
        _seed_closed_trade(engine, pnl_usd=-1000.0)
        _seed_closed_trade(engine, pnl_usd=2000.0)
        dd = compute_max_drawdown(engine)
        # peak stayed at 10000 until recovery brought it to 11000; only dip was -1000
        assert dd > 0
        assert dd < 0.15  # 1000/10000 = 10%


class TestComputeExpectancy:
    def test_empty_returns_zeros(self):
        result = compute_expectancy([])
        assert result["expectancy_usd"] == 0.0
        assert result["positive"] is False

    def test_positive_expectancy(self):
        trades = [
            {"pnl_usd": 200.0}, {"pnl_usd": 150.0},
            {"pnl_usd": -50.0},
        ]
        result = compute_expectancy(trades)
        assert result["positive"] is True
        assert result["win_rate"] == pytest.approx(2 / 3, abs=0.001)
        assert result["expectancy_usd"] > 0

    def test_negative_expectancy(self):
        trades = [
            {"pnl_usd": 50.0},
            {"pnl_usd": -200.0}, {"pnl_usd": -150.0},
        ]
        result = compute_expectancy(trades)
        assert result["positive"] is False
        assert result["expectancy_usd"] < 0

    def test_win_rate_100_percent(self):
        trades = [{"pnl_usd": 100.0}, {"pnl_usd": 200.0}]
        result = compute_expectancy(trades)
        assert result["win_rate"] == 1.0
        assert result["avg_loss_usd"] == 0.0

    def test_formula(self):
        # wins=[200,100] avg_win=150, losses=[50] avg_loss=50
        # win_rate=2/3, loss_rate=1/3
        # expectancy = 150*(2/3) - 50*(1/3) = 100 - 16.67 ≈ 83.33
        trades = [{"pnl_usd": 200.0}, {"pnl_usd": 100.0}, {"pnl_usd": -50.0}]
        result = compute_expectancy(trades)
        expected = 150 * (2 / 3) - 50 * (1 / 3)
        assert abs(result["expectancy_usd"] - expected) < 0.01


class TestComputeCouncilConsistency:
    def test_zero_cycles_returns_zeros(self, engine):
        result = compute_council_consistency(engine)
        assert result["total_cycles"] == 0
        assert result["unanimous_pct"] == 0.0
        assert result["veto_rate"] == 0.0

    def test_unanimous_cycle_counted(self, engine):
        ctx = _make_ctx()
        outputs = _make_outputs("BUY")       # all three directional agents: BUY
        delib = _make_deliberation("BUY")
        log_cycle(engine, ctx, outputs, delib)

        result = compute_council_consistency(engine)
        assert result["total_cycles"] == 1
        assert result["unanimous_pct"] == 1.0

    def test_split_cycle_not_unanimous(self, engine):
        # Manually craft outputs where technical=BUY, others=SELL
        ctx = _make_ctx()
        outputs = _make_outputs("BUY")
        # Mutate sentiment and fundamental to SELL via model_copy
        outputs = CouncilOutputs(
            technical=outputs.technical,
            sentiment=SentimentAnalystOutput(
                direction="SELL", confidence=60, overall_sentiment="bearish",
                dominant_narrative="test", high_impact_events=[],
                sentiment_vs_price_divergence=False,
            ),
            fundamental=FundamentalAnalystOutput(
                direction="SELL", confidence=55, onchain_bias="distribution",
                macro_bias="risk-off", key_factors=["outflows"],
            ),
            risk=outputs.risk,
        )
        delib = _make_deliberation("HOLD")
        log_cycle(engine, ctx, outputs, delib)

        result = compute_council_consistency(engine)
        assert result["unanimous_pct"] == 0.0

    def test_veto_rate(self, engine):
        ctx = _make_ctx()
        vetoed_outputs = _make_outputs("BUY", veto=True)
        normal_outputs = _make_outputs("BUY", veto=False)
        log_cycle(engine, ctx, vetoed_outputs, _make_deliberation("HOLD", veto=True))
        log_cycle(engine, ctx, normal_outputs, _make_deliberation("BUY"))

        result = compute_council_consistency(engine)
        assert result["veto_rate"] == pytest.approx(0.5, abs=0.001)


class TestComputeAllMetrics:
    def test_returns_expected_keys(self, engine):
        result = compute_all_metrics(engine)
        for key in ("trade_count", "sharpe_ratio", "max_drawdown_pct",
                    "expectancy", "consistency",
                    "meets_sharpe_target", "meets_drawdown_target", "meets_expectancy_target"):
            assert key in result

    def test_zero_trades(self, engine):
        result = compute_all_metrics(engine)
        assert result["trade_count"] == 0
        assert result["sharpe_ratio"] is None
        assert result["meets_sharpe_target"] is False

    def test_meets_drawdown_target_initially(self, engine):
        result = compute_all_metrics(engine)
        assert result["meets_drawdown_target"] is True  # 0% < 20%

    def test_target_flags_accurate(self, engine):
        # Seed winning trades that should give positive expectancy
        _seed_closed_trade(engine, pnl_usd=200.0)
        _seed_closed_trade(engine, pnl_usd=300.0)
        result = compute_all_metrics(engine)
        assert result["meets_expectancy_target"] is True
        assert result["meets_drawdown_target"] is True


# ═══════════════════════════════════════════════════════════════════
# weekly.py
# ═══════════════════════════════════════════════════════════════════

class TestBuildWeeklyUserMessage:
    def _make_trade(self, tid: int = 1, pnl: float = 100.0) -> dict:
        return {
            "id": tid, "side": "BUY", "symbol": "BTC/USDT",
            "entry_price": 80000.0, "exit_price": 81000.0,
            "stop_loss": 75000.0, "pnl_usd": pnl, "pnl_pct": 1.25,
            "status": "closed",
        }

    def _make_metrics(self) -> dict:
        return {
            "period_start": "2026-04-01T00:00:00",
            "period_end": "2026-04-07T00:00:00",
            "trade_count": 1,
            "sharpe_ratio": 2.1,
            "max_drawdown_pct": 5.0,
            "expectancy": {"win_rate": 1.0, "avg_win_usd": 100.0,
                           "avg_loss_usd": 0.0, "expectancy_usd": 100.0, "positive": True},
            "consistency": {"total_cycles": 7, "unanimous_pct": 0.71,
                            "avg_confidence_spread": 8.0, "veto_rate": 0.0},
            "meets_sharpe_target": True,
            "meets_drawdown_target": True,
            "meets_expectancy_target": True,
        }

    def test_contains_metrics_section(self):
        msg = _build_weekly_user_message(
            [self._make_trade()], {1: "Tech analyst was accurate."},
            [], self._make_metrics()
        )
        assert "Performance Metrics" in msg
        assert "Sharpe" in msg

    def test_contains_trade_pnl(self):
        msg = _build_weekly_user_message(
            [self._make_trade(pnl=100.0)], {}, [], self._make_metrics()
        )
        assert "80,000" in msg   # entry price formatted

    def test_contains_reflection_text(self):
        msg = _build_weekly_user_message(
            [self._make_trade()],
            {1: "Technical analyst was most predictive."},
            [], self._make_metrics()
        )
        assert "Technical analyst was most predictive." in msg

    def test_flags_missing_reflection(self):
        msg = _build_weekly_user_message(
            [self._make_trade()], {}, [], self._make_metrics()
        )
        assert "none recorded" in msg

    def test_no_trades_handled_gracefully(self):
        msg = _build_weekly_user_message([], {}, [], self._make_metrics())
        assert "No trades closed" in msg

    def test_hold_counts_present(self):
        cycle_rows = [
            {"signal": "HOLD", "conviction": "low", "vetoed": False},
            {"signal": "BUY", "conviction": "high", "vetoed": False},
        ]
        msg = _build_weekly_user_message([], {}, cycle_rows, self._make_metrics())
        assert "HOLD" in msg
        assert "BUY" in msg

    def test_target_checkmarks_present(self):
        msg = _build_weekly_user_message([], {}, [], self._make_metrics())
        assert "✓" in msg or "✗" in msg


class TestRunWeeklySummary:
    async def test_returns_string(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        client = _mock_llm_client("Great week, technical analyst was accurate.")
        result = await run_weekly_summary(engine, client)
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_strips_whitespace(self, engine):
        client = _mock_llm_client("  Summary text.  ")
        result = await run_weekly_summary(engine, client)
        assert result == "Summary text."

    async def test_calls_llm_exactly_once(self, engine):
        client = _mock_llm_client("Summary.")
        await run_weekly_summary(engine, client)
        client.messages.create.assert_called_once()

    async def test_summary_persisted_to_db(self, engine):
        from src.db.store import get_weekly_summaries
        client = _mock_llm_client("Summary saved to DB.")
        await run_weekly_summary(engine, client)
        summaries = get_weekly_summaries(engine)
        assert len(summaries) == 1
        assert "Summary saved to DB." in summaries[0]["summary"]

    async def test_temperature_zero(self, engine):
        client = _mock_llm_client("Summary.")
        await run_weekly_summary(engine, client)
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["temperature"] == 0

    async def test_system_prompt_loaded(self, engine):
        client = _mock_llm_client("Summary.")
        await run_weekly_summary(engine, client)
        kwargs = client.messages.create.call_args.kwargs
        assert len(kwargs["system"]) > 50  # prompt file is non-trivial

    async def test_user_message_contains_metrics(self, engine):
        client = _mock_llm_client("Summary.")
        await run_weekly_summary(engine, client)
        user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Performance Metrics" in user_content

    async def test_raises_agent_error_on_api_failure(self, engine):
        import anthropic as ant
        from src.agents.base import AgentError
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=ant.APIError(
            message="fail", request=MagicMock(), body={}
        ))
        with pytest.raises(AgentError, match="Weekly summary agent API error"):
            await run_weekly_summary(engine, client)

    async def test_empty_week_does_not_crash(self, engine):
        client = _mock_llm_client("No trades this week.")
        result = await run_weekly_summary(engine, client)
        assert isinstance(result, str)

    async def test_includes_trade_data_when_present(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        _seed_closed_trade(engine, pnl_usd=200.0)
        client = _mock_llm_client("Summary with trade data.")
        await run_weekly_summary(engine, client)
        user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Trade #" in user_content

    async def test_prompt_file_loads(self, engine):
        from src.agents.base import load_prompt
        prompt = load_prompt("weekly_summary_v1.txt")
        assert len(prompt) > 100


# ═══════════════════════════════════════════════════════════════════
# store.py — new windowed query functions (TestWeeklySummaries
# and windowed queries extend test_db.py coverage here)
# ═══════════════════════════════════════════════════════════════════

class TestWeeklySummaries:
    async def test_save_and_retrieve(self, engine):
        from src.db.store import get_weekly_summaries, save_weekly_summary
        now = datetime.now(timezone.utc)
        sid = save_weekly_summary(
            engine,
            window_start=now - timedelta(days=7),
            window_end=now,
            summary="Test weekly summary.",
            metrics={"trade_count": 3},
        )
        assert sid > 0
        rows = get_weekly_summaries(engine)
        assert len(rows) == 1
        assert "Test weekly summary." in rows[0]["summary"]

    async def test_get_weekly_summaries_newest_first(self, engine):
        from src.db.store import get_weekly_summaries, save_weekly_summary
        import asyncio
        now = datetime.now(timezone.utc)
        save_weekly_summary(engine, now - timedelta(days=14), now - timedelta(days=7), "week 1", {})
        save_weekly_summary(engine, now - timedelta(days=7), now, "week 2", {})
        rows = get_weekly_summaries(engine)
        assert rows[0]["summary"] == "week 2"

    def test_get_closed_trades_in_window(self, engine):
        from src.db.store import get_closed_trades_in_window
        _seed_closed_trade(engine, pnl_usd=100.0, offset_days=3)   # inside window
        _seed_closed_trade(engine, pnl_usd=200.0, offset_days=10)  # outside window

        since = datetime(2026, 4, 5, tzinfo=timezone.utc) - timedelta(days=5)
        until = datetime(2026, 4, 5, tzinfo=timezone.utc)
        rows = get_closed_trades_in_window(engine, since, until)
        assert len(rows) == 1
        assert abs(rows[0]["pnl_usd"] - 100.0) < 1.0

    def test_get_reflections_for_trades(self, engine):
        from src.db.store import get_reflections_for_trades
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        trade_id = _seed_closed_trade(engine, pnl_usd=100.0)
        save_reflection(engine, trade_id, "Great call by tech analyst.")

        result = get_reflections_for_trades(engine, [trade_id])
        assert result[trade_id] == "Great call by tech analyst."

    def test_get_reflections_empty_list(self, engine):
        from src.db.store import get_reflections_for_trades
        assert get_reflections_for_trades(engine, []) == {}

    def test_get_cycles_in_window(self, engine):
        from src.db.store import get_cycles_in_window
        ctx = _make_ctx()
        outputs = _make_outputs()
        delib = _make_deliberation()

        # Insert two cycles with different timestamps
        from src.db.schema import cycles as cycles_table
        from sqlalchemy import update as sa_update

        cid1 = log_cycle(engine, ctx, outputs, delib)
        cid2 = log_cycle(engine, ctx, outputs, delib)

        # Backdate cid1 to be outside the query window
        with engine.begin() as conn:
            conn.execute(
                sa_update(cycles_table)
                .where(cycles_table.c.id == cid1)
                .values(timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc))
            )

        since = datetime(2026, 4, 1, tzinfo=timezone.utc)
        until = datetime(2026, 4, 30, tzinfo=timezone.utc)
        rows = get_cycles_in_window(engine, since, until)
        assert len(rows) == 1
        assert rows[0]["id"] == cid2
