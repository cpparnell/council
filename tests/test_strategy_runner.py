"""Tests for the generic strategy runner."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import (
    GenericAgentOutput,
    GenericCouncilOutputs,
    IndicatorData,
    MarketContext,
    OnchainData,
    PortfolioData,
    PriceData,
    SentimentData,
    VetoAgentOutput,
)
from src.strategies.config import AgentConfig, DeliberationConfig, RiskRules, SLTPRule, ScoringConfig, StrategyConfig
from src.strategies.runner import GenericCouncilResult, run_strategy_council


# ---------------------------------------------------------------- fixtures


def _make_ctx() -> MarketContext:
    return MarketContext(
        timestamp=datetime.now(timezone.utc),
        price=PriceData(
            current=50_000.0,
            open_24h=50_000.0,
            high_24h=51_000.0,
            low_24h=49_000.0,
            change_pct_24h=0.5,
            volume_24h_usd=1e9,
            volume_vs_7d_avg=1.1,
        ),
        indicators=IndicatorData(
            rsi_14=55.0,
            macd_signal="bullish",
            bb_position="mid",
            ema_20=49_500.0,
            ema_50=48_000.0,
            ema_200=40_000.0,
            atr_14=1_200.0,
            atr_30_avg=1_000.0,
            regime="trending",
        ),
        news=[],
        sentiment=SentimentData(fear_greed_index=60, reddit_sentiment="bullish", social_volume_vs_avg=1.2),
        onchain=OnchainData(exchange_net_flow_btc=-500.0, whale_transactions_24h=30, sopr=1.1),
        portfolio=PortfolioData(
            btc_position_usd=0.0,
            cash_usd=10_000.0,
            current_drawdown_pct=0.0,
            peak_portfolio_value=10_000.0,
        ),
    )


def _two_agent_config(tmp_path: Path) -> StrategyConfig:
    (tmp_path / "a.txt").write_text("You are agent A.")
    (tmp_path / "b.txt").write_text("You are agent B.")
    return StrategyConfig(
        name="Test",
        agents=[
            AgentConfig(name="alpha", model="claude-haiku-4-5",
                        prompt=str(tmp_path / "a.txt"), weight=0.6),
            AgentConfig(name="beta",  model="claude-haiku-4-5",
                        prompt=str(tmp_path / "b.txt"), weight=0.4),
        ],
        scoring=ScoringConfig(confidence_floor=30, trade_threshold=0.25),
        risk=RiskRules(
            stop_loss=SLTPRule(type="atr_multiple", value=2.0),
            take_profit=SLTPRule(type="atr_multiple", value=3.0),
        ),
    )


def _veto_config(tmp_path: Path) -> StrategyConfig:
    (tmp_path / "dir.txt").write_text("directional")
    (tmp_path / "veto.txt").write_text("veto agent")
    return StrategyConfig(
        name="VetoTest",
        agents=[
            AgentConfig(name="trend", model="claude-haiku-4-5",
                        prompt=str(tmp_path / "dir.txt"), weight=1.0),
            AgentConfig(name="guard", model="claude-sonnet-4-6",
                        prompt=str(tmp_path / "veto.txt"), is_veto=True),
        ],
        scoring=ScoringConfig(confidence_floor=30, trade_threshold=0.25),
        risk=RiskRules(
            stop_loss=SLTPRule(type="none"),
            take_profit=SLTPRule(type="none"),
        ),
    )


# ---------------------------------------------------------------- tests


@pytest.mark.asyncio
async def test_all_buy_agents_returns_buy(tmp_path):
    config = _two_agent_config(tmp_path)
    ctx = _make_ctx()

    buy_output = GenericAgentOutput(direction="BUY", confidence=75, reasoning="bullish")

    with patch("src.strategies.runner._call_directional", new_callable=AsyncMock) as mock_dir:
        mock_dir.return_value = buy_output
        result = await run_strategy_council(ctx, config, client=MagicMock())

    assert result.signal == "BUY"
    assert result.score > 0
    assert not result.vetoed


@pytest.mark.asyncio
async def test_all_hold_agents_returns_hold(tmp_path):
    config = _two_agent_config(tmp_path)
    ctx = _make_ctx()

    hold_output = GenericAgentOutput(direction="HOLD", confidence=40, reasoning="uncertain")

    with patch("src.strategies.runner._call_directional", new_callable=AsyncMock) as mock_dir:
        mock_dir.return_value = hold_output
        result = await run_strategy_council(ctx, config, client=MagicMock())

    assert result.signal == "HOLD"
    assert not result.vetoed


@pytest.mark.asyncio
async def test_veto_fires_returns_hold(tmp_path):
    config = _veto_config(tmp_path)
    ctx = _make_ctx()

    dir_output = GenericAgentOutput(direction="BUY", confidence=90, reasoning="strong buy")
    veto_output = VetoAgentOutput(veto=True, veto_reason="drawdown breach", reasoning="risk exceeded")

    with (
        patch("src.strategies.runner._call_directional", new_callable=AsyncMock) as mock_dir,
        patch("src.strategies.runner._call_veto", new_callable=AsyncMock) as mock_veto,
    ):
        mock_dir.return_value = dir_output
        mock_veto.return_value = veto_output
        result = await run_strategy_council(ctx, config, client=MagicMock())

    assert result.signal == "HOLD"
    assert result.vetoed
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_veto_false_does_not_block(tmp_path):
    config = _veto_config(tmp_path)
    ctx = _make_ctx()

    dir_output = GenericAgentOutput(direction="BUY", confidence=90, reasoning="bullish")
    veto_output = VetoAgentOutput(veto=False, veto_reason=None, reasoning="risk acceptable")

    with (
        patch("src.strategies.runner._call_directional", new_callable=AsyncMock) as mock_dir,
        patch("src.strategies.runner._call_veto", new_callable=AsyncMock) as mock_veto,
    ):
        mock_dir.return_value = dir_output
        mock_veto.return_value = veto_output
        result = await run_strategy_council(ctx, config, client=MagicMock())

    assert result.signal == "BUY"
    assert not result.vetoed


@pytest.mark.asyncio
async def test_sl_tp_set_on_result(tmp_path):
    config = _two_agent_config(tmp_path)
    ctx = _make_ctx()  # price=50000, atr=1200

    buy_output = GenericAgentOutput(direction="BUY", confidence=80, reasoning="up")

    with patch("src.strategies.runner._call_directional", new_callable=AsyncMock) as mock_dir:
        mock_dir.return_value = buy_output
        result = await run_strategy_council(ctx, config, client=MagicMock())

    # sl = 50000 - 2*1200 = 47600, tp = 50000 + 3*1200 = 53600
    assert result.sl_price == pytest.approx(47_600.0)
    assert result.tp_price == pytest.approx(53_600.0)


@pytest.mark.asyncio
async def test_conviction_low_for_near_zero_score(tmp_path):
    config = _two_agent_config(tmp_path)
    ctx = _make_ctx()

    hold_output = GenericAgentOutput(direction="HOLD", confidence=40, reasoning="neutral")

    with patch("src.strategies.runner._call_directional", new_callable=AsyncMock) as mock_dir:
        mock_dir.return_value = hold_output
        result = await run_strategy_council(ctx, config, client=MagicMock())

    assert result.conviction == "low"


@pytest.mark.asyncio
async def test_deliberation_failure_is_nonfatal(tmp_path):
    (tmp_path / "a.txt").write_text("agent")
    (tmp_path / "b.txt").write_text("agent b")
    (tmp_path / "delib.txt").write_text("deliberation")

    config = StrategyConfig(
        name="WithDelib",
        agents=[
            AgentConfig(name="x", model="claude-haiku-4-5",
                        prompt=str(tmp_path / "a.txt"), weight=1.0),
        ],
        deliberation=DeliberationConfig(
            enabled=True, model="claude-sonnet-4-6",
            prompt=str(tmp_path / "delib.txt"),
        ),
        scoring=ScoringConfig(),
        risk=RiskRules(stop_loss=SLTPRule(type="none"), take_profit=SLTPRule(type="none")),
    )
    ctx = _make_ctx()

    from src.agents.base import AgentError
    buy_output = GenericAgentOutput(direction="BUY", confidence=80, reasoning="up")

    with (
        patch("src.strategies.runner._call_directional", new_callable=AsyncMock) as mock_dir,
        patch("src.strategies.runner._call_deliberation", new_callable=AsyncMock) as mock_delib,
    ):
        mock_dir.return_value = buy_output
        mock_delib.side_effect = AgentError("API error")
        result = await run_strategy_council(ctx, config, client=MagicMock())

    # Should still return a valid result even if deliberation failed
    assert result.signal == "BUY"
    assert result.narrative is None


@pytest.mark.asyncio
async def test_no_veto_agent_skips_veto_call(tmp_path):
    config = _two_agent_config(tmp_path)  # no veto agent
    ctx = _make_ctx()

    buy_output = GenericAgentOutput(direction="BUY", confidence=80, reasoning="bullish")

    with (
        patch("src.strategies.runner._call_directional", new_callable=AsyncMock) as mock_dir,
        patch("src.strategies.runner._call_veto", new_callable=AsyncMock) as mock_veto,
    ):
        mock_dir.return_value = buy_output
        await run_strategy_council(ctx, config, client=MagicMock())

    mock_veto.assert_not_called()
