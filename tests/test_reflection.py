"""Tests for src/agents/reflection.py using a mocked Anthropic client."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.reflection import run_reflection
from src.agents.base import AgentError
from src.models import (
    AgentWeights,
    CouncilOutputs,
    DeliberationOutput,
    FundamentalAnalystOutput,
    RiskManagerOutput,
    SentimentAnalystOutput,
    TechnicalAnalystOutput,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def outputs():
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
def deliberation():
    return DeliberationOutput(
        final_signal="BUY",
        conviction="high",
        consensus_summary="All agents bullish",
        key_disagreements=[],
        agent_weights_applied=AgentWeights(technical=0.3, sentiment=0.25, fundamental=0.3, risk=0.15),
    )


@pytest.fixture()
def profitable_trade():
    return {
        "id": 1,
        "side": "BUY",
        "entry_price": 83500.0,
        "exit_price": 88000.0,
        "stop_loss": 80200.0,
        "position_size_usd": 1500.0,
        "position_size_btc": 0.018,
        "pnl_usd": 81.0,
        "pnl_pct": 5.4,
        "status": "closed",
        "cycle_id": 1,
    }


@pytest.fixture()
def losing_trade():
    return {
        "id": 2,
        "side": "BUY",
        "entry_price": 83500.0,
        "exit_price": 80200.0,
        "stop_loss": 80200.0,
        "position_size_usd": 1500.0,
        "position_size_btc": 0.018,
        "pnl_usd": -59.4,
        "pnl_pct": -3.96,
        "status": "stopped",
        "cycle_id": 1,
    }


def _mock_client(response_text: str) -> MagicMock:
    content = MagicMock()
    content.text = response_text
    message = MagicMock()
    message.content = [content]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    return client


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestRunReflection:
    async def test_returns_string(self, outputs, deliberation, profitable_trade):
        client = _mock_client("The technical analyst was most accurate. RSI was indeed oversold.")
        result = await run_reflection(outputs, deliberation, profitable_trade, client)
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_strips_whitespace(self, outputs, deliberation, profitable_trade):
        client = _mock_client("  \n  Reflection text.  \n  ")
        result = await run_reflection(outputs, deliberation, profitable_trade, client)
        assert result == "Reflection text."

    async def test_calls_model_with_temperature_zero(self, outputs, deliberation, profitable_trade):
        client = _mock_client("Reflection.")
        await run_reflection(outputs, deliberation, profitable_trade, client)
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    async def test_user_message_contains_trade_outcome(self, outputs, deliberation, profitable_trade):
        client = _mock_client("Reflection.")
        await run_reflection(outputs, deliberation, profitable_trade, client)
        user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "83,500" in user_content   # entry price (formatted with commas)
        assert "88,000" in user_content   # exit price
        assert "closed" in user_content  # status

    async def test_user_message_contains_agent_outputs(self, outputs, deliberation, profitable_trade):
        client = _mock_client("Reflection.")
        await run_reflection(outputs, deliberation, profitable_trade, client)
        user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Technical Analyst" in user_content
        assert "Sentiment Analyst" in user_content
        assert "Fundamental Analyst" in user_content
        assert "Risk Manager" in user_content

    async def test_works_with_losing_trade(self, outputs, deliberation, losing_trade):
        client = _mock_client("Stop was hit. Sentiment was overly optimistic.")
        result = await run_reflection(outputs, deliberation, losing_trade, client)
        assert isinstance(result, str)

    async def test_raises_agent_error_on_api_failure(self, outputs, deliberation, profitable_trade):
        import anthropic
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=anthropic.APIError(
            message="API error", request=MagicMock(), body={}
        ))
        with pytest.raises(AgentError, match="Reflection agent API error"):
            await run_reflection(outputs, deliberation, profitable_trade, client)

    async def test_prompt_file_loads_without_error(self, outputs, deliberation, profitable_trade):
        """Ensure the prompt file exists and is non-empty."""
        from src.agents.base import load_prompt
        prompt = load_prompt("reflection_v1.txt")
        assert len(prompt) > 50
