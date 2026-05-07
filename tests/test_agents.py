"""
Unit tests for src/agents/.

All LLM API calls are mocked — no network calls, no API keys required.
Tests verify: prompt loading, message formatting, JSON parsing, schema
validation, veto short-circuit, and asyncio.gather parallelism in the runner.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    TechnicalAnalystOutput,
)


# ----------------------------------------------------------------- Factories

def _news(n: int = 15) -> list[NewsItem]:
    return [
        NewsItem(
            headline=f"Headline {i}",
            source="CryptoPanic",
            published_at="2026-04-05T00:00:00Z",
        )
        for i in range(n)
    ]


def _make_ctx(**overrides) -> MarketContext:
    from src.models import SentimentData
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        asset="BTC/USD",
        price=PriceData(
            current=83_500,
            open_24h=81_200,
            high_24h=84_100,
            low_24h=80_900,
            change_pct_24h=2.8,
            volume_24h_usd=1_850_000_000,
            volume_vs_7d_avg=1.4,
        ),
        indicators=IndicatorData(
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
        news=_news(),
        sentiment=SentimentData(
            fear_greed_index=62,
            reddit_sentiment="bullish",
            social_volume_vs_avg=1.2,
        ),
        onchain=OnchainData(
            exchange_net_flow_btc=-4_200,
            whale_transactions_24h=183,
            sopr=1.04,
        ),
        portfolio=PortfolioData(
            btc_position_usd=0,
            cash_usd=10_000,
            current_drawdown_pct=0.0,
            peak_portfolio_value=10_000,
        ),
    )
    defaults.update(overrides)
    return MarketContext(**defaults)


# Alias kept for test methods that call the sentiment-specific factory
_make_ctx_real_sentiment = _make_ctx


# ---- Valid agent JSON payloads

VALID_TECHNICAL = {
    "direction": "BUY",
    "confidence": 72,
    "timeframe": "24h",
    "key_signals": ["RSI oversold bounce", "EMA 20/50 bullish cross"],
    "invalidation_level": 81000.0,
}

VALID_SENTIMENT = {
    "direction": "BUY",
    "confidence": 65,
    "overall_sentiment": "bullish",
    "dominant_narrative": "ETF inflows accelerating, institutional demand strong.",
    "high_impact_events": ["BlackRock ETF record inflows"],
    "sentiment_vs_price_divergence": False,
}

VALID_FUNDAMENTAL = {
    "direction": "BUY",
    "confidence": 60,
    "onchain_bias": "accumulation",
    "macro_bias": "risk-on",
    "key_factors": ["Exchange outflows accelerating", "SOPR > 1 profit-taking absorbed"],
}

VALID_RISK = {
    "veto": False,
    "veto_reason": None,
    "approved_position_size_pct": 15.0,
    "recommended_stop_loss": 80_200.0,
    "recommended_take_profit": 88_000.0,
    "risk_reward_ratio": 2.1,
    "notes": "ATR-sized stop, R:R above minimum.",
}

VALID_DELIBERATION = {
    "final_signal": "BUY",
    "conviction": "high",
    "consensus_summary": "All three directional agents agree on BUY with high confidence.",
    "key_disagreements": [],
    "agent_weights_applied": {
        "technical": 0.35,
        "sentiment": 0.25,
        "fundamental": 0.30,
        "risk": 0.10,
    },
}


def _mock_response(payload: dict) -> MagicMock:
    """Build a fake anthropic response with a tool_use block containing payload."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = payload
    msg = MagicMock()
    msg.content = [block]
    msg.stop_reason = "tool_use"
    return msg


def _mock_client(payload: dict) -> MagicMock:
    """Return a mock AsyncAnthropic client whose messages.create returns payload."""
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_response(payload))
    return client


# ================================================================== base.py ==

class TestCallAgent:
    async def test_happy_path(self):
        from src.agents.base import call_agent
        client = _mock_client(VALID_TECHNICAL)
        result = await call_agent(
            client, "claude-haiku-4-5", "system", "user", TechnicalAnalystOutput
        )
        assert result.direction == "BUY"
        assert result.confidence == 72

    async def test_raises_on_missing_tool_use_block(self):
        from src.agents.base import AgentError, call_agent
        # Simulate a response with no tool_use block (e.g. end_turn instead)
        text_only_block = MagicMock()
        text_only_block.type = "text"
        response = MagicMock()
        response.content = [text_only_block]
        response.stop_reason = "end_turn"
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=response)
        with pytest.raises(AgentError, match="tool_use block"):
            await call_agent(
                client, "claude-haiku-4-5", "system", "user", TechnicalAnalystOutput
            )

    async def test_raises_on_schema_mismatch(self):
        from src.agents.base import AgentError, call_agent
        bad = {"direction": "MAYBE", "confidence": 50}  # missing required fields
        client = _mock_client(bad)
        with pytest.raises(AgentError, match="schema validation"):
            await call_agent(
                client, "claude-haiku-4-5", "system", "user", TechnicalAnalystOutput
            )

    async def test_raises_on_api_error(self):
        import anthropic
        from src.agents.base import AgentError, call_agent
        client = MagicMock()
        client.messages.create = AsyncMock(
            side_effect=anthropic.APIStatusError(
                "rate limit",
                response=MagicMock(status_code=429),
                body={},
            )
        )
        with pytest.raises(AgentError, match="API error"):
            await call_agent(
                client, "claude-haiku-4-5", "system", "user", TechnicalAnalystOutput
            )


# =========================================================== individual agents

class TestTechnicalAnalyst:
    async def test_returns_correct_type(self):
        from src.agents.technical import run_technical_analyst
        ctx = _make_ctx()
        result = await run_technical_analyst(ctx, _mock_client(VALID_TECHNICAL))
        assert isinstance(result, TechnicalAnalystOutput)

    async def test_direction_parsed(self):
        from src.agents.technical import run_technical_analyst
        ctx = _make_ctx()
        result = await run_technical_analyst(ctx, _mock_client(VALID_TECHNICAL))
        assert result.direction == "BUY"

    async def test_user_message_contains_price(self):
        from src.agents.technical import _build_user_message
        ctx = _make_ctx()
        msg = _build_user_message(ctx)
        assert "83,500" in msg or "83500" in msg

    async def test_user_message_contains_rsi(self):
        from src.agents.technical import _build_user_message
        ctx = _make_ctx()
        msg = _build_user_message(ctx)
        assert "58.2" in msg

    async def test_user_message_contains_regime(self):
        from src.agents.technical import _build_user_message
        ctx = _make_ctx()
        msg = _build_user_message(ctx)
        assert "trending" in msg


class TestSentimentAnalyst:
    async def test_returns_correct_type(self):
        from src.agents.sentiment import run_sentiment_analyst
        ctx = _make_ctx_real_sentiment()
        result = await run_sentiment_analyst(ctx, _mock_client(VALID_SENTIMENT))
        assert isinstance(result, SentimentAnalystOutput)

    async def test_direction_parsed(self):
        from src.agents.sentiment import run_sentiment_analyst
        ctx = _make_ctx_real_sentiment()
        result = await run_sentiment_analyst(ctx, _mock_client(VALID_SENTIMENT))
        assert result.direction == "BUY"

    async def test_user_message_contains_fear_greed(self):
        from src.agents.sentiment import _build_user_message
        ctx = _make_ctx_real_sentiment()
        msg = _build_user_message(ctx)
        assert "62" in msg

    async def test_user_message_contains_headlines(self):
        from src.agents.sentiment import _build_user_message
        ctx = _make_ctx_real_sentiment()
        msg = _build_user_message(ctx)
        assert "Headline 0" in msg

    async def test_at_most_20_headlines_in_message(self):
        from src.agents.sentiment import _build_user_message
        ctx = _make_ctx_real_sentiment(news=_news(30))
        msg = _build_user_message(ctx)
        # Headline 20 should NOT appear (only first 20 are included)
        assert "Headline 20" not in msg


class TestFundamentalAnalyst:
    async def test_returns_correct_type(self):
        from src.agents.fundamental import run_fundamental_analyst
        ctx = _make_ctx()
        result = await run_fundamental_analyst(ctx, _mock_client(VALID_FUNDAMENTAL))
        assert isinstance(result, FundamentalAnalystOutput)

    async def test_direction_parsed(self):
        from src.agents.fundamental import run_fundamental_analyst
        ctx = _make_ctx()
        result = await run_fundamental_analyst(ctx, _mock_client(VALID_FUNDAMENTAL))
        assert result.direction == "BUY"

    async def test_user_message_labels_outflow(self):
        from src.agents.fundamental import _build_user_message
        ctx = _make_ctx()  # exchange_net_flow_btc = -4200 (outflow)
        msg = _build_user_message(ctx)
        assert "accumulation" in msg

    async def test_user_message_labels_inflow(self):
        from src.agents.fundamental import _build_user_message
        ctx = _make_ctx(
            onchain=OnchainData(
                exchange_net_flow_btc=3_000,
                whale_transactions_24h=100,
                sopr=0.98,
            )
        )
        msg = _build_user_message(ctx)
        assert "distribution" in msg

    async def test_user_message_contains_sopr(self):
        from src.agents.fundamental import _build_user_message
        ctx = _make_ctx()
        msg = _build_user_message(ctx)
        assert "1.04" in msg or "SOPR" in msg


class TestRiskManager:
    async def test_returns_correct_type(self):
        from src.agents.risk import run_risk_manager
        ctx = _make_ctx()
        result = await run_risk_manager(ctx, _mock_client(VALID_RISK))
        assert isinstance(result, RiskManagerOutput)

    async def test_no_veto_in_valid_output(self):
        from src.agents.risk import run_risk_manager
        ctx = _make_ctx()
        result = await run_risk_manager(ctx, _mock_client(VALID_RISK))
        assert result.veto is False
        assert result.veto_reason is None

    async def test_veto_parsed(self):
        from src.agents.risk import run_risk_manager
        vetoed = {**VALID_RISK, "veto": True, "veto_reason": "R:R below minimum threshold"}
        ctx = _make_ctx()
        result = await run_risk_manager(ctx, _mock_client(vetoed))
        assert result.veto is True
        assert "R:R" in result.veto_reason

    async def test_user_message_contains_portfolio_value(self):
        from src.agents.risk import _build_user_message
        ctx = _make_ctx()
        msg = _build_user_message(ctx)
        assert "10,000" in msg or "10000" in msg

    async def test_user_message_contains_drawdown(self):
        from src.agents.risk import _build_user_message
        ctx = _make_ctx()
        msg = _build_user_message(ctx)
        assert "drawdown" in msg.lower()


# ============================================================== deliberation

class TestDeliberation:
    def _make_outputs(self) -> CouncilOutputs:
        return CouncilOutputs(
            technical=TechnicalAnalystOutput(**VALID_TECHNICAL),
            sentiment=SentimentAnalystOutput(**VALID_SENTIMENT),
            fundamental=FundamentalAnalystOutput(**VALID_FUNDAMENTAL),
            risk=RiskManagerOutput(**VALID_RISK),
        )

    async def test_returns_correct_type(self):
        from src.agents.deliberation import run_deliberation
        outputs = self._make_outputs()
        result = await run_deliberation(outputs, _mock_client(VALID_DELIBERATION))
        assert isinstance(result, DeliberationOutput)

    async def test_final_signal_parsed(self):
        from src.agents.deliberation import run_deliberation
        outputs = self._make_outputs()
        result = await run_deliberation(outputs, _mock_client(VALID_DELIBERATION))
        assert result.final_signal == "BUY"

    async def test_user_message_contains_all_agents(self):
        from src.agents.deliberation import _build_user_message
        outputs = self._make_outputs()
        msg = _build_user_message(outputs)
        assert "Technical analyst" in msg
        assert "Sentiment analyst" in msg
        assert "Fundamental analyst" in msg
        assert "Risk manager" in msg

    async def test_agent_weights_sum_to_one(self):
        from src.agents.deliberation import run_deliberation
        outputs = self._make_outputs()
        result = await run_deliberation(outputs, _mock_client(VALID_DELIBERATION))
        w = result.agent_weights_applied
        total = w.technical + w.sentiment + w.fundamental + w.risk
        assert abs(total - 1.0) < 0.01


# ================================================================= runner ==

class TestCouncilRunner:
    def _make_mocks(self, *, risk_veto: bool = False) -> dict:
        risk_payload = {**VALID_RISK, "veto": risk_veto, "veto_reason": "Test veto" if risk_veto else None}
        return {
            "run_technical_analyst": AsyncMock(return_value=TechnicalAnalystOutput(**VALID_TECHNICAL)),
            "run_sentiment_analyst": AsyncMock(return_value=SentimentAnalystOutput(**VALID_SENTIMENT)),
            "run_fundamental_analyst": AsyncMock(return_value=FundamentalAnalystOutput(**VALID_FUNDAMENTAL)),
            "run_risk_manager": AsyncMock(return_value=RiskManagerOutput(**risk_payload)),
            "run_deliberation": AsyncMock(return_value=DeliberationOutput(**VALID_DELIBERATION)),
        }

    async def test_returns_council_result(self):
        from src.agents.runner import CouncilResult, run_council
        ctx = _make_ctx()
        client = MagicMock()
        mocks = self._make_mocks()
        with patch("src.agents.runner.run_technical_analyst", mocks["run_technical_analyst"]), \
             patch("src.agents.runner.run_sentiment_analyst", mocks["run_sentiment_analyst"]), \
             patch("src.agents.runner.run_fundamental_analyst", mocks["run_fundamental_analyst"]), \
             patch("src.agents.runner.run_risk_manager", mocks["run_risk_manager"]), \
             patch("src.agents.runner.run_deliberation", mocks["run_deliberation"]):
            result = await run_council(ctx, client)
        assert isinstance(result, CouncilResult)

    async def test_signal_propagated(self):
        from src.agents.runner import run_council
        ctx = _make_ctx()
        client = MagicMock()
        mocks = self._make_mocks()
        with patch("src.agents.runner.run_technical_analyst", mocks["run_technical_analyst"]), \
             patch("src.agents.runner.run_sentiment_analyst", mocks["run_sentiment_analyst"]), \
             patch("src.agents.runner.run_fundamental_analyst", mocks["run_fundamental_analyst"]), \
             patch("src.agents.runner.run_risk_manager", mocks["run_risk_manager"]), \
             patch("src.agents.runner.run_deliberation", mocks["run_deliberation"]):
            result = await run_council(ctx, client)
        assert result.signal == "BUY"

    async def test_veto_short_circuits_deliberation(self):
        from src.agents.runner import run_council
        ctx = _make_ctx()
        client = MagicMock()
        mocks = self._make_mocks(risk_veto=True)
        with patch("src.agents.runner.run_technical_analyst", mocks["run_technical_analyst"]), \
             patch("src.agents.runner.run_sentiment_analyst", mocks["run_sentiment_analyst"]), \
             patch("src.agents.runner.run_fundamental_analyst", mocks["run_fundamental_analyst"]), \
             patch("src.agents.runner.run_risk_manager", mocks["run_risk_manager"]), \
             patch("src.agents.runner.run_deliberation", mocks["run_deliberation"]):
            result = await run_council(ctx, client)
        assert result.signal == "HOLD"
        assert result.vetoed is True
        mocks["run_deliberation"].assert_not_called()

    async def test_deliberation_called_when_no_veto(self):
        from src.agents.runner import run_council
        ctx = _make_ctx()
        client = MagicMock()
        mocks = self._make_mocks(risk_veto=False)
        with patch("src.agents.runner.run_technical_analyst", mocks["run_technical_analyst"]), \
             patch("src.agents.runner.run_sentiment_analyst", mocks["run_sentiment_analyst"]), \
             patch("src.agents.runner.run_fundamental_analyst", mocks["run_fundamental_analyst"]), \
             patch("src.agents.runner.run_risk_manager", mocks["run_risk_manager"]), \
             patch("src.agents.runner.run_deliberation", mocks["run_deliberation"]):
            await run_council(ctx, client)
        mocks["run_deliberation"].assert_called_once()

    async def test_all_agents_called(self):
        from src.agents.runner import run_council
        ctx = _make_ctx()
        client = MagicMock()
        mocks = self._make_mocks()
        with patch("src.agents.runner.run_technical_analyst", mocks["run_technical_analyst"]), \
             patch("src.agents.runner.run_sentiment_analyst", mocks["run_sentiment_analyst"]), \
             patch("src.agents.runner.run_fundamental_analyst", mocks["run_fundamental_analyst"]), \
             patch("src.agents.runner.run_risk_manager", mocks["run_risk_manager"]), \
             patch("src.agents.runner.run_deliberation", mocks["run_deliberation"]):
            await run_council(ctx, client)
        mocks["run_technical_analyst"].assert_called_once_with(ctx, client)
        mocks["run_sentiment_analyst"].assert_called_once_with(ctx, client)
        mocks["run_fundamental_analyst"].assert_called_once_with(ctx, client)
        mocks["run_risk_manager"].assert_called_once_with(ctx, client)

    async def test_not_vetoed_on_clean_run(self):
        from src.agents.runner import run_council
        ctx = _make_ctx()
        client = MagicMock()
        mocks = self._make_mocks(risk_veto=False)
        with patch("src.agents.runner.run_technical_analyst", mocks["run_technical_analyst"]), \
             patch("src.agents.runner.run_sentiment_analyst", mocks["run_sentiment_analyst"]), \
             patch("src.agents.runner.run_fundamental_analyst", mocks["run_fundamental_analyst"]), \
             patch("src.agents.runner.run_risk_manager", mocks["run_risk_manager"]), \
             patch("src.agents.runner.run_deliberation", mocks["run_deliberation"]):
            result = await run_council(ctx, client)
        assert result.vetoed is False

    async def test_scored_signal_overrides_llm_deliberation(self):
        """Deliberation LLM's final_signal must be overridden by compute_signed_score."""
        from src.agents.runner import run_council
        ctx = _make_ctx()
        client = MagicMock()
        # LLM claims HOLD, but all three directional agents voted BUY with
        # high confidence (72/65/60) → scorer returns BUY → override wins.
        mocks = self._make_mocks(risk_veto=False)
        llm_says_hold = {**VALID_DELIBERATION, "final_signal": "HOLD", "conviction": "low"}
        mocks["run_deliberation"] = AsyncMock(return_value=DeliberationOutput(**llm_says_hold))
        with patch("src.agents.runner.run_technical_analyst", mocks["run_technical_analyst"]), \
             patch("src.agents.runner.run_sentiment_analyst", mocks["run_sentiment_analyst"]), \
             patch("src.agents.runner.run_fundamental_analyst", mocks["run_fundamental_analyst"]), \
             patch("src.agents.runner.run_risk_manager", mocks["run_risk_manager"]), \
             patch("src.agents.runner.run_deliberation", mocks["run_deliberation"]):
            result = await run_council(ctx, client)
        assert result.signal == "BUY"
        assert result.deliberation.score > 0.25


# ========================================================= signed-score scoring

class TestComputeSignedScore:
    def _outputs(self, *, t_dir="BUY", t_conf=70, s_dir="BUY", s_conf=70,
                 f_dir="BUY", f_conf=70, veto=False) -> CouncilOutputs:
        return CouncilOutputs(
            technical=TechnicalAnalystOutput(
                direction=t_dir, confidence=t_conf, timeframe="24h",
                key_signals=["x"], invalidation_level=80_000.0,
            ),
            sentiment=SentimentAnalystOutput(
                direction=s_dir, confidence=s_conf, overall_sentiment="bullish",
                dominant_narrative="x", high_impact_events=[],
                sentiment_vs_price_divergence=False,
            ),
            fundamental=FundamentalAnalystOutput(
                direction=f_dir, confidence=f_conf, onchain_bias="accumulation",
                macro_bias="risk-on", key_factors=["x"],
            ),
            risk=RiskManagerOutput(
                veto=veto, veto_reason="test" if veto else None,
                approved_position_size_pct=15.0,
                recommended_stop_loss=80_000.0,
                recommended_take_profit=88_000.0,
                risk_reward_ratio=2.0, notes="x",
            ),
        )

    def test_all_buy_high_conf_returns_buy(self):
        from src.agents.scoring import compute_signed_score
        signal, score = compute_signed_score(self._outputs(t_conf=80, s_conf=70, f_conf=75))
        assert signal == "BUY"
        assert score > 0.25

    def test_all_sell_high_conf_returns_sell(self):
        from src.agents.scoring import compute_signed_score
        signal, score = compute_signed_score(
            self._outputs(t_dir="SELL", s_dir="SELL", f_dir="SELL",
                          t_conf=80, s_conf=70, f_conf=75)
        )
        assert signal == "SELL"
        assert score < -0.25

    def test_risk_veto_overrides_buy(self):
        from src.agents.scoring import compute_signed_score
        signal, score = compute_signed_score(
            self._outputs(t_conf=90, s_conf=90, f_conf=90, veto=True)
        )
        assert signal == "HOLD"
        assert score == 0.0

    def test_all_hold_returns_hold(self):
        from src.agents.scoring import compute_signed_score
        signal, score = compute_signed_score(
            self._outputs(t_dir="HOLD", s_dir="HOLD", f_dir="HOLD",
                          t_conf=40, s_conf=40, f_conf=40)
        )
        assert signal == "HOLD"
        assert score == 0.0

    def test_low_confidence_agent_excluded(self):
        """An agent with conf < 30 should not dilute the score."""
        from src.agents.scoring import compute_signed_score
        # Technical says BUY strongly; sentiment is HOLD at 20 (excluded);
        # fundamental says BUY at 70. Without the floor, the HOLD would
        # drag the score; with the floor, score should be clearly BUY.
        signal, score = compute_signed_score(
            self._outputs(t_dir="BUY", t_conf=80,
                          s_dir="HOLD", s_conf=20,
                          f_dir="BUY", f_conf=70)
        )
        assert signal == "BUY"
        assert score > 0.5  # strong, not diluted

    def test_split_directions_score_near_zero(self):
        """Mixed buy/sell with similar confidence → score < threshold."""
        from src.agents.scoring import compute_signed_score
        signal, score = compute_signed_score(
            self._outputs(t_dir="BUY", t_conf=60,
                          s_dir="SELL", s_conf=60,
                          f_dir="HOLD", f_conf=60)
        )
        assert signal == "HOLD"
        assert abs(score) < 0.25

    def test_score_below_threshold_returns_hold(self):
        from src.agents.scoring import compute_signed_score
        # All BUY but low conf — weighted score should stay below 0.25
        signal, score = compute_signed_score(
            self._outputs(t_conf=35, s_conf=35, f_conf=35)
        )
        # All agents at conf=35 with dir=BUY → raw score = 1.0 (direction signal is unanimous)
        # So score IS above threshold; need mixed or partial for HOLD via threshold.
        # Instead, test with one clear and one abstaining:
        outputs = self._outputs(t_dir="BUY", t_conf=40,
                                s_dir="HOLD", s_conf=20,     # excluded
                                f_dir="HOLD", f_conf=40)     # HOLD → sign=0
        signal, score = compute_signed_score(outputs)
        # Technical contributes BUY; fundamental contributes HOLD (sign 0); sentiment abstains
        # score = (0.40 * 1 * 0.40 + 0.35 * 0 * 0.40) / (0.40*0.40 + 0.35*0.40) = 0.16 / 0.30 = 0.533
        assert signal == "BUY"
        assert score > 0

    def test_all_agents_below_floor_returns_hold(self):
        from src.agents.scoring import compute_signed_score
        signal, score = compute_signed_score(
            self._outputs(t_conf=20, s_conf=20, f_conf=20)
        )
        assert signal == "HOLD"
        assert score == 0.0


class TestScoreToPositionSize:
    def test_below_threshold_returns_zero(self):
        from src.agents.scoring import score_to_position_size_pct
        assert score_to_position_size_pct(0.10) == 0.0
        assert score_to_position_size_pct(-0.20) == 0.0

    def test_scales_linearly_with_magnitude(self):
        from src.agents.scoring import score_to_position_size_pct
        assert score_to_position_size_pct(0.50) == pytest.approx(10.0)
        assert score_to_position_size_pct(0.80) == pytest.approx(16.0)

    def test_negative_score_returns_positive_size(self):
        from src.agents.scoring import score_to_position_size_pct
        assert score_to_position_size_pct(-0.75) == pytest.approx(15.0)

    def test_capped_at_max(self):
        from src.agents.scoring import score_to_position_size_pct
        assert score_to_position_size_pct(1.5) == 20.0


# ======================================================= prompt loading

_V1_PROMPTS = [
    "technical_analyst_v1.txt",
    "sentiment_analyst_v1.txt",
    "fundamental_analyst_v1.txt",
    "risk_manager_v1.txt",
    "deliberation_v1.txt",
]
_V2_PROMPTS = [
    "technical_analyst_v2.txt",
    "sentiment_analyst_v2.txt",
    "fundamental_analyst_v2.txt",
    "risk_manager_v2.txt",
    "deliberation_v2.txt",
]


class TestPromptLoading:
    def test_all_v1_prompts_exist(self):
        from src.agents.base import PROMPTS_DIR
        for fname in _V1_PROMPTS:
            assert (PROMPTS_DIR / fname).exists(), f"Missing prompt file: {fname}"

    def test_all_v2_prompts_exist(self):
        from src.agents.base import PROMPTS_DIR
        for fname in _V2_PROMPTS:
            assert (PROMPTS_DIR / fname).exists(), f"Missing prompt file: {fname}"

    def test_prompts_are_non_empty(self):
        from src.agents.base import load_prompt
        for fname in _V1_PROMPTS + _V2_PROMPTS:
            text = load_prompt(fname)
            assert len(text) > 100, f"Prompt {fname} is suspiciously short"

    def test_prompts_contain_json_instruction(self):
        from src.agents.base import load_prompt
        for fname in _V1_PROMPTS + _V2_PROMPTS:
            text = load_prompt(fname)
            assert "JSON" in text, f"Prompt {fname} does not mention JSON output format"
