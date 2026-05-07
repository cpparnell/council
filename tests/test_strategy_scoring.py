"""Tests for generic scoring and SL/TP rule computation."""

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
)
from src.strategies.config import RiskRules, SLTPRule, ScoringConfig, StrategyConfig, AgentConfig
from src.strategies.risk_rules import compute_sl_tp
from src.strategies.scoring import compute_generic_score, score_to_position_size_pct


# ---------------------------------------------------------------- helpers


def _make_config(
    agents: list[tuple[str, float]],
    confidence_floor: int = 30,
    trade_threshold: float = 0.25,
) -> StrategyConfig:
    return StrategyConfig(
        name="Test",
        agents=[
            AgentConfig(name=name, model="claude-haiku-4-5", prompt="prompts/agent_template.txt", weight=weight)
            for name, weight in agents
        ],
        scoring=ScoringConfig(confidence_floor=confidence_floor, trade_threshold=trade_threshold),
        risk=RiskRules(
            stop_loss=SLTPRule(type="none"),
            take_profit=SLTPRule(type="none"),
        ),
    )


def _outputs(directions: dict[str, tuple[str, int]]) -> GenericCouncilOutputs:
    return GenericCouncilOutputs(
        directional={
            name: GenericAgentOutput(direction=d, confidence=c, reasoning="test")
            for name, (d, c) in directions.items()
        }
    )


def _ctx(price: float = 50_000.0, atr_14: float = 1_000.0) -> MarketContext:
    from datetime import datetime, timezone
    return MarketContext(
        timestamp=datetime.now(timezone.utc),
        price=PriceData(
            current=price,
            open_24h=price,
            high_24h=price * 1.02,
            low_24h=price * 0.98,
            change_pct_24h=0.0,
            volume_24h_usd=1e9,
            volume_vs_7d_avg=1.0,
        ),
        indicators=IndicatorData(
            rsi_14=50.0,
            macd_signal="neutral",
            bb_position="mid",
            ema_20=price,
            ema_50=price,
            ema_200=price,
            atr_14=atr_14,
            atr_30_avg=900.0,
            regime="ranging",
        ),
        news=[],
        sentiment=SentimentData(fear_greed_index=50, reddit_sentiment="neutral", social_volume_vs_avg=1.0),
        onchain=OnchainData(exchange_net_flow_btc=0.0, whale_transactions_24h=0, sopr=1.0),
        portfolio=PortfolioData(
            btc_position_usd=0.0,
            cash_usd=10_000.0,
            current_drawdown_pct=0.0,
            peak_portfolio_value=10_000.0,
        ),
    )


# ---------------------------------------------------------------- scoring tests


def test_all_buy_high_confidence_gives_buy():
    config = _make_config([("a", 0.5), ("b", 0.5)])
    outputs = _outputs({"a": ("BUY", 80), "b": ("BUY", 80)})
    signal, score = compute_generic_score(outputs, config)
    assert signal == "BUY"
    assert score > 0


def test_all_sell_high_confidence_gives_sell():
    config = _make_config([("a", 0.5), ("b", 0.5)])
    outputs = _outputs({"a": ("SELL", 80), "b": ("SELL", 80)})
    signal, score = compute_generic_score(outputs, config)
    assert signal == "SELL"
    assert score < 0


def test_all_hold_gives_hold():
    config = _make_config([("a", 0.5), ("b", 0.5)])
    outputs = _outputs({"a": ("HOLD", 50), "b": ("HOLD", 50)})
    signal, score = compute_generic_score(outputs, config)
    assert signal == "HOLD"


def test_below_confidence_floor_agent_excluded():
    # Agent 'b' at confidence 20 (below floor=30) should be excluded.
    # Only agent 'a' contributes; it BUYs at 80 confidence.
    config = _make_config([("a", 0.5), ("b", 0.5)], confidence_floor=30)
    outputs = _outputs({"a": ("BUY", 80), "b": ("SELL", 20)})
    signal, score = compute_generic_score(outputs, config)
    assert signal == "BUY"
    assert score > 0


def test_below_trade_threshold_gives_hold():
    # Moderate confidence, mixed signals → score below threshold
    config = _make_config([("a", 0.5), ("b", 0.5)], trade_threshold=0.25)
    outputs = _outputs({"a": ("BUY", 35), "b": ("SELL", 35)})
    _, score = compute_generic_score(outputs, config)
    assert abs(score) < 0.25


def test_weighted_score_matches_manual_calculation():
    # a weight=0.6, BUY conf=70 → contribution = 0.6 * 0.70
    # b weight=0.4, BUY conf=50 → contribution = 0.4 * 0.50
    # weighted_sum = 0.6*0.70 + 0.4*0.50 = 0.42 + 0.20 = 0.62
    # weight_sum   = 0.6*0.70 + 0.4*0.50 = 0.62
    # score = 0.62 / 0.62 = 1.0 (both positive, same direction)
    config = _make_config([("a", 0.6), ("b", 0.4)])
    outputs = _outputs({"a": ("BUY", 70), "b": ("BUY", 50)})
    _, score = compute_generic_score(outputs, config)
    assert score > 0.9  # should be close to 1.0


def test_single_agent_all_weight():
    config = _make_config([("solo", 1.0)])
    outputs = _outputs({"solo": ("BUY", 90)})
    signal, score = compute_generic_score(outputs, config)
    assert signal == "BUY"
    assert score > 0.25


def test_score_to_position_size_zero_below_threshold():
    config = _make_config([("a", 1.0)])
    size = score_to_position_size_pct(0.10, config)
    assert size == 0.0


def test_score_to_position_size_scales_with_score():
    config = _make_config([("a", 1.0)])
    size_low = score_to_position_size_pct(0.30, config)
    size_high = score_to_position_size_pct(0.80, config)
    assert size_high > size_low


def test_score_to_position_size_capped_at_max():
    config = _make_config([("a", 1.0)])
    size = score_to_position_size_pct(1.0, config)
    assert size == config.risk.max_position_pct


# ---------------------------------------------------------------- SL/TP tests


def test_compute_sl_tp_atr_multiple():
    rules = RiskRules(
        stop_loss=SLTPRule(type="atr_multiple", value=2.0),
        take_profit=SLTPRule(type="atr_multiple", value=3.0),
    )
    ctx = _ctx(price=50_000.0, atr_14=1_000.0)
    sl, tp = compute_sl_tp(rules, ctx, entry_price=50_000.0)
    assert sl == pytest.approx(48_000.0)  # 50000 - 2*1000
    assert tp == pytest.approx(53_000.0)  # 50000 + 3*1000


def test_compute_sl_tp_fixed_pct():
    rules = RiskRules(
        stop_loss=SLTPRule(type="fixed_pct", value=0.05),
        take_profit=SLTPRule(type="fixed_pct", value=0.10),
    )
    ctx = _ctx(price=50_000.0)
    sl, tp = compute_sl_tp(rules, ctx, entry_price=50_000.0)
    assert sl == pytest.approx(47_500.0)   # 50000 * 0.95
    assert tp == pytest.approx(55_000.0)   # 50000 * 1.10


def test_compute_sl_tp_none():
    rules = RiskRules(
        stop_loss=SLTPRule(type="none"),
        take_profit=SLTPRule(type="none"),
    )
    ctx = _ctx()
    sl, tp = compute_sl_tp(rules, ctx, entry_price=50_000.0)
    assert sl == 0.0
    assert tp == 0.0


def test_sl_clamped_above_zero():
    # Huge ATR multiple should not produce negative SL
    rules = RiskRules(
        stop_loss=SLTPRule(type="atr_multiple", value=100.0),
        take_profit=SLTPRule(type="none"),
    )
    ctx = _ctx(price=1_000.0, atr_14=500.0)
    sl, _ = compute_sl_tp(rules, ctx, entry_price=1_000.0)
    assert sl > 0


def test_sl_less_than_entry_atr():
    rules = RiskRules(
        stop_loss=SLTPRule(type="atr_multiple", value=2.0),
        take_profit=SLTPRule(type="none"),
    )
    ctx = _ctx(price=50_000.0, atr_14=1_000.0)
    sl, _ = compute_sl_tp(rules, ctx, entry_price=50_000.0)
    assert sl < 50_000.0


def test_tp_greater_than_entry_atr():
    rules = RiskRules(
        stop_loss=SLTPRule(type="none"),
        take_profit=SLTPRule(type="atr_multiple", value=3.0),
    )
    ctx = _ctx(price=50_000.0, atr_14=1_000.0)
    _, tp = compute_sl_tp(rules, ctx, entry_price=50_000.0)
    assert tp > 50_000.0
