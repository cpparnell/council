"""Tests for src/execution/ — orders, state, router — using mocked exchange and DB."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.db.schema import create_all, get_engine
from src.db.store import (
    get_open_trade,
    get_portfolio_state,
    log_cycle,
    open_trade,
    update_portfolio_state,
)
from src.execution.orders import OrderError, place_market_buy, place_market_sell, place_stop_loss
from src.execution.router import RouteResult, route_signal
from src.execution.state import build_portfolio_dict
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
from src.agents.runner import CouncilResult


# ─── Helpers ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def engine():
    eng = get_engine(":memory:")
    create_all(eng)
    return eng


def _mock_exchange(current_price: float = 85000.0) -> MagicMock:
    ex = MagicMock()
    ex.fetch_ticker.return_value = {"last": current_price}
    ex.create_market_buy_order.return_value = {"id": "buy-1", "average": current_price, "price": current_price}
    ex.create_market_sell_order.return_value = {"id": "sell-1", "average": current_price, "price": current_price}
    ex.create_order.return_value = {"id": "sl-1"}
    ex.cancel_order.return_value = {}
    return ex


def _make_ctx(price: float = 83500.0, cash: float = 10000.0, btc_pos: float = 0.0) -> MarketContext:
    return MarketContext(
        timestamp=datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc),
        asset="BTC/USD",
        price=PriceData(
            current=price, open_24h=81200.0, high_24h=84100.0,
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
            btc_position_usd=btc_pos, cash_usd=cash,
            current_drawdown_pct=0.0, peak_portfolio_value=10000.0,
        ),
    )


def _make_outputs(signal: str = "BUY", veto: bool = False, rr: float = 2.1) -> CouncilOutputs:
    return CouncilOutputs(
        technical=TechnicalAnalystOutput(
            direction=signal, confidence=72, timeframe="24h",
            key_signals=["RSI momentum"], invalidation_level=80000.0,
        ),
        sentiment=SentimentAnalystOutput(
            direction=signal, confidence=65, overall_sentiment="bullish",
            dominant_narrative="ETF inflows", high_impact_events=[],
            sentiment_vs_price_divergence=False,
        ),
        fundamental=FundamentalAnalystOutput(
            direction=signal, confidence=68, onchain_bias="accumulation",
            macro_bias="risk-on", key_factors=["Exchange outflows"],
        ),
        risk=RiskManagerOutput(
            veto=veto, veto_reason="drawdown" if veto else None,
            approved_position_size_pct=15.0,
            recommended_stop_loss=80200.0, recommended_take_profit=88000.0,
            risk_reward_ratio=rr, notes="",
        ),
    )


def _make_result(signal: str = "BUY", veto: bool = False, rr: float = 2.1) -> CouncilResult:
    outputs = _make_outputs(signal, veto, rr)
    deliberation = DeliberationOutput(
        final_signal=signal,
        conviction="high",
        consensus_summary="All agents agree",
        key_disagreements=[],
        agent_weights_applied=AgentWeights(technical=0.3, sentiment=0.25, fundamental=0.3, risk=0.15),
    )
    return CouncilResult(outputs=outputs, deliberation=deliberation)


# ─── orders.py ───────────────────────────────────────────────────────────────

class TestPlaceMarketBuy:
    def test_calls_exchange_and_returns_order(self):
        exchange = _mock_exchange()
        order = place_market_buy(exchange, "BTC/USDT", 0.1)
        exchange.create_market_buy_order.assert_called_once_with("BTC/USDT", 0.1)
        assert order["id"] == "buy-1"

    def test_retries_on_network_error(self):
        import ccxt
        exchange = MagicMock()
        exchange.create_market_buy_order.side_effect = [
            ccxt.NetworkError("timeout"),
            {"id": "buy-retry", "average": 83500.0},
        ]
        with patch("src.execution.orders.time.sleep"):
            order = place_market_buy(exchange, "BTC/USDT", 0.1)
        assert order["id"] == "buy-retry"
        assert exchange.create_market_buy_order.call_count == 2

    def test_raises_order_error_after_max_retries(self):
        import ccxt
        exchange = MagicMock()
        exchange.create_market_buy_order.side_effect = ccxt.NetworkError("timeout")
        with patch("src.execution.orders.time.sleep"):
            with pytest.raises(OrderError):
                place_market_buy(exchange, "BTC/USDT", 0.1)

    def test_raises_immediately_on_exchange_error(self):
        import ccxt
        exchange = MagicMock()
        exchange.create_market_buy_order.side_effect = ccxt.ExchangeError("insufficient balance")
        with pytest.raises(OrderError, match="Exchange error"):
            place_market_buy(exchange, "BTC/USDT", 0.1)
        assert exchange.create_market_buy_order.call_count == 1


class TestPlaceMarketSell:
    def test_calls_exchange(self):
        exchange = _mock_exchange()
        order = place_market_sell(exchange, "BTC/USDT", 0.05)
        exchange.create_market_sell_order.assert_called_once_with("BTC/USDT", 0.05)
        assert order["id"] == "sell-1"


class TestPlaceStopLoss:
    def test_places_stop_market(self):
        exchange = _mock_exchange()
        order = place_stop_loss(exchange, "BTC/USDT", 0.1, 80200.0)
        assert order["id"] == "sl-1"

    def test_falls_back_to_stop_limit(self):
        exchange = MagicMock()
        with patch("src.execution.orders._retry") as mock_retry:
            mock_retry.side_effect = [OrderError("no stop_market"), {"id": "sl-limit"}]
            order = place_stop_loss(exchange, "BTC/USDT", 0.1, 80200.0)
        assert order["id"] == "sl-limit"


# ─── state.py ────────────────────────────────────────────────────────────────

class TestBuildPortfolioDict:
    def test_no_open_position(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        d = build_portfolio_dict(engine)
        assert d["btc_position_usd"] == 0.0
        assert d["cash_usd"] == 10000.0
        assert d["current_drawdown_pct"] == 0.0

    def test_with_open_position(self, engine):
        update_portfolio_state(engine, cash_usd=8500.0, peak_value=10000.0)
        # Manually insert an open trade via store
        from src.db.schema import trades as trades_table
        from src.db.schema import cycles as cycles_table
        # Need a cycle first
        from src.db.schema import metadata
        from sqlalchemy import insert

        with engine.begin() as conn:
            conn.execute(
                cycles_table.insert().values(
                    timestamp=datetime(2026, 4, 5, tzinfo=timezone.utc),
                    asset="BTC/USD", signal="BUY", conviction="high",
                    vetoed=False, context_json="{}", council_outputs_json="{}",
                    deliberation_json="{}",
                )
            )
            conn.execute(
                trades_table.insert().values(
                    cycle_id=1, symbol="BTC/USDT", side="BUY",
                    entry_price=83500.0,
                    entry_time=datetime(2026, 4, 5, tzinfo=timezone.utc),
                    position_size_usd=1500.0, position_size_btc=0.018,
                    stop_loss=80200.0, status="open",
                )
            )

        d = build_portfolio_dict(engine)
        assert d["btc_position_usd"] == 1500.0
        assert d["cash_usd"] == 8500.0
        total = 8500.0 + 1500.0
        expected_drawdown = (10000.0 - total) / 10000.0
        assert abs(d["current_drawdown_pct"] - max(0.0, expected_drawdown)) < 1e-9


# ─── router.py ───────────────────────────────────────────────────────────────

class TestRouteSignalBuy:
    def test_buy_places_order_and_logs_trade(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        exchange = _mock_exchange(current_price=83500.0)
        ctx = _make_ctx()
        result = _make_result("BUY")

        route = route_signal(engine, exchange, ctx, result)

        assert route.action == "buy"
        assert route.trade_id is not None
        exchange.create_market_buy_order.assert_called_once()

    def test_buy_deducts_cash(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        exchange = _mock_exchange(current_price=83500.0)
        ctx = _make_ctx()
        result = _make_result("BUY")

        route_signal(engine, exchange, ctx, result)

        state = get_portfolio_state(engine)
        assert state["cash_usd"] < 10000.0

    def test_buy_skipped_when_already_in_position(self, engine):
        update_portfolio_state(engine, cash_usd=8500.0, peak_value=10000.0)
        exchange = _mock_exchange()
        ctx = _make_ctx(btc_pos=1500.0)
        result = _make_result("BUY")

        # Pre-seed an open trade
        cycle_id = log_cycle(engine, ctx, result.outputs, result.deliberation)
        open_trade(engine, cycle_id=cycle_id, symbol="BTC/USDT", side="BUY",
                   entry_price=83500.0, position_size_usd=1500.0, stop_loss=80200.0)

        route = route_signal(engine, exchange, ctx, result)
        assert route.action == "hold"
        exchange.create_market_buy_order.assert_not_called()

    def test_buy_holds_on_insufficient_rr(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        exchange = _mock_exchange()
        ctx = _make_ctx()
        result = _make_result("BUY", rr=1.0)  # below MIN_RISK_REWARD=1.5

        route = route_signal(engine, exchange, ctx, result)
        assert route.action == "hold"
        assert route.error == "insufficient_rr"

    def test_buy_holds_on_invalid_stop_distance(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        exchange = _mock_exchange()
        # Make stop_loss > current_price (invalid)
        ctx = _make_ctx(price=80000.0)
        outputs = _make_outputs("BUY")
        outputs.risk.recommended_stop_loss = 85000.0  # above price
        deliberation = DeliberationOutput(
            final_signal="BUY", conviction="medium",
            consensus_summary="test", key_disagreements=[],
            agent_weights_applied=AgentWeights(technical=0.3, sentiment=0.25, fundamental=0.3, risk=0.15),
        )
        result = CouncilResult(outputs=outputs, deliberation=deliberation)

        route = route_signal(engine, exchange, ctx, result)
        assert route.action == "hold"
        assert route.error == "invalid_stop_distance"


class TestRouteSignalSell:
    def _setup_open_position(self, engine, entry_price=83500.0):
        ctx = _make_ctx()
        result = _make_result("BUY")
        cycle_id = log_cycle(engine, ctx, result.outputs, result.deliberation)
        trade_id = open_trade(engine, cycle_id=cycle_id, symbol="BTC/USDT", side="BUY",
                               entry_price=entry_price, position_size_usd=1500.0,
                               stop_loss=80200.0, sl_order_id="sl-1")
        update_portfolio_state(engine, cash_usd=8500.0, peak_value=10000.0)
        return trade_id

    def test_sell_closes_position(self, engine):
        self._setup_open_position(engine)
        exchange = _mock_exchange(current_price=88000.0)
        ctx = _make_ctx()
        result = _make_result("SELL")

        route = route_signal(engine, exchange, ctx, result)

        assert route.action == "sell"
        assert route.closed_trade is not None
        assert route.closed_trade["status"] == "closed"

    def test_sell_restores_cash(self, engine):
        self._setup_open_position(engine)
        exchange = _mock_exchange(current_price=88000.0)
        ctx = _make_ctx()
        result = _make_result("SELL")

        route_signal(engine, exchange, ctx, result)

        state = get_portfolio_state(engine)
        assert state["cash_usd"] > 8500.0  # cash restored + PnL

    def test_sell_cancels_stop_loss(self, engine):
        self._setup_open_position(engine)
        exchange = _mock_exchange(current_price=88000.0)
        ctx = _make_ctx()
        result = _make_result("SELL")

        route_signal(engine, exchange, ctx, result)
        exchange.cancel_order.assert_called_once_with("sl-1", "BTC/USDT")

    def test_sell_no_position_returns_hold(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        exchange = _mock_exchange()
        ctx = _make_ctx()
        result = _make_result("SELL")

        route = route_signal(engine, exchange, ctx, result)
        assert route.action == "hold"
        exchange.create_market_sell_order.assert_not_called()


class TestRouteSignalHold:
    def test_hold_logs_cycle_only(self, engine):
        update_portfolio_state(engine, cash_usd=10000.0, peak_value=10000.0)
        exchange = _mock_exchange()
        ctx = _make_ctx()
        result = _make_result("HOLD")

        route = route_signal(engine, exchange, ctx, result)
        assert route.action == "hold"
        assert route.cycle_id > 0
        exchange.create_market_buy_order.assert_not_called()
        exchange.create_market_sell_order.assert_not_called()


class TestReconcile:
    def test_no_action_when_no_open_position(self, engine):
        from src.execution.router import reconcile_open_position
        exchange = _mock_exchange(current_price=85000.0)
        result = reconcile_open_position(engine, exchange)
        assert result is None

    def test_stop_triggered_when_price_below_stop(self, engine):
        from src.execution.router import reconcile_open_position
        update_portfolio_state(engine, cash_usd=8500.0, peak_value=10000.0)
        ctx = _make_ctx()
        r = _make_result("BUY")
        cycle_id = log_cycle(engine, ctx, r.outputs, r.deliberation)
        open_trade(engine, cycle_id=cycle_id, symbol="BTC/USDT", side="BUY",
                   entry_price=83500.0, position_size_usd=1500.0, stop_loss=80200.0)

        # Current price below stop
        exchange = _mock_exchange(current_price=79000.0)
        stopped = reconcile_open_position(engine, exchange)

        assert stopped is not None
        assert stopped["status"] == "stopped"
        assert stopped["pnl_usd"] < 0

    def test_no_stop_when_price_above_stop(self, engine):
        from src.execution.router import reconcile_open_position
        update_portfolio_state(engine, cash_usd=8500.0, peak_value=10000.0)
        ctx = _make_ctx()
        r = _make_result("BUY")
        cycle_id = log_cycle(engine, ctx, r.outputs, r.deliberation)
        open_trade(engine, cycle_id=cycle_id, symbol="BTC/USDT", side="BUY",
                   entry_price=83500.0, position_size_usd=1500.0, stop_loss=80200.0)

        exchange = _mock_exchange(current_price=84000.0)
        result = reconcile_open_position(engine, exchange)
        assert result is None
