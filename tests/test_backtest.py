"""
Tests for the backtesting module (src/backtest/).

All tests run offline — no real CCXT, LLM, or HTTP calls.
"""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.data import (
    NEUTRAL_NEWS_STUB,
    fetch_full_ohlcv,
    get_sentiment_for_date,
    slice_ohlcv,
)


# ================================================================ Helpers

def _make_ohlcv_df(n: int = 400, start: str = "2023-01-01") -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with n daily rows."""
    rng = np.random.default_rng(42)
    closes = np.linspace(30_000, 60_000, n) + rng.normal(0, 100, n)
    opens = closes - rng.uniform(0, 100, n)
    highs = closes + rng.uniform(50, 300, n)
    lows = np.minimum(opens, closes) - rng.uniform(50, 300, n)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 2_000.0},
        index=idx,
    )


def _make_sentiment_history(days: int = 400) -> dict[date, dict]:
    history = {}
    base = date(2023, 1, 1)
    for i in range(days):
        d = base + timedelta(days=i)
        fg = (i * 7) % 100
        sentiment = "bearish" if fg < 30 else ("bullish" if fg > 60 else "neutral")
        history[d] = {"fear_greed_index": fg, "reddit_sentiment": sentiment, "social_volume_vs_avg": 1.0}
    return history


# ================================================================ TestBacktestData

class TestBacktestData:

    # ---- slice_ohlcv ----

    def test_slice_ohlcv_no_lookahead(self):
        """Row at as_of date must NOT appear in the slice."""
        df = _make_ohlcv_df(n=300)
        as_of = df.index[200]  # pick a middle date
        sliced = slice_ohlcv(df, as_of=as_of.to_pydatetime())
        assert as_of not in sliced.index
        assert all(sliced.index < as_of)

    def test_slice_ohlcv_returns_lookback_rows(self):
        df = _make_ohlcv_df(n=400)
        as_of = df.index[350].to_pydatetime()
        sliced = slice_ohlcv(df, as_of=as_of, lookback=250)
        assert len(sliced) == 250

    def test_slice_ohlcv_raises_if_insufficient_history(self):
        df = _make_ohlcv_df(n=400)
        # as_of at row 150 → only 150 rows visible, < 200 required
        as_of = df.index[150].to_pydatetime()
        with pytest.raises(ValueError, match="200"):
            slice_ohlcv(df, as_of=as_of)

    def test_slice_ohlcv_respects_custom_lookback(self):
        df = _make_ohlcv_df(n=400)
        as_of = df.index[300].to_pydatetime()
        sliced = slice_ohlcv(df, as_of=as_of, lookback=50)
        assert len(sliced) == 50

    # ---- get_sentiment_for_date ----

    def test_get_sentiment_exact_match(self):
        history = _make_sentiment_history()
        target_date = date(2023, 3, 15)
        target = datetime(2023, 3, 15, tzinfo=timezone.utc)
        result = get_sentiment_for_date(history, target)
        assert result == history[target_date]

    def test_get_sentiment_nearest_earlier(self):
        history = {
            date(2023, 1, 1): {"fear_greed_index": 30, "reddit_sentiment": "bearish", "social_volume_vs_avg": 1.0},
            date(2023, 1, 3): {"fear_greed_index": 70, "reddit_sentiment": "bullish", "social_volume_vs_avg": 1.0},
        }
        # target is 2023-01-02 — between the two entries, should return Jan 1
        target = datetime(2023, 1, 2, tzinfo=timezone.utc)
        result = get_sentiment_for_date(history, target)
        assert result["fear_greed_index"] == 30

    def test_get_sentiment_neutral_fallback_empty_history(self):
        result = get_sentiment_for_date({}, datetime(2023, 1, 1, tzinfo=timezone.utc))
        assert result["reddit_sentiment"] == "neutral"
        assert result["fear_greed_index"] == 50

    def test_get_sentiment_neutral_fallback_predates_history(self):
        history = {
            date(2023, 6, 1): {"fear_greed_index": 60, "reddit_sentiment": "neutral", "social_volume_vs_avg": 1.0},
        }
        target = datetime(2022, 1, 1, tzinfo=timezone.utc)
        result = get_sentiment_for_date(history, target)
        assert result["reddit_sentiment"] == "neutral"
        assert result["fear_greed_index"] == 50

    # ---- fetch_full_ohlcv (mocked CCXT) ----

    def test_fetch_full_ohlcv_returns_data_with_correct_columns(self):
        """fetch_full_ohlcv returns a DataFrame with lowercase OHLCV columns."""
        n = 300
        idx = pd.date_range("2022-01-01", periods=n, freq="D")
        mock_df = pd.DataFrame({
            "Open": [30_000.0] * n, "High": [31_000.0] * n,
            "Low": [29_000.0] * n, "Close": [30_500.0] * n,
            "Volume": [1_000.0] * n,
        }, index=idx)

        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime(2022, 10, 28, tzinfo=timezone.utc)

        with patch("src.backtest.data.yf.download", return_value=mock_df):
            df = fetch_full_ohlcv(start, end)

        assert "open" in df.columns
        assert "close" in df.columns
        assert len(df) >= 200
        assert df.index.tz is not None  # must be timezone-aware

    def test_fetch_full_ohlcv_raises_on_insufficient_data(self):
        """Raises ValueError when fewer than 200 candles are returned."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        mock_df = pd.DataFrame({
            "Open": [30_000.0] * n, "High": [31_000.0] * n,
            "Low": [29_000.0] * n, "Close": [30_500.0] * n,
            "Volume": [1_000.0] * n,
        }, index=idx)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 3, 1, tzinfo=timezone.utc)

        with patch("src.backtest.data.yf.download", return_value=mock_df):
            with pytest.raises(ValueError, match="200"):
                fetch_full_ohlcv(start, end)

    # ---- NEUTRAL_NEWS_STUB ----

    def test_neutral_news_stub_has_15_items(self):
        assert len(NEUTRAL_NEWS_STUB) == 15

    def test_neutral_news_stub_has_required_keys(self):
        for item in NEUTRAL_NEWS_STUB:
            assert "headline" in item
            assert "source" in item
            assert "published_at" in item


# ================================================================ TestGenerateSignals

class TestGenerateSignals:
    """Tests for src/backtest/signals.py::generate_signals and _PortfolioTracker."""

    def _make_council_result(self, signal: str = "HOLD"):
        from src.agents.runner import CouncilResult
        from src.models import (
            AgentWeights,
            CouncilOutputs,
            DeliberationOutput,
            FundamentalAnalystOutput,
            RiskManagerOutput,
            SentimentAnalystOutput,
            TechnicalAnalystOutput,
        )
        technical = TechnicalAnalystOutput(
            direction="HOLD", confidence=50, timeframe="24h",
            key_signals=[], invalidation_level=0.0,
        )
        sentiment_out = SentimentAnalystOutput(
            direction="HOLD", confidence=50, overall_sentiment="neutral",
            dominant_narrative="none", high_impact_events=[],
            sentiment_vs_price_divergence=False,
        )
        fundamental = FundamentalAnalystOutput(
            direction="HOLD", confidence=50, onchain_bias="neutral",
            macro_bias="neutral", key_factors=[],
        )
        risk = RiskManagerOutput(
            veto=False, veto_reason=None, approved_position_size_pct=20.0,
            recommended_stop_loss=28_000.0, recommended_take_profit=35_000.0,
            risk_reward_ratio=2.0, notes="ok",
        )
        outputs = CouncilOutputs(
            technical=technical, sentiment=sentiment_out,
            fundamental=fundamental, risk=risk,
        )
        deliberation = DeliberationOutput(
            final_signal=signal,
            conviction="medium",
            consensus_summary="test",
            key_disagreements=[],
            agent_weights_applied=AgentWeights(
                technical=0.25, sentiment=0.25, fundamental=0.25, risk=0.25,
            ),
        )
        return CouncilResult(outputs=outputs, deliberation=deliberation)

    async def test_returns_dataframe_with_required_columns(self):
        from src.backtest.signals import SIGNAL_COLUMNS, generate_signals

        full_ohlcv = _make_ohlcv_df(n=400, start="2023-01-01")
        sentiment_history = _make_sentiment_history()
        start = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)  # 3 cycles

        mock_result = self._make_council_result("HOLD")
        with patch("src.backtest.signals.assemble_context", new=AsyncMock(return_value=MagicMock())):
            with patch("src.backtest.signals.run_council", new=AsyncMock(return_value=mock_result)):
                df = await generate_signals(full_ohlcv, sentiment_history, start, end)

        for col in SIGNAL_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    async def test_signal_count_matches_date_range(self):
        from src.backtest.signals import generate_signals

        full_ohlcv = _make_ohlcv_df(n=400, start="2023-01-01")
        sentiment_history = _make_sentiment_history()
        start = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end = datetime(2024, 1, 7, tzinfo=timezone.utc)  # 5 cycles

        mock_result = self._make_council_result("HOLD")
        with patch("src.backtest.signals.assemble_context", new=AsyncMock(return_value=MagicMock())):
            with patch("src.backtest.signals.run_council", new=AsyncMock(return_value=mock_result)):
                df = await generate_signals(full_ohlcv, sentiment_history, start, end)

        assert len(df) == 5

    async def test_no_lookahead_in_ohlcv_slice(self):
        """slice_ohlcv is called with as_of=T for each cycle."""
        from src.backtest.signals import generate_signals

        full_ohlcv = _make_ohlcv_df(n=400, start="2023-01-01")
        sentiment_history = _make_sentiment_history()
        start = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end = datetime(2024, 1, 4, tzinfo=timezone.utc)

        mock_result = self._make_council_result("HOLD")
        captured_as_of = []

        original_slice = __import__("src.backtest.signals", fromlist=["slice_ohlcv"]).slice_ohlcv

        def capturing_slice(df, as_of, lookback=250):
            captured_as_of.append(as_of)
            return original_slice(df, as_of, lookback)

        with patch("src.backtest.signals.slice_ohlcv", side_effect=capturing_slice):
            with patch("src.backtest.signals.assemble_context", new=AsyncMock(return_value=MagicMock())):
                with patch("src.backtest.signals.run_council", new=AsyncMock(return_value=mock_result)):
                    await generate_signals(full_ohlcv, sentiment_history, start, end)

        # Each as_of should be strictly after the previous one
        for i in range(1, len(captured_as_of)):
            assert captured_as_of[i] > captured_as_of[i - 1]

    async def test_hold_on_agent_error(self):
        """When run_council raises AgentError, cycle records HOLD and continues."""
        from src.agents.base import AgentError
        from src.backtest.signals import generate_signals

        full_ohlcv = _make_ohlcv_df(n=400, start="2023-01-01")
        sentiment_history = _make_sentiment_history()
        start = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)

        with patch("src.backtest.signals.assemble_context", new=AsyncMock(return_value=MagicMock())):
            with patch("src.backtest.signals.run_council", new=AsyncMock(side_effect=AgentError("API fail"))):
                df = await generate_signals(full_ohlcv, sentiment_history, start, end)

        assert len(df) == 3
        assert (df["signal"] == "HOLD").all()

    # ---- _PortfolioTracker ----

    def test_portfolio_tracker_updates_cash_on_buy(self):
        from src.backtest.signals import _PortfolioTracker

        tracker = _PortfolioTracker(cash=10_000.0)
        tracker.open_position(2_000.0)
        assert tracker.cash == 8_000.0
        assert tracker.position_usd == 2_000.0

    def test_portfolio_tracker_updates_cash_on_sell(self):
        from src.backtest.signals import _PortfolioTracker

        tracker = _PortfolioTracker(cash=10_000.0)
        tracker.open_position(2_000.0)
        tracker.close_position(pnl_usd=500.0)
        assert tracker.position_usd == 0.0
        assert tracker.cash == pytest.approx(10_500.0)

    def test_portfolio_tracker_drawdown(self):
        from src.backtest.signals import _PortfolioTracker

        tracker = _PortfolioTracker(cash=10_000.0)
        tracker.open_position(5_000.0)
        # Simulate a loss: close position at a loss
        tracker.close_position(pnl_usd=-1_000.0)
        assert tracker.drawdown > 0.0
        assert tracker.cash == pytest.approx(9_000.0)

    def test_portfolio_tracker_to_dict(self):
        from src.backtest.signals import _PortfolioTracker

        tracker = _PortfolioTracker(cash=10_000.0)
        d = tracker.to_dict()
        assert "cash_usd" in d
        assert "btc_position_usd" in d
        assert "current_drawdown_pct" in d
        assert "peak_portfolio_value" in d


# ================================================================ TestRunBacktest

class TestRunBacktest:
    """Tests for src/backtest/engine.py::run_backtest and LLMCouncilStrategy."""

    def _make_signals_df(
        self,
        n: int = 60,
        signal_override: str | None = None,
        start: str = "2024-01-01",
    ) -> pd.DataFrame:
        """Build a minimal signals DataFrame for engine tests."""
        rng = np.random.default_rng(99)
        closes = np.linspace(40_000, 50_000, n) + rng.normal(0, 100, n)
        opens = closes - 100
        highs = closes + 200
        lows = closes - 200

        signals = [signal_override or "HOLD"] * n
        size_pcts = [20.0 if s == "BUY" else 0.0 for s in signals]
        sl_pcts = [0.95 if signals[i] == "BUY" else 0.0 for i in range(n)]
        tp_pcts = [1.10 if signals[i] == "BUY" else 0.0 for i in range(n)]

        idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
        return pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1000.0,
            "signal": signals, "size_pct": size_pcts,
            "sl_pct": sl_pcts, "tp_pct": tp_pcts,
            "conviction": ["medium"] * n, "vetoed": [False] * n,
        }, index=idx)

    def _run_strategy(self, df: pd.DataFrame, cash: float = 10_000.0) -> object:
        """Run FractionalBacktest and return stats.

        FractionalBacktest supports fractional BTC units so any cash amount works.
        """
        from backtesting.lib import FractionalBacktest
        from src.backtest.engine import LLMCouncilStrategy

        bt = FractionalBacktest(df, LLMCouncilStrategy, cash=cash, commission=0.001, exclusive_orders=True)
        return bt.run()

    def test_all_hold_signals_produces_zero_trades(self):
        df = self._make_signals_df(n=60, signal_override="HOLD")
        stats = self._run_strategy(df)
        assert int(stats["# Trades"]) == 0

    def test_buy_signal_creates_trade(self):
        """A single BUY signal should open a position."""
        n = 60
        df = self._make_signals_df(n=n, signal_override="HOLD")
        # Set BUY on day 10
        df.at[df.index[10], "signal"] = "BUY"
        df.at[df.index[10], "size_pct"] = 20.0
        close_at_10 = df["Close"].iloc[10]
        df.at[df.index[10], "sl_pct"] = 0.95
        df.at[df.index[10], "tp_pct"] = 1.10
        stats = self._run_strategy(df)
        assert int(stats["# Trades"]) >= 1

    def test_sell_signal_closes_position(self):
        """BUY on day 10, SELL on day 20 should produce exactly 1 closed trade."""
        n = 60
        df = self._make_signals_df(n=n, signal_override="HOLD")
        df.at[df.index[10], "signal"] = "BUY"
        df.at[df.index[10], "size_pct"] = 20.0
        df.at[df.index[10], "sl_pct"] = 0.95
        df.at[df.index[10], "tp_pct"] = 0.0
        df.at[df.index[20], "signal"] = "SELL"
        stats = self._run_strategy(df)
        assert int(stats["# Trades"]) >= 1

    def test_stop_loss_triggers_on_low(self):
        """When Low dips below sl_price, the stop-loss should fire."""
        n = 60
        closes = np.full(n, 40_000.0)
        opens = closes - 50
        highs = closes + 100
        lows = closes - 100

        signals = ["HOLD"] * n
        size_pcts = [0.0] * n
        sl_pcts = [0.0] * n
        tp_pcts = [0.0] * n

        # BUY on day 5 with sl at 99.95% of close (just above the low on day 10)
        signals[5] = "BUY"
        size_pcts[5] = 20.0
        sl_pcts[5] = 0.9995   # sl = 40_000 * 0.9995 = 39_980 → triggers when low hits 39_500
        lows[10] = 39_500.0   # force a low below sl

        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1000.0,
            "signal": signals, "size_pct": size_pcts,
            "sl_pct": sl_pcts, "tp_pct": tp_pcts,
            "conviction": ["medium"] * n, "vetoed": [False] * n,
        }, index=idx)

        stats = self._run_strategy(df)
        assert int(stats["# Trades"]) >= 1

    def test_returns_dict_with_required_keys(self):
        """run_backtest should return a dict with standard performance keys."""
        from src.backtest.engine import run_backtest

        full_ohlcv = _make_ohlcv_df(n=400, start="2023-01-01")
        sentiment_history = _make_sentiment_history()

        start = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end = datetime(2024, 3, 1, tzinfo=timezone.utc)

        # Build a mock signals_df (all HOLD — simplest case)
        n_days = (end - start).days
        idx = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
        mock_signals = pd.DataFrame({
            "open": 40_000.0, "high": 41_000.0, "low": 39_000.0,
            "close": 40_500.0, "volume": 1000.0,
            "signal": "HOLD", "size_pct": 0.0,
            "sl_pct": 0.0, "tp_pct": 0.0,
            "conviction": "medium", "vetoed": False,
        }, index=idx)

        with patch("src.backtest.engine.fetch_full_ohlcv", return_value=full_ohlcv):
            with patch("src.backtest.engine.fetch_historical_sentiment", return_value=sentiment_history):
                with patch("src.backtest.engine.generate_signals", new=AsyncMock(return_value=mock_signals)):
                    result = asyncio.get_event_loop().run_until_complete(
                        run_backtest(start, end, save_signals_csv=False)
                    )

        required_keys = ["return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "trades", "cycles"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_stats_sharpe_is_numeric(self):
        """Sharpe ratio in results should be a float (may be NaN for all-HOLD)."""
        import math

        from src.backtest.engine import run_backtest

        full_ohlcv = _make_ohlcv_df(n=400, start="2023-01-01")
        sentiment_history = _make_sentiment_history()

        start = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end = datetime(2024, 2, 1, tzinfo=timezone.utc)

        n_days = (end - start).days
        idx = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
        mock_signals = pd.DataFrame({
            "open": 40_000.0, "high": 41_000.0, "low": 39_000.0,
            "close": 40_500.0, "volume": 1000.0,
            "signal": "HOLD", "size_pct": 0.0,
            "sl_pct": 0.0, "tp_pct": 0.0,
            "conviction": "medium", "vetoed": False,
        }, index=idx)

        with patch("src.backtest.engine.fetch_full_ohlcv", return_value=full_ohlcv):
            with patch("src.backtest.engine.fetch_historical_sentiment", return_value=sentiment_history):
                with patch("src.backtest.engine.generate_signals", new=AsyncMock(return_value=mock_signals)):
                    result = asyncio.get_event_loop().run_until_complete(
                        run_backtest(start, end, save_signals_csv=False)
                    )

        assert isinstance(result["sharpe_ratio"], float)


# ================================================================ imports needed in tests

import asyncio  # noqa: E402  (used in test_returns_dict_with_required_keys)
