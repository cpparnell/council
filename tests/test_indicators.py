"""
Unit tests for council/data/indicators.py.

All tests use synthetic OHLCV DataFrames — no network calls, no fixtures.
The synthetic data is designed so the expected indicator behaviour is
deterministic and easy to reason about.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import compute_indicators


# ----------------------------------------------------------------- Helpers

def _make_df(
    closes: np.ndarray,
    *,
    high_offset: float = 200.0,
    low_offset: float = 200.0,
    open_offset: float = 100.0,
    volume: float = 2000.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a closes array."""
    n = len(closes)
    rng = np.random.default_rng(seed)
    opens = closes - rng.uniform(0, open_offset, n)
    highs = closes + rng.uniform(0, high_offset, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, low_offset, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


def _uptrend(n: int = 250, start: float = 30_000, end: float = 60_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = np.linspace(start, end, n) + rng.normal(0, 150, n)
    return _make_df(closes, seed=1)


def _downtrend(n: int = 250, start: float = 60_000, end: float = 30_000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    closes = np.linspace(start, end, n) + rng.normal(0, 150, n)
    return _make_df(closes, seed=2)


def _sideways(n: int = 250, midpoint: float = 40_000) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    closes = midpoint + rng.normal(0, 400, n)
    return _make_df(closes, seed=3)


# ------------------------------------------------------------ Schema tests

class TestOutputSchema:
    def test_all_keys_present(self):
        result = compute_indicators(_uptrend())
        expected_keys = {
            "rsi_14", "macd_signal", "bb_position",
            "ema_20", "ema_50", "ema_200",
            "atr_14", "atr_30_avg", "regime",
        }
        assert set(result.keys()) == expected_keys

    def test_raises_on_insufficient_data(self):
        df = _uptrend(n=199)
        with pytest.raises(ValueError, match="200"):
            compute_indicators(df)

    def test_exactly_200_rows_accepted(self):
        df = _uptrend(n=200)
        result = compute_indicators(df)
        assert result["rsi_14"] is not None


# ----------------------------------------------------------------- RSI tests

class TestRSI:
    def test_range(self):
        result = compute_indicators(_uptrend())
        assert 0.0 <= result["rsi_14"] <= 100.0

    def test_high_in_sustained_uptrend(self):
        result = compute_indicators(_uptrend())
        assert result["rsi_14"] > 50, "RSI should be above 50 in a sustained uptrend"

    def test_low_in_sustained_downtrend(self):
        result = compute_indicators(_downtrend())
        assert result["rsi_14"] < 50, "RSI should be below 50 in a sustained downtrend"

    def test_near_50_in_sideways(self):
        result = compute_indicators(_sideways())
        assert 30 <= result["rsi_14"] <= 70, "RSI should be near 50 in a sideways market"


# --------------------------------------------------------------- MACD tests

class TestMACD:
    _valid_signals = {"bullish_cross", "bearish_cross", "bullish", "bearish", "neutral"}

    def test_valid_signal_value(self):
        assert compute_indicators(_uptrend())["macd_signal"] in self._valid_signals

    def test_bullish_in_uptrend(self):
        result = compute_indicators(_uptrend())
        assert result["macd_signal"] in {"bullish_cross", "bullish"}, (
            f"Expected bullish MACD in uptrend, got {result['macd_signal']}"
        )

    def test_bearish_in_downtrend(self):
        result = compute_indicators(_downtrend())
        assert result["macd_signal"] in {"bearish_cross", "bearish"}, (
            f"Expected bearish MACD in downtrend, got {result['macd_signal']}"
        )

    def test_bullish_cross_detected(self):
        """Force a bullish cross: flat price then a sharp upward spike."""
        rng = np.random.default_rng(0)
        n = 250
        # Long flat period so fast and slow EMAs converge
        closes = np.full(n, 40_000.0) + rng.normal(0, 20, n)
        # Final 5 candles ramp sharply upward to create a bullish cross
        closes[-5:] = np.linspace(40_000, 46_000, 5)
        df = _make_df(closes, seed=10)
        result = compute_indicators(df)
        assert result["macd_signal"] in {"bullish_cross", "bullish"}

    def test_bearish_cross_detected(self):
        """Force a bearish cross: flat price then a sharp downward spike."""
        rng = np.random.default_rng(0)
        n = 250
        closes = np.full(n, 40_000.0) + rng.normal(0, 20, n)
        closes[-5:] = np.linspace(40_000, 34_000, 5)
        df = _make_df(closes, seed=11)
        result = compute_indicators(df)
        assert result["macd_signal"] in {"bearish_cross", "bearish"}


# --------------------------------------------------- Bollinger Band tests

class TestBollingerBands:
    _valid_positions = {"upper", "mid", "lower"}

    def test_valid_position_value(self):
        assert compute_indicators(_uptrend())["bb_position"] in self._valid_positions

    def test_upper_after_price_spike(self):
        """Price breaks sharply above a long flat channel → upper band."""
        n = 250
        closes = np.concatenate([
            np.full(230, 40_000.0),
            np.linspace(40_000, 48_000, 20),
        ])
        df = _make_df(closes, seed=20)
        assert compute_indicators(df)["bb_position"] == "upper"

    def test_lower_after_price_crash(self):
        """Price breaks sharply below a long flat channel → lower band."""
        n = 250
        closes = np.concatenate([
            np.full(230, 40_000.0),
            np.linspace(40_000, 32_000, 20),
        ])
        df = _make_df(closes, seed=21)
        assert compute_indicators(df)["bb_position"] == "lower"

    def test_mid_in_sideways(self):
        # Controlled sideways data: the last 20 candles form a full sine
        # cycle over [-π, π] with mean = midpoint and last value = midpoint,
        # so percent-B ≈ 0.5 → "mid" regardless of the preceding noise.
        n = 250
        midpoint = 40_000.0
        closes = np.full(n, midpoint, dtype=float)
        closes[:230] += np.random.default_rng(0).normal(0, 500, 230)
        closes[230:] = midpoint + 300 * np.sin(np.linspace(-np.pi, np.pi, 20))
        df = _make_df(closes, high_offset=100, low_offset=100, seed=3)
        assert compute_indicators(df)["bb_position"] == "mid"


# ----------------------------------------------------------------- EMA tests

class TestEMAs:
    def test_all_positive(self):
        result = compute_indicators(_uptrend())
        assert result["ema_20"] > 0
        assert result["ema_50"] > 0
        assert result["ema_200"] > 0

    def test_bullish_alignment_in_uptrend(self):
        """EMA-20 > EMA-50 > EMA-200 in a long sustained uptrend."""
        result = compute_indicators(_uptrend())
        assert result["ema_20"] > result["ema_50"] > result["ema_200"], (
            "Bullish EMA stack expected: EMA20 > EMA50 > EMA200"
        )

    def test_bearish_alignment_in_downtrend(self):
        """EMA-20 < EMA-50 < EMA-200 in a long sustained downtrend."""
        result = compute_indicators(_downtrend())
        assert result["ema_20"] < result["ema_50"] < result["ema_200"], (
            "Bearish EMA stack expected: EMA20 < EMA50 < EMA200"
        )

    def test_ema20_more_responsive_than_ema200(self):
        """After a sharp move, EMA-20 should deviate more from the old price
        than EMA-200 (which is slower to react)."""
        # Flat baseline for 200 candles then a sharp upward ramp
        n = 250
        closes = np.concatenate([np.full(200, 40_000.0), np.linspace(40_000, 55_000, 50)])
        df = _make_df(closes, seed=30)
        result = compute_indicators(df)
        # EMA-20 should be closer to the new price (55k) than EMA-200
        assert result["ema_20"] > result["ema_200"]


# ----------------------------------------------------------------- ATR tests

class TestATR:
    def test_positive(self):
        assert compute_indicators(_uptrend())["atr_14"] > 0

    def test_atr_30_avg_positive(self):
        assert compute_indicators(_uptrend())["atr_30_avg"] > 0

    def test_higher_atr_in_high_volatility(self):
        """Synthetic high-volatility data should produce a larger ATR than
        low-volatility data at a similar price level."""
        n = 250
        closes_low = np.full(n, 40_000.0) + np.random.default_rng(40).normal(0, 50, n)
        closes_high = np.full(n, 40_000.0) + np.random.default_rng(41).normal(0, 3_000, n)

        df_low = _make_df(closes_low, high_offset=100, low_offset=100, seed=40)
        df_high = _make_df(closes_high, high_offset=5_000, low_offset=5_000, seed=41)

        assert compute_indicators(df_high)["atr_14"] > compute_indicators(df_low)["atr_14"]


# --------------------------------------------------------------- Regime tests

class TestRegime:
    _valid_regimes = {"trending", "ranging", "high_volatility"}

    def test_valid_regime_value(self):
        assert compute_indicators(_uptrend())["regime"] in self._valid_regimes

    def test_trending_in_strong_uptrend(self):
        # A long linear trend should have high ADX → "trending"
        assert compute_indicators(_uptrend())["regime"] == "trending"

    def test_trending_in_strong_downtrend(self):
        assert compute_indicators(_downtrend())["regime"] == "trending"

    def test_ranging_in_sideways(self):
        assert compute_indicators(_sideways())["regime"] == "ranging"

    def test_high_volatility_regime(self):
        """A sudden ATR spike at the end should classify as high_volatility."""
        rng = np.random.default_rng(50)
        n = 250
        # Calm baseline: tiny daily moves
        closes = np.full(n, 40_000.0) + rng.normal(0, 30, n)
        df = _make_df(closes, high_offset=60, low_offset=60, seed=50)

        # Override the last 14 candles with huge swings (ATR spike)
        df_arr = df.copy()
        spike_highs = df_arr["close"].values.copy()
        spike_highs[-14:] += 8_000
        spike_lows = df_arr["close"].values.copy()
        spike_lows[-14:] -= 8_000
        df_arr["high"] = np.maximum(df_arr["high"].values, spike_highs)
        df_arr["low"] = np.minimum(df_arr["low"].values, spike_lows)

        result = compute_indicators(df_arr)
        assert result["regime"] == "high_volatility"
