"""
Unit tests for src/data/macro.py (yfinance macro context).

All yfinance calls are mocked via pandas DataFrames — no network.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.macro import (
    _compute_macro_bias,
    _latest_close,
    _pct_change_over,
    fetch_macro_for_date,
)
from src.models import MacroData


def _df(closes: list[float], start: str = "2023-06-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


class TestLatestClose:
    def test_returns_last_close(self):
        assert _latest_close(_df([10.0, 12.0, 15.0])) == 15.0

    def test_returns_none_on_empty(self):
        assert _latest_close(pd.DataFrame()) is None

    def test_returns_none_on_none(self):
        assert _latest_close(None) is None


class TestPctChangeOver:
    def test_computes_20d_change(self):
        closes = [100.0 + i for i in range(25)]  # 100..124
        df = _df(closes)
        # end=124, start=closes[-21]=104; change = (124-104)/104 * 100 ≈ 19.23%
        assert _pct_change_over(df, window=20) == pytest.approx(19.23, rel=1e-2)

    def test_returns_none_when_too_short(self):
        df = _df([100.0, 101.0, 102.0])
        assert _pct_change_over(df, window=20) is None

    def test_returns_none_on_empty(self):
        assert _pct_change_over(pd.DataFrame(), window=20) is None


class TestComputeMacroBias:
    def test_risk_on_low_vix_weak_dollar_rising_spx(self):
        bias = _compute_macro_bias(vix=15.0, dxy_20d_change_pct=-0.5, spx_20d_change_pct=3.0)
        assert bias == "risk-on"

    def test_risk_off_high_vix(self):
        bias = _compute_macro_bias(vix=30.0, dxy_20d_change_pct=0.0, spx_20d_change_pct=2.0)
        assert bias == "risk-off"

    def test_risk_off_strong_dollar(self):
        bias = _compute_macro_bias(vix=18.0, dxy_20d_change_pct=2.5, spx_20d_change_pct=1.0)
        assert bias == "risk-off"

    def test_neutral_when_mixed(self):
        bias = _compute_macro_bias(vix=22.0, dxy_20d_change_pct=0.5, spx_20d_change_pct=-0.5)
        assert bias == "neutral"

    def test_neutral_when_spx_flat(self):
        bias = _compute_macro_bias(vix=15.0, dxy_20d_change_pct=-0.5, spx_20d_change_pct=-0.1)
        assert bias == "neutral"


class TestFetchMacroForDate:
    def _patch_frames(self, frames: dict):
        def _mock_download(ticker, **kwargs):
            return frames.get(ticker, pd.DataFrame())
        return patch("yfinance.download", side_effect=_mock_download)

    def test_returns_macro_data_on_happy_path(self):
        frames = {
            "^VIX":     _df([15.0] * 25),
            "DX-Y.NYB": _df([103.0 - i * 0.05 for i in range(25)]),   # weakening
            "^GSPC":    _df([4000.0 + i * 5 for i in range(25)]),     # rising
            "^TNX":     _df([39.5] * 25),  # 3.95% yield
        }
        with self._patch_frames(frames):
            result = fetch_macro_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))

        assert isinstance(result, MacroData)
        assert result.vix == 15.0
        assert result.dxy == pytest.approx(103.0 - 24 * 0.05, rel=1e-3)
        assert result.tnx_yield_pct == pytest.approx(3.95, rel=1e-3)
        assert result.spx_20d_change_pct > 0
        assert result.macro_bias == "risk-on"

    def test_tnx_divided_by_ten(self):
        frames = {
            "^VIX":     _df([15.0] * 25),
            "DX-Y.NYB": _df([100.0] * 25),
            "^GSPC":    _df([4000.0 + i for i in range(25)]),
            "^TNX":     _df([42.0] * 25),   # raw; actual yield = 4.20%
        }
        with self._patch_frames(frames):
            result = fetch_macro_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result.tnx_yield_pct == pytest.approx(4.20, rel=1e-3)

    def test_neutral_fallback_on_missing_vix(self):
        frames = {
            "^VIX":     pd.DataFrame(),
            "DX-Y.NYB": _df([100.0] * 25),
            "^GSPC":    _df([4000.0 + i for i in range(25)]),
            "^TNX":     _df([40.0] * 25),
        }
        with self._patch_frames(frames):
            result = fetch_macro_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result.macro_bias == "neutral"

    def test_end_date_strictly_before_as_of(self):
        """yfinance end param must be as_of date (exclusive upper bound)."""
        captured = {}

        def _mock_download(ticker, **kwargs):
            captured[ticker] = kwargs
            return _df([100.0] * 25)

        with patch("yfinance.download", side_effect=_mock_download):
            fetch_macro_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))

        # end should equal 2023-06-15 (yfinance treats end as exclusive, so
        # the last bar returned is 2023-06-14 — strictly before as_of).
        assert captured["^VIX"]["end"] == "2023-06-15"

    def test_handles_yfinance_exception(self):
        with patch("yfinance.download", side_effect=Exception("network down")):
            result = fetch_macro_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result.macro_bias == "neutral"
