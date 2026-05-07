"""
Unit tests for src/data/onchain_historical.py.

All HTTP and yfinance calls are mocked — no network required.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from src.data.onchain_historical import (
    _compute_momentum,
    _fetch_mvrv_proxy,
    _latest_value,
    fetch_onchain_for_date,
)


def _btc_df(prices: list[float]) -> pd.DataFrame:
    """Build a minimal yfinance-like Close DataFrame."""
    idx = pd.date_range("2022-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def _mock_blockchain(values: list[float]) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = {"values": [{"x": i, "y": v} for i, v in enumerate(values)]}
    resp.raise_for_status.return_value = None
    resp.status_code = 200
    return resp


class TestComputeMomentum:
    def test_positive_when_recent_higher(self):
        values = [10.0] * 8 + [20.0] * 7   # recent avg 20, prior avg 10
        assert _compute_momentum(values) > 0

    def test_negative_when_recent_lower(self):
        values = [20.0] * 8 + [10.0] * 7
        assert _compute_momentum(values) < 0

    def test_zero_on_too_few(self):
        assert _compute_momentum([1.0, 2.0, 3.0]) == 0.0

    def test_scale_divides_result(self):
        values = [10.0] * 8 + [20.0] * 7
        unscaled = _compute_momentum(values, scale=1.0)
        scaled = _compute_momentum(values, scale=10.0)
        assert scaled == pytest.approx(unscaled / 10.0)


class TestLatestValue:
    def test_returns_last(self):
        assert _latest_value([1.0, 2.0, 3.0]) == 3.0

    def test_empty_returns_zero(self):
        assert _latest_value([]) == 0.0


class TestFetchMvrvProxy:
    def test_returns_ratio_above_one_when_price_above_sma(self):
        # 200 closes of 50_000, then final close of 60_000 → ratio ≈ 1.1
        prices = [50_000.0] * 199 + [60_000.0]
        with patch("yfinance.download", return_value=_btc_df(prices)):
            result = _fetch_mvrv_proxy(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result > 1.0

    def test_returns_ratio_below_one_when_price_below_sma(self):
        prices = [50_000.0] * 199 + [40_000.0]
        with patch("yfinance.download", return_value=_btc_df(prices)):
            result = _fetch_mvrv_proxy(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result < 1.0

    def test_returns_one_on_insufficient_data(self):
        with patch("yfinance.download", return_value=_btc_df([50_000.0] * 50)):
            result = _fetch_mvrv_proxy(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result == 1.0

    def test_returns_one_on_empty_df(self):
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = _fetch_mvrv_proxy(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result == 1.0

    def test_returns_one_on_exception(self):
        with patch("yfinance.download", side_effect=Exception("network down")):
            result = _fetch_mvrv_proxy(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result == 1.0


class TestFetchOnchainForDate:
    def _setup_mocks(self, tx_values: list[float], addr_values: list[float],
                     btc_prices: list[float]):
        """Return context manager patching yfinance and httpx.get."""
        btc_df = _btc_df(btc_prices)
        yf_patch = patch("yfinance.download", return_value=btc_df)

        blockchain_responses = {
            "n-transactions": _mock_blockchain(tx_values),
            "n-unique-addresses": _mock_blockchain(addr_values),
        }

        def _httpx_side_effect(url, **kwargs):
            for key, resp in blockchain_responses.items():
                if key in url:
                    return resp
            raise httpx.HTTPError(f"unexpected url: {url}")

        httpx_patch = patch("httpx.get", side_effect=_httpx_side_effect)
        return yf_patch, httpx_patch

    def test_returns_valid_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.onchain_historical.CACHE_DIR", tmp_path)
        tx = [300_000.0] * 14 + [350_000.0] * 7
        addr = [900_000.0] * 21
        prices = [50_000.0] * 200
        yf_p, httpx_p = self._setup_mocks(tx, addr, prices)
        with yf_p, httpx_p:
            result = fetch_onchain_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))

        assert "exchange_net_flow_btc" in result
        assert "whale_transactions_24h" in result
        assert "sopr" in result
        assert isinstance(result["whale_transactions_24h"], int)

    def test_whale_proxy_scales_from_addresses(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.onchain_historical.CACHE_DIR", tmp_path)
        addr = [700_000.0] * 21
        tx = [300_000.0] * 21
        prices = [50_000.0] * 200
        yf_p, httpx_p = self._setup_mocks(tx, addr, prices)
        with yf_p, httpx_p:
            result = fetch_onchain_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result["whale_transactions_24h"] == 700

    def test_whale_proxy_capped_at_10000(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.onchain_historical.CACHE_DIR", tmp_path)
        addr = [500_000_000.0] * 21
        tx = [300_000.0] * 21
        prices = [50_000.0] * 200
        yf_p, httpx_p = self._setup_mocks(tx, addr, prices)
        with yf_p, httpx_p:
            result = fetch_onchain_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result["whale_transactions_24h"] == 10_000

    def test_net_flow_positive_on_rising_tx_volume(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.onchain_historical.CACHE_DIR", tmp_path)
        tx = [200_000.0] * 14 + [400_000.0] * 7   # doubling in last 7 days
        addr = [900_000.0] * 21
        prices = [50_000.0] * 200
        yf_p, httpx_p = self._setup_mocks(tx, addr, prices)
        with yf_p, httpx_p:
            result = fetch_onchain_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))
        assert result["exchange_net_flow_btc"] > 0

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.onchain_historical.CACHE_DIR", tmp_path)
        tx = [300_000.0] * 21
        addr = [900_000.0] * 21
        prices = [50_000.0] * 200
        as_of = datetime(2023, 6, 15, tzinfo=timezone.utc)
        yf_p, httpx_p = self._setup_mocks(tx, addr, prices)
        with yf_p, httpx_p:
            first = fetch_onchain_for_date(as_of)

        cache_file = tmp_path / "2023-06-15.json"
        assert cache_file.exists()
        assert json.loads(cache_file.read_text())["sopr"] == first["sopr"]

        # Second call must not hit network
        with patch("httpx.get", side_effect=AssertionError("network hit")), \
             patch("yfinance.download", side_effect=AssertionError("yf hit")):
            second = fetch_onchain_for_date(as_of)
        assert second == first

    def test_graceful_on_blockchain_http_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.onchain_historical.CACHE_DIR", tmp_path)
        prices = [50_000.0] * 200
        with patch("yfinance.download", return_value=_btc_df(prices)), \
             patch("httpx.get", side_effect=httpx.HTTPError("boom")):
            result = fetch_onchain_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))
        # Should return neutral defaults for tx/addr fields but still have MVRV from yfinance
        assert result["sopr"] > 0
        assert result["exchange_net_flow_btc"] == 0.0
