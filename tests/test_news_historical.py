"""
Unit tests for src/data/news_historical.py (GDELT historical news).

All HTTP calls are mocked — no network required.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.data.news_historical import (
    GDELT_ALLOWLIST,
    _extract_domain,
    fetch_news_for_date,
)


def _mock_response(articles: list[dict]) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = {"articles": articles}
    resp.raise_for_status.return_value = None
    resp.status_code = 200
    return resp


class TestExtractDomain:
    def test_strips_www_prefix(self):
        assert _extract_domain("https://www.reuters.com/article/123") == "reuters.com"

    def test_handles_bare_domain(self):
        assert _extract_domain("https://coindesk.com/foo") == "coindesk.com"

    def test_returns_empty_on_bad_url(self):
        assert _extract_domain("") == ""


class TestFetchNewsForDate:
    def test_filters_by_allowlist(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.news_historical.CACHE_DIR", tmp_path)
        articles = [
            {"url": "https://reuters.com/a", "title": "Good", "seendate": "20230615T100000Z"},
            {"url": "https://randomblog.com/x", "title": "Spam", "seendate": "20230615T100000Z"},
            {"url": "https://coindesk.com/b", "title": "Also good", "seendate": "20230615T110000Z"},
        ]
        with patch("httpx.get", return_value=_mock_response(articles)):
            result = fetch_news_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))

        assert len(result) == 2
        assert all(item["source"] in GDELT_ALLOWLIST for item in result)
        assert [item["headline"] for item in result] == ["Good", "Also good"]

    def test_returns_empty_on_http_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.news_historical.CACHE_DIR", tmp_path)
        with patch("httpx.get", side_effect=httpx.HTTPError("boom")):
            result = fetch_news_for_date(datetime(2023, 6, 15, tzinfo=timezone.utc))

        assert result == []

    def test_respects_max_items(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.news_historical.CACHE_DIR", tmp_path)
        articles = [
            {"url": f"https://reuters.com/a{i}", "title": f"H{i}", "seendate": "20230615T100000Z"}
            for i in range(40)
        ]
        with patch("httpx.get", return_value=_mock_response(articles)):
            result = fetch_news_for_date(
                datetime(2023, 6, 15, tzinfo=timezone.utc),
                max_items=5,
            )

        assert len(result) == 5

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.news_historical.CACHE_DIR", tmp_path)
        articles = [
            {"url": "https://bloomberg.com/a", "title": "Cached", "seendate": "20230615T100000Z"},
        ]
        as_of = datetime(2023, 6, 15, tzinfo=timezone.utc)

        with patch("httpx.get", return_value=_mock_response(articles)) as mock_get:
            first = fetch_news_for_date(as_of)
        assert first[0]["headline"] == "Cached"

        cache_file = tmp_path / "2023-06-15.json"
        assert cache_file.exists()
        assert json.loads(cache_file.read_text())[0]["headline"] == "Cached"

        # Second call should hit the cache, not the network
        with patch("httpx.get", side_effect=AssertionError("network hit")) as mock_get2:
            second = fetch_news_for_date(as_of)
        assert second == first

    def test_publish_lag_cutoff_24h(self, tmp_path, monkeypatch):
        """The query window must END at as_of - 24h to avoid look-ahead."""
        monkeypatch.setattr("src.data.news_historical.CACHE_DIR", tmp_path)
        as_of = datetime(2023, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        captured = {}

        def _capture(url, params=None, timeout=None):
            captured["params"] = params
            return _mock_response([])

        with patch("httpx.get", side_effect=_capture):
            fetch_news_for_date(as_of)

        # enddatetime should be as_of - 24h = 20230614000000
        assert captured["params"]["enddatetime"] == "20230614000000"
        assert captured["params"]["startdatetime"] == "20230613000000"

    def test_naive_datetime_treated_as_utc(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.news_historical.CACHE_DIR", tmp_path)
        articles = [
            {"url": "https://reuters.com/a", "title": "X", "seendate": "20230615T100000Z"},
        ]
        naive = datetime(2023, 6, 15)
        with patch("httpx.get", return_value=_mock_response(articles)):
            result = fetch_news_for_date(naive)
        assert len(result) == 1
