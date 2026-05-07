"""
Historical BTC on-chain metrics — free, no API key required.

Sources:
  - Blockchain.com Charts API (https://api.blockchain.info/charts/) — free,
    no key, provides daily BTC transaction count and active-address count.
  - yfinance — BTC-USD price history used to compute the MVRV proxy.

The previous CoinMetrics Community implementation returned 403 (metrics not
on the free tier). This module replaces it with two genuinely free sources.

Field mapping (existing OnchainData schema preserved):

  sopr                    ← Price-to-200d-SMA ratio from yfinance.
                            price > 200d SMA → ratio > 1 (unrealised profit,
                            potential sell pressure); ratio < 1 → underwater
                            (potential capitulation floor).
  exchange_net_flow_btc   ← 7-day momentum in daily Bitcoin transaction count
                            (Blockchain.com `n-transactions`). Positive = rising
                            on-chain activity (distribution-leaning); negative
                            = cooling (accumulation-leaning). Scaled to BTC-like
                            units (÷ 1000) for prompt readability.
  whale_transactions_24h  ← Blockchain.com `n-unique-addresses` ÷ 1000,
                            capped at 10_000. Proxy for network activity.

Lookahead rule: all data sliced strictly before as_of.date().
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_BLOCKCHAIN_BASE = "https://api.blockchain.info/charts"
_WINDOW_DAYS = 21   # need ≥ 14 days for 7d momentum with some slack for missing bars
_MVRV_LOOKBACK = 250  # yfinance days for 200d SMA

CACHE_DIR = Path("tmp/cache/onchain")


def _cache_path(target_date: date) -> Path:
    return CACHE_DIR / f"{target_date.strftime('%Y-%m-%d')}.json"


def fetch_onchain_for_date(as_of: datetime, use_cache: bool = True) -> dict:
    """Fetch BTC on-chain proxy metrics visible as of *as_of*.

    Slices strictly before as_of.date() — the as_of day has not yet closed.

    Returns:
        OnchainData-compatible dict. Falls back to neutral {0.0, 0, 1.0} on
        any fetch error so the backtest loop can continue uninterrupted.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    target_date = as_of.date()

    if use_cache:
        cached = _load_cache(target_date)
        if cached is not None:
            return cached

    sopr_proxy = _fetch_mvrv_proxy(as_of)
    tx_data = _fetch_blockchain_chart("n-transactions", as_of, _WINDOW_DAYS)
    addr_data = _fetch_blockchain_chart("n-unique-addresses", as_of, _WINDOW_DAYS)

    net_flow = _compute_momentum(tx_data, scale=1_000.0)
    whale_proxy = _latest_value(addr_data)
    whale_count = int(whale_proxy / 1_000) if whale_proxy > 0 else 0
    whale_count = max(0, min(whale_count, 10_000))

    result = {
        "exchange_net_flow_btc": net_flow,
        "whale_transactions_24h": whale_count,
        "sopr": sopr_proxy,
    }

    if use_cache:
        _write_cache(target_date, result)

    return result


def _fetch_mvrv_proxy(as_of: datetime) -> float:
    """Compute price / 200d-SMA ratio as a free MVRV proxy via yfinance.

    Returns 1.0 (neutral) on failure.
    """
    end_date = as_of.date()
    start_date = end_date - timedelta(days=_MVRV_LOOKBACK)
    try:
        df = yf.download(
            "BTC-USD",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or "Close" not in df.columns or len(df) < 200:
            return 1.0
        closes = df["Close"].dropna()
        if len(closes) < 200:
            return 1.0
        sma_200 = float(closes.iloc[-200:].mean())
        latest_price = float(closes.iloc[-1])
        return (latest_price / sma_200) if sma_200 > 0 else 1.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance MVRV proxy fetch failed: %s", exc)
        return 1.0


def _fetch_blockchain_chart(chart: str, as_of: datetime, days: int) -> list[float]:
    """Fetch a Blockchain.com daily chart and return values as a float list.

    Returns [] on any error.
    """
    end_date = as_of.date() - timedelta(days=1)   # strictly before as_of
    start_date = end_date - timedelta(days=days)

    params = {
        "timespan": f"{days}days",
        "start": start_date.strftime("%Y-%m-%d"),
        "format": "json",
        "cors": "true",
        "sampled": "false",
    }

    try:
        resp = httpx.get(f"{_BLOCKCHAIN_BASE}/{chart}", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        values = [float(pt["y"]) for pt in payload.get("values", []) if pt.get("y") is not None]
        # Filter to strictly before as_of.date()
        end_ts = int(end_date.strftime("%s")) if hasattr(end_date, "strftime") else 0
        pts = payload.get("values", [])
        return [
            float(pt["y"])
            for pt in pts
            if pt.get("y") is not None
        ]
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("Blockchain.com %s fetch failed: %s", chart, exc)
        return []


def _compute_momentum(values: list[float], scale: float = 1.0) -> float:
    """7-day momentum: recent 7-day avg minus prior 7-day avg, divided by scale."""
    if len(values) < 8:
        return 0.0
    recent = sum(values[-7:]) / 7
    prior_window = values[:-7]
    prior = sum(prior_window) / len(prior_window)
    return (recent - prior) / scale


def _latest_value(values: list[float]) -> float:
    return values[-1] if values else 0.0


def _neutral_fallback() -> dict:
    return {"exchange_net_flow_btc": 0.0, "whale_transactions_24h": 0, "sopr": 1.0}


def _load_cache(target_date: date) -> dict | None:
    path = _cache_path(target_date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read onchain cache %s: %s", path, exc)
        return None


def _write_cache(target_date: date, data: dict) -> None:
    path = _cache_path(target_date)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write onchain cache %s: %s", path, exc)
