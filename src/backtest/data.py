"""
Historical data fetching and slicing for backtesting.

Provides:
- fetch_full_ohlcv: Download daily OHLCV candles via Yahoo Finance (BTC-USD).
- slice_ohlcv: Return a lookback window with strict no-lookahead enforcement.
- fetch_historical_sentiment: Download Fear & Greed history from Alternative.me.
- get_sentiment_for_date: Look up the closest sentiment snapshot for a given date.
- NEUTRAL_NEWS_STUB: Placeholder news list used for all backtest cycles.

Note: yfinance (Yahoo Finance) is used for historical data because Kraken's
public OHLC API only provides the most recent ~720 daily candles (~2 years).
yfinance provides BTC-USD history back to 2014 with no API key required.
"""

from datetime import date, datetime, timezone

import httpx
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------ Constants

NEUTRAL_NEWS_STUB: list[dict] = [
    {
        "headline": f"Bitcoin market update #{i + 1}",
        "source": "stub",
        "published_at": "2024-01-01T00:00:00Z",
    }
    for i in range(15)
]

NEUTRAL_ONCHAIN_STUB: dict = {
    "exchange_net_flow_btc": 0.0,
    "whale_transactions_24h": 0,
    "sopr": 1.0,
}

_FNG_URL = "https://api.alternative.me/fng/"


# ------------------------------------------------------------------ OHLCV


def fetch_full_ohlcv(
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Download daily BTC-USD OHLCV candles via Yahoo Finance.

    Uses yfinance which provides history back to 2014 with no API key.
    The end_date is exclusive (consistent with Python date-range convention).

    Args:
        start_date: Inclusive start of the desired range (UTC).
        end_date: Exclusive end of the desired range (UTC).

    Returns:
        DataFrame with columns open/high/low/close/volume indexed by UTC date.

    Raises:
        ValueError: If no data is returned or fewer than 200 candles are
                    available (insufficient for EMA-200 computation).
    """
    df = yf.download(
        "BTC-USD",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError("No OHLCV data returned for the requested date range.")

    # yfinance 0.2+ returns MultiIndex columns — flatten to first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if len(df) < 200:
        raise ValueError(
            f"Only {len(df)} candles returned for the date range; "
            "need ≥ 200 to compute EMA-200."
        )

    return df


def slice_ohlcv(
    full_df: pd.DataFrame,
    as_of: datetime,
    lookback: int = 250,
) -> pd.DataFrame:
    """Return the *lookback* most recent rows with index STRICTLY BEFORE as_of.

    This is the single lookahead-prevention point. No candle from as_of or
    later will ever appear in the slice returned to the agents.

    Args:
        full_df: Full OHLCV DataFrame (output of fetch_full_ohlcv).
        as_of: The simulated decision time. Only rows before this timestamp
               are visible.
        lookback: Maximum number of rows to return (default 250).

    Returns:
        DataFrame slice with at most *lookback* rows.

    Raises:
        ValueError: If fewer than 200 rows are available before as_of
                    (insufficient for indicator computation).
    """
    cutoff = pd.Timestamp(as_of, tz="UTC") if as_of.tzinfo is None else pd.Timestamp(as_of)
    visible = full_df[full_df.index < cutoff]
    if len(visible) < 200:
        raise ValueError(
            f"Only {len(visible)} candles available before {as_of}; "
            "need ≥ 200 for indicator computation."
        )
    return visible.iloc[-lookback:]


# ---------------------------------------------------------------- Sentiment


def fetch_historical_sentiment(limit: int = 1000) -> dict[date, dict]:
    """Download historical Fear & Greed index data from Alternative.me.

    Args:
        limit: Number of days to fetch (max supported by the API is 1000).

    Returns:
        Mapping of {date: sentiment_dict} where sentiment_dict is compatible
        with the SentimentData schema:
            {"fear_greed_index": int, "reddit_sentiment": str, "social_volume_vs_avg": float}

        reddit_sentiment is derived from the Fear & Greed value:
            < 30  → "bearish"
            > 60  → "bullish"
            else  → "neutral"

        social_volume_vs_avg defaults to 1.0 (no historical data available).

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
    """
    response = httpx.get(_FNG_URL, params={"limit": limit, "format": "json"}, timeout=30)
    response.raise_for_status()
    payload = response.json()

    result: dict[date, dict] = {}
    for entry in payload.get("data", []):
        ts = datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc)
        fg_value = int(entry["value"])

        if fg_value < 30:
            reddit_sentiment = "bearish"
        elif fg_value > 60:
            reddit_sentiment = "bullish"
        else:
            reddit_sentiment = "neutral"

        result[ts.date()] = {
            "fear_greed_index": fg_value,
            "reddit_sentiment": reddit_sentiment,
            "social_volume_vs_avg": 1.0,
        }

    return result


def get_sentiment_for_date(
    history: dict[date, dict],
    target: datetime,
) -> dict:
    """Return the sentiment snapshot at or before *target* date.

    Falls back to the nearest earlier date if an exact match is missing.
    Falls back to a neutral stub if history is empty or target predates all data.

    Args:
        history: Output of fetch_historical_sentiment.
        target: The date for which to retrieve a sentiment snapshot.

    Returns:
        A sentiment dict compatible with SentimentData.
    """
    _NEUTRAL = {
        "fear_greed_index": 50,
        "reddit_sentiment": "neutral",
        "social_volume_vs_avg": 1.0,
    }

    if not history:
        return _NEUTRAL

    target_date = target.date() if isinstance(target, datetime) else target

    # Exact match
    if target_date in history:
        return history[target_date]

    # Nearest earlier date
    earlier = [d for d in history if d <= target_date]
    if not earlier:
        return _NEUTRAL

    return history[max(earlier)]
