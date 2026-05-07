"""
Fetches BTC/USD OHLCV data from Kraken.

Kraken is US-accessible with a free public API. Public endpoints require no
credentials — OHLCV history is available back to 2013. Authenticated
credentials (KRAKEN_API_KEY / KRAKEN_SECRET) are only needed for live order
placement. Kraken has no spot sandbox; paper trading uses DRY_RUN=1 instead.
"""

import os

import ccxt
import pandas as pd

SYMBOL = "BTC/USD"
TIMEFRAME = "1d"


def get_exchange(sandbox: bool = False) -> ccxt.Exchange:
    """Return a ccxt Kraken exchange instance.

    Kraken has no spot sandbox, so the *sandbox* parameter is accepted for
    interface compatibility but has no effect — the exchange always points at
    Kraken mainnet. For paper trading, set DRY_RUN=1 to skip order execution.

    Args:
        sandbox: Ignored. Kept for interface compatibility with callers that
                 pass sandbox=True for paper mode.
    """
    kwargs: dict = {}
    api_key = os.getenv("KRAKEN_API_KEY")
    secret = os.getenv("KRAKEN_SECRET")
    if api_key:
        kwargs["apiKey"] = api_key
    if secret:
        kwargs["secret"] = secret
    return ccxt.kraken(kwargs)


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    limit: int = 250,
    since: int | None = None,
) -> pd.DataFrame:
    """Fetch the most recent *limit* daily candles as a DataFrame.

    Args:
        exchange: ccxt exchange instance.
        limit: Number of candles to fetch.
        since: Start timestamp in milliseconds since epoch (ccxt convention).
               When None, fetches the most recent candles.

    Columns: open, high, low, close, volume  (indexed by UTC timestamp).
    Raises ValueError if fewer than 200 candles are returned — the minimum
    needed for EMA-200 computation.
    """
    raw = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=limit)
    if len(raw) < 200:
        raise ValueError(
            f"Only {len(raw)} candles returned; need ≥ 200 for EMA-200."
        )

    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def build_price_data(df: pd.DataFrame) -> dict:
    """Derive the price section of the context object from OHLCV data.

    Uses the most recent completed candle.  *change_pct_24h* measures the
    move from that candle's open to its close (intra-day).  *volume_vs_7d_avg*
    compares the latest candle's USD volume against the 7-candle average
    of the preceding week.
    """
    latest = df.iloc[-1]
    current = float(latest["close"])
    open_24h = float(latest["open"])
    high_24h = float(latest["high"])
    low_24h = float(latest["low"])
    volume_24h_usd = float(latest["volume"]) * current

    change_pct_24h = (current - open_24h) / open_24h * 100 if open_24h else 0.0

    # 7-candle average (excluding the latest candle)
    week_slice = df.iloc[-8:-1]
    avg_vol_usd = (week_slice["volume"] * week_slice["close"]).mean()
    volume_vs_7d_avg = volume_24h_usd / avg_vol_usd if avg_vol_usd > 0 else 1.0

    return {
        "current": round(current, 2),
        "open_24h": round(open_24h, 2),
        "high_24h": round(high_24h, 2),
        "low_24h": round(low_24h, 2),
        "change_pct_24h": round(change_pct_24h, 4),
        "volume_24h_usd": round(volume_24h_usd, 2),
        "volume_vs_7d_avg": round(volume_vs_7d_avg, 4),
    }
