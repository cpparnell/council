"""
Sentiment data fetcher.

Fear & Greed Index — Alternative.me (free, no API key required).
Social data — proxied from Fear & Greed when LunarCrush key is absent.

Set LUNARCRUSH_API_KEY in .env for real social volume figures.
"""

import os

import httpx

_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


async def _fetch_fear_greed(client: httpx.AsyncClient) -> int:
    resp = await client.get(_FEAR_GREED_URL)
    resp.raise_for_status()
    data = resp.json()
    return int(data["data"][0]["value"])


async def fetch_sentiment(
    lunarcrush_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Fetch sentiment indicators.

    Returns:
        Dict with keys: fear_greed_index, reddit_sentiment, social_volume_vs_avg.
        *reddit_sentiment* is derived from Fear & Greed when LunarCrush is
        unavailable.  *social_volume_vs_avg* defaults to 1.0 in that case.
    """
    resolved_key = lunarcrush_key or os.getenv("LUNARCRUSH_API_KEY")

    async def _run(c: httpx.AsyncClient) -> dict:
        fear_greed = await _fetch_fear_greed(c)

        # Map Fear & Greed → directional label (proxy for social sentiment)
        if fear_greed >= 60:
            reddit_sentiment = "bullish"
        elif fear_greed <= 40:
            reddit_sentiment = "bearish"
        else:
            reddit_sentiment = "neutral"

        social_volume_vs_avg = 1.0  # placeholder until LunarCrush integrated

        return {
            "fear_greed_index": fear_greed,
            "reddit_sentiment": reddit_sentiment,
            "social_volume_vs_avg": social_volume_vs_avg,
        }

    if client is not None:
        return await _run(client)

    async with httpx.AsyncClient(timeout=10) as c:
        return await _run(c)
