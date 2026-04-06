"""
News feed fetcher — CryptoPanic (free tier, BTC-filtered).

Returns an empty list rather than raising when the API key is absent so
the Tier-0 validator can emit the appropriate "insufficient news" warning
rather than a raw exception.

Set CRYPTOPANIC_API_KEY in .env to enable live fetching.
"""

import os

import httpx

_BASE_URL = "https://cryptopanic.com/api/v1/posts/"


async def fetch_news(
    api_key: str | None = None,
    limit: int = 20,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Fetch the latest BTC news headlines from CryptoPanic.

    Args:
        api_key: CryptoPanic auth token.  Defaults to CRYPTOPANIC_API_KEY env var.
        limit: Maximum number of items to return.
        client: Optional pre-built httpx client (useful for testing).

    Returns:
        List of dicts with keys: headline, source, published_at.
        Returns [] if no API key is configured.
    """
    resolved_key = api_key or os.getenv("CRYPTOPANIC_API_KEY")
    if not resolved_key:
        return []

    params = {
        "auth_token": resolved_key,
        "currencies": "BTC",
        "kind": "news",
        "public": "true",
    }

    async def _get(c: httpx.AsyncClient) -> list[dict]:
        resp = await c.get(_BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        items = []
        for post in data.get("results", [])[:limit]:
            items.append(
                {
                    "headline": post.get("title", ""),
                    "source": post.get("source", {}).get("title", ""),
                    "published_at": post.get("published_at", ""),
                }
            )
        return items

    if client is not None:
        return await _get(client)

    async with httpx.AsyncClient(timeout=30) as c:
        return await _get(c)
