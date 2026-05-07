"""
Historical BTC news headlines via GDELT DOC 2.0 artlist API.

Used by the backtesting loop to replace NEUTRAL_NEWS_STUB with real dated
headlines so the sentiment agent has meaningful signal.

Source allowlist is applied post-fetch to reduce blogspam. Queries use a
strict `as_of - 24h` cutoff to respect publish-lag and avoid look-ahead:
a headline timestamped 2023-06-15 10:00 UTC is NOT allowed to inform a
decision made at 2023-06-15 00:00 UTC.

Results are cached to tmp/cache/gdelt/YYYY-MM-DD.json to avoid repeated
API calls across backtest runs. The cache is the primary rate-control
mechanism — every date is fetched at most once per machine.

Rate limiting: GDELT does not publish explicit limits but enforces them
aggressively. We sleep 2 s after every uncached API call and retry up to
3 times with exponential backoff on 429.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Sources we trust for BTC reporting. Everything else is filtered out post-fetch.
GDELT_ALLOWLIST = frozenset({
    "reuters.com",
    "bloomberg.com",
    "coindesk.com",
    "wsj.com",
    "ft.com",
    "cointelegraph.com",
    "forbes.com",
    "cnbc.com",
    "theblock.co",
    "decrypt.co",
})

CACHE_DIR = Path("tmp/cache/gdelt")

# Seconds to sleep before each GDELT request (proactive rate control)
_PRE_REQUEST_SLEEP = 2.0
# Additional seconds to sleep after each uncached fetch
_POST_FETCH_SLEEP = 1.0
# Max retry attempts on 429
_MAX_RETRIES = 3


def _cache_path(target_date: datetime) -> Path:
    return CACHE_DIR / f"{target_date.strftime('%Y-%m-%d')}.json"


def _extract_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _gdelt_get(params: dict) -> dict | None:
    """GET GDELT with proactive rate pacing and exponential-backoff retry on 429.

    Sleeps _PRE_REQUEST_SLEEP seconds before every attempt so that the request
    rate is controlled from the very first call rather than only after a 429.
    """
    retry_delay = 10.0
    for attempt in range(_MAX_RETRIES):
        time.sleep(_PRE_REQUEST_SLEEP)
        try:
            resp = httpx.get(GDELT_URL, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning(
                    "GDELT rate-limited (attempt %d/%d); sleeping %.0fs",
                    attempt + 1, _MAX_RETRIES, retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay *= 3
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError as exc:
            logger.warning("GDELT HTTP error: %s", exc)
            return None
    logger.warning("GDELT still rate-limited after %d attempts; skipping.", _MAX_RETRIES)
    return None


def fetch_news_for_date(
    as_of: datetime,
    max_items: int = 20,
    use_cache: bool = True,
) -> list[dict]:
    """Fetch BTC news headlines visible as of *as_of* via GDELT DOC 2.0 artlist.

    The query window is [as_of - 48h, as_of - 24h] — a 24-hour slice ending
    24 hours before the decision timestamp. The 24-hour buffer accounts for
    GDELT publish lag and prevents any article from the decision day itself
    from leaking into the context.

    Args:
        as_of: Decision timestamp. Must be tz-aware UTC.
        max_items: Cap on returned headlines (post-filter).
        use_cache: When True, read/write tmp/cache/gdelt/YYYY-MM-DD.json.

    Returns:
        List of dicts with keys: headline, source, published_at.
        Returns [] on fetch error or empty result — the caller is responsible
        for the stub fallback.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    if use_cache:
        cached = _load_cache(as_of)
        if cached is not None:
            return cached[:max_items]

    end_dt = as_of - timedelta(hours=24)
    start_dt = end_dt - timedelta(hours=24)

    params = {
        "query": "bitcoin sourcelang:eng",
        "mode": "artlist",
        "format": "json",
        "maxrecords": "75",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "sort": "hybridrel",
    }

    payload = _gdelt_get(params)

    # Always sleep after a live API call, whether it succeeded or not, so
    # back-to-back cycles don't hammer the endpoint.
    time.sleep(_POST_FETCH_SLEEP)

    if payload is None:
        return []

    raw_articles = payload.get("articles", []) or []
    filtered: list[dict] = []
    for art in raw_articles:
        domain = _extract_domain(art.get("url", ""))
        if domain not in GDELT_ALLOWLIST:
            continue
        filtered.append({
            "headline": art.get("title", "").strip(),
            "source": domain,
            "published_at": art.get("seendate", ""),
        })

    if use_cache:
        _write_cache(as_of, filtered)

    return filtered[:max_items]


def _load_cache(as_of: datetime) -> list[dict] | None:
    path = _cache_path(as_of)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read GDELT cache %s: %s", path, exc)
        return None


def _write_cache(as_of: datetime, items: list[dict]) -> None:
    path = _cache_path(as_of)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write GDELT cache %s: %s", path, exc)
