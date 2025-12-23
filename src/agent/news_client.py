"""
NewsAPI client with defensive error handling.

NewsAPI (https://newsapi.org) offers a free "Developer" tier API key
sufficient for this project (no payment required, rate-limited to 100
requests/day and historical articles limited to the last month). If no key
is configured, or the request fails for any reason (network error, rate
limit, invalid key), this client degrades gracefully by returning an empty
list rather than raising — callers (RAG ingestion, the alert agent) are
expected to fall back to synthetic/cached content in that case.
"""
from __future__ import annotations

import time
import requests

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
REQUEST_TIMEOUT_SECONDS = 10
MAX_ATTEMPTS = 3


class NewsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.newsapi_key

    def fetch_disruption_news(self, query: str = "supply chain disruption", page_size: int = 10, language: str = "en") -> list[dict]:
        """Fetch recent news articles matching `query`. Returns [] on any failure."""
        if not self.api_key or self.api_key == "your_newsapi_key_here":
            logger.info("NEWSAPI_KEY not configured; skipping live news fetch (caller should use fallback content).")
            return []

        params = {
            "q": query,
            "pageSize": page_size,
            "language": language,
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
        }

        response = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                if attempt + 1 == MAX_ATTEMPTS:
                    logger.error("NewsAPI request failed after %d attempts (%s)", MAX_ATTEMPTS, type(exc).__name__)
                    return []
                logger.warning(
                    "NewsAPI attempt %d/%d failed (%s)", attempt + 1, MAX_ATTEMPTS, type(exc).__name__,
                )
                time.sleep(2 ** attempt)

        if response is None:
            logger.error("NewsAPI request did not return a response")
            return []
        try:
            payload = response.json()
            articles = payload.get("articles", [])
        except (ValueError, KeyError) as exc:
            logger.error("Failed to parse NewsAPI response: %s", exc)
            return []

        results = []
        for a in articles:
            results.append({
                "title": a.get("title") or "",
                "description": a.get("description") or "",
                "url": a.get("url") or "",
                "published_at": a.get("publishedAt") or "",
                "source_name": (a.get("source") or {}).get("name") or "unknown",
            })

        logger.info("Fetched %d news articles for query='%s'", len(results), query)
        return results
