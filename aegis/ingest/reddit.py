from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html import unescape
import re

import requests

from aegis.config import SETTINGS
from aegis.ingest.base import SourceConnector
from aegis.schemas import TextItem


_TAG = re.compile(r"<[^>]+>")


class RedditConnector(SourceConnector):
    """Public Reddit search connector for hackathon demos.

    Reddit may rate-limit unauthenticated requests. For production, use OAuth and platform
    terms-compliant quotas. Failure is surfaced to the caller rather than silently fabricated.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SETTINGS.user_agent})

    def fetch(self, query: str, limit: int = 25) -> list[TextItem]:
        url = f"https://www.reddit.com/r/{SETTINGS.reddit_subreddits}/search.json"
        params = {"q": query, "restrict_sr": "1", "sort": "new", "t": "day", "limit": min(100, limit), "raw_json": 1}
        r = self.session.get(url, params=params, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        data = r.json()
        out = []
        for c in data.get("data", {}).get("children", []):
            p = c.get("data", {})
            title = p.get("title", "")
            body = unescape(_TAG.sub(" ", p.get("selftext", "")))
            permalink = p.get("permalink", "")
            full_url = f"https://www.reddit.com{permalink}" if permalink.startswith("/") else p.get("url", "")
            ts = datetime.fromtimestamp(float(p.get("created_utc", 0) or 0), tz=timezone.utc)
            item_id = str(p.get("id") or hashlib.sha1(full_url.encode()).hexdigest())
            out.append(TextItem(
                source="Reddit",
                title=title,
                text=body or title,
                published_at=ts,
                url=full_url,
                author=str(p.get("author", "")),
                source_domain="reddit.com",
                metadata={
                    "subreddit": p.get("subreddit"), "score": p.get("score", 0),
                    "num_comments": p.get("num_comments", 0), "upvote_ratio": p.get("upvote_ratio"),
                },
                item_id=item_id,
            ))
        return out
