from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib

import requests

from aegis.config import SETTINGS
from aegis.ingest.base import SourceConnector
from aegis.schemas import TextItem


class GDELTConnector(SourceConnector):
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SETTINGS.user_agent})

    def fetch(self, query: str, limit: int = 25) -> list[TextItem]:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": min(max(1, limit), 250),
            "timespan": SETTINGS.news_lookback,
            "sort": "datedesc",
        }
        r = self.session.get(self.endpoint, params=params, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        payload = r.json()
        rows = []
        for a in payload.get("articles", []):
            url = a.get("url", "")
            domain = a.get("domain") or urlparse(url).netloc.lower().removeprefix("www.")
            dt = _parse_gdelt_date(a.get("seendate", ""))
            title = a.get("title", "")
            # DOC API returns article metadata/title; title is the reliable unstructured text field.
            # A production connector can add licensed full-text extraction.
            text = a.get("title", "")
            item_id = hashlib.sha1(f"gdelt|{url}|{title}".encode("utf-8", "ignore")).hexdigest()
            rows.append(TextItem(
                source="GDELT",
                title=title,
                text=text,
                published_at=dt,
                url=url,
                source_domain=domain,
                metadata={"language": a.get("language"), "sourcecountry": a.get("sourcecountry")},
                item_id=item_id,
            ))
        return rows


def _parse_gdelt_date(value: str) -> datetime:
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)
