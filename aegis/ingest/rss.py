from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse
import hashlib
import xml.etree.ElementTree as ET

import requests

from aegis.config import SETTINGS
from aegis.ingest.base import SourceConnector
from aegis.schemas import TextItem


class GoogleNewsRSSConnector(SourceConnector):
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SETTINGS.user_agent})

    def fetch(self, query: str, limit: int = 25) -> list[TextItem]:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        r = self.session.get(url, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = (item.findtext("description") or "").strip()
            pub = item.findtext("pubDate") or ""
            try:
                dt = parsedate_to_datetime(pub)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
            domain = urlparse(link).netloc.lower().removeprefix("www.")
            item_id = hashlib.sha1(f"rss|{link}|{title}".encode("utf-8", "ignore")).hexdigest()
            out.append(TextItem(
                source="GoogleNewsRSS",
                title=title,
                text=desc or title,
                published_at=dt,
                url=link,
                source_domain=domain,
                item_id=item_id,
            ))
        return out
