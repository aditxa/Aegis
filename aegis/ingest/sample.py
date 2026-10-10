from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from aegis.ingest.base import SourceConnector
from aegis.schemas import TextItem


class SampleConnector(SourceConnector):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, query: str = "", limit: int = 100) -> list[TextItem]:
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            text = f"{r.get('title','')} {r.get('text','')}"
            if query and query.lower() not in text.lower():
                # Query filter is optional for sample mode; ticker/company search still works.
                pass
            dt = datetime.fromisoformat(r["published_at"].replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            out.append(TextItem(
                source=r["source"], title=r["title"], text=r.get("text", ""),
                published_at=dt, url=r.get("url", ""), author=r.get("author", ""),
                source_domain=r.get("source_domain", ""), metadata=r.get("metadata", {}),
                item_id=r.get("item_id", ""),
            ))
            if len(out) >= limit:
                break
        return out
