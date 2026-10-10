from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

AUTHORITATIVE = {
    "sec.gov", "federalreserve.gov", "ecb.europa.eu", "bis.org", "rbi.org.in", "bankofengland.co.uk",
    "justice.gov", "ftc.gov", "cftc.gov", "treasury.gov",
}
TOP_NEWS = {"reuters.com", "apnews.com", "ft.com", "wsj.com", "bloomberg.com"}
MAJOR_NEWS = {"cnbc.com", "marketwatch.com", "finance.yahoo.com", "economictimes.indiatimes.com", "moneycontrol.com", "forbes.com", "businessinsider.com"}
SOCIAL = {"reddit.com", "x.com", "twitter.com", "stocktwits.com", "youtube.com", "tiktok.com"}


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _in(domain: str, group: set[str]) -> bool:
    return domain in group or any(domain.endswith("."+x) for x in group)


def source_quality(source: str, source_domain: str) -> float:
    d = (source_domain or "").lower().removeprefix("www.")
    s = (source or "").lower()
    if _in(d, AUTHORITATIVE): return 0.99
    if _in(d, TOP_NEWS): return 0.94
    if _in(d, MAJOR_NEWS): return 0.79
    if _in(d, SOCIAL): return 0.40
    # Investor-relations/corporate domains can be authoritative about what a company said,
    # but not necessarily independent evidence about its economic interpretation.
    if any(x in d for x in ("investor", "ir.", "corp")) or "press release" in s: return 0.76
    if "gdelt" in s: return 0.62  # aggregator; article domain should dominate when present.
    if "reddit" in s or "social" in s: return 0.40
    return 0.58
