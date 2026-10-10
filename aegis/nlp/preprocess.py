from __future__ import annotations

import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" URL ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def sentence_split(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    # Conservative split: avoids breaking decimal numbers and many abbreviations.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]
