from __future__ import annotations

import math
import re


SENSATIONAL = {"guaranteed", "100%", "insider", "urgent", "must buy", "must sell", "exploding", "moon", "rug", "secret", "leaked", "breaking!!!", "panic"}


def manipulation_score(text: str, source_quality: float, contradiction: float, duplicate_probability: float, uncertainty: float) -> float:
    t = text or ""
    lo = t.lower()
    sensational = sum(x in lo for x in SENSATIONAL)
    caps = re.findall(r"\b[A-Z]{4,}\b", t)
    exclam = min(5, t.count("!"))
    url_spam = len(re.findall(r"https?://|www\.", t))
    raw = (
        1.0 * max(0.0, 0.60 - source_quality)
        + 0.55 * contradiction
        + 0.35 * duplicate_probability * max(0.0, 0.65-source_quality)
        + 0.35 * uncertainty
        + 0.22 * sensational
        + 0.08 * len(caps)
        + 0.05 * exclam
        + 0.06 * max(0, url_spam-1)
    )
    return max(0.0, min(1.0, 1.0 - math.exp(-raw)))
