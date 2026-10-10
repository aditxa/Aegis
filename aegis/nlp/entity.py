from __future__ import annotations

import csv
import re
from pathlib import Path

from aegis.schemas import EntityMatch
from aegis.nlp.preprocess import clean_text, sentence_split


ACTOR_VERBS = r"announced|reported|warned|said|filed|launched|acquired|sued|faces|cut|raised|issued|disclosed|confirmed|denied|resigned"
TARGET_PATTERNS = r"against|targeting|affecting|impacting|from|by"


class EntityLinker:
    """High-precision alias linker with short-ticker ambiguity protection and rough event role tagging."""
    def __init__(self, companies_csv: str | Path):
        self.entries: list[tuple[str, str, list[str], str]] = []
        with open(companies_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                aliases = [a.strip() for a in row.get("aliases", "").split("|") if a.strip()]
                aliases.extend([row["company"]])
                self.entries.append((row["ticker"].upper(), row["company"], sorted(set(aliases), key=len, reverse=True), row.get("sector", "")))

    @staticmethod
    def _ticker_hit(raw: str, ticker: str) -> bool:
        if re.search(rf"(?<!\w)\${re.escape(ticker)}(?!\w)", raw, re.I):
            return True
        # Short tickers are dangerous English tokens; require exact upper-case form.
        if len(ticker) <= 3:
            return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", raw))
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", raw, re.I))

    @staticmethod
    def _alias_hit(raw: str, alias: str) -> bool:
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", raw, re.I))

    @staticmethod
    def _role(text: str, alias: str) -> str:
        for s in sentence_split(text):
            m = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", s, re.I)
            if not m:
                continue
            left = s[max(0, m.start()-70):m.start()].lower()
            right = s[m.end():m.end()+100].lower()
            if re.search(rf"^\s*(?:{ACTOR_VERBS})\b|\b(?:{ACTOR_VERBS})\b", right):
                return "actor"
            if re.search(rf"\b(?:{TARGET_PATTERNS})\s*$", left):
                return "target"
        return "mentioned"

    def link(self, text: str) -> list[EntityMatch]:
        raw = clean_text(text)
        matches: list[EntityMatch] = []
        for ticker, company, aliases, sector in self.entries:
            hit_alias = ""
            conf = 0.0
            if self._ticker_hit(raw, ticker):
                hit_alias, conf = ticker, 0.97 if len(ticker) > 3 else 0.91
            else:
                for alias in aliases:
                    if self._alias_hit(raw, alias):
                        hit_alias = alias
                        conf = 0.98 if alias.lower() == company.lower() else 0.94
                        break
            if hit_alias:
                matches.append(EntityMatch(ticker=ticker, company=company, confidence=conf, matched_alias=hit_alias, sector=sector, role=self._role(raw, hit_alias)))
        return matches
