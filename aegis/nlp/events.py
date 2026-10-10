from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from aegis.config import SETTINGS
from aegis.nlp.preprocess import clean_text


EVENT_LABELS = [
    "Geopolitical", "Macroeconomic", "Credit Event", "Merger/Acquisition", "Earnings",
    "Regulatory/Legal", "Cybersecurity", "Supply Chain", "Management/Governance",
    "Operational Incident", "Product Launch", "Market/Trading", "Financing/Liquidity", "Other",
]

EVENT_PRIORS = {
    "Geopolitical": 0.86, "Macroeconomic": 0.72, "Credit Event": 0.96,
    "Merger/Acquisition": 0.60, "Earnings": 0.56, "Regulatory/Legal": 0.75,
    "Cybersecurity": 0.90, "Supply Chain": 0.72, "Management/Governance": 0.69,
    "Operational Incident": 0.80, "Product Launch": 0.38, "Market/Trading": 0.67,
    "Financing/Liquidity": 0.84, "Other": 0.42,
}

KEYWORDS = {
    "Geopolitical": {"war": 1.5, "missile": 1.4, "invasion": 1.8, "sanction": 1.2, "sanctions": 1.2, "embargo": 1.4, "conflict": 1.1, "tariff": 0.9},
    "Macroeconomic": {"inflation": 1.0, "cpi": 1.1, "gdp": 1.0, "recession": 1.5, "unemployment": 1.1, "central bank": 1.3, "rate hike": 1.4, "rate cut": 1.2, "yield": 0.7},
    "Credit Event": {"default": 2.0, "bankruptcy": 2.2, "insolvency": 2.0, "downgrade": 1.4, "missed payment": 1.8, "covenant breach": 1.8, "credit event": 1.8},
    "Merger/Acquisition": {"merger": 1.7, "acquisition": 1.7, "acquire": 1.3, "takeover": 1.6, "buyout": 1.7, "deal": 0.5, "bid": 0.7},
    "Earnings": {"earnings": 1.5, "revenue": 0.9, "guidance": 1.4, "profit": 0.8, "eps": 1.1, "quarter": 0.5, "forecast": 0.8, "outlook": 0.8},
    "Regulatory/Legal": {"regulator": 1.1, "regulatory": 1.0, "lawsuit": 1.5, "antitrust": 1.7, "probe": 1.5, "investigation": 1.5, "fine": 1.2, "sec": 1.0, "ftc": 1.0, "doj": 1.0},
    "Cybersecurity": {"cyberattack": 2.0, "cyber attack": 2.0, "ransomware": 2.1, "data breach": 1.9, "breach": 1.4, "hack": 1.4, "malware": 1.6},
    "Supply Chain": {"supply chain": 1.8, "shortage": 1.5, "supplier": 0.9, "factory": 0.7, "port closure": 1.6, "shipping": 0.8, "logistics": 0.9, "inventory": 0.6},
    "Management/Governance": {"ceo resigns": 1.8, "cfo resigns": 1.8, "resignation": 1.3, "board": 0.6, "governance": 1.2, "accounting": 1.0, "restatement": 1.7},
    "Operational Incident": {"outage": 1.6, "accident": 1.3, "fire": 1.2, "explosion": 1.6, "shutdown": 1.5, "trading halt": 1.1, "recall": 1.4, "disruption": 1.1},
    "Product Launch": {"launch": 1.2, "unveil": 1.3, "release": 0.8, "product approval": 1.5, "fda approval": 1.7},
    "Market/Trading": {"short squeeze": 1.8, "margin call": 1.8, "volatility": 1.0, "selloff": 1.5, "flash crash": 2.2, "volume spike": 1.3, "limit down": 1.8},
    "Financing/Liquidity": {"liquidity crisis": 2.0, "cash crunch": 1.8, "funding stress": 1.8, "refinancing": 1.0, "capital raise": 1.0, "debt issuance": 0.8, "withdrawals": 1.2, "bank run": 2.2},
}

@dataclass
class EventCandidate:
    label: str
    confidence: float
    prior_severity: float
    evidence_score: float = 0.0

@dataclass
class EventResult:
    label: str
    confidence: float
    prior_severity: float
    model: str
    candidates: list[EventCandidate] = field(default_factory=list)


class EventClassifier:
    """Multi-label event recognizer with ontology-aware rule fallback.

    The case-study schema still receives one primary event label, but secondary candidates are
    retained so multi-event articles do not collapse into one brittle class.
    """
    def __init__(self) -> None:
        self._pipe = None
        if SETTINGS.use_transformers:
            try:
                from transformers import pipeline
                self._pipe = pipeline("zero-shot-classification", model=SETTINGS.event_model, device=SETTINGS.model_device)
            except Exception:
                pass

    def _nli(self, text: str) -> EventResult | None:
        if self._pipe is None:
            return None
        try:
            out = self._pipe(
                clean_text(text)[:6000], EVENT_LABELS[:-1],
                hypothesis_template="This financial text contains a {} event.", multi_label=True,
            )
            cands = []
            for label, score in zip(out["labels"], out["scores"]):
                s = float(score)
                if s >= 0.18 or not cands:
                    cands.append(EventCandidate(label, s, EVENT_PRIORS[label], s))
                if len(cands) >= 4:
                    break
            cands.sort(key=lambda x: x.confidence * (0.75 + 0.25 * x.prior_severity), reverse=True)
            top = cands[0]
            return EventResult(top.label, top.confidence, top.prior_severity, SETTINGS.event_model, cands)
        except Exception:
            return None

    @staticmethod
    def _rules(text: str) -> EventResult:
        t = clean_text(text).lower()
        scores = {}
        for label, kws in KEYWORDS.items():
            score = 0.0
            for kw, weight in kws.items():
                pat = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
                n = len(re.findall(pat, t))
                if n:
                    # Repeating the same keyword in headline + body is usually syndication/echo,
                    # not independent evidence. Cap repetition aggressively so a generic term
                    # (e.g. "outage") cannot swamp a more specific causal event
                    # (e.g. "cyberattack").
                    score += weight * (1.0 + 0.20 * (min(3, n) - 1))
            scores[label] = score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked or ranked[0][1] <= 0:
            c = EventCandidate("Other", 0.38, EVENT_PRIORS["Other"], 0.0)
            return EventResult("Other", 0.38, EVENT_PRIORS["Other"], "event-ontology-rules", [c])
        top_score = ranked[0][1]
        # softmax-ish confidence over rule evidence prevents strong overconfidence from one token.
        vals = [math.exp(min(5.0, s)) for _, s in ranked[:6]]
        den = sum(vals) + 1e-9
        cands = []
        for (label, s), ev in zip(ranked[:4], vals[:4]):
            if s <= 0:
                continue
            conf = min(0.94, 0.45 + 0.5 * ev / den + 0.05 * min(s, 4.0))
            cands.append(EventCandidate(label, conf, EVENT_PRIORS[label], s))
        top = cands[0]
        return EventResult(top.label, top.confidence, top.prior_severity, "event-ontology-rules", cands)

    def classify(self, text: str) -> EventResult:
        return self._nli(text) or self._rules(text)
