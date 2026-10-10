from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean, pstdev

from aegis.config import SETTINGS
from aegis.nlp.preprocess import clean_text, sentence_split


@dataclass
class SentimentResult:
    score: float
    label: str
    confidence: float
    disagreement: float
    model: str
    components: list[dict]


# Fallback lexicon is intentionally finance/event oriented and handles negation/intensity.
POSITIVE = {
    "beat": 1.2, "beats": 1.2, "surge": 1.2, "growth": 0.8, "upgrade": 1.1,
    "upgraded": 1.1, "record": 0.6, "profit": 0.7, "strong": 0.6, "outperform": 1.0,
    "approval": 0.8, "approved": 0.8, "launch": 0.4, "win": 0.8, "rebound": 0.8,
    "raises": 0.7, "raised": 0.7, "expands": 0.6, "accretive": 1.0, "resilient": 0.6,
    "settles": 0.35, "cleared": 0.8, "dismissed": 0.7, "resolved": 0.6, "recovery": 0.7,
}
NEGATIVE = {
    "miss": 1.2, "misses": 1.2, "plunge": 1.5, "drop": 0.8, "falls": 0.9,
    "downgrade": 1.2, "downgraded": 1.2, "loss": 0.8, "losses": 0.9, "weak": 0.6,
    "default": 1.8, "bankruptcy": 2.0, "fraud": 1.8, "probe": 1.2, "investigation": 1.2,
    "lawsuit": 1.1, "recall": 1.0, "breach": 1.4, "hack": 1.4, "cyberattack": 1.6,
    "outage": 1.1, "layoffs": 0.8, "war": 1.7, "sanctions": 1.2, "crisis": 1.7,
    "shortage": 1.0, "slump": 1.2, "recession": 1.4, "warning": 0.9, "restatement": 1.3,
    "fine": 0.9, "penalty": 1.0, "delay": 0.6, "disruption": 1.0, "shortfall": 1.0,
    "liquidity": 0.45, "margin": 0.25, "impairment": 1.2, "write-down": 1.2,
}
NEGATIONS = {"not", "no", "never", "without", "hardly", "unlikely", "denies", "denied"}
INTENSIFIERS = {
    "sharply": 1.4, "severely": 1.5, "significantly": 1.25, "unexpectedly": 1.25,
    "massive": 1.45, "major": 1.25, "material": 1.3, "slightly": 0.65, "modest": 0.75,
}


class SentimentAnalyzer:
    """Entity-conditioned ensemble financial sentiment.

    MAX mode ensembles multiple public finance-domain classifiers. The ensemble deliberately
    exposes disagreement: uncertainty between models is itself useful risk metadata and is
    not silently averaged away. Offline mode falls back to a deterministic finance lexicon.
    """

    def __init__(self) -> None:
        self._pipes: list[tuple[str, object]] = []
        if SETTINGS.use_transformers:
            models = SETTINGS.sentiment_models if SETTINGS.max_profile else SETTINGS.sentiment_models[:1]
            for name in models:
                try:
                    from transformers import pipeline
                    p = pipeline(
                        "text-classification",
                        model=name,
                        tokenizer=name,
                        device=SETTINGS.model_device,
                        truncation=True,
                        top_k=None,
                    )
                    self._pipes.append((name, p))
                except Exception:
                    continue

    @staticmethod
    def entity_context(text: str, aliases: list[str], radius: int = 1) -> str:
        sentences = sentence_split(text)
        if not sentences:
            return clean_text(text)[:6000]
        idx = []
        for i, s in enumerate(sentences):
            if any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", s, re.I) for a in aliases if a):
                idx.append(i)
        if not idx:
            return clean_text(text)[:6000]
        keep = set()
        for i in idx:
            keep.update(range(max(0, i - radius), min(len(sentences), i + radius + 1)))
        return " ".join(sentences[i] for i in sorted(keep))[:6000]

    @staticmethod
    def _normalize_labels(raw: list[dict], pipe) -> tuple[float, float, float]:
        # Use model config labels whenever possible; many finance checkpoints use LABEL_0 etc.
        id2label = {}
        try:
            id2label = {str(k): str(v).lower() for k, v in pipe.model.config.id2label.items()}
        except Exception:
            pass
        probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for row in raw:
            label = str(row.get("label", "")).lower()
            score = float(row.get("score", 0.0))
            resolved = label
            if label.startswith("label_"):
                key = label.split("_")[-1]
                resolved = id2label.get(key, label)
            # Known mappings when id2label is generic.
            if resolved.startswith("label_"):
                # Prosus usually exposes semantic labels; finbert-tone generic order is neutral,pos,neg.
                idx = int(resolved.split("_")[-1])
                mapping = {0: "neutral", 1: "positive", 2: "negative"}
                resolved = mapping.get(idx, "neutral")
            if "pos" in resolved:
                probs["positive"] += score
            elif "neg" in resolved:
                probs["negative"] += score
            else:
                probs["neutral"] += score
        z = sum(probs.values()) or 1.0
        return probs["positive"] / z, probs["negative"] / z, probs["neutral"] / z

    def _transformer_ensemble(self, text: str) -> SentimentResult | None:
        if not self._pipes:
            return None
        components = []
        scores = []
        confs = []
        for name, pipe in self._pipes:
            try:
                out = pipe(clean_text(text)[:6000], top_k=None)
                if out and isinstance(out[0], list):
                    out = out[0]
                pos, neg, neu = self._normalize_labels(out, pipe)
                s = max(-1.0, min(1.0, pos - neg))
                c = max(pos, neg, neu)
                # Entropy-aware confidence: a high max probability is useful only if distribution is concentrated.
                ent = -sum(p * math.log(max(p, 1e-12)) for p in (pos, neg, neu)) / math.log(3.0)
                c_adj = c * (1.0 - 0.35 * ent)
                components.append({"model": name, "score": s, "confidence": c_adj, "p_pos": pos, "p_neg": neg, "p_neu": neu})
                scores.append(s)
                confs.append(c_adj)
            except Exception:
                continue
        if not scores:
            return None
        # Confidence-weighted robust mean; clipping prevents one overconfident checkpoint dominating.
        ws = [max(0.15, min(0.95, c)) for c in confs]
        weighted = sum(w * s for w, s in zip(ws, scores)) / sum(ws)
        disagreement = min(1.0, pstdev(scores) / 0.75) if len(scores) > 1 else 0.0
        confidence = max(0.0, min(1.0, mean(confs) * (1.0 - 0.55 * disagreement)))
        label = "positive" if weighted > 0.10 else "negative" if weighted < -0.10 else "neutral"
        return SentimentResult(weighted, label, confidence, disagreement, "ensemble:" + "+".join(x[0] for x in self._pipes), components)

    @staticmethod
    def _lexicon(text: str) -> SentimentResult:
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", clean_text(text).lower())
        raw, evidence = 0.0, 0
        for i, w in enumerate(words):
            val = POSITIVE.get(w, 0.0) - NEGATIVE.get(w, 0.0)
            if not val:
                continue
            evidence += 1
            window = words[max(0, i - 4):i]
            if any(n in window for n in NEGATIONS):
                val *= -0.9
            if i > 0 and words[i - 1] in INTENSIFIERS:
                val *= INTENSIFIERS[words[i - 1]]
            raw += val
        score = math.tanh(raw / max(2.0, math.sqrt(max(1, evidence)) * 2.0))
        confidence = min(0.86, 0.42 + 0.09 * evidence + 0.16 * abs(score))
        label = "positive" if score > 0.10 else "negative" if score < -0.10 else "neutral"
        return SentimentResult(score, label, confidence, 0.35 if evidence < 2 else 0.18, "finance-lexicon-fallback", [])

    def analyze(self, text: str) -> SentimentResult:
        return self._transformer_ensemble(text) or self._lexicon(text)
