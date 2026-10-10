from __future__ import annotations

import re
from dataclasses import dataclass

from aegis.config import SETTINGS
from aegis.nlp.preprocess import clean_text


@dataclass
class StanceResult:
    direction: float  # -1 harmful, +1 beneficial
    confidence: float
    model: str


HARM = {"default", "bankruptcy", "breach", "lawsuit", "probe", "investigation", "recall", "outage", "shortage", "downgrade", "cuts guidance", "misses", "fraud", "sanctions", "shutdown", "restatement", "fine", "warning"}
BENEFIT = {"beats", "beat expectations", "raises guidance", "approval", "approved", "upgrade", "record profit", "wins", "launches", "rebound", "accretive", "dismissed", "cleared"}
REVERSALS = {"dismissed", "closed without action", "no evidence", "denied", "resolved", "recovers", "restored"}


class ImpactStanceAnalyzer:
    def __init__(self) -> None:
        self._pipe = None
        if SETTINGS.use_transformers:
            try:
                from transformers import pipeline
                self._pipe = pipeline("zero-shot-classification", model=SETTINGS.event_model, device=SETTINGS.model_device)
            except Exception:
                pass

    def analyze(self, text: str, company: str) -> StanceResult:
        t = clean_text(text)[:5000]
        if self._pipe is not None:
            try:
                out = self._pipe(
                    t,
                    [f"harmful for {company}", f"beneficial for {company}", f"neutral for {company}"],
                    hypothesis_template="The described financial event is {}.",
                    multi_label=False,
                )
                lab = out["labels"][0].lower()
                conf = float(out["scores"][0])
                d = -1.0 if "harmful" in lab else 1.0 if "beneficial" in lab else 0.0
                return StanceResult(d * conf, conf, SETTINGS.event_model)
            except Exception:
                pass
        lo = t.lower()
        harm = sum(1.0 for x in HARM if x in lo)
        good = sum(1.0 for x in BENEFIT if x in lo)
        for x in REVERSALS:
            if x in lo:
                good += 1.25
                harm *= 0.6
        # basic negation around adverse words
        for m in re.finditer(r"\b(no|not|denies|denied|without)\b.{0,45}\b(investigation|breach|default|fraud|outage|shortage)\b", lo):
            harm = max(0.0, harm - 1.0)
            good += 0.5
        total = harm + good
        if total == 0:
            return StanceResult(0.0, 0.35, "stance-rules")
        d = (good - harm) / total
        return StanceResult(max(-1.0, min(1.0, d)), min(0.90, 0.55 + 0.08 * total), "stance-rules")
