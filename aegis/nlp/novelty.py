from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

from aegis.config import SETTINGS
from aegis.nlp.preprocess import clean_text


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", clean_text(text).lower())


def _shingles(text: str, n: int = 4) -> set[str]:
    toks = _tokens(text)
    if len(toks) < n:
        return set(toks)
    return {" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _cos(a, b) -> float:
    try:
        import numpy as np
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


@dataclass
class NoveltyResult:
    novelty: float
    duplicate_probability: float
    cluster_id: str
    lexical_similarity: float
    semantic_similarity: float


class NoveltyTracker:
    """Story-chain clustering, not just same-domain duplicate detection.

    Rewrites across independent domains often evade word-count cosine duplicate filters. We combine
    4-shingle overlap with optional sentence embeddings and assign a persistent story cluster id.
    """
    def __init__(self, max_history: int = 512) -> None:
        self.history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._embedder = None
        if SETTINGS.max_profile and SETTINGS.use_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu" if SETTINGS.model_device < 0 else None)
            except Exception:
                pass

    def _embed(self, text: str):
        if self._embedder is None:
            return None
        try:
            return self._embedder.encode([clean_text(text)[:5000]], normalize_embeddings=True, show_progress_bar=False)[0]
        except Exception:
            return None

    def analyze(self, entity: str, text: str) -> NoveltyResult:
        sh = _shingles(text)
        emb = self._embed(text)
        best = None
        best_score = 0.0
        best_lex = best_sem = 0.0
        # Prefer same-entity stories but permit cross-entity matching for syndicated macro stories.
        for row in self.history:
            lex = _jaccard(sh, row["shingles"])
            sem = _cos(emb, row["embedding"]) if emb is not None and row["embedding"] is not None else 0.0
            entity_boost = 0.06 if row["entity"] == entity else 0.0
            # Lexical overlap is high precision; semantic catches paraphrases but receives lower weight.
            combined = min(1.0, 0.68 * lex + 0.32 * max(0.0, (sem - 0.45) / 0.55) + entity_boost)
            if combined > best_score:
                best, best_score, best_lex, best_sem = row, combined, lex, sem
        dup = max(0.0, min(1.0, (best_score - 0.20) / 0.70))
        novelty = max(0.0, min(1.0, 1.0 - dup))
        if best is not None and (best_lex >= 0.42 or best_sem >= 0.88 or best_score >= 0.62):
            cluster = best["cluster_id"]
        else:
            canonical = " ".join(_tokens(text)[:40])
            cluster = hashlib.sha1(canonical.encode("utf-8", "ignore")).hexdigest()[:14]
        self.history.append({"entity": entity, "shingles": sh, "embedding": emb, "cluster_id": cluster})
        return NoveltyResult(novelty, dup, cluster, best_lex, best_sem)

    def score(self, entity: str, text: str) -> float:
        return self.analyze(entity, text).novelty
