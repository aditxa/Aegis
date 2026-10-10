from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TextItem:
    source: str
    title: str
    text: str
    published_at: datetime
    url: str = ""
    author: str = ""
    source_domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    item_id: str = ""

    def __post_init__(self) -> None:
        if self.published_at.tzinfo is None:
            self.published_at = self.published_at.replace(tzinfo=timezone.utc)

    @property
    def full_text(self) -> str:
        if self.title and self.text:
            return f"{self.title}. {self.text}"
        return self.title or self.text


@dataclass
class EntityMatch:
    ticker: str
    company: str
    confidence: float
    matched_alias: str
    sector: str = ""
    role: str = "mentioned"


@dataclass
class MarketStress:
    return_z: float = 0.0
    abnormal_return_z: float = 0.0
    volume_z: float = 0.0
    vol_ratio: float = 1.0
    drawdown: float = 0.0
    jump_score: float = 0.0
    illiquidity_score: float = 0.0
    gap_score: float = 0.0
    tail_score: float = 0.0
    regime_score: float = 0.0
    stress_score: float = 0.0
    available: bool = False
    timestamp_aligned: bool = False


@dataclass
class RiskSignal:
    item_id: str
    timestamp: str
    source: str
    source_domain: str
    url: str
    ticker: str
    company: str
    sentiment_score: float
    sentiment_label: str
    sentiment_confidence: float
    sentiment_disagreement: float
    event_classification: str
    secondary_events: list[dict[str, Any]]
    event_confidence: float
    event_direction: float
    impact_score: float
    early_warning_score: float
    risk_priority_score: float
    alert_level: str
    novelty_score: float
    story_cluster_id: str
    duplicate_probability: float
    corroboration_score: float
    contradiction_score: float
    source_quality: float
    source_independence: float
    burst_score: float
    risk_velocity: float
    change_score: float
    hawkes_score: float
    graph_contagion_score: float
    systemic_resonance: float
    market_stress: float
    microstructure_stress: float
    tail_risk_score: float
    regime_score: float
    manipulation_score: float
    uncertainty_score: float
    model_confidence: float
    evidence: list[str] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
