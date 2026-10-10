from __future__ import annotations

import math
import re
from dataclasses import dataclass

from aegis.risk.online import clamp, sigmoid

UNCERTAINTY_TERMS = {
    "rumor","rumour","reportedly","unconfirmed","may","might","could","possible","alleged","allegedly",
    "sources","speculation","uncertain","warning","risk","preliminary","considering","exploring",
}


def uncertainty_score(text: str) -> float:
    toks = re.findall(r"[a-z]+", (text or "").lower())
    hits = sum(t in UNCERTAINTY_TERMS for t in toks)
    return clamp(1.0-math.exp(-hits/3.0))


@dataclass
class FusionInputs:
    sentiment: float
    sentiment_conf: float
    sentiment_disagreement: float
    event_prior: float
    event_conf: float
    event_direction: float
    novelty: float
    duplicate_probability: float
    corroboration: float
    contradiction: float
    source_quality: float
    source_independence: float
    burst: float
    velocity: float
    change: float
    hawkes: float
    graph_contagion: float
    systemic: float
    market_stress: float
    microstructure_stress: float
    tail_risk: float
    regime: float
    manipulation: float
    uncertainty: float


@dataclass
class FusionResult:
    # Impact = conditional severity if the extracted event is real.
    impact_score: float
    # Early warning = evidence-aware probability-like prioritisation signal (not calibrated unless a learned head is fitted).
    early_warning_score: float
    # Priority combines severity, evidence and early warning; use this for alert routing.
    risk_priority_score: float
    confidence: float
    contributions: dict[str,float]


def fuse_risk(x: FusionInputs) -> FusionResult:
    """Dual-score nonlinear fusion: conditional severity + evidence-aware early warning.

    The previous implementation conflated *severity* with *credibility*: a single credible report of
    a catastrophic event could get an artificially low impact score merely because it lacked a
    second source.  Here impact answers "how severe, if true?" while confidence/early-warning answer
    "how much should we act on this evidence now?".  That separation is crucial for rumors,
    corrections, breaking news and low-latency monitoring.
    """
    adverse = clamp((1.0-x.event_direction)/2.0)  # harmful=-1 ->1, beneficial=+1 ->0
    sentiment_risk = clamp(max(0.0,-x.sentiment) * (0.55+0.45*x.sentiment_conf))
    semantic = clamp(
        0.62*x.event_prior*x.event_conf*(0.50+0.50*adverse)
        + 0.25*sentiment_risk
        + 0.13*abs(x.sentiment)*x.sentiment_conf
    )
    evidence = clamp(
        (0.30+0.70*x.source_quality)
        * (0.45+0.55*x.source_independence)
        * (1.0-0.62*x.manipulation)
        * (1.0-0.50*x.contradiction)
    )
    information = clamp(
        0.18*x.novelty + 0.24*x.burst + 0.19*x.velocity + 0.19*x.change + 0.20*x.hawkes
    )
    market = clamp(0.40*x.market_stress + 0.22*x.microstructure_stress + 0.23*x.tail_risk + 0.15*x.regime)
    network = clamp(0.55*x.graph_contagion + 0.45*x.systemic)
    uncertainty_watch = clamp(0.55*x.uncertainty + 0.45*x.sentiment_disagreement)

    echo_penalty = x.duplicate_probability * (1.0-x.corroboration) * 0.55
    sem_verified = semantic * evidence * (1.0-echo_penalty)

    # Conditional severity should not collapse just because the event is breaking and has one source.
    sev_sm = semantic * market
    sev_nr = network * max(x.regime, x.market_stress, x.microstructure_stress)
    sev_tail = x.tail_risk * max(x.hawkes,x.graph_contagion)
    z_severity = (
        -1.90
        + 3.25*semantic
        + 1.05*information
        + 1.55*market
        + 0.85*network
        + 1.05*sev_sm
        + 0.65*sev_nr
        + 0.50*sev_tail
    )
    severity_p = sigmoid(z_severity)
    impact = 1.0 + 9.0*severity_p

    # Alert/early-warning channel is much stricter about provenance and manipulation.
    verified_sm = sem_verified * market
    verified_nr = network * max(x.regime, x.market_stress, x.microstructure_stress)
    novel_burst = x.burst * x.novelty * x.source_quality
    z_early = (
        -2.65
        + 3.00*sem_verified
        + 1.55*information
        + 1.05*network
        + 1.20*market
        + 1.10*verified_sm
        + 0.80*verified_nr
        + 0.95*novel_burst
        + 0.25*uncertainty_watch
    )
    early = sigmoid(z_early)

    confidence = clamp(
        0.34*x.event_conf + 0.22*x.sentiment_conf + 0.20*x.source_quality
        + 0.14*x.source_independence + 0.10*(1.0-x.sentiment_disagreement)
    ) * (1.0-0.48*x.contradiction) * (1.0-0.42*x.manipulation)
    confidence = clamp(confidence)

    # Operational priority: severe-but-unconfirmed events stay visible without becoming an automatic
    # panic alert; corroboration/market confirmation can quickly lift them.
    priority_unit = clamp(
        0.55*severity_p*(0.60+0.40*confidence)
        + 0.35*early
        + 0.10*max(market, network*max(x.regime,x.market_stress,x.microstructure_stress))
    )
    priority = 1.0 + 9.0*priority_unit

    contributions = {
        "semantic_severity": round(semantic,4), "verified_semantic": round(sem_verified,4),
        "information_dynamics": round(information,4), "market_microstructure": round(market,4),
        "network_contagion": round(network,4), "semantic_x_market": round(sev_sm,4),
        "network_x_regime": round(sev_nr,4), "novel_burst": round(novel_burst,4),
        "tail_x_contagion": round(sev_tail,4), "echo_penalty": round(echo_penalty,4),
        "evidence_gate": round(evidence,4), "severity_latent": round(severity_p,4),
    }
    return FusionResult(round(impact,2), round(early,4), round(priority,2), round(confidence,4), contributions)


def impact_score(x: FusionInputs) -> float:
    return fuse_risk(x).impact_score


def alert_level(score: float) -> str:
    if score >= 8.5: return "CRITICAL"
    if score >= 7.0: return "HIGH"
    if score >= 5.0: return "ELEVATED"
    if score >= 3.0: return "WATCH"
    return "LOW"
