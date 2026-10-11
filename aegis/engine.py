from __future__ import annotations

from collections import defaultdict
from datetime import timezone
from pathlib import Path
import hashlib

from aegis.config import SETTINGS
from aegis.schemas import RiskSignal, TextItem, EntityMatch
from aegis.nlp import EntityLinker, EventClassifier, NoveltyTracker, SentimentAnalyzer, ImpactStanceAnalyzer
from aegis.risk import (
    FusionInputs, fuse_risk, YahooMarketProvider, CorroborationTracker, BurstTracker,
    RiskVelocityTracker, PageHinkleyBank, BOCPDBank, MarkedHawkesContagion, DynamicRiskGraph,
    SystemicResonance, LimitOrderBookStress, alert_level, domain_from_url, source_quality, uncertainty_score, manipulation_score,
)
from aegis.storage import SignalStore
from aegis.risk.impact_model import TailImpactEnsemble


class AegisRiskEngine:
    """AegisSignal-X unified multimodal early-risk engine.

    The engine is intentionally a *mixture of independent evidence channels*: financial-language
    models, story-chain provenance, online temporal detectors, event-time market microstructure,
    tail-risk statistics, self-exciting contagion and graph propagation. This avoids asking any one
    classifier to solve a problem it cannot solve reliably.
    """
    def __init__(self, companies_csv: str | Path, sqlite_path: str | Path | None = None, use_market_context: bool = False, relations_csv: str | Path | None = None) -> None:
        self.companies_csv = Path(companies_csv)
        self.linker = EntityLinker(companies_csv)
        self.sentiment = SentimentAnalyzer()
        self.events = EventClassifier()
        self.stance = ImpactStanceAnalyzer()
        self.novelty = NoveltyTracker()
        self.corroboration = CorroborationTracker()
        self.burst = BurstTracker()
        self.velocity = RiskVelocityTracker()
        self.change_fast = PageHinkleyBank()
        self.change_bayes = BOCPDBank()
        self.hawkes = MarkedHawkesContagion()
        self.graph = DynamicRiskGraph(companies_csv, relations_csv)
        self.systemic = SystemicResonance()
        self.microstructure = LimitOrderBookStress()
        self.market = YahooMarketProvider() if use_market_context else None
        self.store = SignalStore(sqlite_path) if sqlite_path else None
        self.learned_impact = TailImpactEnsemble(SETTINGS.impact_model_path or None)
        self.recent: dict[str, list[RiskSignal]] = defaultdict(list)

    def analyze_item(self, item: TextItem) -> list[RiskSignal]:
        text = item.full_text
        entities = self.linker.link(text)
        if not entities:
            entities = [EntityMatch(ticker="MARKET", company="Broad Market", confidence=0.70, matched_alias="", sector="Broad Market", role="macro")]
        self.graph.update_comention([e.ticker for e in entities if e.ticker != "MARKET"])

        out: list[RiskSignal] = []
        for ent in entities:
            aliases = [ent.ticker, ent.company, ent.matched_alias]
            local_text = self.sentiment.entity_context(text, aliases) if ent.ticker != "MARKET" else text
            event = self.events.classify(local_text)
            sent = self.sentiment.analyze(local_text)
            stance = self.stance.analyze(local_text, ent.company)
            novelty = self.novelty.analyze(ent.ticker, text)

            domain = item.source_domain or domain_from_url(item.url)
            srcq = source_quality(item.source, domain)
            if item.metadata.get("synthetic_fixture") is True and "source_quality" in item.metadata:
                try:
                    srcq = max(0.0, min(1.0, float(item.metadata["source_quality"])))
                except (TypeError, ValueError):
                    pass
            direction_scalar = stance.direction if abs(stance.direction) >= 0.12 else sent.score
            direction = 1 if direction_scalar > 0.12 else -1 if direction_scalar < -0.12 else 0
            corroboration, contradiction, independence = self.corroboration.score(
                ent.ticker, event.label, domain, item.published_at, direction,
                cluster_id=novelty.cluster_id, duplicate_probability=novelty.duplicate_probability,
            )
            burst = self.burst.score(f"{ent.ticker}|{event.label}", item.published_at)
            uncertainty = uncertainty_score(text)

            adverse = max(0.0, min(1.0, (1.0 - stance.direction) / 2.0))
            semantic_pre = max(0.0, min(1.0,
                0.58 * event.prior_severity * event.confidence * (0.55 + 0.45 * adverse)
                + 0.24 * max(0.0, -sent.score)
                + 0.10 * burst
                + 0.08 * uncertainty
            ))
            velocity = self.velocity.score(ent.ticker, semantic_pre, item.published_at)
            ph = self.change_fast.update(ent.ticker, semantic_pre)
            bocpd = self.change_bayes.update(ent.ticker, semantic_pre)
            change = max(ph, min(1.0, max(0.0, (bocpd - 0.15) / 0.65)))
            propagation_mark = semantic_pre * (0.25 + 0.75 * adverse)
            hawkes = self.hawkes.update(item.published_at, ent.ticker, ent.sector, propagation_mark)
            graph_contagion = self.graph.update(ent.ticker, propagation_mark) if ent.ticker != "MARKET" else 0.0
            systemic = self.systemic.update(item.published_at, ent.ticker, propagation_mark, ent.sector)

            market = self.market.get(ent.ticker, item.published_at) if self.market and ent.ticker != "MARKET" else None
            mstress = market.stress_score if market else 0.0
            micro = self.microstructure.from_features(item.metadata.get("microstructure", {})) if isinstance(item.metadata, dict) else None
            microstress = micro.score if micro and micro.available else 0.0
            tail = market.tail_score if market else 0.0
            regime = market.regime_score if market else 0.0
            manip = manipulation_score(text, srcq, contradiction, novelty.duplicate_probability, uncertainty)

            fi = FusionInputs(
                sentiment=sent.score, sentiment_conf=sent.confidence, sentiment_disagreement=sent.disagreement,
                event_prior=event.prior_severity, event_conf=event.confidence, event_direction=stance.direction,
                novelty=novelty.novelty, duplicate_probability=novelty.duplicate_probability,
                corroboration=corroboration, contradiction=contradiction, source_quality=srcq,
                source_independence=independence, burst=burst, velocity=velocity, change=change,
                hawkes=hawkes, graph_contagion=graph_contagion, systemic=systemic,
                market_stress=mstress, microstructure_stress=microstress, tail_risk=tail, regime=regime, manipulation=manip, uncertainty=uncertainty,
            )
            fused = fuse_risk(fi)
            impact = fused.impact_score
            early = fused.early_warning_score
            priority = fused.risk_priority_score
            learned_features = {k: float(v) for k, v in {
                "sentiment": sent.score, "sentiment_conf": sent.confidence, "sentiment_disagreement": sent.disagreement,
                "event_prior": event.prior_severity, "event_conf": event.confidence, "event_direction": stance.direction,
                "novelty": novelty.novelty, "duplicate_probability": novelty.duplicate_probability,
                "corroboration": corroboration, "contradiction": contradiction, "source_quality": srcq,
                "source_independence": independence, "burst": burst, "velocity": velocity, "change": change,
                "hawkes": hawkes, "graph_contagion": graph_contagion, "systemic": systemic,
                "market_stress": mstress, "microstructure_stress": microstress, "tail_risk": tail, "regime": regime, "manipulation": manip, "uncertainty": uncertainty,
            }.items()}
            learned = self.learned_impact.predict(learned_features)
            if learned is not None:
                b = max(0.0, min(0.75, SETTINGS.impact_tail_blend))
                learned_sev = 0.55 * learned["median"] + 0.45 * learned["tail90"]
                impact = round(max(1.0, min(10.0, (1.0-b)*impact + b*learned_sev)), 2)
                if "p_high" in learned:
                    early = round(max(0.0, min(1.0, 0.65*early + 0.35*learned["p_high"])), 4)
                    priority = round(max(priority, 1.0 + 9.0*(0.55*((impact-1.0)/9.0)*(0.60+0.40*fused.confidence)+0.35*early)), 2)

            evidence = [
                f"entity_role={ent.role}; sentiment={sent.score:+.3f} conf={sent.confidence:.2f} disagree={sent.disagreement:.2f} ({sent.model})",
                f"event={event.label} conf={event.confidence:.2f}; stance={stance.direction:+.2f} ({stance.model})",
                f"story={novelty.cluster_id}; novelty={novelty.novelty:.2f}; duplicate_p={novelty.duplicate_probability:.2f}",
                f"corroboration={corroboration:.2f}; contradiction={contradiction:.2f}; source_quality={srcq:.2f}; independence={independence:.2f}",
                f"burst={burst:.2f}; velocity={velocity:.2f}; change={change:.2f}; hawkes={hawkes:.2f}",
                f"graph_contagion={graph_contagion:.2f}; systemic={systemic:.2f}; manipulation={manip:.2f}; microstructure={microstress:.2f}",
            ]
            if market and market.available:
                evidence.append(
                    f"market={mstress:.2f}; abnormal_return_z={market.abnormal_return_z:+.2f}; volume_z={market.volume_z:+.2f}; "
                    f"vol_ratio={market.vol_ratio:.2f}; jump={market.jump_score:.2f}; illiquidity={market.illiquidity_score:.2f}; "
                    f"tail={market.tail_score:.2f}; regime={market.regime_score:.2f}; time_aligned={market.timestamp_aligned}"
                )
            else:
                evidence.append("market_context=disabled/unavailable; no market-confirmation contribution used")

            item_id = item.item_id or hashlib.sha1(f"{item.source}|{item.url}|{item.title}|{item.published_at.isoformat()}".encode("utf-8","ignore")).hexdigest()
            sig = RiskSignal(
                item_id=item_id, timestamp=item.published_at.astimezone(timezone.utc).isoformat(), source=item.source,
                source_domain=domain, url=item.url, ticker=ent.ticker, company=ent.company,
                sentiment_score=round(sent.score,4), sentiment_label=sent.label, sentiment_confidence=round(sent.confidence,4),
                sentiment_disagreement=round(sent.disagreement,4), event_classification=event.label,
                secondary_events=[{"label":c.label,"confidence":round(c.confidence,4),"prior_severity":c.prior_severity} for c in event.candidates[1:]],
                event_confidence=round(event.confidence,4), event_direction=round(stance.direction,4), impact_score=impact,
                early_warning_score=early, risk_priority_score=priority, alert_level=alert_level(priority), novelty_score=round(novelty.novelty,4),
                story_cluster_id=novelty.cluster_id, duplicate_probability=round(novelty.duplicate_probability,4),
                corroboration_score=round(corroboration,4), contradiction_score=round(contradiction,4), source_quality=round(srcq,4),
                source_independence=round(independence,4), burst_score=round(burst,4), risk_velocity=round(velocity,4),
                change_score=round(change,4), hawkes_score=round(hawkes,4), graph_contagion_score=round(graph_contagion,4),
                systemic_resonance=round(systemic,4), market_stress=round(mstress,4), microstructure_stress=round(microstress,4), tail_risk_score=round(tail,4),
                regime_score=round(regime,4), manipulation_score=round(manip,4), uncertainty_score=round(uncertainty,4),
                model_confidence=fused.confidence, evidence=evidence, contributions=fused.contributions,
            )
            out.append(sig)
            self.recent[ent.ticker].append(sig)
            self.recent[ent.ticker] = self.recent[ent.ticker][-200:]
            if self.store:
                self.store.put(sig)
        return out

    def analyze(self, items: list[TextItem]) -> list[RiskSignal]:
        signals: list[RiskSignal] = []
        for item in sorted(items, key=lambda x: x.published_at):
            signals.extend(self.analyze_item(item))
        return signals

    def aggregate_for_rebalancer(self, tickers: list[str]) -> dict[str, dict[str,float]]:
        out = {}
        for ticker in tickers:
            rows = self.recent.get(ticker.upper(), [])[-24:]
            if not rows:
                out[ticker.upper()] = {"sentiment":0.0,"impact":1.0,"confidence":0.35,"early_warning":0.0,"risk":0.0}
                continue
            weights = [0.88 ** (len(rows)-1-i) for i in range(len(rows))]
            sw = sum(weights)
            avg = lambda attr: sum(w*float(getattr(r,attr)) for w,r in zip(weights,rows))/sw
            sentiment = avg("sentiment_score")
            impact = avg("impact_score")
            conf = avg("model_confidence")
            early = avg("early_warning_score")
            out[ticker.upper()] = {
                "sentiment":sentiment, "impact":impact, "confidence":conf, "early_warning":early,
                "risk":max(0.0,min(1.0,(impact-1.0)/9.0)),
            }
        return out
