from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
import math


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class CorroborationTracker:
    """Direction-aware corroboration that discounts syndicated/duplicate story chains."""
    def __init__(self, window_minutes: int = 120) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.events = defaultdict(deque)

    def score(
        self, entity: str, event: str, source_domain: str, ts: datetime, direction: int = 0,
        cluster_id: str = "", duplicate_probability: float = 0.0,
    ) -> tuple[float, float, float]:
        key = (entity, event)
        q = self.events[key]
        cutoff = ts - self.window
        while q and q[0][0] < cutoff:
            q.popleft()
        domain = source_domain or "unknown"
        # Highly duplicated stories across domains count as one wire/story-chain, not independent witnesses.
        independence_key = f"story:{cluster_id}" if cluster_id and duplicate_probability >= 0.72 else f"domain:{domain}"
        q.append((ts, domain, independence_key, int(max(-1, min(1, direction)))))

        latest_by_group = {}
        domains = set()
        for _, d, g, sign in q:
            latest_by_group[g] = sign
            domains.add(d)
        if direction == 0:
            consistent = sum(sign == 0 for sign in latest_by_group.values())
            opposite = sum(sign != 0 for sign in latest_by_group.values())
        else:
            consistent = sum(sign == direction for sign in latest_by_group.values())
            opposite = sum(sign == -direction for sign in latest_by_group.values())
        corroboration = clamp(1.0 - math.exp(-max(0, consistent - 1) / 1.6))
        contradiction = clamp(1.0 - math.exp(-opposite / 1.2))
        corroboration *= (1.0 - 0.75 * contradiction)
        # 1 group -> 0.35; 3+ truly independent groups -> high.
        independence = clamp(1.0 - math.exp(-len(latest_by_group) / 2.2))
        return clamp(corroboration), contradiction, independence


class BurstTracker:
    """Bayesian-smoothed event-count burst detector using 15m versus 3h baselines."""
    def __init__(self, max_hours: float = 6.0) -> None:
        self.window = timedelta(hours=max_hours)
        self.events = defaultdict(deque)

    def score(self, key: str, ts: datetime) -> float:
        q = self.events[key]
        q.append(ts)
        cutoff = ts - self.window
        while q and q[0] < cutoff:
            q.popleft()
        recent_cut = ts - timedelta(minutes=15)
        base_cut = ts - timedelta(hours=3)
        recent = sum(x >= recent_cut for x in q)
        baseline = [x for x in q if base_cut <= x < recent_cut]
        # A single first-seen story is not a burst. This removes the old cold-start artifact
        # where every issuer's first observation received a ~0.49 burst score.
        if len(q) < 2 or recent < 2:
            return 0.0
        # Gamma-Poisson smoothing: expected 15m count from previous 165 minutes.
        expected = 0.25 * (len(baseline) + 1.5) / (2.75 + 0.5)
        z = (recent - expected) / math.sqrt(expected + 0.75)
        return clamp(sigmoid(1.15 * (z - 1.0)))


class RiskVelocityTracker:
    """Time-aware acceleration of semantic risk intensity."""
    def __init__(self, maxlen: int = 48) -> None:
        self.hist = defaultdict(lambda: deque(maxlen=maxlen))

    def score(self, entity: str, risk_value: float, ts: datetime | None = None) -> float:
        if ts is None:
            ts = datetime.utcnow()
        h = self.hist[entity]
        h.append((ts, float(risk_value)))
        if len(h) < 4:
            return 0.0
        now = h[-1][0]
        recent = [(t, v) for t, v in h if (now - t).total_seconds() <= 3600]
        older = [(t, v) for t, v in h if 3600 < (now - t).total_seconds() <= 6 * 3600]
        if not older:
            vals = [v for _, v in h]
            older = list(h)[:max(1, len(vals)-3)]
        r = sum(v for _, v in recent) / max(1, len(recent))
        o = sum(v for _, v in older) / max(1, len(older))
        acceleration = r - o
        return clamp(2.0 * max(0.0, sigmoid(4.0 * acceleration) - 0.5))


class PageHinkleyBank:
    """Page-Hinkley retained as a complementary fast detector; reset after strong change."""
    def __init__(self, delta: float = 0.015, threshold: float = 0.45) -> None:
        self.delta = delta
        self.threshold = threshold
        self.states = defaultdict(lambda: {"n": 0, "mean": 0.0, "cum": 0.0, "min": 0.0})

    def update(self, key: str, x: float) -> float:
        st = self.states[key]
        st["n"] += 1
        st["mean"] += (x - st["mean"]) / st["n"]
        st["cum"] += x - st["mean"] - self.delta
        st["min"] = min(st["min"], st["cum"])
        ph = st["cum"] - st["min"]
        score = clamp(ph / self.threshold)
        if score >= 1.0 and st["n"] > 8:
            self.states[key] = {"n": 1, "mean": x, "cum": 0.0, "min": 0.0}
        return score
