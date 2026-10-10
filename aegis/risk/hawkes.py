from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
import math

from aegis.config import SETTINGS


class MarkedHawkesContagion:
    """Online marked multivariate Hawkes proxy for clustered/cascading risk events.

    Each event excites its own issuer most strongly, same-sector issuers moderately, and the broad
    market weakly. The exponential kernel is exactly Hawkes-shaped; the interaction matrix is an
    interpretable online proxy rather than a claim of MLE-fitted exchange-level Hawkes parameters.
    """
    def __init__(self, half_life_minutes: float | None = None, horizon_hours: float = 6.0) -> None:
        hl = half_life_minutes or SETTINGS.hawkes_half_life_minutes
        self.beta = math.log(2.0) / max(1.0, hl)
        self.horizon = timedelta(hours=horizon_hours)
        self.events: deque[tuple[datetime, str, str, float]] = deque()

    def update(self, ts: datetime, ticker: str, sector: str, mark: float) -> float:
        cutoff = ts - self.horizon
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()
        intensity = 0.0
        for t0, src, src_sector, m in self.events:
            dtm = max(0.0, (ts - t0).total_seconds() / 60.0)
            kernel = math.exp(-self.beta * dtm)
            coupling = 0.75 if src == ticker else 0.38 if sector and src_sector == sector else 0.10
            intensity += coupling * m * kernel
        score = 1.0 - math.exp(-intensity / 1.8)
        self.events.append((ts, ticker, sector, max(0.0, min(1.0, mark))))
        return max(0.0, min(1.0, score))
