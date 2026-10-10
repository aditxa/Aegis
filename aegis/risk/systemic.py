from __future__ import annotations

from collections import deque, defaultdict
from datetime import datetime, timedelta
import math


class SystemicResonance:
    """CISS-inspired breadth/concentration/resonance aggregator over entity and sector stress."""
    def __init__(self, window_minutes: int = 90) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.rows: deque[tuple[datetime,str,str,float]] = deque()

    def update(self, ts: datetime, ticker: str, risk_intensity: float, sector: str = "") -> float:
        self.rows.append((ts,ticker,sector,max(0.0,min(1.0,risk_intensity))))
        cutoff = ts-self.window
        while self.rows and self.rows[0][0] < cutoff:
            self.rows.popleft()
        latest = {}; sector_max = defaultdict(float)
        for _,t,s,r in self.rows:
            latest[t] = max(latest.get(t,0.0),r)
            if s: sector_max[s] = max(sector_max[s],r)
        if not latest: return 0.0
        vals = list(latest.values())
        mean = sum(vals)/len(vals)
        breadth = sum(v>=0.55 for v in vals)/len(vals)
        sector_breadth = sum(v>=0.50 for v in sector_max.values())/max(1,len(sector_max))
        hhi = sum(v*v for v in vals)/max(1e-9,sum(vals)**2)
        diversification = max(0.0,1.0-hhi)
        # Wide, multi-sector stress receives greater weight than one isolated name.
        raw = mean*(0.45+0.75*breadth+0.45*sector_breadth+0.35*diversification)
        return max(0.0,min(1.0,raw))
