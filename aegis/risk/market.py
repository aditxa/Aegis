from __future__ import annotations

from datetime import datetime, timezone
import math
import numpy as np
import requests

from aegis.config import SETTINGS
from aegis.schemas import MarketStress
from aegis.risk.tail import robust_z, bns_jump_score, garch_vol_ratio, evt_tail_score
from aegis.risk.regime import RegimeFilter


class YahooMarketProvider:
    """Credential-free, event-time-aligned market context.

    This is suitable for a hackathon prototype, not exchange-grade HFT. Crucially, features are
    computed only from bars timestamped at or before the event timestamp, preventing the historical
    look-ahead bug in the previous implementation.
    """
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SETTINGS.user_agent})
        self.raw_cache: dict[tuple[str,str,str], dict] = {}
        self.regimes: dict[str, RegimeFilter] = {}

    def _fetch(self, ticker: str, interval: str, range_: str) -> dict:
        key = (ticker, interval, range_)
        if key in self.raw_cache:
            return self.raw_cache[key]
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": range_, "interval": interval, "includePrePost": "true", "events": "div,splits"}
        r = self.session.get(url, params=params, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        result = r.json()["chart"]["result"][0]
        self.raw_cache[key] = result
        return result

    @staticmethod
    def _bars(result: dict, ts: datetime) -> list[dict]:
        timestamps = result.get("timestamp") or []
        q = (result.get("indicators",{}).get("quote") or [{}])[0]
        closes = q.get("close") or []
        opens = q.get("open") or []
        highs = q.get("high") or []
        lows = q.get("low") or []
        vols = q.get("volume") or []
        cutoff = int(ts.astimezone(timezone.utc).timestamp())
        out = []
        for i, t in enumerate(timestamps):
            if t > cutoff:
                break
            c = closes[i] if i < len(closes) else None
            if c is None or c <= 0:
                continue
            out.append({
                "ts": int(t), "close": float(c),
                "open": float(opens[i]) if i < len(opens) and opens[i] is not None else float(c),
                "high": float(highs[i]) if i < len(highs) and highs[i] is not None else float(c),
                "low": float(lows[i]) if i < len(lows) and lows[i] is not None else float(c),
                "volume": float(vols[i]) if i < len(vols) and vols[i] is not None else 0.0,
            })
        return out

    @staticmethod
    def _returns(bars: list[dict]) -> list[float]:
        c = [x["close"] for x in bars]
        return [math.log(b/a) for a,b in zip(c,c[1:]) if a > 0 and b > 0]

    @staticmethod
    def _beta_residual_z(asset_rets: list[float], bench_rets: list[float]) -> float:
        n = min(len(asset_rets), len(bench_rets))
        if n < 15:
            return 0.0
        a = np.asarray(asset_rets[-n:], dtype=float)
        b = np.asarray(bench_rets[-n:], dtype=float)
        train_a, train_b = a[:-1], b[:-1]
        vb = float(np.var(train_b))
        beta = float(np.cov(train_a, train_b, ddof=0)[0,1] / vb) if vb > 1e-12 else 0.0
        alpha = float(np.mean(train_a) - beta*np.mean(train_b))
        residuals = train_a - (alpha + beta*train_b)
        latest = float(a[-1] - (alpha + beta*b[-1]))
        return robust_z(latest, residuals.tolist())

    def get(self, ticker: str, event_time: datetime | None = None) -> MarketStress:
        event_time = event_time or datetime.now(timezone.utc)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        try:
            interval = SETTINGS.market_intraday_interval
            range_ = SETTINGS.market_intraday_range
            asset = self._bars(self._fetch(ticker, interval, range_), event_time)
            bench = self._bars(self._fetch(SETTINGS.market_benchmark, interval, range_), event_time)
            if len(asset) < 20:
                # Older timestamps may not be covered by intraday history. Daily fallback remains time aligned.
                asset = self._bars(self._fetch(ticker, "1d", "1y"), event_time)
                bench = self._bars(self._fetch(SETTINGS.market_benchmark, "1d", "1y"), event_time)
            if len(asset) < 12:
                raise ValueError("insufficient pre-event market history")
            ar = self._returns(asset); br = self._returns(bench)
            latest_ret = ar[-1]
            rz = robust_z(latest_ret, ar[:-1])
            arz = self._beta_residual_z(ar, br)
            volumes = [x["volume"] for x in asset]
            vz = robust_z(volumes[-1], volumes[:-1]) if len(volumes) > 6 else 0.0
            vr = garch_vol_ratio(ar)
            closes = [x["close"] for x in asset]
            peak = max(closes[-min(120,len(closes)):])
            dd = min(0.0, closes[-1]/peak - 1.0)
            jump = bns_jump_score(ar)
            dollar_vol = [max(1.0, b["close"] * max(0.0,b["volume"])) for b in asset[1:]]
            amihud = [abs(r)/dv for r,dv in zip(ar,dollar_vol)]
            ill_z = robust_z(amihud[-1], amihud[:-1]) if len(amihud) > 6 else 0.0
            ill = max(0.0, min(1.0, ill_z/5.0))
            last = asset[-1]
            gap = abs(last["open"] / asset[-2]["close"] - 1.0) if len(asset)>1 and asset[-2]["close"] else 0.0
            gap_hist = [abs(asset[i]["open"]/asset[i-1]["close"]-1.0) for i in range(1,len(asset)-1) if asset[i-1]["close"]]
            gap_score = max(0.0, min(1.0, robust_z(gap, gap_hist)/5.0)) if gap_hist else 0.0
            tail = evt_tail_score(ar)
            regime = self.regimes.setdefault(ticker, RegimeFilter()).update(vr, min(5.0,abs(arz)), 0.0)
            raw = (
                0.75*max(0.0,-arz/4.0) + 0.25*abs(rz)/5.0 + 0.35*max(0.0,vz/5.0)
                + 0.65*max(0.0,vr-1.0) + 0.60*jump + 0.45*ill + 0.40*gap_score
                + 0.65*tail + 0.40*regime + min(0.8,abs(dd)*6.0)
            )
            stress = max(0.0, min(1.0, 1.0-math.exp(-raw)))
            last_ts = datetime.fromtimestamp(asset[-1]["ts"], tz=timezone.utc)
            aligned = last_ts <= event_time.astimezone(timezone.utc)
            return MarketStress(
                return_z=rz, abnormal_return_z=arz, volume_z=vz, vol_ratio=vr, drawdown=dd,
                jump_score=jump, illiquidity_score=ill, gap_score=gap_score, tail_score=tail,
                regime_score=regime, stress_score=stress, available=True, timestamp_aligned=aligned,
            )
        except Exception:
            return MarketStress()
