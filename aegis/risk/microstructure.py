from __future__ import annotations

"""Market-microstructure stress features.

The engine does not pretend a free Yahoo endpoint is Level-2/Level-3 data.  This module is an
adapter for *licensed or user-supplied* order-book/event feeds.  It turns raw snapshots or already
standardised exchange features into a bounded liquidity/microstructure stress score that can be
fused with text and network evidence.
"""

from dataclasses import dataclass, asdict
import math
import numpy as np


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _robust_z(x: float, hist: list[float]) -> float:
    if len(hist) < 8:
        return 0.0
    a = np.asarray(hist, dtype=float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return 0.0 if mad < 1e-12 else 0.6745 * (float(x) - med) / mad


@dataclass
class MicrostructureStress:
    spread_z: float = 0.0
    depth_collapse: float = 0.0
    order_imbalance: float = 0.0
    ofi_z: float = 0.0
    cancel_trade_stress: float = 0.0
    vpin_like: float = 0.0
    kyle_lambda_z: float = 0.0
    score: float = 0.0
    available: bool = False

    def to_dict(self):
        return asdict(self)


class LimitOrderBookStress:
    """Extract robust LOB/liquidity stress from a sequence of order-book snapshots.

    Expected snapshot fields (minimal):
        bid, ask, bid_size, ask_size
    Optional fields:
        signed_volume, trade_volume, cancel_count, trade_count, mid_return

    For deeper books, callers can pre-aggregate the first N levels into bid_size/ask_size or pass
    `bid_sizes`/`ask_sizes`; the analyzer sums them.  The score is an alert feature, not a trading
    signal or a claim of execution alpha.
    """

    @staticmethod
    def _depth(r: dict, side: str) -> float:
        arr = r.get(f"{side}_sizes")
        if isinstance(arr, (list, tuple)) and arr:
            return float(sum(max(0.0, float(x)) for x in arr))
        return max(0.0, float(r.get(f"{side}_size", 0.0) or 0.0))

    def analyze(self, snapshots: list[dict]) -> MicrostructureStress:
        if len(snapshots) < 10:
            return MicrostructureStress()
        spread_bps: list[float] = []
        depth: list[float] = []
        imbalance: list[float] = []
        ofi: list[float] = []
        signed_vol: list[float] = []
        total_vol: list[float] = []
        cancel_ratio: list[float] = []
        rets: list[float] = []

        prev_bd = prev_ad = None
        for r in snapshots:
            bid = float(r.get("bid", 0.0) or 0.0)
            ask = float(r.get("ask", 0.0) or 0.0)
            if bid <= 0 or ask <= bid:
                continue
            mid = 0.5 * (bid + ask)
            spread_bps.append(1e4 * (ask - bid) / mid)
            bd, ad = self._depth(r, "bid"), self._depth(r, "ask")
            dep = bd + ad
            depth.append(dep)
            imbalance.append((bd - ad) / max(dep, 1e-12))
            if prev_bd is not None:
                # Compact OFI proxy: signed change in displayed bid/ask depth.
                ofi.append((bd - prev_bd) - (ad - prev_ad))
            prev_bd, prev_ad = bd, ad
            sv = float(r.get("signed_volume", 0.0) or 0.0)
            tv = float(r.get("trade_volume", abs(sv)) or abs(sv))
            signed_vol.append(sv); total_vol.append(max(tv, abs(sv), 0.0))
            cc = float(r.get("cancel_count", 0.0) or 0.0); tc = float(r.get("trade_count", 0.0) or 0.0)
            cancel_ratio.append(cc / max(1.0, tc))
            if "mid_return" in r:
                rets.append(float(r.get("mid_return", 0.0) or 0.0))

        if len(spread_bps) < 10:
            return MicrostructureStress()
        spread_z = max(0.0, _robust_z(spread_bps[-1], spread_bps[:-1]))
        dep_med = float(np.median(depth[:-1])) if len(depth) > 1 else depth[-1]
        depth_collapse = _clamp(1.0 - depth[-1] / max(dep_med, 1e-12))
        oi = abs(float(imbalance[-1]))
        ofi_z = abs(_robust_z(ofi[-1], ofi[:-1])) if len(ofi) >= 9 else 0.0
        ctr_z = max(0.0, _robust_z(cancel_ratio[-1], cancel_ratio[:-1])) if len(cancel_ratio) >= 9 else 0.0
        cancel_stress = _clamp(ctr_z / 5.0)

        # VPIN-like toxicity proxy over recent flow; intentionally labelled "like" because true
        # VPIN requires volume-bucket construction and trade classification choices.
        sv = np.asarray(signed_vol[-20:], dtype=float)
        tv = np.asarray(total_vol[-20:], dtype=float)
        vpin_like = _clamp(float(np.sum(np.abs(sv))) / max(float(np.sum(tv)), 1e-12))

        # Kyle-lambda proxy if signed flow and returns are provided: price response per unit flow,
        # standardized against rolling single-point lambdas.
        kyle_z = 0.0
        if len(rets) >= 10 and len(signed_vol) >= len(rets):
            q = np.asarray(signed_vol[-len(rets):], dtype=float)
            rr = np.asarray(rets, dtype=float)
            den = float(q @ q)
            lam = float((q @ rr) / den) if den > 1e-12 else 0.0
            local = [abs(rr[i] / q[i]) for i in range(len(rr)-1) if abs(q[i]) > 1e-9]
            kyle_z = max(0.0, _robust_z(abs(lam), local)) if len(local) >= 8 else 0.0

        raw = (
            0.80 * min(1.0, spread_z / 5.0)
            + 1.05 * depth_collapse
            + 0.60 * oi
            + 0.70 * min(1.0, ofi_z / 5.0)
            + 0.65 * cancel_stress
            + 0.70 * vpin_like
            + 0.55 * min(1.0, kyle_z / 5.0)
        )
        score = 1.0 - math.exp(-raw / 2.4)
        return MicrostructureStress(
            spread_z=spread_z, depth_collapse=depth_collapse, order_imbalance=oi, ofi_z=ofi_z,
            cancel_trade_stress=cancel_stress, vpin_like=vpin_like, kyle_lambda_z=kyle_z,
            score=_clamp(score), available=True,
        )

    def from_features(self, f: dict) -> MicrostructureStress:
        """Use externally-calculated exchange features without reconstructing snapshots."""
        if not f:
            return MicrostructureStress()
        spread_z = max(0.0, float(f.get("spread_z", 0.0)))
        depth = _clamp(float(f.get("depth_collapse", 0.0)))
        imb = _clamp(abs(float(f.get("order_imbalance", 0.0))))
        ofi = max(0.0, abs(float(f.get("ofi_z", 0.0))))
        ctr = _clamp(float(f.get("cancel_trade_stress", 0.0)))
        vpin = _clamp(float(f.get("vpin_like", 0.0)))
        kyle = max(0.0, float(f.get("kyle_lambda_z", 0.0)))
        raw = 0.80*min(1,spread_z/5)+1.05*depth+0.60*imb+0.70*min(1,ofi/5)+0.65*ctr+0.70*vpin+0.55*min(1,kyle/5)
        return MicrostructureStress(spread_z,depth,imb,ofi,ctr,vpin,kyle,_clamp(1-math.exp(-raw/2.4)),True)
