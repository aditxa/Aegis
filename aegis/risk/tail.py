from __future__ import annotations

import math
import numpy as np


def robust_z(x: float, hist: list[float]) -> float:
    if len(hist) < 5:
        return 0.0
    a = np.asarray(hist, dtype=float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return 0.0 if mad < 1e-12 else 0.6745 * (x - med) / mad


def bns_jump_score(returns: list[float], window: int = 24) -> float:
    r = np.asarray(returns[-window:], dtype=float)
    if len(r) < 6:
        return 0.0
    rv = float(np.sum(r * r))
    bpv = float((math.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    if rv <= 1e-12:
        return 0.0
    jump_share = max(0.0, (rv - bpv) / rv)
    return float(min(1.0, 1.8 * jump_share))


def garch_vol_ratio(returns: list[float], alpha: float = 0.08, beta: float = 0.90) -> float:
    if len(returns) < 8:
        return 1.0
    r = np.asarray(returns, dtype=float)
    var0 = float(np.var(r[:max(5, len(r)//3)])) + 1e-10
    omega = max(1e-12, var0 * (1.0 - alpha - beta))
    v = var0
    hist = []
    for x in r:
        v = omega + alpha * x * x + beta * v
        hist.append(v)
    long = float(np.median(hist[:-1])) if len(hist) > 2 else var0
    return float(math.sqrt(hist[-1] / max(long, 1e-12)))


def evt_tail_score(returns: list[float], current_return: float | None = None, q: float = 0.90) -> float:
    """POT/GPD downside tail anomaly score with empirical fallback."""
    if len(returns) < 40:
        if not returns:
            return 0.0
        losses = np.maximum(0.0, -np.asarray(returns, dtype=float))
        x = max(0.0, -(current_return if current_return is not None else returns[-1]))
        return float(np.mean(losses <= x) ** 4) if x > 0 else 0.0
    losses = np.maximum(0.0, -np.asarray(returns[:-1] if current_return is None else returns, dtype=float))
    x = max(0.0, -(returns[-1] if current_return is None else current_return))
    u = float(np.quantile(losses, q))
    exc = losses[losses > u] - u
    if x <= u or len(exc) < 8:
        # map empirical percentile above median to [0, .45]
        pct = float(np.mean(losses <= x))
        return max(0.0, min(0.45, (pct - 0.5) * 0.9))
    try:
        from scipy.stats import genpareto
        shape, _, scale = genpareto.fit(exc, floc=0.0)
        sf = float(genpareto.sf(x - u, shape, loc=0.0, scale=max(scale, 1e-10)))
        # unconditional tail exceedance probability; rarer -> closer to 1.
        p = max(1e-8, (1.0 - q) * sf)
        return float(max(0.0, min(1.0, -math.log10(p) / 5.0)))
    except Exception:
        pct = float(np.mean(losses <= x))
        return max(0.0, min(1.0, (pct - q) / max(1e-6, 1.0 - q)))
