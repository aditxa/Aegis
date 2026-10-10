from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))


def _stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


@dataclass
class LiquiditySnapshot:
    ticker: str
    price: float
    adv_inr: float
    spread_bps: float
    daily_vol: float
    depth_10bps_inr: float
    participation_cap: float = 0.12

    def to_dict(self) -> dict:
        return asdict(self)


INDIA_LIQUIDITY: dict[str, LiquiditySnapshot] = {
    "RELIANCE": LiquiditySnapshot("RELIANCE", 2987.50, 14.5e9, 2.2, 0.0145, 1.40e9),
    "HDFCBANK": LiquiditySnapshot("HDFCBANK", 1762.40, 12.4e9, 1.9, 0.0130, 1.25e9),
    "ICICIBANK": LiquiditySnapshot("ICICIBANK", 1468.20, 10.8e9, 1.8, 0.0140, 1.05e9),
    "INFY": LiquiditySnapshot("INFY", 1588.70, 8.9e9, 2.4, 0.0165, 0.86e9),
    "TCS": LiquiditySnapshot("TCS", 3342.10, 7.1e9, 2.5, 0.0150, 0.78e9),
    "BHARTIARTL": LiquiditySnapshot("BHARTIARTL", 2064.30, 7.5e9, 2.3, 0.0140, 0.80e9),
    "SBIN": LiquiditySnapshot("SBIN", 1044.60, 11.7e9, 1.8, 0.0170, 1.20e9),
    "LT": LiquiditySnapshot("LT", 3788.90, 5.8e9, 2.7, 0.0155, 0.62e9),
    "ITC": LiquiditySnapshot("ITC", 485.35, 6.4e9, 2.0, 0.0120, 0.70e9),
    "AXISBANK": LiquiditySnapshot("AXISBANK", 1324.75, 7.9e9, 2.1, 0.0150, 0.82e9),
    "MARUTI": LiquiditySnapshot("MARUTI", 13384.00, 3.6e9, 3.5, 0.0160, 0.36e9),
    "SUNPHARMA": LiquiditySnapshot("SUNPHARMA", 1806.60, 3.3e9, 3.1, 0.0145, 0.34e9),
    "TATAMOTORS": LiquiditySnapshot("TATAMOTORS", 962.15, 6.2e9, 2.8, 0.0210, 0.58e9),
    "KOTAKBANK": LiquiditySnapshot("KOTAKBANK", 2058.20, 5.0e9, 2.5, 0.0145, 0.52e9),
    "ADANIENT": LiquiditySnapshot("ADANIENT", 3022.40, 5.7e9, 4.8, 0.0300, 0.42e9),
    "BAJFINANCE": LiquiditySnapshot("BAJFINANCE", 9312.30, 4.6e9, 3.6, 0.0200, 0.45e9),
    "HINDUNILVR": LiquiditySnapshot("HINDUNILVR", 2718.60, 3.2e9, 2.8, 0.0125, 0.37e9),
    "NTPC": LiquiditySnapshot("NTPC", 356.75, 4.2e9, 2.4, 0.0150, 0.49e9),
    "ONGC": LiquiditySnapshot("ONGC", 287.40, 4.0e9, 2.6, 0.0175, 0.46e9),
    "ASIANPAINT": LiquiditySnapshot("ASIANPAINT", 3098.80, 2.5e9, 3.5, 0.0155, 0.28e9),
}


def liquidity_for(ticker: str) -> LiquiditySnapshot:
    t = ticker.upper()
    if t in INDIA_LIQUIDITY:
        return INDIA_LIQUIDITY[t]
    rng = np.random.default_rng(_stable_seed(t))
    return LiquiditySnapshot(
        t,
        float(rng.uniform(250, 4000)),
        float(rng.uniform(1.5e9, 8.0e9)),
        float(rng.uniform(2.0, 6.0)),
        float(rng.uniform(0.012, 0.03)),
        float(rng.uniform(0.15e9, 0.8e9)),
    )


def execution_estimate(ticker: str, trade_value_inr: float, side: str = "BUY", horizon_minutes: int = 45) -> dict[str, float | str]:
    """Deterministic institutional-demo execution estimate.

    Uses a square-root impact term plus half spread and a participation/urgency term. It is a
    transparent simulator for dashboard risk controls, not an exchange-certified TCA model.
    """
    liq = liquidity_for(ticker)
    q = abs(float(trade_value_inr))
    adv_frac = q / max(liq.adv_inr, 1.0)
    participation = min(1.0, adv_frac * (390.0 / max(5.0, float(horizon_minutes))))
    sigma = liq.daily_vol
    # Square-root impact in decimal price terms. Coefficient deliberately conservative for demo TCA.
    impact_dec = 0.42 * sigma * math.sqrt(max(adv_frac, 0.0))
    impact_bps = 1e4 * impact_dec
    urgency_bps = max(0.0, participation - 0.08) * 16.0
    if q == 0:
        impact_bps = urgency_bps = slippage_bps = all_in_bps = 0.0
        est_cost = 0.0
        status, style = "NO TRADE", "—"
    else:
        slippage_bps = 0.50 * liq.spread_bps + impact_bps + urgency_bps
        fees_bps = 1.2
        all_in_bps = slippage_bps + fees_bps
        est_cost = q * all_in_bps / 1e4
        if adv_frac > 0.12 or participation > 0.20:
            status, style = "LIQUIDITY RISK", "POV / staged"
        elif adv_frac > 0.05 or participation > 0.10:
            status, style = "REVIEW", "VWAP / POV"
        else:
            status, style = "EXECUTABLE", "VWAP"
    return {
        "ticker": ticker.upper(),
        "side": side.upper(),
        "trade_value_inr": q,
        "adv_inr": liq.adv_inr,
        "adv_pct": 100.0 * adv_frac,
        "spread_bps": liq.spread_bps,
        "daily_vol_pct": 100.0 * sigma,
        "participation_pct": 100.0 * participation,
        "impact_cost_bps": impact_bps,
        "slippage_bps": slippage_bps,
        "all_in_cost_bps": all_in_bps,
        "estimated_cost_inr": est_cost,
        "depth_10bps_inr": liq.depth_10bps_inr,
        "status": status,
        "execution_style": style,
    }


def synthetic_iv_surface(spot: float = 25000.0, seed: int = 17, stress: float = 0.35) -> pd.DataFrame:
    """Produce a smooth, arbitrage-conscious *synthetic* NIFTY-like IV surface.

    The surface is parameterised in log-moneyness and sqrt-time with a downside skew, smile,
    term premium and stress bump. It is intended for UI/demo work when licensed option-chain data
    are unavailable; it is explicitly not represented as live NSE data.
    """
    rng = np.random.default_rng(seed)
    tenors = np.array([7, 14, 21, 30, 45, 60, 90, 120, 180], dtype=float)
    mny = np.array([0.78, 0.82, 0.86, 0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10, 1.15, 1.20, 1.25])
    rows = []
    for d in tenors:
        t = d / 365.0
        base = 0.135 + 0.026 * math.exp(-d / 55.0) + 0.015 * math.sqrt(t)
        for k_ratio in mny:
            x = math.log(k_ratio)
            skew = -0.105 * x
            smile = 0.33 * x * x
            left_tail = 0.055 * max(-x - 0.05, 0.0) ** 1.35
            short_stress = stress * 0.045 * math.exp(-d / 32.0) * (1.0 + 1.4 * max(-x, 0.0))
            micro_noise = rng.normal(0, 0.0012)
            iv = max(0.075, min(0.65, base + skew + smile + left_tail + short_stress + micro_noise))
            rows.append({
                "tenor_days": int(d),
                "strike": round(spot * k_ratio / 50.0) * 50.0,
                "moneyness": k_ratio,
                "log_moneyness": x,
                "iv": iv,
                "iv_pct": 100.0 * iv,
            })
    return pd.DataFrame(rows)


def synthetic_market_tape(tickers: Iterable[str], periods: int = 96, freq_minutes: int = 5, seed: int = 73) -> pd.DataFrame:
    """High-density synthetic intraday tape with stress, liquidity and order-flow fields."""
    tickers = [t.upper() for t in tickers]
    rng = np.random.default_rng(seed)
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    rows = []
    market_shock = rng.standard_t(5, size=periods) * 0.0021
    vol_state = 0.0
    for j in range(periods):
        vol_state = 0.90 * vol_state + 0.10 * abs(market_shock[j]) * 100
        ts = start + timedelta(minutes=freq_minutes * j)
        for i, t in enumerate(tickers):
            liq = liquidity_for(t)
            idio = rng.standard_t(5) * liq.daily_vol / math.sqrt(78)
            ret = 0.55 * market_shock[j] + 0.45 * idio
            spread = liq.spread_bps * (1.0 + 0.9 * vol_state + 0.25 * abs(rng.normal()))
            imbalance = float(np.tanh(rng.normal(0, 0.72) - 8 * ret))
            depth = liq.depth_10bps_inr * max(0.22, 1.0 - 0.55 * vol_state - 0.18 * abs(imbalance))
            rows.append({
                "timestamp": ts,
                "ticker": t,
                "return_bps": 1e4 * ret,
                "spread_bps": spread,
                "depth_10bps_inr": depth,
                "order_imbalance": imbalance,
                "realized_vol_pct": 100 * liq.daily_vol * (1.0 + 0.7 * vol_state),
                "liquidity_stress": min(1.0, 0.30 * (spread / max(liq.spread_bps, 0.1) - 1.0) + 0.70 * (1.0 - depth / max(liq.depth_10bps_inr, 1.0))),
            })
    return pd.DataFrame(rows)
