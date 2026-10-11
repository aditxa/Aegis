"""Independent, inspectable quantitative extensions to AegisSignal X.

DEMO/SIMULATION ONLY. Nothing in this module is a trained crash predictor or licensed
market-data feed. Numerical checks are diagnostics, not a certification of no arbitrage.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Iterable
import math
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import norm

IST = ZoneInfo('Asia/Kolkata')
SESSION_START = time(9,15)
SESSION_END = time(15,30)
SESSION_MINUTES = 375


def session_minutes_remaining(now: datetime) -> int:
    """Minutes until Indian normal equity cash session close. Calendar holiday controls omitted."""
    t = now.astimezone(IST)
    start = t.replace(hour=9,minute=15,second=0,microsecond=0)
    end = t.replace(hour=15,minute=30,second=0,microsecond=0)
    if t.weekday() >= 5: return 0  # exchange-specific public holidays still require an official calendar
    if t <= start: return SESSION_MINUTES
    if t >= end: return 0
    return math.floor((end-t).total_seconds()/60)


def order_flow_imbalance(prev:dict, curr:dict) -> float:
    """Best-level OFI in shares from Cont/Kukanov/Stoikov's publicly described statistic.

    Input keys: bid_price, bid_size, ask_price, ask_size. A negative OFI denotes selling pressure.
    """
    keys=('bid_price','bid_size','ask_price','ask_size')
    p={k:float(prev[k]) for k in keys}; c={k:float(curr[k]) for k in keys}
    if not (0 < p['bid_price'] < p['ask_price'] and 0 < c['bid_price'] < c['ask_price']):
        raise ValueError('Invalid crossed/locked quote')
    if min(p['bid_size'],p['ask_size'],c['bid_size'],c['ask_size'])<0: raise ValueError('Negative size')
    b=int(c['bid_price']>=p['bid_price'])*c['bid_size']-int(c['bid_price']<=p['bid_price'])*p['bid_size']
    a=int(c['ask_price']<=p['ask_price'])*c['ask_size']-int(c['ask_price']>=p['ask_price'])*p['ask_size']
    return b-a


def execution_gate(trade_inr:float, adv_inr:float, horizon_minutes:float,
                   participation_cap:float=0.12, max_adv_fraction:float=0.12,
                   depth_10bps_inr:float|None=None) -> dict:
    """Conservative pre-trade guardrail, not approval to route an order.

    adv_inr is traded *value* per whole 375 minute cash session; production must derive
    a time-of-day volume curve and broker constraints, not uniform participation.
    """
    if not (math.isfinite(trade_inr) and math.isfinite(adv_inr) and adv_inr>0):
        raise ValueError('Trade and ADV must be finite and ADV positive')
    if not 0<horizon_minutes<=SESSION_MINUTES: raise ValueError('Bad execution horizon')
    q=abs(trade_inr)
    pct_adv=100*q/adv_inr
    available=adv_inr*(horizon_minutes/SESSION_MINUTES)
    pov=100*q/max(available,1)
    depth_ratio=q/depth_10bps_inr if depth_10bps_inr and depth_10bps_inr>0 else None
    flags=[]
    if pct_adv>100*max_adv_fraction: flags.append('daily_ADV_cap')
    if pov>100*participation_cap: flags.append('horizon_POV_cap')
    if depth_ratio is not None and depth_ratio>1: flags.append('visible_depth_10bps')
    return dict(trade_inr=q,adv_pct=round(pct_adv,3),projected_pov_pct=round(pov,3),
                volume_available_inr=round(available,2),depth_ratio=depth_ratio,
                decision='REVIEW_REQUIRED' if flags else 'WITHIN_SIMULATED_LIMITS',breaches=flags)


def black_scholes_call(spot:float,strike:float,t:float,vol:float,r:float=0.06,dividend:float=0.0)->float:
    if min(spot,strike,t,vol)<=0: raise ValueError('BS parameters must be positive')
    d1=(math.log(spot/strike)+(r-dividend+0.5*vol*vol)*t)/(vol*math.sqrt(t))
    d2=d1-vol*math.sqrt(t)
    return spot*math.exp(-dividend*t)*norm.cdf(d1)-strike*math.exp(-r*t)*norm.cdf(d2)


def surface_diagnostics(surface:pd.DataFrame,spot:float,r:float=0.06,tolerance:float=0.0001)->dict:
    """Finite-grid calendar/vertical/butterfly diagnostics for synthetic EUROPEAN calls.

    Butterfly test uses undiscounted finite difference of slopes; local-vol surfaces
    require much stricter calibration, bid/ask cleaning and dividend/futures inputs.
    """
    for f in ['tenor_days','strike','iv']:
        if f not in surface: raise ValueError(f'Missing {f}')
    x=surface.copy()
    if len(x)==0 or x[['tenor_days','strike','iv']].isna().any().any():
        raise ValueError('Incomplete IV grid')
    if (x[['tenor_days','strike','iv']]<=0).any().any(): raise ValueError('Nonpositive tenor, strike or IV')
    if x.duplicated(['tenor_days','strike']).any(): raise ValueError('Duplicate option keys')
    x['call']=x.apply(lambda row:black_scholes_call(spot,float(row.strike),float(row.tenor_days)/365,float(row.iv),r),axis=1)
    x['undiscounted_call']=x['call']*np.exp(r*x['tenor_days']/365)
    x['total_variance']=x['iv']**2*x['tenor_days']/365
    invalid_vertical=0; invalid_bfly=0; invalid_calendar=0; tested=0
    for tenor,g in x.groupby('tenor_days'):
        g=g.sort_values('strike')
        k=g.strike.to_numpy(dtype=float); v=g.undiscounted_call.to_numpy(dtype=float)
        slope=np.diff(v)/np.diff(k)
        invalid_vertical += int(np.sum(slope>tolerance))+int(np.sum(slope < -1-tolerance))
        invalid_bfly += int(np.sum(np.diff(slope)<-tolerance))
        tested+=len(slope)
    # Compare TOTAL VARIANCE at same fixed strike; necessary but not sufficient for calendar arb.
    for strike,g in x.groupby('strike'):
        g=g.sort_values('tenor_days')
        invalid_calendar+=int(np.sum(np.diff(g.total_variance.to_numpy())<-tolerance))
    return {'rows':int(len(x)),'vertical_violations':invalid_vertical,
            'butterfly_violations':invalid_bfly,'calendar_total_variance_violations':invalid_calendar,
            'checks':'discrete diagnostic only; not arbitrage-free certification',
            'passed_grid_checks':not any([invalid_vertical,invalid_bfly,invalid_calendar])}


def provenance_check(frame:pd.DataFrame, source_col='data_origin', timestamp_col='asof_utc') -> dict:
    if frame.empty:return {'status':'NO_DATA','reason':'empty frame'}
    if source_col not in frame or timestamp_col not in frame:
        return {'status':'NOT_EXECUTABLE','reason':'missing provenance or as-of timestamp'}
    if frame[source_col].isna().any() or frame[timestamp_col].isna().any():
        return {'status':'NOT_EXECUTABLE','reason':'null provenance fields'}
    return {'status':'RESEARCH_ONLY' if any(frame[source_col].astype(str).str.contains('synthetic|demo',case=False)) else 'NOT_VERIFIED',
            'rows':len(frame),'reason':'No licensed feed or broker integration has been independently verified'}


@dataclass
class MonitorRow:
    ticker:str; timestamp:str; change_z:float; cusum:float; fused_risk:float
    warning:str; explanation:str


def sequential_risk_monitor(tape:pd.DataFrame,window:int=24,cusum_k:float=0.5,
                            cusum_h:float=4.5,cooldown_bars:int=6)->pd.DataFrame:
    """Causal, warm-start-safe anomaly detector on historical observed tape values.

    Uses prior-only rolling median/MAD; emits no anomalies before at least window
    observations. Event-time outcomes are NOT labels and this is NOT validated prediction.
    """
    cols={'ticker','timestamp','return_bps','spread_bps','liquidity_stress'}
    if not cols.issubset(tape.columns): raise ValueError(f'missing {cols-set(tape.columns)}')
    if window<6: raise ValueError('window must be >=6')
    o=[]
    for sym,grp in tape.sort_values('timestamp').groupby('ticker',sort=False):
        grp=grp.sort_values('timestamp')
        risk=(0.45*np.abs(grp.return_bps.to_numpy(float))+
              1.8*np.maximum(0,grp.spread_bps.to_numpy(float))+
              25*np.maximum(0,grp.liquidity_stress.to_numpy(float)))
        cus=0.0; last_alarm=-10**9
        for i,(row,s) in enumerate(zip(grp.to_dict('records'),risk)):
            if i<window:
                o.append(asdict(MonitorRow(sym,str(row['timestamp']),0.0,0.0,0.0,'WARMUP','Insufficient prior observations')))
                continue
            past=risk[i-window:i]
            median=float(np.median(past)); mad=float(np.median(np.abs(past-median)))
            z=float(np.clip((s-median)/max(1.4826*mad,2),-12,12))
            cus=max(0.0,cus+z-cusum_k)
            evidence=max(0.0,float(row['liquidity_stress']))
            fused=float(np.clip(0.58*min(max(z,0)/7,1)+0.27*min(cus/(2*cusum_h),1)+0.15*evidence,0,1))
            alarm=(z>=3.5 or cus>=cusum_h) and (i-last_alarm)>=cooldown_bars
            if alarm:last_alarm=i
            warning='ELEVATED' if alarm else 'WATCH' if fused>=0.55 else 'NORMAL'
            o.append(asdict(MonitorRow(sym,str(row['timestamp']),round(z,2),round(cus,2),round(fused,3),warning,
                                           'prior-only robust z + one-sided CUSUM; no future data')))
    return pd.DataFrame(o)


def validated_portfolio(rows:list[dict],nominal_aum_inr:float|None=None)->dict:
    if not rows:raise ValueError('portfolio empty')
    ids=[str(r.get('asset_id','')) for r in rows]
    if len(set(ids))!=len(ids) or '' in ids: raise ValueError('Duplicate or blank asset IDs')
    vals=np.asarray([float(r['market_value']) for r in rows],dtype=float)
    if not np.all(np.isfinite(vals)) or np.any(vals<0):raise ValueError('Portfolio contains invalid market value')
    total=float(np.sum(vals))
    if nominal_aum_inr is not None and abs(total-nominal_aum_inr)>max(1.0,nominal_aum_inr*1e-8):
        raise ValueError(f'AUM mismatch: {total} vs {nominal_aum_inr}')
    return {'assets':len(rows),'aum_inr':total,'reconciled':True}
