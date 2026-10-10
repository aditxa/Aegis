from __future__ import annotations

import math
from dataclasses import dataclass, asdict
import numpy as np


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, v) for v in weights.values())
    if s <= 0:
        n = max(1, len(weights)); return {k:1.0/n for k in weights}
    return {k:max(0.0,v)/s for k,v in weights.items()}


def _bounded_simplex(raw: dict[str,float], min_w: float, max_w: float) -> dict[str,float]:
    w = _normalize(raw); keys = list(w)
    if min_w*len(keys)>1.0 or max_w*len(keys)<1.0: raise ValueError("Infeasible min/max weight constraints")
    for _ in range(80):
        low = [k for k in keys if w[k] < min_w]; high = [k for k in keys if w[k] > max_w]
        if not low and not high: break
        fixed = {k:min_w for k in low}; fixed.update({k:max_w for k in high})
        free = [k for k in keys if k not in fixed]
        rem = 1.0-sum(fixed.values())
        fs = sum(max(0,w[k]) for k in free)
        w.update(fixed)
        for k in free: w[k] = rem*(max(0,w[k])/fs if fs>0 else 1/len(free))
    return _normalize(w)


@dataclass
class RebalanceRow:
    ticker: str
    old_weight: float
    new_weight: float
    delta: float
    sentiment: float
    impact: float
    early_warning: float
    score: float
    marginal_risk: float
    def to_dict(self): return asdict(self)


class TacticalRebalancer:
    """Risk-aware tactical index optimizer.

    Signal alpha is constrained by variance, heavy-tailed scenario CVaR, turnover and position limits.
    If numerical optimization fails it falls back to the bounded exponentiated update rather than
    emitting unstable weights.
    """
    def __init__(self, eta: float=1.0, risk_penalty: float=0.80, min_w: float=0.02, max_w: float=0.20,
                 turnover_limit: float=0.25, variance_aversion: float=12.0, cvar_aversion: float=2.2,
                 transaction_cost: float=0.0025, random_seed: int=42) -> None:
        self.eta=eta; self.risk_penalty=risk_penalty; self.min_w=min_w; self.max_w=max_w
        self.turnover_limit=turnover_limit; self.variance_aversion=variance_aversion
        self.cvar_aversion=cvar_aversion; self.transaction_cost=transaction_cost; self.random_seed=random_seed

    def _signal(self, sig: dict[str,float]) -> tuple[float,float,float,float]:
        sentiment=float(sig.get("sentiment",0.0)); impact=float(sig.get("impact",1.0)); conf=float(sig.get("confidence",0.35)); early=float(sig.get("early_warning",0.0))
        risk=float(sig.get("risk",max(0.0,min(1.0,(impact-1)/9))))
        alpha = conf*sentiment - self.risk_penalty*risk*max(0.25,early)*(0.6+0.4*max(0.0,-sentiment))
        return alpha,risk,sentiment,early

    def rebalance(self, current_weights: dict[str,float], signals: dict[str,dict[str,float]]) -> list[RebalanceRow]:
        current=_normalize(current_weights); keys=list(current); n=len(keys)
        if n==0: return []
        w0=np.array([current[k] for k in keys],dtype=float)
        vals=[self._signal(signals.get(k,{})) for k in keys]
        alpha=np.array([v[0] for v in vals]); risks=np.array([v[1] for v in vals])
        # Alpha is a ranking signal rather than a claimed expected-return forecast.
        mu=0.012*alpha
        vols=0.14*(0.65+1.15*risks)
        corr=0.22*np.ones((n,n))+0.78*np.eye(n)
        cov=np.outer(vols,vols)*corr
        rng=np.random.default_rng(self.random_seed)
        df=5.0; m=700
        z=rng.multivariate_normal(np.zeros(n),cov,size=m)
        scales=np.sqrt(df/rng.chisquare(df,size=m))[:,None]
        scenarios=z*scales

        def objective(w):
            port_var=float(w@cov@w)
            losses=-(scenarios@w)
            q=np.quantile(losses,0.90)
            cvar=float(losses[losses>=q].mean()) if np.any(losses>=q) else float(q)
            turnover_smooth=float(np.sum(np.sqrt((w-w0)**2+1e-10)))
            return -float(mu@w)+self.variance_aversion*port_var+self.cvar_aversion*cvar+self.transaction_cost*turnover_smooth

        bounds=[(self.min_w,self.max_w) for _ in keys]
        # Directional constraints preserve case-study semantics when jointly feasible.
        directional=[]
        for i,(a,_,sent,_) in enumerate(vals):
            if sent>0.15: directional.append((i,"up"))
            elif sent<-0.15: directional.append((i,"down"))
        cons=[{"type":"eq","fun":lambda w: float(np.sum(w)-1.0)},
              {"type":"ineq","fun":lambda w: float(self.turnover_limit-0.5*np.sum(np.abs(w-w0)))}]
        for i,d in directional:
            if d=="up": cons.append({"type":"ineq","fun":lambda w,i=i: float(w[i]-w0[i])})
            else: cons.append({"type":"ineq","fun":lambda w,i=i: float(w0[i]-w[i])})
        try:
            from scipy.optimize import minimize
            res=minimize(objective,w0,method="SLSQP",bounds=bounds,constraints=cons,options={"maxiter":350,"ftol":1e-10,"disp":False})
            if not res.success or not np.all(np.isfinite(res.x)):
                # Relax directional constraints before falling back; all-positive/all-negative sets may make them infeasible.
                res=minimize(objective,w0,method="SLSQP",bounds=bounds,constraints=cons[:2],options={"maxiter":350,"ftol":1e-10,"disp":False})
            if not res.success: raise RuntimeError(res.message)
            target={k:float(v) for k,v in zip(keys,res.x)}
        except Exception:
            raw={k:current[k]*math.exp(self.eta*alpha[i]) for i,k in enumerate(keys)}
            target=_bounded_simplex(raw,self.min_w,self.max_w)
            turn=0.5*sum(abs(target[k]-current[k]) for k in keys)
            if turn>self.turnover_limit and turn>0:
                a=self.turnover_limit/turn
                target={k:current[k]+a*(target[k]-current[k]) for k in keys}
                target=_bounded_simplex(target,self.min_w,self.max_w)

        w=np.array([target[k] for k in keys])
        marginal=cov@w
        rows=[]
        for i,k in enumerate(keys):
            sig=signals.get(k,{})
            rows.append(RebalanceRow(k,round(current[k],6),round(target[k],6),round(target[k]-current[k],6),
                round(float(sig.get("sentiment",0.0)),4),round(float(sig.get("impact",1.0)),2),round(float(sig.get("early_warning",0.0)),4),
                round(float(alpha[i]),4),round(float(marginal[i]),6)))
        return sorted(rows,key=lambda r:r.new_weight,reverse=True)
