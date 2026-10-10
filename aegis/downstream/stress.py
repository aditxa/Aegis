from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np


@dataclass
class StressResult:
    asset_id: str; name: str; asset_type: str; before: float; pnl: float; after: float; pnl_pct: float
    def to_dict(self): return asdict(self)


class StressTester:
    """Event-conditioned deterministic + heavy-tailed Monte-Carlo stress engine."""
    def __init__(self, shock_library_path: str | Path, random_seed: int = 42) -> None:
        self.shocks=json.loads(Path(shock_library_path).read_text(encoding="utf-8")); self.random_seed=random_seed

    def shock_for(self,event_class: str,impact_score: float)->dict[str,float]:
        base=self.shocks.get(event_class,self.shocks["Other"])
        # Convex severity map: high-impact events accelerate rather than scale linearly forever.
        x=max(0.0,min(1.0,(impact_score-1.0)/9.0)); scale=0.20+1.25*(x**1.35)
        return {k:float(v)*scale for k,v in base.items()}

    @staticmethod
    def _pnl(row: dict, shock: dict[str,float]) -> float:
        typ=str(row.get("asset_type","equity")).lower(); v=float(row.get("market_value",0.0))
        if typ=="equity":
            return v*float(row.get("beta",1.0) or 1.0)*shock.get("equity_pct",0.0)
        if typ=="bond":
            d=float(row.get("duration",0.0) or 0.0); c=float(row.get("convexity",0.0) or 0.0); sd=float(row.get("spread_duration",0.0) or 0.0)
            dy=shock.get("rate_bps",0.0)/10000.0; ds=shock.get("credit_spread_bps",0.0)/10000.0
            return v*(-d*dy+0.5*c*dy*dy-sd*ds)
        if typ=="loan":
            sd=float(row.get("spread_duration",0.0) or 0.0); lgd=float(row.get("lgd",0.45) or 0.45)
            return v*(-sd*shock.get("credit_spread_bps",0.0)/10000.0-lgd*shock.get("default_prob_shift",0.0))
        if typ in {"derivative","option"}:
            delta=float(row.get("delta",0.0) or 0.0); gamma=float(row.get("gamma",0.0) or 0.0); vega=float(row.get("vega",0.0) or 0.0)
            eq=shock.get("equity_pct",0.0); vol=shock.get("vol_points",0.0)/100.0
            return v*(delta*eq+0.5*gamma*eq*eq+vega*vol)
        if typ=="cash":
            return v*float(row.get("fx_exposure",0.0) or 0.0)*shock.get("fx_pct",0.0)
        return v*shock.get("equity_pct",0.0)*0.5

    def monte_carlo(self,portfolio:list[dict],event_class:str,impact_score:float,n_sims:int=2500)->dict:
        base=self.shock_for(event_class,impact_score)
        keys=["equity_pct","rate_bps","credit_spread_bps","fx_pct","vol_points","default_prob_shift"]
        mu=np.array([base.get(k,0.0) for k in keys],dtype=float)
        floors=np.array([0.012,22.0,30.0,0.008,3.0,0.003])
        sd=np.maximum(np.abs(mu)*0.35,floors)
        corr=np.array([
            [1.00,-0.25,-0.55,0.20,-0.65,-0.45],[-0.25,1.00,0.25,-0.10,0.15,0.15],
            [-0.55,0.25,1.00,-0.15,0.55,0.70],[0.20,-0.10,-0.15,1.00,-0.10,-0.10],
            [-0.65,0.15,0.55,-0.10,1.00,0.45],[-0.45,0.15,0.70,-0.10,0.45,1.00],
        ])
        cov=np.outer(sd,sd)*corr
        rng=np.random.default_rng(self.random_seed+int(round(impact_score*10)))
        z=rng.multivariate_normal(np.zeros(6),cov,size=n_sims)
        df=5.0; z*=np.sqrt(df/rng.chisquare(df,size=n_sims))[:,None]
        scen=mu+z
        # Jump mixture: more severe textual events increase the chance of an additional same-direction shock.
        jump_p=0.03+0.20*((max(1.0,min(10.0,impact_score))-1.0)/9.0)**1.5
        mask=rng.random(n_sims)<jump_p
        if np.any(mask):
            jump_mult=rng.lognormal(mean=-0.25,sigma=0.45,size=mask.sum())[:,None]
            scen[mask]+=jump_mult*mu
        totals=[]
        for s in scen:
            sh={k:float(v) for k,v in zip(keys,s)}
            totals.append(sum(self._pnl(r,sh) for r in portfolio))
        p=np.asarray(totals,dtype=float); before=sum(float(r.get("market_value",0.0)) for r in portfolio)
        q05=float(np.quantile(p,0.05)); q01=float(np.quantile(p,0.01))
        es95=float(p[p<=q05].mean()) if np.any(p<=q05) else q05
        es99=float(p[p<=q01].mean()) if np.any(p<=q01) else q01
        return {
            "simulations":n_sims,"median_pnl":round(float(np.median(p)),2),"var95_loss":round(max(0.0,-q05),2),
            "cvar95_loss":round(max(0.0,-es95),2),"var99_loss":round(max(0.0,-q01),2),"cvar99_loss":round(max(0.0,-es99),2),
            "prob_loss_gt_5pct":round(float(np.mean(p<=-0.05*before)),4) if before else 0.0,
            "prob_loss_gt_10pct":round(float(np.mean(p<=-0.10*before)),4) if before else 0.0,"jump_probability":round(jump_p,4),
        }

    def run(self,portfolio:list[dict],event_class:str,impact_score:float,monte_carlo:bool=True)->dict:
        shock=self.shock_for(event_class,impact_score); out=[]
        for row in portfolio:
            v=float(row.get("market_value",0.0)); pnl=self._pnl(row,shock); after=v+pnl; typ=str(row.get("asset_type","equity")).lower()
            out.append(StressResult(str(row.get("asset_id",row.get("ticker","asset"))),str(row.get("name",row.get("ticker","asset"))),typ,round(v,2),round(pnl,2),round(after,2),round(100*pnl/v,3) if v else 0.0))
        before=sum(x.before for x in out); pnl=sum(x.pnl for x in out)
        result={"event_classification":event_class,"impact_score":impact_score,"shock":{k:round(v,6) for k,v in shock.items()},
                "portfolio_before":round(before,2),"portfolio_after":round(before+pnl,2),"portfolio_pnl":round(pnl,2),
                "portfolio_pnl_pct":round(100*pnl/before,3) if before else 0.0,"assets":[x.to_dict() for x in out]}
        if monte_carlo: result["distributional_risk"]=self.monte_carlo(portfolio,event_class,impact_score)
        return result

    def reverse_stress(self,portfolio:list[dict],event_class:str,target_loss_pct:float=-10.0)->dict:
        if target_loss_pct>=0: raise ValueError("target_loss_pct must be negative")
        lo,hi=1.0,10.0; hi_r=self.run(portfolio,event_class,hi,monte_carlo=False)
        if hi_r["portfolio_pnl_pct"]>target_loss_pct:
            return {"reachable":False,"target_loss_pct":target_loss_pct,"impact_score_required":None,"loss_at_impact_10":hi_r["portfolio_pnl_pct"]}
        for _ in range(35):
            mid=(lo+hi)/2; r=self.run(portfolio,event_class,mid,monte_carlo=False)
            if r["portfolio_pnl_pct"]<=target_loss_pct: hi=mid
            else: lo=mid
        r=self.run(portfolio,event_class,hi,monte_carlo=False)
        return {"reachable":True,"target_loss_pct":target_loss_pct,"impact_score_required":round(hi,3),"portfolio_pnl_pct":r["portfolio_pnl_pct"],"shock":r["shock"]}
