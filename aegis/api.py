from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aegis.config import SETTINGS
from aegis.downstream import StressTester, TacticalRebalancer
from aegis.engine import AegisRiskEngine
from aegis.ingest import GDELTConnector, GoogleNewsRSSConnector, RedditConnector, SECEdgarConnector
from aegis.schemas import TextItem
from aegis.institutional import execution_estimate, synthetic_iv_surface, synthetic_market_tape, INDIA_LIQUIDITY

ROOT=Path(__file__).resolve().parents[1]
COMPANY_FILE = ROOT/"data"/"india_companies.csv"
engine=AegisRiskEngine(COMPANY_FILE,ROOT/SETTINGS.sqlite_path,use_market_context=False)
stress_tester=StressTester(ROOT/"data"/"shock_library.json")
rebalancer=TacticalRebalancer()
app=FastAPI(title="AegisSignal-X Financial Early-Risk Engine",version="0.3.0")

class AnalyzeRequest(BaseModel):
    source:str="manual"; title:str=""; text:str
    published_at:datetime=Field(default_factory=lambda:datetime.now(timezone.utc))
    url:str=""; source_domain:str=""
    metadata:dict[str,Any]=Field(default_factory=dict)
    market_context:bool=False
class PipelineRequest(AnalyzeRequest):
    current_weights:dict[str,float]|None=None; portfolio:list[dict[str,Any]]|None=None
class IngestRequest(BaseModel):
    query:str; limit:int=Field(default=20,ge=1,le=100); sources:list[str]=["gdelt","rss"]
    market_context:bool=False
class RebalanceRequest(BaseModel):
    current_weights:dict[str,float]
    aum_inr:float|None=Field(default=None,gt=0)
    execution_horizon_minutes:int=Field(default=45,ge=5,le=390)
class StressRequest(BaseModel):
    event_classification:str; impact_score:float=Field(ge=1,le=10); portfolio:list[dict[str,Any]]; monte_carlo:bool=True

@app.get("/health")
def health():
    return {"status":"ok","profile":"max" if SETTINGS.max_profile else "standard","transformers_enabled":SETTINGS.use_transformers,
            "sentiment_models":list(SETTINGS.sentiment_models if SETTINGS.max_profile else SETTINGS.sentiment_models[:1]),
            "event_model":SETTINGS.event_model,"market_context_default":False,"universe":"india_demo","version":"0.3.0"}

@app.post("/analyze")
def analyze(req:AnalyzeRequest):
    item=TextItem(source=req.source,title=req.title,text=req.text,published_at=req.published_at,url=req.url,source_domain=req.source_domain,metadata=req.metadata)
    eng=engine if not req.market_context else AegisRiskEngine(COMPANY_FILE,use_market_context=True)
    return [x.to_dict() for x in eng.analyze_item(item)]

@app.post("/ingest")
def ingest(req:IngestRequest):
    connectors={"gdelt":GDELTConnector,"reddit":RedditConnector,"rss":GoogleNewsRSSConnector,"edgar":SECEdgarConnector}
    # market_context requires a fresh engine because it changes data dependencies.
    eng=engine if not req.market_context else AegisRiskEngine(COMPANY_FILE,use_market_context=True)
    items=[]; errors={}
    for name in req.sources:
        ctor=connectors.get(name.lower())
        if not ctor: errors[name]="unknown source"; continue
        try: items.extend(ctor().fetch(req.query,req.limit))
        except Exception as e: errors[name]=str(e)
    if not items and errors: raise HTTPException(status_code=502,detail={"message":"all requested sources failed","errors":errors})
    sigs=eng.analyze(items)
    return {"items_ingested":len(items),"signals":[s.to_dict() for s in sigs],"source_errors":errors}

@app.get("/signals")
def signals(ticker:str|None=None,limit:int=100):
    if engine.store is None:return []
    return engine.store.latest(ticker=ticker,limit=min(500,max(1,limit)))

@app.post("/rebalance")
def rebalance(req:RebalanceRequest):
    agg=engine.aggregate_for_rebalancer(list(req.current_weights)); rows=rebalancer.rebalance(req.current_weights,agg)
    result={"signals":agg,"weights":[r.to_dict() for r in rows]}
    if req.aum_inr:
        execution=[]
        for r in rows:
            trade=(r.new_weight-r.old_weight)*req.aum_inr
            execution.append({**r.to_dict(),**execution_estimate(r.ticker,trade,"BUY" if trade>0 else "SELL" if trade<0 else "FLAT",req.execution_horizon_minutes)})
        result["execution"]=execution
    return result

@app.get("/institutional/vol-surface")
def vol_surface(spot:float=25000.0,stress:float=0.35):
    stress=max(0.0,min(1.0,float(stress)))
    return {"synthetic":True,"underlying":"NIFTY-like demo","spot":spot,"points":synthetic_iv_surface(spot=spot,stress=stress).to_dict("records")}

@app.get("/institutional/market-tape")
def market_tape(periods:int=78):
    periods=max(6,min(390,int(periods)))
    frame=synthetic_market_tape(list(INDIA_LIQUIDITY)[:14],periods=periods)
    frame["timestamp"]=frame["timestamp"].astype(str)
    return {"synthetic":True,"rows":frame.to_dict("records")}

@app.post("/stress-test")
def stress_test(req:StressRequest):
    return stress_tester.run(req.portfolio,req.event_classification,req.impact_score,monte_carlo=req.monte_carlo)

@app.post("/pipeline")
def pipeline(req:PipelineRequest):
    item=TextItem(source=req.source,title=req.title,text=req.text,published_at=req.published_at,url=req.url,source_domain=req.source_domain,metadata=req.metadata)
    eng=engine if not req.market_context else AegisRiskEngine(COMPANY_FILE,use_market_context=True)
    sigs=eng.analyze_item(item); result:dict[str,Any]={"signals":[s.to_dict() for s in sigs]}
    if req.current_weights:
        agg=eng.aggregate_for_rebalancer(list(req.current_weights)); result["rebalance"]=[r.to_dict() for r in rebalancer.rebalance(req.current_weights,agg)]
    if req.portfolio and sigs:
        top=max(sigs,key=lambda s:s.impact_score)
        if top.impact_score>=SETTINGS.risk_alert_threshold:
            result["stress_triggered"]=True; result["stress_test"]=stress_tester.run(req.portfolio,top.event_classification,top.impact_score)
        else:
            result["stress_triggered"]=False; result["stress_reason"]=f"impact {top.impact_score:.2f} below threshold {SETTINGS.risk_alert_threshold:.2f}"
    return result
