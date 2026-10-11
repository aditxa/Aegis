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
from aegis.institutional_plus import execution_gate,surface_diagnostics,sequential_risk_monitor,validated_portfolio
from aegis.institutional import execution_estimate, synthetic_iv_surface, synthetic_market_tape, INDIA_LIQUIDITY

ROOT=Path(__file__).resolve().parents[1]
COMPANY_FILE = ROOT/"data"/"india_companies.csv"
engine=AegisRiskEngine(COMPANY_FILE,ROOT/SETTINGS.sqlite_path,use_market_context=False)
stress_tester=StressTester(ROOT/"data"/"shock_library.json")
rebalancer=TacticalRebalancer()
app=FastAPI(title="AegisSignal-X Financial Early-Risk Engine",version="0.5.0")

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
    execution_horizon_minutes:int=Field(default=45,ge=5,le=375)
class StressRequest(BaseModel):
    event_classification:str; impact_score:float=Field(ge=1,le=10); portfolio:list[dict[str,Any]]; monte_carlo:bool=True

@app.get("/health")
def health():
    return {"status":"ok","profile":"max" if SETTINGS.max_profile else "standard","transformers_enabled":SETTINGS.use_transformers,
            "sentiment_models":list(SETTINGS.sentiment_models if SETTINGS.max_profile else SETTINGS.sentiment_models[:1]),
            "event_model":SETTINGS.event_model,"market_context_default":False,"universe":"india_demo","version":"0.5.0"}

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
def market_tape(periods:int=75):
    periods=max(6,min(75,int(periods)))
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

@app.get("/institutional/iv-quality")
def iv_quality(spot:float=25000.0,stress:float=0.35):
    surf=synthetic_iv_surface(spot=spot,stress=min(1,max(0,stress)))
    return {**surface_diagnostics(surf,spot),"synthetic":True,"not_for_trading":True}

@app.get("/institutional/anomaly-monitor")
def anomaly_monitor(periods:int=75,window:int=20):
    periods=max(30,min(int(periods),75)); window=max(6,min(int(window),periods-1))
    market=synthetic_market_tape(list(INDIA_LIQUIDITY)[:14],periods=periods)
    rows=sequential_risk_monitor(market,window=window)
    return {"synthetic":True,"validated_predictor":False,"alarms":rows[rows.warning=='ELEVATED'].to_dict('records'),
            "rows_processed":len(rows),"warmup_bars":window}

# ---------------------------------------------------------------------------
# Ω Evidence Intelligence: explicitly simulation/research-only endpoints
# ---------------------------------------------------------------------------
from aegis.omega import Evidence,analyze_case,replay_cases,guarded_rebalance
import csv

class OmegaAssessmentRequest(BaseModel):
    evidence:list[dict[str,Any]]
    decision_at:str
    portfolio:list[dict[str,Any]]|None=None

class OmegaTextRequest(AnalyzeRequest):
    publisher_group:str
    source_type:str='other_news'
    first_seen_at:datetime|None=None
    claim_id:str=''

@app.get('/omega/demo')
def omega_demo():
    from scripts.omega_demo import build
    return build()

@app.post('/omega/assess')
def omega_assess(req:OmegaAssessmentRequest):
    from scripts.omega_demo import load_portfolio,sector_map
    try:
        records=[Evidence(**x) for x in req.evidence]
        return analyze_case(records,req.portfolio or load_portfolio(),sector_map(),req.decision_at)
    except (ValueError,TypeError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc

@app.post('/omega/analyze-text')
def omega_analyze_text(req:OmegaTextRequest):
    """Existing NLP result -> a single observation. One source must abstain from execution."""
    from scripts.omega_demo import load_portfolio,sector_map
    from aegis.omega import _as_utc
    first_seen=req.first_seen_at or datetime.now(timezone.utc)
    if first_seen.tzinfo is None:raise HTTPException(422,'First-seen timestamp requires timezone')
    item=TextItem(source=req.source,title=req.title,text=req.text,published_at=req.published_at,
                  url=req.url,source_domain=req.source_domain,metadata=req.metadata)
    try:
        signals=engine.analyze_item(item)
        evidence=[]
        for j,sig in enumerate(signals):
            evidence.append(Evidence(evidence_id=sig.item_id+'-'+str(j),ticker=sig.ticker,
                event_type=sig.event_classification,headline=req.title or req.text[:120],
                publisher=req.source_domain or req.source,
                publisher_group=req.publisher_group,source_type=req.source_type,
                stance='support' if sig.event_direction<0 else 'uncertain',
                published_at=req.published_at.isoformat(),first_seen_at=first_seen.isoformat(),
                confidence=max(0,min(1,sig.event_confidence)),severity=sig.impact_score,
                sentiment=sig.sentiment_score,url=req.url,
                claim_id=req.claim_id or sig.item_id))
        output=[analyze_case([e],load_portfolio(),sector_map(),first_seen.isoformat()) for e in evidence]
    except (ValueError,TypeError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    return {'extracted_signals':[s.to_dict() for s in signals], 'omega_assessments':output,
            'single_source_safety':'Single independent source cannot authorize portfolio action',
            'unverified_metadata':'publisher_group and source_type need a trusted server-side source registry'}
