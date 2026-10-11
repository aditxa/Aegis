"""AegisSignal Ω: evidence-to-decision laboratory.

All numerical 'supports' are *heuristic evidence scores*, NOT calibrated probabilities.
All quantitative losses are conditional scenario outputs, NOT future-price predictions.
No trading actions or order routing are performed.

Principles: deterministic provenance groups, time-safe replay, transparent conflicts,
counterfactual evidence dependence, bounded network exposure and abstention.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Any
import math
import hashlib
import json

SOURCE_WEIGHTS = {'regulatory_filing': .84, 'exchange_notice': .80,
                  'company_disclosure': .72, 'established_news': .51,
                  'other_news': .33, 'social': .15}
ALLOWED_STANCES = {'support', 'contradict', 'uncertain'}


def _as_utc(s: str | datetime) -> datetime:
    dt = datetime.fromisoformat(s.replace('Z', '+00:00')) if isinstance(s, str) else s
    if dt.tzinfo is None:
        raise ValueError('Every clock must specify its timezone')
    return dt.astimezone(timezone.utc)


def _clamp(x, a=0., b=1.): return max(a, min(b, float(x)))


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    ticker: str
    event_type: str
    headline: str
    publisher: str
    publisher_group: str
    source_type: str
    stance: str
    published_at: str
    first_seen_at: str
    confidence: float
    severity: float   # expert or model conditional severity 1..10; not learned market prediction
    sentiment: float  # issuer-conditioned -1..1
    url: str = ''
    claim_id: str = ''

    def validate(self, decision_at: str):
        if not (self.evidence_id and self.publisher and self.publisher_group and self.ticker and self.event_type and self.claim_id):
            raise ValueError('Missing required evidence identifiers')
        if self.source_type not in SOURCE_WEIGHTS: raise ValueError('Unsupported source type')
        if self.stance not in ALLOWED_STANCES: raise ValueError('Unsupported evidence stance')
        if not (0<=self.confidence<=1 and 1<=self.severity<=10 and -1<=self.sentiment<=1):
            raise ValueError('Evidence model ranges invalid')
        pub, seen, decision = map(_as_utc, (self.published_at, self.first_seen_at, decision_at))
        if seen < pub - timedelta(minutes=1): raise ValueError('First-seen before publication')
        if seen > decision: raise ValueError('Future evidence cannot affect past decisions')
        if pub > decision: raise ValueError('Future publication leakage')
        if self.url and not self.url.startswith(('http://','https://')): raise ValueError('URL scheme rejected')
        return True


def _score(observations: list[Evidence], decision_at: str) -> dict:
    """Deduplicate by publisher group. Accumulate independent *evidence strength*.

    Principal counterfactual: removing a story syndicate must remove its whole group,
    not just one duplicated URL. The score has no frequentist/Bayesian calibration claim.
    """
    if not observations:
        return dict(support_index=0.0, opposing_index=0.0, contested=False,
                    independent_supporters=0, independent_opponents=0,
                    publishers=[], dominance=0., severity=1., sentiment=0., notes=['No evidence'])
    for e in observations:e.validate(decision_at)
    if len({(e.ticker,e.event_type,e.claim_id) for e in observations})>1:
        raise ValueError('Never combine distinct claims, tickers or event types')
    now=_as_utc(decision_at)
    groups:dict[str,list[Evidence]]=defaultdict(list)
    for e in observations:groups[e.publisher_group.strip().lower()].append(e)
    chosen=[]
    for group, items in sorted(groups.items()):
        # A publisher may issue corrected stories. Prefer latest available version for each group,
        # and do not double count the old and corrected statements.
        e=max(items, key=lambda x:(_as_utc(x.first_seen_at),SOURCE_WEIGHTS[x.source_type],x.evidence_id))
        age_hours=(now-_as_utc(e.first_seen_at)).total_seconds()/3600
        freshness=max(.25,math.exp(-max(age_hours,0)/144.0))
        weight=SOURCE_WEIGHTS[e.source_type]*e.confidence*freshness
        if e.stance=='uncertain':weight*=.3
        chosen.append((group,e,round(weight,8),len(items)))
    p_sup=1.;p_opp=1.
    for _,e,w,_ in chosen:
        if e.stance=='support':p_sup*=1-w
        elif e.stance=='contradict':p_opp*=1-w
    supporting=1-p_sup; opposing=1-p_opp
    # Subtract meaningful contradiction; display raw support and opposition separately.
    effective=_clamp(supporting*(1-.82*opposing))
    support_items=[(g,e,w) for g,e,w,n in chosen if e.stance=='support']
    max_group=max((w for _,_,w in support_items),default=0.)
    dominance=max_group/sum((w for _,_,w in support_items),start=1e-10)
    weight_sum=sum(w for _,_,w in support_items)
    severity=sum(e.severity*w for _,e,w in support_items)/weight_sum if weight_sum else 1.
    sentiment=sum(e.sentiment*w for _,e,w in support_items)/weight_sum if weight_sum else 0.
    return dict(support_index=round(effective,5),raw_support_index=round(supporting,5),
                opposing_index=round(opposing,5),contested=opposing>.15,
                independent_supporters=len(support_items),
                independent_opponents=sum(e.stance=='contradict' for _,e,w,n in chosen),
                publishers=[dict(publisher_group=g,headline=e.headline,source=e.source_type,
                                 stance=e.stance,weight=round(w,4),syndicated_records=n,
                                 first_seen_at=e.first_seen_at,url=e.url)
                            for g,e,w,n in chosen],
                dominance=round(dominance,4), severity=round(severity,3),
                sentiment=round(sentiment,4),notes=['Publisher groups deduplicated','Score is not a calibrated probability'])


def counterfactual_evidence(observations:list[Evidence],decision_at:str)->dict:
    base=_score(observations,decision_at)
    groups=sorted({e.publisher_group.strip().lower() for e in observations})
    effects=[]
    for group in groups:
        remaining=[e for e in observations if e.publisher_group.strip().lower()!=group]
        altered=_score(remaining,decision_at)
        effects.append(dict(publisher_group=group,delta_support=round(base['support_index']-altered['support_index'],5),
                            support_without=altered['support_index']))
    effects.sort(key=lambda x:abs(x['delta_support']),reverse=True)
    return dict(base=base,remove_one_group=effects,
                fragility=abs(effects[0]['delta_support']) if effects else 0.)


def risk_decision(state:dict, market_confirmation:float|None=None,
                  threshold:float=.53, minimum_sources:int=2)->dict:
    """Conservative gating; no estimate of realized crash probability."""
    score=state['support_index']; independent=state['independent_supporters']
    uncertain=(state['contested'] or independent<minimum_sources or state['dominance']>.74)
    risk_strength=_clamp((state['severity']-1)/9)*score
    if state['independent_opponents']>0:action='INVESTIGATE_CONFLICT'
    elif uncertain:action='VERIFY_SOURCE'
    elif score>=threshold and state['severity']>=7:action='REVIEW_STRESS'
    elif score>=threshold:action='MONITOR'
    else:action='COLLECT_EVIDENCE'
    if market_confirmation is not None:
        if not 0<=market_confirmation<=1:raise ValueError('Invalid market confirmation')
        # May upgrade human monitoring urgency but does not override provenance abstention.
        if market_confirmation>.8 and action=='MONITOR':action='REVIEW_STRESS'
    return dict(action=action,decision_class='HUMAN_REVIEW_ONLY',
                risk_strength_proxy=round(risk_strength,4),
                support_index=score, independent_publishers=independent,
                contradictory_publishers=state['independent_opponents'],
                missing_evidence=uncertain,not_a_crash_probability=True)


def counterfactual_portfolio(portfolio:list[dict],risk_ticker:str,
                             sector_map:dict[str,str],severity:float, evidence_index:float,
                             spread_stress_bps:float=130.0, equity_drawdown_pct:float=-.08) -> dict:
    """Deterministic event-conditional exposure propagation (NOT DebtRank or price forecast).

    Shocks: targeted issuer, peers, aggregate equity factor, bonds, loans and derivatives;
    impacts are linearized by risk sensitivities. Explicitly dimensioned in INR.
    """
    from aegis.institutional_plus import validated_portfolio
    validated_portfolio(portfolio)
    if not (1<=severity<=10 and 0<=evidence_index<=1):raise ValueError('invalid risk indices')
    severity_multiplier=(severity/10)**1.4
    sector=sector_map.get(risk_ticker,'Unknown')
    sh=equity_drawdown_pct*severity_multiplier
    results=[]
    for row in portfolio:
        typ=str(row.get('asset_type','equity')).lower()
        ticker=str(row.get('ticker',''))
        v=float(row['market_value'])
        if typ=='equity':
            transmission=(2.3 if ticker==risk_ticker else 1.2 if sector_map.get(ticker)==sector and sector!='Unknown' else .45)
            transmission=min(2.5,transmission)
            pnl=v*float(row.get('beta',1.) or 1.)*sh*transmission
        elif typ=='bond':
            dy=(spread_stress_bps*0.36)/10000*severity_multiplier
            ds=spread_stress_bps/10000*severity_multiplier
            duration=float(row.get('duration',0.) or 0.)
            c=float(row.get('convexity',0.) or 0.)
            sd=float(row.get('spread_duration',0.) or 0.)
            pnl=v*(-duration*dy+.5*c*dy*dy-sd*ds)
        elif typ=='loan':
            ds=spread_stress_bps/10000*severity_multiplier
            dp=.018*severity_multiplier
            pnl=v*(-float(row.get('spread_duration',0.) or 0.)*ds - float(row.get('lgd',.45) or .45)*dp)
        elif typ in ('derivative','option'):
            delta=float(row.get('delta',0.) or 0.)
            gamma=float(row.get('gamma',0.) or 0.)
            vega=float(row.get('vega',0.) or 0.)
            vol_shift=.09*severity_multiplier
            pnl=v*(delta*sh+.5*gamma*sh*sh+vega*vol_shift)
        elif typ=='cash':pnl=v*float(row.get('fx_exposure',0.) or 0.)*.018*severity_multiplier
        else:raise ValueError('Unknown asset type: '+typ)
        results.append(dict(asset_id=row['asset_id'],ticker=ticker,asset_type=typ,
                            before=round(v,2),conditional_pnl=round(pnl,2),after=round(v+pnl,2)))
    nav=sum(x['before'] for x in results);total=sum(x['conditional_pnl'] for x in results)
    # IMPORTANT: evidence strength is not probability, hence the adjusted number is NOT expected loss.
    return dict(event_ticker=risk_ticker,event_sector=sector,event_severity=round(severity,2),
                support_index=evidence_index,nav_inr=round(nav,2),conditional_pnl_inr=round(total,2),
                conditional_pnl_pct=round(100*total/nav,3) if nav else 0.,
                evidence_scaled_loss_proxy_inr=round(min(0.,total)*evidence_index,2),
                assets=results,synthetic=True,calibrated=False,
                disclaimer='Conditional scenario P&L, NOT a probability weighted expected return or tradable quote')


def bounded_sector_diffusion(issuer:str,sector_map:dict[str,str],seed_strength:float,
                             damping:float=.38,iterations:int=14)->dict:
    """Exposure-proxy diffusion, NOT observed supplier/credit relationships.

    A[i,j] is a row-stochastic synthetic peer/factor-link matrix. Using damping<1
    makes the Neumann-series propagation bounded; avoid creating false real-world
    edges from co-mentions. This is mathematical scenario sensitivity, not causal ID.
    """
    import numpy as np
    if issuer not in sector_map: raise ValueError('Unknown issuer; cannot impute graph edges')
    if not 0<=seed_strength<=1 or not 0<=damping<1 or iterations<1:
        raise ValueError('invalid graph parameters')
    names=sorted(sector_map)
    n=len(names); matrix=np.zeros((n,n),dtype=float)
    for i,ni in enumerate(names):
        for j,nj in enumerate(names):
            if i!=j:
                matrix[i,j]=1.0 if sector_map[ni]==sector_map[nj] else .06
        z=matrix[i].sum()
        if z>0:matrix[i]/=z
    impulse=np.zeros(n);impulse[names.index(issuer)]=seed_strength
    propagated=impulse.copy()
    for _ in range(iterations):propagated=impulse+damping*matrix.T@propagated
    # Expose normalized bounded diffusion as a diagnostic not a loss forecast.
    spill=[dict(ticker=k,score=round(float(propagated[i]),5),
                direct=bool(k==issuer),sector=sector_map[k]) for i,k in enumerate(names)]
    return dict(issuer=issuer,source='synthetic-sector-proxy',damping=damping,
                spectral_bound='row-stochastic W, alpha < 1',
                total_influence=round(float(propagated.sum()),5),
                propagated=sorted(spill,key=lambda x:-x['score']),
                warning='Synthetic sector links, not a verified corporate/credit exposure graph')


def guarded_rebalance(current_weights:dict[str,float],assessments:list[dict],
                      aum_inr:float,adv_inr_by_ticker:dict[str,float],
                      horizon_minutes:float=45.,minimum_credible_sources:int=2)->dict:
    """Produce review-only hypothetical target weights; never send orders."""
    from aegis.downstream.rebalance import TacticalRebalancer
    from aegis.institutional_plus import execution_gate
    if not current_weights or aum_inr<=0:raise ValueError('Missing weights or AUM')
    if any(not math.isfinite(float(w)) or float(w)<0 for w in current_weights.values()):
        raise ValueError('Invalid current weights')
    wsum=sum(current_weights.values())
    if abs(wsum-1)>1e-6:raise ValueError('Current weights must reconcile to 1')
    eligible={r['ticker']:r for r in assessments if r['decision']['action'] in ('REVIEW_STRESS','MONITOR')
              and r['state']['independent_supporters']>=minimum_credible_sources
              and not r['state']['contested']}
    if not eligible:
        return dict(eligible=0,reason='No adequately corroborated risk event',
                    execution_permitted=False,
                    rows=[dict(ticker=k,old_weight=v,new_weight=v,delta=0.,
                               status='ABSTAIN') for k,v in current_weights.items()])
    signals={}
    for ticker,row in eligible.items():
        s=row['state'];signals[ticker]=dict(sentiment=s['sentiment'],impact=s['severity'],
            early_warning=s['support_index'], confidence=s['support_index'],
            risk=(s['severity']-1)/9*s['support_index'])
    n=len(current_weights)
    opt=TacticalRebalancer(min_w=0,max_w=max(.20,1.0/n,max(current_weights.values())),turnover_limit=.12)
    rows=opt.rebalance(current_weights,signals)
    output=[]
    for r in rows:
        trade=(r.new_weight-r.old_weight)*aum_inr
        adv=adv_inr_by_ticker.get(r.ticker)
        if adv is None or not math.isfinite(float(adv)) or float(adv)<=0:
            gate=dict(decision='REVIEW_REQUIRED',breaches=['missing_verified_ADV'],adv_pct=None,projected_pov_pct=None)
        else:
            gate=execution_gate(trade,float(adv),horizon_minutes)
        output.append(dict(**r.to_dict(),side='BUY' if trade>0 else 'SELL' if trade<0 else 'FLAT',
                           trade_inr=round(trade,2),gate=gate))
    return dict(eligible=len(eligible),execution_permitted=False,rows=output,
                gross_traded_inr=round(sum(abs(x['trade_inr']) for x in output),2),
                approval='Never approved to route. Human/exchange/broker review required')


def adversarial_check(observations:list[Evidence],decision_at:str)->dict:
    original=counterfactual_evidence(observations,decision_at)
    state=original['base']
    return dict(
        story_volume=len(observations), independent_groups=len(state['publishers']),
        copy_amplification=round(len(observations)/max(1,len(state['publishers'])),2),
        largest_publisher_dependence=round(original['fragility'],5),
        opposing_independent_sources=state['independent_opponents'],
        human_review_required=state['contested'] or state['dominance']>.74 or state['independent_supporters']<2,
        evidence_actions=original['remove_one_group'])


def analyze_case(observations:list[Evidence],portfolio:list[dict],sector_map:dict[str,str],
                 decision_at:str) -> dict:
    if not observations:raise ValueError('Case needs at least one observation')
    c=counterfactual_evidence(observations,decision_at)
    state=c['base'];dec=risk_decision(state)
    stress=counterfactual_portfolio(portfolio,observations[0].ticker,sector_map,
                                   state['severity'],state['support_index'])
    return dict(claim_id=observations[0].claim_id,ticker=observations[0].ticker,
                event_type=observations[0].event_type,asof_utc=_as_utc(decision_at).isoformat(),
                state=state,decision=dec,adversarial=adversarial_check(observations,decision_at),
                portfolio=stress,
                diffusion=bounded_sector_diffusion(observations[0].ticker,sector_map,
                    state['support_index']*(state['severity']-1)/9) if observations[0].ticker in sector_map else None,
                simulation=True,
                scope='Demonstration only. No financial forecasts or executable trading advice')


def replay_cases(events:list[Evidence],portfolio:list[dict],sector_map:dict[str,str],
                 checkpoint_times:list[str]) -> list[dict]:
    """Event-time replay. No future records or labels may enter a checkpoint."""
    outputs=[]
    for checkpoint in checkpoint_times:
        valid=[e for e in events if _as_utc(e.first_seen_at)<=_as_utc(checkpoint)]
        buckets=defaultdict(list)
        for e in valid:buckets[(e.ticker,e.event_type,e.claim_id)].append(e)
        for key,group in sorted(buckets.items()):
            outputs.append(analyze_case(group,portfolio,sector_map,checkpoint))
    return outputs


def chronological_impact_interval(prediction:float,history:list[dict],asof_utc:str,
                                   alpha:float=.1,min_calibration:int=20)->dict:
    """Empirical residual interval on observed *matured* outcomes only.

    This is a baseline conformal-style diagnostic; temporal/dependence shifts mean
    finite-sample conformal coverage is NOT guaranteed. It is deliberately unavailable
    without historical labels.
    """
    if not (1<=prediction<=10 and 0<alpha<1):raise ValueError('invalid prediction or alpha')
    now=_as_utc(asof_utc)
    residuals=[]
    for row in history:
        if _as_utc(row['outcome_observed_at'])>now:continue
        if _as_utc(row['prediction_asof'])>_as_utc(row['outcome_observed_at']):
            raise ValueError('Misordered outcome history')
        pred=float(row['predicted_impact']); y=float(row['realized_impact'])
        if not (1<=pred<=10 and 1<=y<=10):raise ValueError('Impact values outside schema')
        residuals.append(abs(y-pred))
    if len(residuals)<min_calibration:
        return {'status':'ABSTAIN','reason':'Insufficient matured chronological labels',
                'calibration_samples':len(residuals),'minimum':min_calibration}
    n=len(residuals); rank=min(n,math.ceil((n+1)*(1-alpha)))
    q=sorted(residuals)[rank-1]
    return dict(status='EMPIRICAL_ONLY',lower=round(max(1.,prediction-q),3),
                upper=round(min(10.,prediction+q),3),calibration_samples=n,
                coverage_not_guaranteed_under_shift=True)

ANALOG_FEATURES = ('severity','pre_event_vol_pct','spread_bps','volume_ratio')

def historical_analogues(query:dict,history:list[dict],asof_utc:str,
                         k:int=7,min_history:int=20)->dict:
    """Research-only nearest historical regime/event analogues, no non-public algorithm.

    A case becomes eligible only *after its outcome is known*. Explicit case-type and
    event-time filters defend against contaminated 'analogues from the future'.
    Median outcomes describe a selected cohort; NOT predictive/calibrated forecasts.
    """
    import numpy as np
    if k<1 or min_history<k:raise ValueError('Invalid k/minimum')
    for name in ANALOG_FEATURES:
        if name not in query or not math.isfinite(float(query[name])):
            raise ValueError('missing/nonfinite query feature '+name)
    asof=_as_utc(asof_utc)
    valid=[]
    for h in history:
        if h.get('event_type')!=query.get('event_type'):continue
        if h.get('sector')!=query.get('sector'):continue
        try:
            incident=_as_utc(h['event_first_seen_at'])
            outcome=_as_utc(h['outcome_observed_at'])
            if outcome<incident:raise ValueError('Event outcome predates event')
            if incident>=asof or outcome>=asof:continue
            x=[float(h[n]) for n in ANALOG_FEATURES]
            y=float(h['forward_abs_move_bps'])
            if not all(map(math.isfinite,x+[y])) or y<0:continue
            valid.append((h,x,y))
        except (KeyError,TypeError):continue
    if len(valid)<min_history:
        return dict(status='ABSTAIN',eligible=len(valid),needed=min_history,
                    reason='Missing sufficiently many *matured* same-sector/same-event historical analogues')
    a=np.array([x for h,x,y in valid]); q=np.array([float(query[n]) for n in ANALOG_FEATURES])
    # Robust IQR floor; not fit on any future data or target labels.
    iqr=np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0)
    floor=np.array([1.,2.,5.,.5])
    scale=np.maximum(iqr,floor)
    d=np.sqrt(np.sum(((a-q)/scale)**2,axis=1))
    idx=sorted(range(len(valid)),key=lambda i:(float(d[i]),str(valid[i][0].get('event_id',''))))[:k]
    outcomes=np.array([valid[i][2] for i in idx])
    return dict(status='DESCRIPTIVE_COHORT',eligible=len(valid),neighbors=k,
                median_future_abs_move_bps=round(float(np.median(outcomes)),3),
                cohort_p80_abs_move_bps=round(float(np.quantile(outcomes,.8)),3),
                nearest=[dict(event_id=valid[i][0].get('event_id','?'),
                              distance=round(float(d[i]),4),
                              historical_forward_abs_move_bps=valid[i][2]) for i in idx],
                warning='Nearest-neighbor historical cohort, NOT a calibrated market forecast. No real data bundled.')
