"""Deterministic adversarial / causal tests for Ω. No live-market validation claimed."""
import json
import math
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import pytest
from fastapi.testclient import TestClient

from aegis.omega import (Evidence,_score,analyze_case,bounded_sector_diffusion,
                         counterfactual_evidence,risk_decision,replay_cases,
                         chronological_impact_interval,guarded_rebalance)
from scripts.omega_demo import ASOF,fixtures,load_portfolio,sector_map,build


def groups():
    f=fixtures()
    return {
      'r':[x for x in f if x.ticker=='RELIANCE'],
      'h':[x for x in f if x.ticker=='HDFCBANK'],
      'i':[x for x in f if x.ticker=='INFY']}


def test_replay_no_future_leakage():
    ts='2026-10-01T04:32:00+00:00'
    o=replay_cases(groups()['r'],load_portfolio(),sector_map(),[ts])
    assert len(o)==1
    assert o[0]['state']['independent_supporters']==1
    assert o[0]['decision']['action']=='VERIFY_SOURCE'


def test_duplicated_wire_is_one():
    f=groups()['r']
    a=_score([f[0],f[1]],ASOF)
    b=_score([f[0]]+f[1:-1],ASOF)
    assert a['independent_supporters']==2==b['independent_supporters']
    assert b['raw_support_index']==a['raw_support_index']


def test_independent_corrob_raises_support():
    f=groups()['r']
    assert _score(f,ASOF)['support_index']>_score([f[0]],ASOF)['support_index']


def test_contradiction_blocks_execution():
    st=_score(groups()['h'],ASOF)
    assert st['contested']
    assert risk_decision(st)['action']=='INVESTIGATE_CONFLICT'


def test_single_source_must_verify():
    assert risk_decision(_score(groups()['i'],ASOF))['action']=='VERIFY_SOURCE'


def test_corrob_triggers_human_review():
    assert risk_decision(_score(groups()['r'],ASOF))['action']=='REVIEW_STRESS'


def test_remove_publishers_counterfactual():
    out=counterfactual_evidence(groups()['r'],ASOF)
    assert out['remove_one_group'][0]['delta_support']>0
    assert len(out['remove_one_group'])==3


def test_same_claim_required():
    x,y=groups()['r'][0],groups()['h'][0]
    with pytest.raises(ValueError,match='distinct claims'):_score([x,y],ASOF)


def test_future_bars_rejected():
    a=replace(groups()['r'][0],first_seen_at='2026-10-02T08:00:00+00:00',published_at='2026-10-02T08:00:00+00:00')
    with pytest.raises(ValueError,match='Future'):_score([a],ASOF)


def test_timezone_missing_rejected():
    a=replace(groups()['r'][0],first_seen_at='2026-10-01T04:30:00')
    with pytest.raises(ValueError,match='timezone'):_score([a],ASOF)


def test_first_seen_before_publication_rejected():
    a=replace(groups()['r'][0],published_at='2026-10-01T06:00:00+00:00')
    with pytest.raises(ValueError,match='First-seen'):_score([a],ASOF)


def test_future_higher_severity_never_changes_past():
    r=groups()['r'];a=replay_cases(r,load_portfolio(),sector_map(),['2026-10-01T04:35:00+00:00'])[0]
    b=replay_cases([r[0]],load_portfolio(),sector_map(),['2026-10-01T04:35:00+00:00'])[0]
    assert a['state']['support_index']==b['state']['support_index']


def test_invalid_unsupported_source():
    a=replace(groups()['r'][0],source_type='made_up_very_trusted')
    with pytest.raises(ValueError,match='source type'):_score([a],ASOF)


def test_synthetic_portfolio_reconciles():
    case=build()['cases'][0]['portfolio']; rows=case['assets']
    assert abs(sum(x['before'] for x in rows)-case['nav_inr'])<1
    assert abs(sum(x['conditional_pnl'] for x in rows)-case['conditional_pnl_inr'])<1
    assert abs(sum(x['after'] for x in rows)-(case['nav_inr']+case['conditional_pnl_inr']))<1


def test_risk_sector_diffusion_bounded():
    g=bounded_sector_diffusion('RELIANCE',sector_map(),.8,damping=.4,iterations=30)
    assert g['total_influence']<=.8/(1-.4)+.0001
    assert g['total_influence']>=.8
    assert g['propagated'][0]['ticker']=='RELIANCE'


def test_invalid_graph_rejected():
    with pytest.raises(ValueError,match='graph parameters'):
        bounded_sector_diffusion('RELIANCE',sector_map(),.5,damping=1.0)


def test_no_labeled_history_abstains():
    r=chronological_impact_interval(8.,[],ASOF)
    assert r['status']=='ABSTAIN'


def test_conformal_reject_future_labels():
    hist=[dict(prediction_asof='2026-10-01T04:00:00+00:00',
               outcome_observed_at='2026-10-03T04:00:00+00:00',predicted_impact=8.,realized_impact=7.)]*30
    assert chronological_impact_interval(8.,hist,ASOF)['status']=='ABSTAIN'


def test_conformal_bounded_interval():
    hist=[dict(prediction_asof='2026-09-01T04:00:00+00:00',
               outcome_observed_at='2026-09-05T04:00:00+00:00',
               predicted_impact=7.,realized_impact=8.) for _ in range(25)]
    r=chronological_impact_interval(8.,hist,ASOF)
    assert r['status']=='EMPIRICAL_ONLY' and r['lower']==7 and r['upper']==9


def test_unverified_events_block_rebalance():
    cs=[analyze_case(groups()['h'],load_portfolio(),sector_map(),ASOF)]
    result=guarded_rebalance({'HDFCBANK':.5,'RELIANCE':.5},cs,5e9,{})
    assert not result['execution_permitted'] and result['eligible']==0
    assert all(abs(x['delta'])<1e-8 for x in result['rows'])


def test_rebalance_never_routes():
    cs=[analyze_case(groups()['r'],load_portfolio(),sector_map(),ASOF)]
    result=guarded_rebalance({'HDFCBANK':.5,'RELIANCE':.5},cs,5e9,{'HDFCBANK':4e9,'RELIANCE':5e9})
    assert result['eligible']==1 and result['execution_permitted'] is False
    assert abs(sum(x['new_weight'] for x in result['rows'])-1)<1e-5
    assert all(x['gate']['decision']!='APPROVED' for x in result['rows'])


def test_omega_api_demo_and_validation():
    from aegis.api import app
    client=TestClient(app)
    r=client.get('/omega/demo');assert r.status_code==200
    assert len(r.json()['cases'])==3
    z=client.post('/omega/assess',json=dict(evidence=[vars(groups()['r'][0])],
               decision_at=ASOF))
    assert z.status_code==200 and z.json()['decision']['action']=='VERIFY_SOURCE'
    bad=client.post('/omega/assess',json=dict(evidence=[vars(groups()['r'][0])],
               decision_at='2026-09-01T00:00:00+00:00'))
    assert bad.status_code==422


def test_analogues_abstain_without_matured_history():
    from aegis.omega import historical_analogues
    query={'event_type':'Credit Event','sector':'Financials','severity':8,
           'pre_event_vol_pct':21,'spread_bps':3,'volume_ratio':1.5}
    assert historical_analogues(query,[],ASOF)['status']=='ABSTAIN'


def test_analogues_never_use_future_outcomes():
    from aegis.omega import historical_analogues
    query={'event_type':'Credit Event','sector':'Financials','severity':8,
           'pre_event_vol_pct':21,'spread_bps':3,'volume_ratio':1.5}
    h=[dict(event_type='Credit Event',sector='Financials',severity=8,
       pre_event_vol_pct=21,spread_bps=3,volume_ratio=1.5,
       event_first_seen_at='2026-09-30T04:00:00+00:00',
       outcome_observed_at='2026-10-02T04:00:00+00:00',forward_abs_move_bps=120)]*30
    assert historical_analogues(query,h,ASOF)['eligible']==0


def test_analogues_outcome_cohort_only():
    from aegis.omega import historical_analogues
    query={'event_type':'Credit Event','sector':'Financials','severity':8,
           'pre_event_vol_pct':21,'spread_bps':3,'volume_ratio':1.5}
    h=[dict(event_id=f'CASE-{i}',event_type='Credit Event',sector='Financials',
       severity=8,pre_event_vol_pct=21,spread_bps=3,volume_ratio=1.5,
       event_first_seen_at='2026-09-01T04:00:00+00:00',
       outcome_observed_at='2026-09-02T04:00:00+00:00',forward_abs_move_bps=40+i)
       for i in range(24)]
    o=historical_analogues(query,h,ASOF)
    assert o['status']=='DESCRIPTIVE_COHORT' and o['neighbors']==7


def test_missing_claim_id_rejected():
    e=replace(groups()['r'][0],claim_id='')
    with pytest.raises(ValueError,match='identifiers'):_score([e],ASOF)
