from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from aegis.institutional_plus import (execution_gate, order_flow_imbalance, session_minutes_remaining,
    sequential_risk_monitor, surface_diagnostics,validated_portfolio,provenance_check)
from aegis.institutional import synthetic_iv_surface, synthetic_market_tape
from aegis.ingest.structural_extract import parse_document,relevant_change

def test_session_and_participation():
    from zoneinfo import ZoneInfo
    ist=ZoneInfo('Asia/Kolkata')
    assert session_minutes_remaining(datetime(2026,10,5,9,15,tzinfo=ist))==375
    assert session_minutes_remaining(datetime(2026,10,5,15,31,tzinfo=ist))==0
    assert execution_gate(1e9,10e9,45)['decision']=='REVIEW_REQUIRED'
    assert execution_gate(1e6,10e9,90)['decision']=='WITHIN_SIMULATED_LIMITS'

def test_real_best_level_ofi_and_crossed_quote():
    prev=dict(bid_price=100,bid_size=100,ask_price=101,ask_size=120)
    curr=dict(bid_price=100,bid_size=150,ask_price=101,ask_size=80)
    assert order_flow_imbalance(prev,curr)==90
    with pytest.raises(ValueError):order_flow_imbalance(prev,{**curr,'ask_price':99})

def test_sequential_causes_no_warmup_spikes():
    tape=synthetic_market_tape(['RELIANCE'],periods=40)
    out=sequential_risk_monitor(tape,window=12)
    assert (out.warning.iloc[:12]=='WARMUP').all()
    assert out.fused_risk.between(0,1).all()
    out2=sequential_risk_monitor(tape.iloc[:20],window=12)
    assert np.allclose(out.fused_risk.iloc[:20],out2.fused_risk)

def test_iv_diagnostics_and_reconciliation():
    s=synthetic_iv_surface()
    d=surface_diagnostics(s,25000)
    assert d['rows']==len(s)
    assert isinstance(d['passed_grid_checks'],bool)
    with pytest.raises(ValueError):validated_portfolio([{'asset_id':'a','market_value':10}],100)

def test_source_and_html_change():
    assert provenance_check(pd.DataFrame([{'data_origin':'synthetic','asof_utc':'2026-10-01T00:00:00Z'}]))['status']=='RESEARCH_ONLY'
    a=parse_document('<html><title>Filing</title><p>Debt: 100</p></html>')
    b=parse_document('<html><title>Filing</title><p>Debt: 250</p></html>')
    assert relevant_change(a,b)['key_value_changes']['Debt']['after']=='250'

def test_tape_market_session_boundary():
    from aegis.institutional import synthetic_market_tape
    with pytest.raises(ValueError):synthetic_market_tape(['RELIANCE'],periods=78)
    tape=synthetic_market_tape(['RELIANCE'],periods=75)
    assert str(tape.timestamp.max()).find('15:')>=0

def test_snapshot_outputs_have_real_columns_and_no_null_financial_basis():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    a=pd.read_csv(root/'data'/'india_portfolio.csv')
    assert a.market_value.notna().all()
    assert round(a.market_value.sum())==5_000_000_000

def test_sample_query_excludes_unrelated_documents(tmp_path):
    import json
    from aegis.ingest.sample import SampleConnector
    path=tmp_path/'stories.jsonl'
    records=[{'source':'demo','title':'Reliance earnings','text':'good', 'published_at':'2026-10-01T09:00:00Z'},
             {'source':'demo','title':'Infosys guidance','text':'down','published_at':'2026-10-01T09:00:01Z'}]
    path.write_text('\n'.join(map(json.dumps,records)))
    out=SampleConnector(path).fetch(query='Reliance',limit=100)
    assert len(out)==1 and 'Reliance' in out[0].title
