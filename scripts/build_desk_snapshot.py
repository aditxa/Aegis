"""Generate an OFFLINE interactive, fully self-contained institutional risk desk snapshot."""
from __future__ import annotations
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aegis.engine import AegisRiskEngine
from aegis.ingest import SampleConnector
from aegis.downstream import StressTester,TacticalRebalancer
from aegis.institutional import INDIA_LIQUIDITY,synthetic_market_tape,synthetic_iv_surface,execution_estimate
from aegis.institutional_plus import sequential_risk_monitor,surface_diagnostics,validated_portfolio

ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs';O.mkdir(exist_ok=True)
NAV=5_000_000_000
CURRENT={'RELIANCE':.14,'HDFCBANK':.13,'ICICIBANK':.115,'INFY':.10,'TCS':.095,'BHARTIARTL':.09,'SBIN':.085,'LT':.08,'ITC':.075,'AXISBANK':.09}
BLUE='#7d9dd7';GREEN='#63c89e';RED='#e87079';AMBER='#e7bc76';BG='#0c1118';TEXT='#dee8ef'

def export():
    eng=AegisRiskEngine(ROOT/'data/india_companies.csv',use_market_context=False)
    events=SampleConnector(ROOT/'data/india_sample_articles.jsonl').fetch(limit=10_000)
    raw=eng.analyze(events)
    df=pd.DataFrame([x.to_dict() for x in raw])
    df['ts']=pd.to_datetime(df.timestamp,utc=True).dt.tz_convert('Asia/Kolkata')
    df['HH:MM:SS IST']=df.ts.dt.strftime('%H:%M:%S')
    df.drop(columns=['ts']).to_csv(O/'desk_signals.csv',index=False)
    agg=eng.aggregate_for_rebalancer(list(CURRENT))
    rows=TacticalRebalancer(min_w=.04,max_w=.17,turnover_limit=.12,transaction_cost=.002).rebalance(CURRENT,agg)
    reb=pd.DataFrame([r.to_dict() for r in rows]);execs=[]
    for r in rows:
        trade=(r.new_weight-r.old_weight)*NAV
        execs.append({**r.to_dict(),**execution_estimate(r.ticker,trade,'BUY' if trade>0 else 'SELL' if trade<0 else 'FLAT')})
    ex=pd.DataFrame(execs)
    ex.drop(columns=['guardrail']).to_csv(O/'desk_execution.csv',index=False)
    portfolio=pd.read_csv(ROOT/'data/india_portfolio.csv').fillna(0).to_dict('records')
    validated_portfolio(portfolio,NAV)
    stress=StressTester(ROOT/'data/shock_library.json').run(portfolio,'Credit Event',8.0)
    (O/'desk_stress.json').write_text(json.dumps(stress,indent=2))
    surf=synthetic_iv_surface(spot=25000,stress=.35)
    surf.to_csv(O/'desk_iv_surface.csv',index=False)
    diag=surface_diagnostics(surf,25000)
    market=synthetic_market_tape(list(INDIA_LIQUIDITY),periods=75)
    monitor=sequential_risk_monitor(market,window=20)
    monitor.to_csv(O/'desk_anomaly_monitor.csv',index=False)
    ix=market.groupby('timestamp',sort=True).return_bps.mean()
    index=25000*np.exp(ix.cumsum().to_numpy()/1e4)
    # Plotly 3D is real mouse-interactive in the self-contained HTML.
    piv=surf.pivot(index='tenor_days',columns='moneyness',values='iv_pct')
    fvol=go.Figure(go.Surface(x=piv.columns.to_numpy(),y=piv.index.to_numpy(),z=piv.to_numpy(),
                 colorscale='Cividis',colorbar=dict(title='IV %',thickness=9),
                 hovertemplate='K / Spot %{x:.2f}<br>Tenor %{y:.0f} days<br>IV %{z:.2f}%<extra></extra>'))
    fvol.update_layout(scene=dict(xaxis_title='Strike / spot',yaxis_title='Tenor (days)',zaxis_title='IV (%)',bgcolor=BG,
                  camera=dict(eye=dict(x=1.6,y=-1.6,z=.75))),paper_bgcolor=BG,font=dict(color=TEXT,family='Arial'),
                  margin=dict(l=0,r=0,t=5,b=0),height=425)
    riskby=df.groupby('ticker').agg(priority=('risk_priority_score','max'),impact=('impact_score','max'),events=('item_id','count')).sort_values('priority')
    fr=go.Figure(go.Bar(x=riskby.priority.values,y=riskby.index.tolist(),orientation='h',
                    marker_color=[RED if x>=7 else AMBER if x>=5 else BLUE for x in riskby.priority],
                    hovertemplate='%{y}<br>Priority %{x:.2f}<extra></extra>'))
    fr.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=dict(color=TEXT,family='Arial'),margin=dict(l=85,r=20,t=15,b=30),height=425,
                     xaxis=dict(range=[0,10],title='Peak signal priority',gridcolor='#27313a'),yaxis=dict(tickfont=dict(size=10)))
    fx=go.Figure(go.Bar(x=(ex.new_weight-ex.old_weight)*100,y=ex.ticker,orientation='h',
           marker_color=[GREEN if v>=0 else RED for v in ex.new_weight-ex.old_weight],
           hovertemplate='%{y}<br>Change %{x:+.2f} percentage points<extra></extra>'))
    fx.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=dict(color=TEXT,family='Arial'),margin=dict(l=90,r=10,t=15,b=40),height=345,
                     xaxis=dict(title='Change in portfolio weight (pp)',gridcolor='#27313a'))
    fa=go.Figure(go.Scatter(x=[str(x) for x in ix.index],y=index,mode='lines',
                  line=dict(color=BLUE,width=2),hovertemplate='%{x}<br>Synthetic proxy %{y:,.0f}<extra></extra>'))
    fa.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=dict(color=TEXT,family='Arial'),margin=dict(l=55,r=20,t=10,b=40),height=265,
                    xaxis=dict(title='Demo session, IST',showticklabels=False),yaxis=dict(title='Simulated index',gridcolor='#27313a'))
    # Rich, bounded front-of-book table with clear per-row units.
    latest=df.sort_values('ts',ascending=False).head(24)
    feed=''.join('<tr><td>'+html.escape(r['ts'].strftime('%H:%M:%S'))+'</td><td><strong>'+html.escape(str(r['ticker']))+'</strong></td><td>'+html.escape(str(r['event_classification']))+'</td>'+
                 f'<td>{r["sentiment_score"]:+.2f}</td><td>{r["early_warning_score"]:.2f}</td><td>{r["risk_priority_score"]:.2f}</td>'+
                 '<td><span class="alert '+('bad' if r['alert_level'] in ('HIGH','CRITICAL') else 'watch' if r['alert_level'] in ('ELEVATED','WATCH') else '')+'">'+html.escape(str(r['alert_level']))+'</span></td></tr>'
                 for _,r in latest.iterrows())
    erows=''.join(f'<tr><td><strong>{html.escape(str(r.ticker))}</strong></td><td>{r.old_weight*100:.2f}%</td><td>{r.new_weight*100:.2f}%</td><td>{(r.new_weight-r.old_weight)*100:+.2f} pp</td><td>₹{r.trade_value_inr/1e7:.2f} Cr</td><td>{r.adv_pct:.2f}%</td><td>{r.participation_pct:.2f}%</td><td>{r.impact_cost_bps:.2f} bps</td><td><span class="alert {'bad' if r.status=='LIQUIDITY RISK' else 'watch' if r.status=='REVIEW' else 'ok'}">{r.status}</span></td></tr>' for _,r in ex.iterrows())
    assets=stress['assets']
    trows=''.join(f'<tr><td>{html.escape(r["name"])}</td><td>{html.escape(r["asset_type"].upper())}</td><td>₹{r["before"]/1e7:.2f} Cr</td><td class="{'neg' if r['pnl']<0 else 'pos'}">₹{r["pnl"]/1e7:+.2f} Cr</td><td>₹{r["after"]/1e7:.2f} Cr</td><td>{r["pnl_pct"]:+.2f}%</td></tr>' for r in assets)
    def chart(g, includejs):return g.to_html(full_html=False,include_plotlyjs='inline' if includejs else False,config={'responsive':True,'displaylogo':False})
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AEGIS / Institutional Risk Workstation</title>
    <style>:root{{--bg:#070c11;--panel:#0c1118;--line:#27313a;--text:#dfe7ee;--muted:#8998a7;--green:#63c89e;--red:#e87079;--amber:#e7bc76}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:13px Arial,Helvetica,sans-serif}}.wrap{{max-width:1740px;margin:auto;padding:20px 24px}}header{{display:flex;justify-content:space-between;gap:16px;align-items:center;padding:8px 0 22px;border-bottom:1px solid var(--line)}}h1{{font-size:22px;margin:0;font-weight:700;letter-spacing:-.6px}}h2{{font-size:12px;letter-spacing:1px;color:#cbd9e5;margin:0 0 12px;text-transform:uppercase}}.small{{color:var(--muted);font-size:11px}}.tag{{border:1px solid #6b4e28;color:#ffd295;background:#2a2014;padding:7px 11px;font:11px Consolas,monospace;white-space:nowrap}}.mono,.metric .value,td{{font-variant-numeric:tabular-nums}}.grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin:14px 0}}.metric,.panel{{border:1px solid var(--line);background:var(--panel)}}.metric{{padding:12px}}.metric .label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px}}.metric .value{{font:21px Consolas,monospace;margin:7px 0 4px}}.metric .foot{{font-size:10px;color:var(--muted)}}.pair{{display:grid;grid-template-columns:1.4fr 1fr;gap:12px;margin:12px 0}}.pair.half{{grid-template-columns:1fr 1fr}}.panel{{padding:14px;min-width:0}}.wide{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:11px}}td,th{{border-bottom:1px solid #1d2831;padding:9px 7px;text-align:right;white-space:nowrap}}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){{text-align:left}}th{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}}.alert{{padding:3px 5px;font-size:10px;border:1px solid #516273;color:#bac9d4}}.alert.bad{{color:#ffb8b9;background:#3f2227;border-color:#743b45}}.alert.watch{{color:#ffd08c;background:#382b1d;border-color:#71522f}}.alert.ok{{color:#8fe7bf;background:#173529;border-color:#285843}}.neg{{color:var(--red)}}.pos{{color:var(--green)}}.disclaimer{{font-size:11px;line-height:1.55;color:#a8b5bd;padding:13px 0 7px;border-top:1px solid var(--line)}}@media(max-width:1020px){{.grid{{grid-template-columns:repeat(3,1fr)}}.pair,.pair.half{{grid-template-columns:1fr}}.wrap{{padding:12px}}}}@media(max-width:600px){{.grid{{grid-template-columns:repeat(2,1fr)}}header{{align-items:start;flex-direction:column}}}}</style></head><body><div class="wrap">
    <header><div><div class="small">RESEARCH DESK  /  RISK & EXECUTION</div><h1>AEGIS<span style="color:#7d9dd7">/</span> SIGNAL X</h1><div class="small">Institutional-style demonstrator &nbsp;·&nbsp; INR book &nbsp;·&nbsp; 10 Oct 2026 software build</div></div><div class="tag">SYNTHETIC SCENARIO · NOT LIVE · NO ORDER ROUTING</div></header>
    <section class="grid">
    <div class="metric"><div class="label">Book NAV</div><div class="value">₹500 Cr</div><div class="foot">Reconciled synthetic holdings</div></div>
    <div class="metric"><div class="label">Signals analysed</div><div class="value">{len(df):,}</div><div class="foot">17 Aug–01 Oct 2026 synthetic</div></div>
    <div class="metric"><div class="label">Latest alert priority</div><div class="value">{df.risk_priority_score.max():.2f}<span class="small"> / 10</span></div><div class="foot">Heuristic, uncalibrated</div></div>
    <div class="metric"><div class="label">Credit shock P&amp;L</div><div class="value {'neg' if stress['portfolio_pnl']<0 else 'pos'}">₹{stress['portfolio_pnl']/1e7:+.2f} Cr</div><div class="foot">Severity 8 / 10 · deterministic</div></div>
    <div class="metric"><div class="label">Est. execution cost</div><div class="value">₹{ex.estimated_cost_inr.sum()/1e5:.2f} L</div><div class="foot">Synthetic TCA, not broker quote</div></div>
    <div class="metric"><div class="label">Vol grid diagnostics</div><div class="value">{'PASS' if diag['passed_grid_checks'] else 'WARN'}</div><div class="foot">Finite checks only; not certification</div></div>
    </section><div class="pair"><div class="panel"><h2>Event & early-risk tape</h2><div class="wide"><table><thead><tr><th>IST</th><th>Ticker</th><th>Event</th><th>Sentiment</th><th>Early risk</th><th>Priority</th><th>Alert</th></tr></thead><tbody>{feed}</tbody></table></div></div><div class="panel"><h2>Risk concentration · 20 issuers</h2>{chart(fr,True)}</div></div>
    <div class="pair half"><div class="panel"><h2>3D implied volatility surface · NIFTY-like simulated quotes</h2>{chart(fvol,False)}</div><div class="panel"><h2>Simulated cash-index market stress tape · 01 Oct 2026</h2>{chart(fa,False)}<div class="small">Index proxy from artificial correlated returns. Not an NSE price history. IV is constructed, not calibrated from options.</div></div></div>
    <div class="pair half"><div class="panel"><h2>Execution-aware tactical rebalance · 10-stock basket</h2><div class="wide"><table><thead><tr><th>Ticker</th><th>Current</th><th>Target</th><th>Δ wt.</th><th>Trade</th><th>% ADV</th><th>POV</th><th>Impact</th><th>Review gate</th></tr></thead><tbody>{erows}</tbody></table></div></div><div class="panel"><h2>Portfolio weight delta · percentage points</h2>{chart(fx,False)}</div></div>
    <div class="panel"><h2>Cross-asset stress decomposed · deterministic credit scenario</h2><div class="wide"><table><thead><tr><th>Asset</th><th>Class</th><th>Before (₹)</th><th>P&amp;L (₹)</th><th>After (₹)</th><th>P&amp;L %</th></tr></thead><tbody>{trows}</tbody></table></div></div>
    <p class="disclaimer">DATA INTEGRITY: All event stories, market quotes, ADV/depth assumptions, implied volatilities, model risk scores and option sensitivities shown here are simulated. No live NSE/BSE market data, L2/L3 feed, external news validation, actual trade fills, regulatory compliance checks or calibrated crash-forecast skill are represented. No broker integration or live execution. Source code and methodology are included in the accompanying project archive.</p>
    </div></body></html>'''
    (O/'desk_snapshot.html').write_text(doc,encoding='utf-8')
    result={'signals':len(df),'unique_issuers':int(df.ticker.nunique()),'iv':diag,'portfolio_aum':NAV,
            'stress_pnl':stress['portfolio_pnl'],'execution_liquidity_reviews':int(ex.status.eq('LIQUIDITY RISK').sum()),
            'anomaly_alarms':int(monitor.warning.eq('ELEVATED').sum()),'not_live':True}
    (O/'desk_build_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':export()
