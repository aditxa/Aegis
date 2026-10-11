from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path
import math

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError as e:
    raise SystemExit("Install dashboard extras: pip install streamlit plotly pandas") from e

from aegis.engine import AegisRiskEngine
from aegis.ingest import SampleConnector
from aegis.downstream import TacticalRebalancer, StressTester
from aegis.institutional_plus import (surface_diagnostics, sequential_risk_monitor,
    validated_portfolio, execution_gate)
from aegis.institutional import (
    INDIA_LIQUIDITY,
    execution_estimate,
    synthetic_iv_surface,
    synthetic_market_tape,
)

ROOT = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30))

# -----------------------------------------------------------------------------
# Page / visual system
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AegisSignal X | Risk & Execution", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      :root {
        --bg:#071017; --panel:#0b151d; --panel2:#101c25; --line:#1d2c36;
        --text:#dce5eb; --muted:#82929e; --cyan:#5cc8d7; --green:#4fd1a1;
        --amber:#f2b84b; --red:#ff6678; --blue:#74a7ff;
      }
      html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: Arial, "Helvetica Neue", sans-serif !important;
      }
      [data-testid="stAppViewContainer"] { background: var(--bg); color:var(--text); }
      [data-testid="stHeader"] { background:rgba(7,16,23,.92); height:34px; }
      .block-container { padding: .65rem 1.15rem 1.8rem 1.15rem; max-width: 100%; }
      h1,h2,h3 { letter-spacing:-0.025em; }
      h1 { font-size:1.55rem !important; margin:.1rem 0 .1rem 0 !important; font-weight:650 !important; }
      h2 { font-size:1.03rem !important; margin:.6rem 0 .2rem 0 !important; font-weight:650 !important; }
      h3 { font-size:.90rem !important; }
      p, label, .stCaption { color:var(--muted) !important; }
      .terminal-title { display:flex; align-items:center; gap:.55rem; margin-bottom:.05rem; }
      .brandmark { width:22px; height:22px; border:1px solid #36525e; display:grid; place-items:center; color:#77dbe7; font-weight:800; }
      .status-chip { display:inline-block; padding:2px 6px; border:1px solid #2d4653; border-radius:3px; color:#aab9c3; font-size:10px; letter-spacing:.08em; text-transform:uppercase; }
      .status-chip.demo { color:#f2b84b; border-color:#705727; background:#241c0d; }
      .subline { font-size:11px; color:#71838f; margin-top:1px; }
      .market-strip { display:grid; grid-template-columns:repeat(8,minmax(0,1fr)); gap:1px; background:#1b2a34; border:1px solid #1b2a34; margin:.55rem 0 .65rem 0; }
      .quote { background:#0b151d; padding:7px 9px; min-height:53px; }
      .quote .qname { font-size:9px; color:#788994; letter-spacing:.06em; text-transform:uppercase; }
      .quote .qval { font-family:Consolas,"SFMono-Regular",monospace; font-size:15px; color:#e8eef2; margin-top:1px; }
      .quote .qchg { font-family:Consolas,"SFMono-Regular",monospace; font-size:10px; }
      .up{color:#4fd1a1}.down{color:#ff6678}.warn{color:#f2b84b}.cyan{color:#5cc8d7}
      .kpi-grid { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:7px; margin:.25rem 0 .55rem 0; }
      .kpi { background:#0b151d; border:1px solid #1d2c36; padding:8px 10px; border-radius:3px; }
      .kpi .label { color:#748691; font-size:9px; text-transform:uppercase; letter-spacing:.06em; }
      .kpi .value { font-family:Consolas,"SFMono-Regular",monospace; color:#e5edf2; font-size:18px; margin-top:2px; }
      .kpi .foot { font-size:9px; color:#667984; margin-top:1px; }
      div[data-testid="stDataFrame"] { border:1px solid #1d2c36; border-radius:3px; }
      div[data-baseweb="tab-list"] { gap:0px; border-bottom:1px solid #1d2c36; }
      button[data-baseweb="tab"] { padding:7px 13px !important; font-size:11px !important; color:#8fa0ab !important; }
      button[data-baseweb="tab"][aria-selected="true"] { color:#dce5eb !important; border-bottom-color:#5cc8d7 !important; }
      [data-testid="stMetric"] { background:#0b151d; border:1px solid #1d2c36; padding:7px 9px; border-radius:3px; }
      [data-testid="stMetricLabel"] { font-size:9px; text-transform:uppercase; letter-spacing:.04em; }
      [data-testid="stMetricValue"] { font-family:Consolas,"SFMono-Regular",monospace; font-size:18px; }
      .stSelectbox, .stSlider { font-size:11px; }
      hr { border-color:#1d2c36 !important; }
      .fineprint { font-size:9px; color:#61737f; line-height:1.35; }
      @media (max-width: 1000px) { .market-strip{grid-template-columns:repeat(4,1fr)} .kpi-grid{grid-template-columns:repeat(3,1fr)} }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT = dict(
    paper_bgcolor="#071017",
    plot_bgcolor="#071017",
    font=dict(family="Arial, Helvetica Neue", color="#93a4ae", size=10),
    margin=dict(l=35, r=15, t=30, b=35),
)
GRID = "#1a2a34"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def inr(v: float, compact: bool = True) -> str:
    sign = "-" if v < 0 else ""
    x = abs(float(v))
    if compact:
        if x >= 1e9: return f"{sign}₹{x/1e7:,.1f} Cr"
        if x >= 1e7: return f"{sign}₹{x/1e7:,.2f} Cr"
        if x >= 1e5: return f"{sign}₹{x/1e5:,.1f} L"
    return f"{sign}₹{x:,.0f}"


def pct(v: float, n: int = 2) -> str:
    return f"{float(v):.{n}f}%"


def bps(v: float, n: int = 1) -> str:
    return f"{float(v):.{n}f} bps"


def kpi(label: str, value: str, foot: str = "", cls: str = "") -> str:
    return f'<div class="kpi"><div class="label">{label}</div><div class="value {cls}">{value}</div><div class="foot">{foot}</div></div>'


def _alert_css(v: str) -> str:
    m = {"LOW":"color:#81909a", "WATCH":"color:#f2b84b;font-weight:700", "ELEVATED":"color:#ff9f5d;font-weight:700", "HIGH":"color:#ff6678;font-weight:800", "CRITICAL":"background:#7c1f31;color:white;font-weight:800"}
    return m.get(str(v), "")


def _sent_css(v: float) -> str:
    if v <= -0.45: return "color:#ff6678"
    if v >= 0.45: return "color:#4fd1a1"
    return "color:#aab8c0"


def _priority_css(v: float) -> str:
    if v >= 7: return "background-color:#421925;color:#ff8a99;font-weight:700"
    if v >= 5: return "background-color:#3b2c14;color:#f4c464;font-weight:700"
    return ""


@st.cache_data(show_spinner=False)
def load_signals(limit: int = 2400) -> pd.DataFrame:
    engine = AegisRiskEngine(ROOT / "data" / "india_companies.csv", use_market_context=False)
    items = SampleConnector(ROOT / "data" / "india_sample_articles.jsonl").fetch(limit=limit)
    signals = engine.analyze(items)
    frame = pd.DataFrame([x.to_dict() for x in signals])
    return frame


@st.cache_resource(show_spinner=False)
def get_rebalance_engine() -> AegisRiskEngine:
    return AegisRiskEngine(ROOT / "data" / "india_companies.csv", use_market_context=False)


@st.cache_data(show_spinner=False)
def rebalance_payload() -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = get_rebalance_engine()
    # Rehydrate state from the same dense demo feed once; cache prevents rerun on widgets.
    if not any(engine.recent.values()):
        items = SampleConnector(ROOT / "data" / "india_sample_articles.jsonl").fetch(limit=2400)
        engine.analyze(items)
    current = {
        "RELIANCE":.14, "HDFCBANK":.13, "ICICIBANK":.115, "INFY":.10, "TCS":.095,
        "BHARTIARTL":.09, "SBIN":.085, "LT":.08, "ITC":.075, "AXISBANK":.09,
    }
    agg = engine.aggregate_for_rebalancer(list(current))
    rows = TacticalRebalancer(min_w=.04, max_w=.17, turnover_limit=.12, transaction_cost=.0020).rebalance(current, agg)
    rdf = pd.DataFrame([r.to_dict() for r in rows])
    aum = 5_000_000_000.0
    ex = []
    for _, r in rdf.iterrows():
        trade = (r["new_weight"] - r["old_weight"]) * aum
        est = execution_estimate(r["ticker"], trade, "BUY" if trade > 0 else "SELL" if trade < 0 else "FLAT")
        ex.append({**r.to_dict(), **est})
    return rdf, pd.DataFrame(ex)


# -----------------------------------------------------------------------------
# Header and market strip
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="terminal-title"><div class="brandmark">A</div><div>
      <h1>AegisSignal X</h1>
      <div class="subline">Financial risk intelligence · event surveillance · liquidity-aware execution · portfolio stress</div>
    </div><span class="status-chip">IST</span></div>
    """,
    unsafe_allow_html=True,
)

# Market context is simulated from the same time-stamped tape, never presented as live NSE data.
from_demo = synthetic_market_tape(list(INDIA_LIQUIDITY)[:14], periods=75)
demo_by_time = from_demo.groupby('timestamp').agg({'return_bps':'mean','realized_vol_pct':'mean','liquidity_stress':'mean','spread_bps':'mean'}).sort_index()
demo_index = 25000.0 * np.exp(demo_by_time['return_bps'].cumsum().to_numpy() / 1e4)
demo_index_change = 100*(demo_index[-1]/demo_index[0]-1.0)
demo_risk = float(demo_by_time.liquidity_stress.iloc[-1])
demo_spread = float(from_demo.spread_bps.quantile(.95))
demo_iv = synthetic_iv_surface().query('tenor_days == 30 and moneyness == 1.0').iv_pct.iloc[0]
demo_max_dd = float(np.min(demo_index/np.maximum.accumulate(demo_index)-1)*100)
demo_stamp = from_demo.timestamp.max().strftime('%d %b %Y, %H:%M IST')
quotes = [
    ('INDEX PROXY', f'{demo_index[-1]:,.2f}', f'{demo_index_change:+.2f}% simulated', 'up' if demo_index_change>=0 else 'down'),
    ('SYNTH ATM IV', f'{demo_iv:.2f}%', '30-day simulated', 'warn'),
    ('SYNTH RV', f'{demo_by_time.realized_vol_pct.iloc[-1]:.2f}%', 'realized-vol proxy', 'cyan'),
    ('SYNTH SPREAD P95', f'{demo_spread:.1f} bp', '14 demo securities', 'warn'),
    ('LIQUIDITY STRESS', f'{demo_risk:.2f}', 'unitless 0–1', 'warn'),
    ('DEMO MAX DD', f'{demo_max_dd:+.2f}%', 'session drawdown', 'down'),
    ('PORTFOLIO AUM', '₹500 Cr', 'synthetic multi-asset book', 'cyan'),
    ('DATA AS OF', '01 OCT 2026', 'NOT LIVE · SIM', 'warn'),
]
st.caption(f"Scenario clock: {demo_stamp} · Source: deterministic research simulation · Not a live exchange feed")
st.markdown("<div class='market-strip'>" + "".join(
    f"<div class='quote'><div class='qname'>{n}</div><div class='qval'>{v}</div><div class='qchg {c}'>{d}</div></div>" for n,v,d,c in quotes
) + "</div>", unsafe_allow_html=True)

with st.spinner("Building dense risk state…"):
    df = load_signals(2400)

if df.empty:
    st.error("No signals available.")
    st.stop()

df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
latest = df["ts"].max()
critical_count = int(df["alert_level"].isin(["HIGH","CRITICAL"]).sum())
elevated_count = int(df["alert_level"].isin(["ELEVATED","HIGH","CRITICAL"]).sum())
negative_share = float((df["sentiment_score"] < -0.10).mean() * 100)
max_priority = float(df["risk_priority_score"].max())

st.markdown("<div class='kpi-grid'>" + "".join([
    kpi("Signals", f"{len(df):,}", "multi-session simulated feed"),
    kpi("Elevated+", f"{elevated_count}", f"{critical_count} high/critical", "warn" if elevated_count else ""),
    kpi("Max impact", f"{df['impact_score'].max():.2f}/10", "conditional severity"),
    kpi("Max priority", f"{max_priority:.2f}/10", "evidence-gated urgency", "warn" if max_priority >= 5 else ""),
    kpi("Negative tape", f"{negative_share:.1f}%", "sentiment < -0.10"),
    kpi("Sources", f"{df['source_domain'].nunique()}", latest.strftime("updated %H:%M:%S IST")),
]) + "</div>", unsafe_allow_html=True)

risk_tab, omega_tab, exec_tab, vol_tab, stress_tab, health_tab = st.tabs([
    "RISK MONITOR", "EVIDENCE DECISION LAB", "EXECUTION / REBALANCE", "VOLATILITY SURFACE", "STRESS LAB", "MODEL / DATA HEALTH"
])

# -----------------------------------------------------------------------------
# Risk monitor
# -----------------------------------------------------------------------------
with risk_tab:
    left, right = st.columns([1.42, .58], gap="small")
    with left:
        st.subheader("Event & risk tape")
        show = df.sort_values("ts", ascending=False).copy().head(120)
        tape = pd.DataFrame({
            "Time": show["ts"].dt.strftime("%H:%M:%S"),
            "Ticker": show["ticker"],
            "Event": show["event_classification"],
            "Sent": show["sentiment_score"].round(2),
            "Impact": show["impact_score"].round(2),
            "EW": show["early_warning_score"].round(2),
            "Priority": show["risk_priority_score"].round(2),
            "Conf": show["model_confidence"].round(2),
            "Alert": show["alert_level"],
            "Source": show["source_domain"],
        })
        styled = tape.style.map(_alert_css, subset=["Alert"]).map(_sent_css, subset=["Sent"]).map(_priority_css, subset=["Priority"])
        st.dataframe(
            styled, hide_index=True, use_container_width=True, height=430,
            column_config={
                "Time": st.column_config.TextColumn(width="small"),
                "Ticker": st.column_config.TextColumn(width="small"),
                "Event": st.column_config.TextColumn(width="medium"),
                "Sent": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Impact": st.column_config.NumberColumn(format="%.2f", width="small"),
                "EW": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Priority": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.2f", width="small"),
                "Conf": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Alert": st.column_config.TextColumn(width="small"),
            },
        )

    with right:
        st.subheader("Cross-sectional risk heatmap")
        dims = ["impact_score","early_warning_score","market_stress","microstructure_stress","tail_risk_score","hawkes_score","graph_contagion_score"]
        labels = ["Impact","Early warn","Market","Microstruct","Tail","Hawkes","Graph"]
        latest_by_t = df.sort_values("ts").groupby("ticker", as_index=False).tail(1)
        top_t = df.groupby("ticker")["risk_priority_score"].max().sort_values(ascending=False).head(14).index
        hm = latest_by_t[latest_by_t.ticker.isin(top_t)].set_index("ticker").reindex(top_t)
        z = hm[dims].to_numpy(float)
        # Bring impact to a 0..1 scale for comparability.
        z[:,0] = (z[:,0] - 1.0) / 9.0
        fig = go.Figure(go.Heatmap(z=z, x=labels, y=hm.index.tolist(), zmin=0, zmax=1, colorscale="RdYlGn_r", colorbar=dict(title="risk", thickness=8)))
        fig.update_layout(**{**PLOT, 'height': 430, 'margin': dict(l=70, r=20, t=10, b=35)})        
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    c1, c2 = st.columns([1.15,.85], gap="small")
    with c1:
        st.subheader("Risk breadth & information arrival")
        t15 = df.copy()
        t15["bucket"] = t15["ts"].dt.floor("15min")
        agg = t15.groupby("bucket").agg(
            signals=("item_id","count"),
            avg_priority=("risk_priority_score","mean"),
            high_impact=("impact_score", lambda x: int((x >= 7).sum())),
            avg_burst=("burst_score","mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg.bucket, y=agg.signals, name="signals", marker_color="#34596d", yaxis="y"))
        fig.add_trace(go.Scatter(x=agg.bucket, y=agg.avg_priority, name="avg priority", line=dict(color="#f2b84b", width=1.5), yaxis="y2"))
        fig.add_trace(go.Scatter(x=agg.bucket, y=agg.high_impact, name="impact≥7", line=dict(color="#ff6678", width=1.3), yaxis="y"))
        fig.update_layout(**PLOT, height=265, legend=dict(orientation="h", y=1.12, x=0), yaxis=dict(title="events", gridcolor=GRID), yaxis2=dict(title="priority", overlaying="y", side="right", range=[0,10], showgrid=False))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.subheader("Risk concentration")
        conc = df.groupby("ticker").agg(priority=("risk_priority_score","max"), impact=("impact_score","max"), alerts=("item_id","count")).sort_values("priority", ascending=True).tail(12)
        fig = go.Figure(go.Bar(x=conc.priority, y=conc.index, orientation="h", marker=dict(color=conc.priority, colorscale="YlOrRd", cmin=1, cmax=10), text=conc.impact.round(1), textposition="inside"))
        fig.update_layout(**PLOT, height=265, xaxis=dict(title="max priority", range=[0,10], gridcolor=GRID), yaxis=dict(title=""), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# -----------------------------------------------------------------------------
# Execution / rebalance
# -----------------------------------------------------------------------------
with exec_tab:
    rdf, exdf = rebalance_payload()
    turnover = 0.5 * float((rdf.new_weight - rdf.old_weight).abs().sum())
    gross_trade = float(exdf.trade_value_inr.abs().sum())
    est_cost = float(exdf.estimated_cost_inr.sum())
    at_risk = int(exdf.status.isin(["REVIEW", "LIQUIDITY RISK"]).sum())
    if at_risk: st.warning(f"{at_risk} proposed trades require human liquidity review. Simulation is NOT an order-routing approval.")
    st.markdown("<div class='kpi-grid'>" + "".join([
        kpi("AUM", "₹500.0 Cr", "institutional demo book"),
        kpi("One-way turnover", f"{turnover*100:.2f}%", "optimizer constrained"),
        kpi("Gross trade", inr(gross_trade), "absolute notional"),
        kpi("Est. execution cost", inr(est_cost), f"{1e4*est_cost/max(gross_trade,1):.1f} bps of traded notional"),
        kpi("Liquidity reviews", str(max(0,at_risk)), "ADV / participation controls", "warn" if at_risk else ""),
        kpi("Names", str(len(exdf)), "tactical basket"),
    ]) + "</div>", unsafe_allow_html=True)

    st.subheader("Execution-aware tactical rebalance")
    exec_view = pd.DataFrame({
        "Ticker": exdf.ticker,
        "Current": 100*exdf.old_weight,
        "Target": 100*exdf.new_weight,
        "Δ wt": 100*exdf.delta,
        "Side": exdf.side,
        "Trade ₹Cr": exdf.trade_value_inr / 1e7,
        "% ADV": exdf.adv_pct,
        "Part.%": exdf.participation_pct,
        "Spread": exdf.spread_bps,
        "Impact": exdf.impact_cost_bps,
        "All-in": exdf.all_in_cost_bps,
        "Cost ₹L": exdf.estimated_cost_inr / 1e5,
        "POV breach": exdf.guardrail.apply(lambda g: "YES" if g["breaches"] else "NO"),
        "Sent": exdf.sentiment,
        "Risk": exdf.impact,
        "Status": exdf.status,
        "Style": exdf.execution_style,
    })
    def status_style(v):
        return {"LIQUIDITY RISK":"background:#571c27;color:#ffabb5;font-weight:700", "REVIEW":"background:#443414;color:#ffd178;font-weight:700", "MODEL PASS":"color:#63d6aa", "NO TRADE":"color:#71838f"}.get(str(v), "")
    ev = exec_view.style.map(status_style, subset=["Status"]).map(_sent_css, subset=["Sent"])
    st.dataframe(ev, hide_index=True, use_container_width=True, height=365, column_config={
        "Current": st.column_config.NumberColumn(format="%.2f%%"),
        "Target": st.column_config.NumberColumn(format="%.2f%%"),
        "Δ wt": st.column_config.NumberColumn(format="%+.2f%%"),
        "Trade ₹Cr": st.column_config.NumberColumn(format="%+.2f"),
        "% ADV": st.column_config.NumberColumn(format="%.2f%%"),
        "Part.%": st.column_config.NumberColumn(format="%.1f%%"),
        "Spread": st.column_config.NumberColumn(format="%.1f bps"),
        "Impact": st.column_config.NumberColumn(format="%.1f bps"),
        "All-in": st.column_config.NumberColumn(format="%.1f bps"),
        "Cost ₹L": st.column_config.NumberColumn(format="%.2f"),
        "Sent": st.column_config.NumberColumn(format="%+.2f"),
        "Risk": st.column_config.NumberColumn(format="%.2f"),
    })

    a, b = st.columns([1.1,.9], gap="small")
    with a:
        st.subheader("Target change — compact delta view")
        plot = exec_view.sort_values("Δ wt")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot["Δ wt"], y=plot.Ticker, orientation="h",
            marker_color=np.where(plot["Δ wt"] >= 0, "#4fd1a1", "#ff6678"),
            text=[f"{x:+.2f}%" for x in plot["Δ wt"]], textposition="outside",
        ))
        fig.update_layout(**PLOT, height=330, xaxis=dict(title="weight change (percentage points)", zeroline=True, zerolinecolor="#52636e", gridcolor=GRID), yaxis=dict(title=""), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with b:
        st.subheader("Market-impact / liquidity map")
        fig = go.Figure(go.Scatter(
            x=exec_view["% ADV"], y=exec_view["All-in"], mode="markers+text",
            text=exec_view.Ticker, textposition="top center",
            marker=dict(size=np.clip(exec_view["Trade ₹Cr"].abs()*2.6+8,8,34), color=exec_view["Risk"], colorscale="YlOrRd", cmin=1,cmax=10, line=dict(width=.5,color="#071017")),
            hovertemplate="%{text}<br>%ADV=%{x:.2f}%<br>all-in=%{y:.1f} bps<extra></extra>",
        ))
        fig.add_vline(x=.12*100, line_dash="dot", line_color="#ff6678", annotation_text="12% ADV")
        fig.update_layout(**PLOT, height=330, xaxis=dict(title="trade as % of ADV", gridcolor=GRID), yaxis=dict(title="estimated all-in cost (bps)", gridcolor=GRID))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown("<div class='fineprint'>Execution analytics use a transparent square-root impact simulator plus spread/participation terms. Values are synthetic and for architecture demonstration; a production deployment should replace these inputs with broker/exchange TCA, venue-specific fees, live order-book depth and calibrated impact coefficients.</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Volatility surface
# -----------------------------------------------------------------------------
with vol_tab:
    v1, v2, v3, v4 = st.columns(4)
    stress_level = v1.slider("Risk shock", 0.0, 1.0, 0.35, 0.05, help="Synthetic event-to-IV stress bump")
    spot = v2.number_input("NIFTY spot", 15000.0, 40000.0, 25000.0, 50.0)
    view_mode = v3.selectbox("Surface view", ["3D IV surface", "2D heatmap"], index=0)
    surf = synthetic_iv_surface(spot=spot, stress=stress_level)
    iv_atm = surf.loc[(surf.tenor_days==30)&(surf.moneyness==1.0),'iv_pct'].iloc[0]
    v4.metric("Synthetic 30D ATM IV", f"{iv_atm:.2f}%", "scenario, not a market quote")
    quality = surface_diagnostics(surf,spot)
    if quality['passed_grid_checks']:
        st.caption('Grid checks: no detected discrete butterfly, vertical or total-variance calendar violations · not a full no-arbitrage certification')
    else:
        st.warning(f"Surface quality check failed: {quality['butterfly_violations']} butterfly, {quality['vertical_violations']} vertical, {quality['calendar_total_variance_violations']} calendar warnings. Do not use for pricing.")
    pivot = surf.pivot(index="tenor_days", columns="strike", values="iv_pct")
    if view_mode == "3D IV surface":
        fig = go.Figure(data=[go.Surface(
            x=pivot.columns.to_numpy(), y=pivot.index.to_numpy(), z=pivot.to_numpy(),
            colorscale="Turbo", colorbar=dict(title="IV %", thickness=10),
            contours={"z":{"show":True,"usecolormap":True,"highlightcolor":"#ffffff","project_z":True}},
            hovertemplate="Strike %{x:.0f}<br>Tenor %{y:.0f}d<br>IV %{z:.2f}%<extra></extra>",
        )])
        fig.update_layout(
            **PLOT, height=585,
            scene=dict(
                bgcolor="#071017",
                xaxis=dict(title="Strike", gridcolor=GRID, backgroundcolor="#071017"),
                yaxis=dict(title="Days to expiry", gridcolor=GRID, backgroundcolor="#071017"),
                zaxis=dict(title="Implied volatility (%)", gridcolor=GRID, backgroundcolor="#071017"),
                camera=dict(eye=dict(x=1.55,y=-1.6,z=.92)),
            ),
        )
    else:
        fig = go.Figure(go.Heatmap(x=pivot.columns, y=pivot.index, z=pivot.values, colorscale="Turbo", colorbar=dict(title="IV %")))
        fig.update_layout(**PLOT, height=585, xaxis=dict(title="Strike"), yaxis=dict(title="Days to expiry"))
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3, gap="small")
    atm = surf.iloc[(surf.moneyness-1).abs().argsort()[:9]].sort_values("tenor_days")
    front = surf[surf.tenor_days == surf.tenor_days.min()].sort_values("strike")
    put25 = surf[surf.moneyness <= .90].groupby("tenor_days").iv_pct.mean()
    call25 = surf[surf.moneyness >= 1.10].groupby("tenor_days").iv_pct.mean()
    skew = (put25 - call25).rename("put-call skew")
    with c1:
        st.subheader("ATM term structure")
        f = go.Figure(go.Scatter(x=atm.tenor_days, y=atm.iv_pct, mode="lines+markers", line=dict(color="#5cc8d7")))
        f.update_layout(**PLOT,height=235,xaxis=dict(title="days",gridcolor=GRID),yaxis=dict(title="IV %",gridcolor=GRID))
        st.plotly_chart(f,use_container_width=True,config={"displayModeBar":False})
    with c2:
        st.subheader("Front-expiry smile / skew")
        f=go.Figure(go.Scatter(x=front.strike,y=front.iv_pct,mode="lines+markers",line=dict(color="#f2b84b")))
        f.add_vline(x=spot,line_dash="dot",line_color="#637581")
        f.update_layout(**PLOT,height=235,xaxis=dict(title="strike",gridcolor=GRID),yaxis=dict(title="IV %",gridcolor=GRID))
        st.plotly_chart(f,use_container_width=True,config={"displayModeBar":False})
    with c3:
        st.subheader("Downside skew by tenor")
        f=go.Figure(go.Bar(x=skew.index,y=skew.values,marker_color="#9f76d4"))
        f.update_layout(**PLOT,height=235,xaxis=dict(title="days",gridcolor=GRID),yaxis=dict(title="put wing − call wing (vol pts)",gridcolor=GRID))
        st.plotly_chart(f,use_container_width=True,config={"displayModeBar":False})
    st.markdown("<div class='fineprint'>The volatility surface is a clearly labelled synthetic NIFTY-like demo surface generated in log-moneyness / maturity space. A production desk should use licensed option-chain quotes, forward/dividend curves, robust no-arbitrage filtering and a calibrated surface such as SVI/SABR or spline-in-total-variance depending on desk methodology.</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Stress lab
# -----------------------------------------------------------------------------
with stress_tab:
    portfolio = pd.read_csv(ROOT / "data" / "india_portfolio.csv").fillna(0).to_dict("records")
    validated_portfolio(portfolio,5_000_000_000)
    tester = StressTester(ROOT / "data" / "shock_library.json")
    q1,q2,q3 = st.columns([1.0,1.0,2.2])
    event = q1.selectbox("Scenario", sorted(tester.shocks.keys()), index=sorted(tester.shocks.keys()).index("Credit Event"))
    impact = q2.slider("Severity", 1.0, 10.0, 8.0, .1)
    result = tester.run(portfolio,event,impact)
    d = result.get("distributional_risk",{})
    st.markdown("<div class='kpi-grid'>" + "".join([
        kpi("Baseline NAV", inr(result["portfolio_before"]), "pre-shock exposure"),
        kpi("Stressed NAV", inr(result["portfolio_after"]), "deterministic scenario"),
        kpi("Scenario P&L", inr(result["portfolio_pnl"]), pct(result["portfolio_pnl_pct"]), "down" if result["portfolio_pnl"] < 0 else "up"),
        kpi("95% CVaR", inr(d.get("cvar95_loss",0)), "heavy-tail Monte Carlo"),
        kpi("99% CVaR", inr(d.get("cvar99_loss",0)), "tail loss"),
        kpi("P(loss >5%)", pct(100*d.get("prob_loss_gt_5pct",0),1), f"jump p={100*d.get('jump_probability',0):.1f}%"),
    ]) + "</div>", unsafe_allow_html=True)

    assets = pd.DataFrame(result["assets"])
    stress_view = pd.DataFrame({
        "Asset": assets.name,
        "Type": assets.asset_type.str.upper(),
        "Baseline": assets.before,
        "P&L": assets.pnl,
        "Stressed": assets.after,
        "P&L %": assets.pnl_pct,
    })
    def pnl_style(v): return "color:#ff6678" if float(v)<0 else "color:#4fd1a1" if float(v)>0 else ""
    sv=stress_view.style.map(pnl_style,subset=["P&L","P&L %"])
    a,b=st.columns([1.22,.78],gap="small")
    with a:
        st.subheader("Asset-level stress decomposition")
        st.dataframe(sv,hide_index=True,use_container_width=True,height=355,column_config={
            "Baseline":st.column_config.NumberColumn(format="₹%.0f"),
            "P&L":st.column_config.NumberColumn(format="₹%+.0f"),
            "Stressed":st.column_config.NumberColumn(format="₹%.0f"),
            "P&L %":st.column_config.NumberColumn(format="%+.2f%%"),
        })
    with b:
        st.subheader("Loss contribution")
        p = stress_view.sort_values("P&L")
        fig=go.Figure(go.Bar(x=p["P&L"]/1e7,y=p.Asset,orientation="h",marker_color=np.where(p["P&L"]<0,"#ff6678","#4fd1a1")))
        fig.update_layout(**PLOT,height=355,xaxis=dict(title="P&L (₹ Cr)",gridcolor=GRID),yaxis=dict(title=""),showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    st.subheader("Scenario shock vector")
    shock = pd.DataFrame([{"Factor":k,"Shock":v} for k,v in result["shock"].items()])
    st.dataframe(shock,hide_index=True,use_container_width=True,height=205)

# -----------------------------------------------------------------------------
# Model / data health
# -----------------------------------------------------------------------------
with health_tab:
    tape = synthetic_market_tape(list(INDIA_LIQUIDITY)[:14], periods=75)
    anomaly = sequential_risk_monitor(tape,window=20)
    alarms = anomaly.loc[anomaly.warning == "ELEVATED"]
    st.caption(f"Causal simulated anomaly monitor: {len(alarms)} alarms across {len(tape):,} rows; no predictive accuracy has been measured. Warmup/cooldown enforced.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk records", f"{len(df):,}")
    c2.metric("Market tape rows", f"{len(tape):,}")
    c3.metric("Synthetic news sources", f"{df.source_domain.nunique()}")
    c4.metric("Data label", "SYNTHETIC", "no live execution")
    a,b=st.columns([1.05,.95],gap="small")
    with a:
        st.subheader("Intraday microstructure stress")
        mt = tape.groupby("timestamp").agg(spread=("spread_bps","median"), liquidity=("liquidity_stress","mean"), vol=("realized_vol_pct","median")).reset_index()
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=mt.timestamp,y=mt.spread,name="median spread (bps)",line=dict(color="#f2b84b")))
        fig.add_trace(go.Scatter(x=mt.timestamp,y=10*mt.liquidity,name="liquidity stress ×10",line=dict(color="#ff6678")))
        fig.update_layout(**PLOT,height=320,xaxis=dict(gridcolor=GRID),yaxis=dict(gridcolor=GRID),legend=dict(orientation="h",y=1.12))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with b:
        st.subheader("Source / model diagnostics")
        health = pd.DataFrame({
            "Metric":["Median model confidence","Median source quality","Median novelty","Mean disagreement","Mean contradiction","Mean manipulation","Mean duplicate probability"],
            "Value":[df.model_confidence.median(),df.source_quality.median(),df.novelty_score.median(),df.sentiment_disagreement.mean(),df.contradiction_score.mean(),df.manipulation_score.mean(),df.duplicate_probability.mean()],
        })
        st.dataframe(health,hide_index=True,use_container_width=True,height=320,column_config={"Value":st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.3f")})
    st.markdown("""
    <div class='fineprint'>
    <b>Production gaps intentionally exposed rather than hidden:</b> this demo has no licensed live NSE/BSE feed, no broker order-management connection, no real Level-2/3 depth, no live derivatives chain, no calibrated venue-specific transaction-cost model, no position/P&L reconciliation service, no entitlement layer, no kill-switch workflow, and no audited model-risk / change-control process. Those belong in the production integration layer rather than being faked in a hackathon UI.
    </div>
    """,unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Evidence decision lab. Real pipeline supports one-source abstention and a
# deterministic multi-source attack/corroboration fixture for judged demos.
# -----------------------------------------------------------------------------
with omega_tab:
    from scripts.omega_demo import build as omega_build
    with st.spinner('Checking independent evidence, reversals and exposures…'):
        omega=omega_build()
    st.info('SYNTHETIC DECISION LAB — all stories, dates, weights and scenarios are educational fixtures. The source evidence score is NOT a measured probability of truth.')
    cases={c['ticker']:c for c in omega['cases']}
    sel=st.selectbox('Investigated risk hypothesis',list(cases),format_func=lambda s:f"{s} / {cases[s]['event_type']}",key='omega_case')
    c=cases[sel];state=c['state']; dec=c['decision'];p=c['portfolio']
    colors={'REVIEW_STRESS':'#ff7788','INVESTIGATE_CONFLICT':'#f4b45c','VERIFY_SOURCE':'#f4b45c',
            'MONITOR':'#4bcbab','COLLECT_EVIDENCE':'#8698a4'}
    tagcolor=colors.get(dec['action'],'#b3c6cf')
    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #28404e;padding:8px 0'><span style='font-size:15px;color:#dbe8ed;font-weight:700'>DECISION INTELLIGENCE | {sel}</span><span style='color:{tagcolor};border:1px solid {tagcolor};padding:5px 8px;font:12px Consolas,monospace'>{dec['action'].replace('_',' ')}</span></div>",unsafe_allow_html=True)
    st.markdown("<div class='kpi-grid'>"+''.join([
        kpi('Support index',f"{state['support_index']:.2f}",'heuristic / not calibrated'),
        kpi('Opposition',f"{state['opposing_index']:.2f}",'contradictory independent sources'),
        kpi('Independent support',str(state['independent_supporters']),'publisher groups'),
        kpi('Copies suppressed',str(sum(x['syndicated_records']-1 for x in state['publishers'])),'syndications deduplicated'),
        kpi('Scenario P&L',inr(p['conditional_pnl_inr']),f"{p['conditional_pnl_pct']:+.2f}% NAV; not a forecast",'down'),
        kpi('Evidence fragility',f"{c['adversarial']['largest_publisher_dependence']:.2f}",'remove top publisher group'),
    ])+'</div>',unsafe_allow_html=True)
    l,r=st.columns([1.12,.88],gap='small')
with l:
    st.subheader('Provenance — independent evidence only')
    ed = pd.DataFrame(state['publishers'])
    st.dataframe(
        ed[['publisher_group', 'stance', 'source', 'weight',
            'syndicated_records', 'headline']],
        hide_index=True,
        use_container_width=True,
        height=275,
        column_config={
            'weight': st.column_config.NumberColumn(
                'Strength', format='%.2f'
            )
        }
    )

    st.subheader('Counterfactual: remove one publishing group')
    vals = c['adversarial']['evidence_actions']
    cf = pd.DataFrame(vals)

    fig = go.Figure(go.Bar(
        x=cf.delta_support,
        y=cf.publisher_group,
        orientation='h',
        marker_color=[
            '#ff7788' if x > 0 else '#4bcbab'
            for x in cf.delta_support
        ]
    ))

    fig.update_layout(**PLOT)
    fig.update_layout(
        height=225,
        margin=dict(l=120, r=15, t=12, b=28),
        xaxis=dict(
            title='Change in support index',
            gridcolor=GRID
        ),
        yaxis=dict(autorange='reversed')
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={'displayModeBar': False}
    )

with r:
    st.subheader('Exposure proxy network')
    gr = c.get('diffusion')

    if gr:
        gl = pd.DataFrame(gr['propagated']).head(12).iloc[::-1]

        fig = go.Figure(go.Bar(
            x=gl.score,
            y=gl.ticker,
            orientation='h',
            marker_color=[
                '#ff7788' if d else '#638c9e'
                for d in gl.direct
            ]
        ))

        fig.update_layout(**PLOT)
        fig.update_layout(
            height=300,
            margin=dict(l=90, r=10, t=8, b=30),
            xaxis=dict(
                title='Synthetic diffusion intensity',
                gridcolor=GRID
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

        st.caption(
            'Network edges are synthetic sector-proxy relationships, '
            'NOT established credit/supplier linkages.'
        )

    st.subheader('Asset-class contribution to conditional P&L')

    fd = (
        pd.DataFrame(p['assets'])
        .groupby('asset_type', as_index=False)
        .conditional_pnl.sum()
        .sort_values('conditional_pnl')
    )

    fd['pnl_cr'] = fd.conditional_pnl / 1e7

    st.dataframe(
        fd[['asset_type', 'pnl_cr']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'pnl_cr': st.column_config.NumberColumn(
                '₹ crore', format='%+.2f'
            )
        }
    )

st.caption(
    'ACTION POLICY · VERIFY_SOURCE / INVESTIGATE_CONFLICT means abstain. '
    'REVIEW_STRESS means scenario assessment only; no routing. '
    'Impact scores and downside scenarios are not calibrated '
    'market return forecasts.'
)