# AegisSignal-X

**Research-grade multi-source financial early-risk intelligence engine**

AegisSignal-X turns real-time unstructured financial information into issuer-level, machine-readable risk signals and drives both downstream modules in the case study:

- **Module A — Tactical Index Rebalancing:** constrained variance/CVaR/turnover-aware tactical optimization.
- **Module B — Strategic Portfolio Stress Testing:** event-conditioned deterministic shocks plus correlated heavy-tailed jump Monte Carlo and reverse stress.

The redesign was built after a deliberately harsh audit of the first prototype. The largest conceptual change is that the engine no longer treats “sentiment” as risk or collapse severity, credibility and action priority into one arbitrary score.

> **Scope:** this is a serious hackathon/research prototype using public research, public APIs and open models. It does not claim access to private HFT algorithms, Bloomberg proprietary data/models, exchange Level-3 feeds or a perfect crash predictor. Those claims would be indefensible without proprietary data and locked out-of-sample validation.

---

## Why X is materially different from the first version

The original prototype had several weaknesses that can create deceptively poor or unstable output: lexical fallback NLP, duplicated-news confirmation, cold-start burst/change artifacts, event-time market leakage risk, a mostly additive impact score, no genuine microstructure interface, no trainable graph model and an impact score that mixed severity with source credibility.

AegisSignal-X fixes those failure modes with:

1. **Finance-domain NLP ensemble** — optional FinBERT / financial RoBERTa ensemble, model disagreement, entity-local context.
2. **Multi-event extraction + issuer stance** — event topic is separate from whether the event is harmful or beneficial.
3. **Story-chain provenance** — semantic/lexical duplicate detection, independent-source corroboration and contradiction handling.
4. **Adversarial robustness signals** — low-trust/echo/manipulation and uncertainty features instead of blindly trusting classifier confidence.
5. **Online information dynamics** — Bayesian-smoothed bursts, risk velocity, Page-Hinkley and warm-start-safe BOCPD.
6. **Marked Hawkes contagion** — self/same-sector/broad-market event excitation.
7. **Dynamic risk graph** — sector/relation/co-mention graph diffusion plus an optional trainable Temporal Graph Attention + GRU model.
8. **Event-time market alignment** — bars are truncated at or before article time; no future-bar contamination.
9. **Tail / regime statistics** — abnormal return, volatility expansion, BNS jump proxy, Amihud-like illiquidity, EVT tail score and sticky stress regime.
10. **Optional real L2/L3 adapter** — spread, depth collapse, OFI, cancel/trade stress, VPIN-like flow toxicity and Kyle-lambda proxy. Free retail data are never mislabeled as Level 3.
11. **Dual-score nonlinear risk fusion** — conditional impact, evidence-aware early warning and operational risk priority are separate.
12. **Learned tail-impact stack** — optional q50/q90 gradient boosting + high-impact classifier + chronological split-conformal calibration.
13. **Robust Module A** — variance + heavy-tailed CVaR + turnover + cost + position constraints.
14. **Distributional Module B** — correlated Student-t stress factors + event-severity-dependent jumps + VaR/CVaR-style reporting.
15. **Leakage-resistant evaluation harness** — PR-AUC, Brier, recall at a false-alert budget, severity error and a rigorous walk-forward protocol.

Read the exact old-failure → fix mapping in **`docs/REVIEW_AND_UPGRADE.md`**.

---

## Case-study requirement coverage

| Requirement | Implementation |
|---|---|
| At least two text sources | GDELT news, Reddit; also Google News RSS and SEC EDGAR |
| Sentiment score | Entity-conditioned financial sentiment, `[-1,1]` |
| Event classification | 14-class financial ontology; primary + secondary events |
| Impact score | Conditional severity, `1..10` |
| Structured output | JSON, CSV, SQLite, FastAPI |
| Module A | Robust constrained tactical rebalancing |
| Module B | Event-driven deterministic + stochastic stress testing |
| Visualization | Streamlit + Plotly dashboard |
| Explainability | Channel contributions + evidence strings |
| Real-time readiness | Live-source connectors; online stateful detectors |

---

## Architecture in one line

```text
news/social/filings
  → entity + event + sentiment + stance
  → story provenance + online change + Hawkes
  → market/tail/regime + optional LOB microstructure
  → dynamic graph/systemic propagation
  → severity + early-warning + priority
  → rebalance / stress test / API / dashboard
```

Full diagram: `docs/ARCHITECTURE.md`.

---

# Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
python scripts/demo.py
```

Windows shortcut:

```text
run_demo.bat
```

The deterministic demo writes:

```text
outputs/signals.json
outputs/signals.csv
outputs/rebalance.json
outputs/execution_rebalance.json
outputs/stress_test.json
```

The bundled stories are explicitly **synthetic fixtures**. Their source-quality metadata exists only to exercise provenance logic and must not be interpreted as benchmark performance.

---


## Institutional dashboard (v0.3)

The default dashboard is now a dense India-focused trading/risk workstation rather than a generic hackathon page. It uses an explicit **synthetic institutional demo** label and includes:

- compact IST event/risk tape with conditional formatting and no raw ISO timestamp bloat;
- cross-sectional issuer × risk-channel heatmap;
- risk breadth, information-arrival and concentration panels;
- execution-aware rebalancing with trade notional, %ADV, participation, spread, square-root impact, all-in cost, status and suggested execution style;
- interactive 3D NIFTY-like implied-volatility surface plus 2D heatmap, smile/skew and term-structure diagnostics;
- ₹500 Cr synthetic India portfolio with baseline/stressed NAV, CVaR-style tail metrics and asset-level loss decomposition;
- model/data-health page exposing source quality, disagreement, duplication, manipulation and synthetic microstructure stress.

Run it with:

```bash
python -m streamlit run aegis/dashboard.py
```

Windows shortcut: `run_dashboard.bat`.

The packaged India demo contains **420 synthetic news/event records**, 20 NSE/BSE-style issuers, a dense intraday synthetic microstructure tape, and a 117-point synthetic NIFTY-like IV surface. These are realism-oriented fixtures, not live NSE data. Production deployment should replace them with licensed exchange/broker feeds and calibrated transaction-cost/volatility models.

# MAX NLP / research profile

```bash
pip install -r requirements-max.txt
```

PowerShell:

```powershell
$env:AEGIS_USE_TRANSFORMERS="true"
$env:AEGIS_MAX_PROFILE="true"
python scripts/demo.py
```

Linux/macOS:

```bash
export AEGIS_USE_TRANSFORMERS=true
export AEGIS_MAX_PROFILE=true
python scripts/demo.py
```

Default MAX components include:

- `ProsusAI/finbert`
- `yiyanghkust/finbert-tone`
- `soleimanian/financial-roberta-large-sentiment`
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- sentence-transformer semantic story clustering

For CUDA, set `AEGIS_MODEL_DEVICE=0`.

Model licenses and provider terms remain with the original authors/providers.

---

# Run API

```bash
uvicorn aegis.api:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

Important endpoints:

- `POST /analyze`
- `POST /ingest`
- `GET /signals`
- `POST /rebalance`
- `POST /stress-test`
- `POST /pipeline`

Example:

```json
{
  "source": "manual",
  "title": "Issuer warns of disruption",
  "text": "NVIDIA warns that a supplier disruption may constrain shipments next quarter.",
  "source_domain": "example.com",
  "market_context": false
}
```

Optional licensed/external microstructure features can be supplied in `metadata`:

```json
{
  "metadata": {
    "microstructure": {
      "spread_z": 4.2,
      "depth_collapse": 0.67,
      "order_imbalance": -0.81,
      "ofi_z": 3.4,
      "cancel_trade_stress": 0.74,
      "vpin_like": 0.69,
      "kyle_lambda_z": 2.8
    }
  }
}
```

In a hardened production deployment, client-supplied `source_quality` should not be trusted; the demo supports it only for controlled fixtures.

---

# Live source example

```bash
python scripts/live_poll.py "NVIDIA OR NVDA" --sources gdelt,reddit --limit 20
```

With free market context:

```bash
python scripts/live_poll.py "JPMorgan OR JPM" --sources gdelt,rss --market
```

Available connectors:

- GDELT
- Reddit
- Google News RSS
- SEC EDGAR filing text

Connectors fail explicitly if a source is unavailable; missing data are not fabricated.

---

# The three scores you should not confuse

## 1. `impact_score` — 1 to 10

Conditional severity:

> **If the extracted event is true, how material could it be?**

A severe but unconfirmed bank-run rumour can therefore have a high impact score.

## 2. `early_warning_score` — 0 to 1

Evidence-aware warning signal driven by provenance, information acceleration, network effects and market/liquidity confirmation.

It is **not a calibrated probability** until trained/calibrated on historical event studies.

## 3. `risk_priority_score` — 1 to 10

Operational routing score combining severity, confidence, early-warning evidence and market/network confirmation. `alert_level` is based on this score so a dramatic low-quality rumour does not automatically become a critical alert.

---

# Example structured output

```json
{
  "ticker": "MSFT",
  "sentiment_score": -0.69,
  "event_classification": "Cybersecurity",
  "event_direction": -0.91,
  "impact_score": 7.3,
  "early_warning_score": 0.34,
  "risk_priority_score": 5.2,
  "alert_level": "ELEVATED",
  "story_cluster_id": "...",
  "corroboration_score": 0.0,
  "source_quality": 0.93,
  "hawkes_score": 0.08,
  "graph_contagion_score": 0.10,
  "systemic_resonance": 0.88,
  "market_stress": 0.0,
  "microstructure_stress": 0.0,
  "tail_risk_score": 0.0,
  "model_confidence": 0.74
}
```

Numbers vary with state, source sequence and model profile. This is a schema illustration, not a real Microsoft claim.

---

# Module A — Tactical Index Rebalancing

The engine aggregates recent issuer signals with recency decay. Tactical optimization conceptually solves

```text
min_w  -mu'w
       + lambda_v * w'Σw
       + lambda_c * CVaR_90(loss)
       + transaction_cost * |w-w0|
```

subject to:

- weights sum to 1;
- min/max position limits;
- turnover cap;
- case-study directional semantics where feasible.

Student-t scenarios make the risk penalty heavier-tailed than a Gaussian optimizer. NLP is treated as a ranking signal — **not** a guaranteed expected-return estimate.

---

# Module B — Strategic Portfolio Stress Testing

An event maps to shocks in:

- equity prices;
- interest rates;
- credit spreads;
- FX;
- volatility;
- default probability.

The engine produces:

- deterministic pre/post portfolio value;
- P&L by instrument;
- 2,500-simulation correlated Student-t + jump distribution by default;
- 95%/99% VaR-style loss;
- 95%/99% CVaR-style expected shortfall;
- probability of >5% and >10% loss;
- reverse-stress impact required to breach a target loss.

This remains a transparent hackathon stress engine, not a bank-grade full revaluation stack.

---

# Optional Temporal GAT

`aegis/risk/temporal_gat.py` implements a dense multi-head graph-attention encoder followed by a per-node GRU risk head without requiring PyTorch Geometric.

Expected training data:

```text
X: [samples, time, nodes, features]
A: [nodes,nodes] or dynamic adjacency
Y: [samples,nodes] future risk labels
```

Train:

```bash
python scripts/train_temporal_gat.py path/to/data.npz --out models/temporal_gat.pt
```

The live engine intentionally keeps graph diffusion as its fallback. Activating an untrained GNN would make the project look more sophisticated while making it worse.

---

# Train the tail-impact model correctly

Prepare chronologically ordered historical events with all feature columns in `aegis/risk/impact_model.py` plus:

```text
target_impact
```

Optional `timestamp` is used for sorting by the training script.

```bash
python scripts/train_impact_model.py data/labeled_events.csv --out models/impact_tail.joblib
```

Then:

```bash
export AEGIS_IMPACT_MODEL_PATH=models/impact_tail.joblib
export AEGIS_IMPACT_TAIL_BLEND=0.45
```

The learned stack fits q50/q90 conditional-severity heads, a high-impact classifier when data permit, and reserves the final chronological slice for split-conformal calibration.

---

# Evaluate instead of guessing performance

```bash
python scripts/backtest_alerts.py path/to/out_of_sample_scores.csv \
  --score risk_priority_score \
  --threshold 7
```

Expected label column:

```text
realized_stress
```

The evaluator can report:

- PR-AUC;
- ROC-AUC;
- Brier score;
- recall under a specified false-alert budget;
- threshold precision/recall/FPR;
- severity MAE/Spearman when `realized_severity` exists.

The full leakage-resistant protocol is in `docs/EVALUATION.md`.

---

# Project tree

```text
AegisSignal_X/
├── aegis/
│   ├── api.py
│   ├── dashboard.py
│   ├── engine.py
│   ├── schemas.py
│   ├── ingest/
│   │   ├── gdelt.py
│   │   ├── reddit.py
│   │   ├── rss.py
│   │   ├── edgar.py
│   │   └── sample.py
│   ├── nlp/
│   │   ├── entity.py
│   │   ├── sentiment.py
│   │   ├── events.py
│   │   ├── stance.py
│   │   └── novelty.py
│   ├── risk/
│   │   ├── fusion.py
│   │   ├── market.py
│   │   ├── microstructure.py
│   │   ├── tail.py
│   │   ├── regime.py
│   │   ├── online.py
│   │   ├── bocpd.py
│   │   ├── hawkes.py
│   │   ├── graph.py
│   │   ├── temporal_gat.py
│   │   ├── systemic.py
│   │   ├── adversarial.py
│   │   └── impact_model.py
│   └── downstream/
│       ├── rebalance.py
│       └── stress.py
├── data/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── REVIEW_AND_UPGRADE.md
│   ├── RESEARCH_AND_MATH.md
│   ├── FAILURE_MODES.md
│   ├── EVALUATION.md
│   └── REFERENCES.md
├── scripts/
│   ├── demo.py
│   ├── live_poll.py
│   ├── train_impact_model.py
│   ├── train_temporal_gat.py
│   └── backtest_alerts.py
└── tests/
```

---

# Research basis

The design is grounded in public work including:

- FinBERT / finance-domain transformers;
- BloombergGPT / FinGPT / FinMA as evidence for domain specialization;
- FIRE financial relation extraction;
- BIS machine-learning work on predicting the **distribution/tails** of financial-market stress;
- BIS AI market-monitoring work combining large market feature sets with LLM-based contextual explanation;
- ECB CISS-style systemic breadth;
- Hawkes processes in high-frequency finance;
- BNS jump variation;
- EVT/POT tail modelling;
- DeepLOB/TLOB microstructure modelling;
- dynamic graph risk models;
- conformalized quantile regression;
- public patent disclosures on relevance, novelty, source, volume, company-specific sentiment and structured financial-event extraction.

See `docs/REFERENCES.md` and `docs/RESEARCH_AND_MATH.md`.

---

# Important limitations

No public hackathon repository can honestly promise the “slightest crash” will always be caught. Some events are endogenous, data arrive late, liquidity can disappear abruptly, and false positives have real costs.

The scientifically defensible target is:

> **maximize stress-event recall and lead time subject to a controlled false-alert budget, with calibrated uncertainty and strict out-of-sample timestamps.**

Also intentionally **not** included:

- claimed proprietary/hidden HFT secrets;
- executive “stress biometrics” from voice;
- invented Level-3 data;
- a pretrained impact model fitted on the synthetic demo;
- fake accuracy/backtest numbers.

That restraint is a feature, not a missing component.
