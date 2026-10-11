# AegisSignal Ω v0.5 — Competition research, mathematics, failure modes, and verification

**Research date:** 10 October 2026. **Scope:** S&P Global × CRISIL-style financial NLP risk-engine challenge, plus Module A and Module B. **Status:** Runnable research/hackathon prototype; **not** trained, measured, or certified as a crash predictor or trading execution system. All quoted demo issuer events are fictional and every dashboard quote, surface, and market observation is synthetic.

## 0. Hard conclusion: the competition is about evidence-to-decision, not HFT

The actual brief requires two distinct unstructured text sources; issuer/event-conditioned sentiment in [-1,1]; event classification; impact severity in [1,10]; machine-readable results; and **at least one** downstream consumer. Module A varies mock-index weights with sentiment, while Module B estimates scenario consequences when an event score crosses a threshold. Thus a high-frequency order-book forecaster on its own is **not** a satisfactory submission. A winning technical hypothesis is: **financial risk signals must be corroborated, causally timestamped, tied to exposures, and permitted to abstain rather than hallucinate confidence.**

The strongest feasible entry pairs a sound common baseline (FinBERT, finance-specific event ontology, news/social ingestion, API, optimization and stress testing) with a deeply tested differentiator: *counterfactual publisher independence, contradiction arbitration, risk-exposure mapping, abstention and execution-aware conditional decisions*. This project realizes that functionality in executable Python and an interactive offline institutional decision interface.

## 1. What genuinely differentiates Ω

| Concern | Ordinary hackathon submission | Ω v0.5 behavior | Demonstrated or aspirational? |
|---|---|---|---|
| “Five articles agree” | Count five headlines | Deduplicate by verified-in-demo publisher organization; removal-of-one-group counterfactual | Implemented, simulated publisher groups |
| Contradicted event | Average sentiment over both sides | Keep supporting and opposing independent evidence separately; veto action | Implemented |
| Isolated rumor | Score it and trigger red banner | `VERIFY_SOURCE` and zero execution | Implemented |
| Financial downside | Show an event score or falling price | Reconcile portfolio NAV, apply conditional asset-type shocks and show INR contributions | Implemented, synthetic |
| Systemic propagation | Arbitrary correlations / pretty network | Bounded sector-proxy diffusion with explicit noncausal data-label | Implemented as proxy only |
| Alert thresholds | Absolute sentiment cutoff | Conditional severity × evidence index with independent-source gates and contradiction checks | Heuristic; not empirically calibrated |
| Quant research validation | Random split / accuracy percentage | Time-safe replay, first-seen clock checks, matured-label-only diagnostics, label abstention | Tests implemented; empirical historical dataset absent |
| Execution | Send a simulated buy/sell | Rebalancing with human gate, ADV/time-of-day participation review, no routing | Implemented; no broker |
| Derivatives analytics | Decorate page with IV surface | 3D synthetic term smile plus v0.4 static-arbitrage grid diagnostics, WebGL fallback | Demo, not exchange-derived |
| Historical precedent | Hindsight “similar events” | Matched past cases only when the future outcome of those *past* cases was already observed by current decision time | Implemented, no real history bundled |

### Architecture

```
Legitimate news/feed/social/filing/HTML bytes
       ├─ v0.4 adapters: GDELT, news RSS, Reddit, SEC EDGAR
       └─ v0.4 structural HTML table / key-value extractor
           ↓
AegisRiskEngine: issuer matching, sentiment, financial event class,
  issuer stance, novelty, evidence and regime scores
           ↓
Ω Claim ledger (entity, event, claim-id, publisher group, published_at, first_seen_at)
           ↓
Temporal constraints → organization-aware syndication suppression → contradiction
           ↓
Publisher-removal counterfactual → abstention or human stress review
           ↓
Conditional asset-level scenario + synthetic bounded spillover network
           ↓
Module A: risk-constrained mock index rebalance + execution review
Module B: event-conditioned stress results + exposure reconciliation
           ↓
FastAPI / Streamlit / offline Plotly terminal / JSON evidence journal
```

**Important:** `source_type` and `publisher_group` are supplied as test fixture/HTTP input in Ω. Those fields MUST be validated against a signed ingestion-side registry before production; otherwise an attacker could fabricate apparent independent endorsements.

## 2. Patent intelligence — verified public records, not alleged trade secrets

1. **S&P Global Inc., US20250265305A1, *Method and System for Web Data Extraction Using Meta-Path Graph*.** Published application, not confirmed granted patent. Graph-structured HTML regions and extraction patterns are relevant to page-change resilience, table discovery and source structure. We independently implemented simpler HTML structural extraction and content-change checks in v0.4; Ω does **not** reproduce, assert infringement clearance, or claim their internal production weights. https://patents.google.com/patent/US20250265305A1/en
2. **S&P Global Inc., US20260073020A1, *Large language model for financial news events detection and categorization*.** Published application relevant to event classification, annotated news and retraining. Ω instead implements independent model/publisher gating on top of v0.4 NLP. https://patents.google.com/patent/US20260073020A1/en
3. **Kensho Technologies LLC, US11373244B2, *Searching pre-generated data structures for event impact discovery*.** Granted patent. Discusses fast retrieval of historical event precedents and subsequent market-statistic patterns. Ω contains an independently designed, causal-eligibility-filtered nearest-neighbor *descriptive* baseline. It abstains when the historical cohorts are missing; it does not claim to implement the patent claims. https://patents.google.com/patent/US11373244B2/en
4. **Kensho Technologies LLC, US10726071B2, *Content search engine*.** Granted patent, relevant to structured/faceted entity retrieval, but not evidence of a secret alpha-generation model. https://patents.google.com/patent/US10726071B2/en
5. **Nasdaq SMARTS market-surveillance discussions:** US20230237579A1 describes dynamic monitoring configurations, surveillance alerting and market replay, including background descriptions of Nasdaq OMX SMARTS. This is a surveillance reference, not proof that a specific neural method reliably predicts crashes. https://patents.google.com/patent/US20230237579A1/en

**Defensible interpretation:** patents describe claimed techniques, not production model checkpoints, validation results, internal market advantage, or trade secrets. Competitors cannot be ranked by patent count. There is no verified access to proprietary HFT methods from IMC, Citadel, Jane Street, HRT, Optiver or Tower Research. Their public research and competition datasets are valuable; secret alpha/routing claims would be inventions.

## 3. Proven public quantitative evidence and practical integration

### 3.1 Market microstructure

* **Cont, Kukanov & Stoikov**, *The Price Impact of Order Book Events*, Journal of Financial Econometrics. Finds price changes at short horizons linked to best-level order-flow imbalance with an inverse-depth impact relation. v0.4 implements a finite best-level OFI statistic and participation guardrails. Paper: https://arxiv.org/abs/1011.6402
* **DeepLOB**, Zhang, Zohren, Roberts, IEEE TSP 2019. Convolutional neural networks + LSTMs over limit-order-book data. Relevant only when true chronological exchange L2/L3 observations are licensed, replayable, and properly cross-validated. https://arxiv.org/abs/1808.03668
* **BDLOB**, Bayesian uncertainty over LOB predictions and position-sizing. Useful research motivation for abstention, not a warrant for reliable directional alpha without actual LOB feeds. https://arxiv.org/abs/1811.10041

### 3.2 Financial language, relations, numerical grounding

* **FinBERT** (ProsusAI): financial tone classifier and an appropriate baseline for mandatory sentiment. https://huggingface.co/ProsusAI/finbert
* **FinRED** (2023): finance-specific relations from financial text and earnings transcripts; useful for issuer-target relations rather than generic mention graphs. https://arxiv.org/abs/2306.03736
* **REFinD** (2023): ~29K relation instances and 22 relation categories over real financial documents; useful to evaluate who affects whom. https://arxiv.org/abs/2305.18322
* **FinQA** (EMNLP 2021): executable multistep financial numerical reasoning benchmarks; important for preventing LLMs from misstating magnitudes, percentages, fiscal periods or denominators. https://aclanthology.org/2021.emnlp-main.300/
* **FinGPT** project: alternative open financial instruction model and financial sentiment evaluation framework. Real weights/licensing/resources must be checked independently. https://github.com/AI4Finance-Foundation/FinGPT

**Practical upgrade sequence:** first annotate 300–500 issuer-centric, including sarcasm, denials, factual good-news/negative-news divergence, targets versus acquirers, earnings-versus-guidance, superseding corrections and one-line rumor. Compare lexical/FinBERT/FinGPT/NLI under a strictly held-out date and issuer window; only then fine-tune the best configuration. Do not deploy an LLM “judge” that verifies its own citations.

### 3.3 Uncertainty, calibration and false-alarm quality

* **Xu & Xie**, *Conformal prediction interval for dynamic time-series*, ICML 2021 / EnbPI. Validity results rely on assumptions; indiscriminate exchangeable conformal intervals fail in markets with distribution shifts. https://proceedings.mlr.press/v139/xu21h.html
* **Gibbs & Candès**, *Adaptive Conformal Inference Under Distribution Shift*. Important for adapting coverage frequency under drift; not a magic guarantee for conditional rare-event tail coverage. https://arxiv.org/abs/2106.00170
* **Bailey & López de Prado**, *Deflated Sharpe Ratio*. Search over many models/thresholds produces severe selection bias. Positive hackathon backtests on same crisis windows cannot validate “superiority”. https://doi.org/10.3905/jpm.2014.40.5.094

### 3.4 Contagion and systemic fragility

* **Battiston et al.**, *DebtRank: Too Central to Fail?* Scientific Reports 2012. Financial networks can transmit dispersed shocks; identifying true linkages is crucial. Ω uses sector-proxy diffusion and labels it honestly, rather than passing a correlation graph off as a corporate exposure network. https://www.nature.com/articles/srep00541

### 3.5 What public competition winners *actually* did

* **Optiver Realized Volatility Prediction first-place write-up:** reverse-engineered temporal ordering, nearest-neighbor aggregations, LightGBM, MLP, 1D CNN ensembles. Subsequent author ablation details 5 LightGBMs + 3 CNNs + 3 MLPs. This supports feature validity, sensible ensembles and robust temporal validation, not “deepest model wins.” https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970 and https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/302626
* **Jane Street Real-Time Market Data Forecasting:** sequentially evaluated anonymized multi-feature financial tape; hidden future sample, nonstationarity, tail risk and feature drift matter. No hidden winning model is implied by the dataset. https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data

## 4. Mathematical design — precise claims, limitations and tests

### A. Evidence trust and independence

For a claim, a single publisher group contributes no more than one chosen version. Each group receives a *heuristic*, provenance-tagged contribution:

`w_i = source_type_weight_i × extraction_confidence_i × freshness_i`.

Independent supporting groups are summarized with `S = 1 − product_i (1 − w_i)` and opposing groups analogously as `O`. The **effective evidence index** is `S × (1 − 0.82 × O)`, clipped to [0,1]. This is **not a Bayesian posterior** and must not be represented as the probability that the event is true. Group selection prefers latest publication then higher-authority record for identical observed timestamps. If sources are not independent, the formula still overstates evidence; therefore a signed media-ownership/syndication dependency graph is needed in production.

Counterfactual publisher importance is `Δ_g = S_all − S_without_group_g`. This exposes fragility and “many domains, one wire agency.” A contradiction or a single independent source triggers abstention from portfolio action.

### B. Temporal integrity

`published_at ≤ first_seen_at ≤ decision_at` within a one-minute publication tolerance. Every clock must carry a timezone. Future source records and historical price outcomes are barred from earlier replay. If a system uses GDELT indexing time as publication time, it MUST retain both fields separately and report the inherent freshness lag.

### C. Exposure diffusion

With a nonnegative synthetic row-stochastic weight matrix `W`, impulse `u`, and `0 ≤ α < 1`, iterate:

`x^(k+1) = u + α Wᵀ x^(k)`.

Since `||Wᵀ||_1 = 1`, the total mass is bounded by `||u||_1 / (1 − α)` and the iteration converges. In Ω the matrix is **not factual**: same-sector proxy weight 1, cross-sector proxy 0.06, diagonal 0, normalized. Its output is a risk-exploration map and **must not be interpreted as contractual or inferred supplier/counterparty exposures**. Use actual issuer dependencies, verified credit exposures, disclosures and instrument mappings for deployment.

### D. Conditional stress

For equities a shock is scaled by β and a synthetic direct/sector-factor coefficient. For bonds: `ΔP/P ≈ −D Δy + 0.5 C (Δy)^2 − SD × Δspread`. Loans add simplified `−LGD × ΔPD`. Derivatives use local delta/gamma/vega approximations. These are local, conditional scenarios: no options full re-pricing, nonlinear rate-curve shifts or basis effects are solved. The loss is reconciled asset-by-asset to the ₹500 crore synthetic NAV, and negative signs and currency units appear explicitly.

**Do not multiply conditional loss by an evidence-support heuristic and call the product expected loss.** Ω labels such a number *evidence-scaled loss proxy* only and does not report it as a calibrated expected P&L.

### E. Historical-analogue research baseline

Use only **past historical events with already matured post-event labels**. Filter on event class and sector. Scale decision-time features by the IQR of eligible historical cases with stability floors, calculate robust nearest-neighbor distances, and report the median and p80 historical absolute moves across selected neighbors. This is descriptive retrieval, not causal inference or a validated estimator of the current forward move. Real historical cases are **not bundled**, and the endpoint returns `ABSTAIN` without them.

### F. Sequential controls / false-alert budgets

The prior v0.4 CUSUM uses historical-only rolling robust location and scale, warm-up, and cooldown. It is a useful *change diagnostic*, but **not** an anytime-valid calibrated market-crash p-value. An upgraded live validation should fix an alert threshold **before** held-out evaluation, specify 15m/1h/1d horizons, and measure true alarms per 1,000 issuer-hours, median/95th-percentile lead time, precision-recall AUC, Brier for calibrated event outcomes, downside recall at a fixed false-alarm budget, and *after-cost* decision utility. Report high-confidence ablations, not anecdotes.

### G. Execution-aware Module A

The existing optimizer uses variance/CVaR, turnover caps and simplex/position constraints, plus an estimated ADV/POV liquidity gate. Ω suppresses a rebalance entirely if the event is disputed or insufficiently corroborated and returns `execution_permitted=false` even on a reviewed scenario. Indian equity cash regular-session **09:15–15:30 IST** is correctly 375 minutes, but exchange holidays, pre/post markets, symbol restrictions, impact parameter drift, order queue state, auction windows, taxes and bid/ask dynamics still require authentic data.

## 5. Data sourcing, where real-world advantage can emerge

| Layer | Candidate | What it can establish | Caveat |
|---|---|---|---|
| Global news | GDELT DOC + Context APIs | 15m-scale monitoring and snippet/context | Index latency, syndication, coverage biases; URL ≠ legal text rights |
| Indian disclosures | NSE RSS Corporate Announcements + BSE official channels | Authoritative announcements and first-seen updates | Terms, allowed rate limits, archive/replay and enterprise licensing |
| US disclosures | SEC EDGAR and official RSS feeds | Filed disclosures, institution-grade source hierarchy | Timing, amendments, issuer-to-ticker mapping |
| Short-form chatter | Authorized Reddit or X source | Rumor breadth, social emergence, disagreement | Coordinated manipulation, paid access, ToS, botnets |
| Model benchmark | Financial PhraseBank/FinRED/REFinD/FinQA | Sentiment, entity relation and numerical extraction | Domain mismatch, annotation noise, entity drift |
| Market baseline | Optiver/Kaggle datasets | LOB and volatility forecasting features | Cannot be passed off as Indian live L2/L3 order books |
| Trade-ready features | Licensed NSE/BSE market-data vendor | True depth, trades, quotes, event-time | Paid agreements, market-hours and survivorship controls |

Official NSE RSS: https://www.nseindia.com/static/rss-feed; SEC feeds: https://www.sec.gov/files/about/secrss.shtml; GDELT Context: https://blog.gdeltproject.org/announcing-the-gdelt-context-2-0-api/ .

**Best high-return research investment:** build an immutable issuer-specific *as-of* event dataset: original publication timestamp, first indexed timestamp, first acquired timestamp, source family, copyright/terms, correction chain, independently assessed event truth, and prices/returns at *decision-aligned* horizons. Positive results are not credible without this.

## 6. Red-team review: failure modes likely to sink a competition submission

| Severity | Failure mode | Evidence from existing project / general mechanism | Mitigation | Current status |
|---|---|---|---|---|
| CRITICAL | Web text/market tape labeled live though synthetic | Existing India demo is deterministic | Synthetic watermark and data-origin field on every panel and API | Done for Ω demos; live data not connected |
| CRITICAL | Social duplication counted as confirmation | Numerous article copies from same syndicate | Publisher-root grouping and remove-one-root counterfactual | Implemented and tested with 15 copies |
| CRITICAL | Filing denies a rumor but score still generates sell | Stance and source disagreement | Keep opposition index, require investigation | Implemented and tested |
| CRITICAL | Future labels / market bars inform past | Incorrect join by article date vs first-seen | Strict `first_seen <= decision_time`, outcome maturity | Implemented / tests |
| CRITICAL | Event score mistaken for crash probability | Arbitrary 1–10 severity | Three concepts: conditional severity, evidence support, operational action | Disentangled; not calibrated |
| CRITICAL | Fabricated counterparty graph | Co-mentions assumed supply chain links | Explicit synthetic sector-proxy disclaimer; ingest verified disclosures later | Flagged; true network not built |
| HIGH | Price-impact costs ignore real liquidity | Static ADV and uniform intraday volume | Session-aware POV, no trade approval, licensed depth needed | Review-only |
| HIGH | Negative-P&L denominators and format mismatches | Missing before/after assets, unbounded precision | Reconciled ₹500 Cr NAV, signed ₹Cr and percent | Implemented/tested |
| HIGH | Financial NLP confuses acquirer and target | Pure document sentiment not issuer directed | Existing issuer stance + future REFinD relational testing | Baseline available; external evaluation missing |
| HIGH | Market changes regime | Spreads, variance and source behavior shift after shock | Temporal walk-forward and drift monitoring | Design; no real validation |
| HIGH | Misreported near-perfect crash hit rate | Tiny number of rare crises, threshold search | PR-AUC, false alert budget, lockbox test, DSR/selection correction | Protocol only |
| HIGH | Fake account/domain impersonates issuer | Social spoofing and publisher roots supplied by client | Trusted domain registry, signatures, authenticated fetch | NOT implemented: insecure for production |
| HIGH | Option-volatility surface looks arbitrage-free but isn't | Grid smile diagnostics are necessary not sufficient | Quotes, bid-ask, SVI/SSVI, no-arb calibration | v0.4 finite diagnostic only |
| MEDIUM | Too much information on trading screen | Many competing graphs | Dense decision banner, evidence roots, counterfactuals, NAV exposure, IV at glance | Browser tested |
| MEDIUM | Live environment cannot render WebGL | 3D Plotly depends on WebGL support | Conditional 2D heatmap fallback | Browser tested |
| MEDIUM | API allows caller-supplied provenance | Caller could spoof `publisher_group` | Sign/derive server-side; authorized producer credentials | Outstanding before deployment |
| MEDIUM | Contradiction can refer to separate claim | Entity/event only is too broad | Enforce explicit `claim_id`, separate hypothesis ledger | Implemented for provided IDs; claim coreference model missing |

## 7. Out-of-sample experiment to run before claiming superior risk accuracy

### Fixed targets

For an issuer-specific event at first actionable discovery time `t`, define three separate endpoints:

1. **Extraction**: issuer/event/stance ground truth against annotators' original evidence, macro-F1 and cross-source precision/recall, including contradiction cases.
2. **Risk**: realized abnormal absolute return or variance over 15m, 60m, next session; define baseline using market and sector control models fit strictly before `t`. Include *regime-stratified* 95th-percentile tail-event outcomes.
3. **Decision**: alert lead-time before abnormal moves versus false alerts per issuer-hour, review latency and after-cost utility of a **hypothetical** mock index if every constraint is met. Avoid trading cost-free alpha claims.

### Split strategy

Training: oldest period; validation: subsequent period; locked test: newest period. Purge overlapping forward horizons and enforce embargo across adjacent training/test windows. Treat issuer mergers, changes in company names, amendments and media syndication families as grouped entities. Freeze event labeling and severity thresholds before unsealing the held-out set. Report confidence intervals by *block bootstrap over dates*, not individual articles. Use at least a financial-news tone baseline, class-only baseline, FinBERT baseline and v0.4 baseline before adding Ω.

### Ablations

- Remove independent-publisher dedup → measure rumor false-positive increase.
- Remove contradictions → count adverse rumor actions.
- Remove first-seen constraints → expose score inflation from hindsight.
- Remove sector-proxy graph → test whether it adds genuine held-out risk utility; drop it if not.
- Remove confidence gating → compare alert precision versus missed early events.
- Remove market microstructure → test incremental accuracy only if authentic L2 feed exists.
- Remove historical analogs → measure change in conditional tail-risk calibration only after true history is available.
- Replace deep models with LightGBM + carefully engineered time-safe features → a genuine low-latency challenger.

**Ablations are required, not yet performed on real market data.** This implementation currently passes deterministic integrity/unit tests; numerical tests cannot certify predictive advantage.

## 8. Judge demonstration — 3-minute sequence

* **00:00–00:30**: “Our innovation is not predicting that every negative headline will crash an index. We distinguish source credibility, market severity and actionable risk.” Show required signals and two-source ingestion integrations.
* **00:30–01:15**: Select RELIANCE fictional case: 17 article records, only 3 independent publisher roots; 14 copies suppressed, conditional loss traced to ₹500 Cr multi-asset book. Display source-removal counterfactual.
* **01:15–02:00**: Switch to HDFCBANK: copied unverified social reports do not overrule contradictory disclosure. Action becomes `INVESTIGATE_CONFLICT`. The platform explicitly abstains.
* **02:00–02:25**: Switch to INFY: isolated source -> `VERIFY_SOURCE`. Show time-travel leakage test and independence guard.
* **02:25–02:45**: Return to desk: exposure propagation (synthetic, deliberately labeled), local stress-test P&L, rotatable IV surface with fallback, executable JSON API.
* **02:45–03:00**: Show `pytest` suite and explain missing genuine data, disciplined evaluation plan and no broker routing.

## 9. Deliverables / exact reproducibility

- Core code: `aegis/omega.py`.
- Original NLP, multi-source connectors, both downstream modules: `aegis/` from v0.4, maintained.
- FastAPI: `GET /omega/demo`, `POST /omega/assess`, `POST /omega/analyze-text` (one-source safety).
- Interface: Streamlit **EVIDENCE DECISION LAB** tab in `aegis/dashboard.py`.
- Offline HTML: `outputs/Omega_Decision_Terminal_Offline.html` (self-contained Plotly and local HTML/CSS/JS).
- Deterministic synthetic cases: `scripts/omega_demo.py` produces `outputs/omega_decision_lab.json`.
- Offline HTML rebuild: `python scripts/build_omega_terminal.py`.
- Tests: `tests/test_omega.py` + original tests; **46/46 passed** on this environment, but no prospective price-forecast benchmark.
- Third-party models/data remain subject to their licenses; patents are research context, **not** a freedom-to-operate opinion.

## 10. Prioritized next work (avoid tool vanity)

**P0 (highest scoring improvement):** at least 3 months of real immutable, timestamp-aligned news/filing records and held-out issuer-specific events with return/volatility outcomes. A real 100-event annotated contradiction/syndication set matters more than a larger UI.

**P1:** exact issuer-role relation benchmark (FinRED/REFinD), audited source ownership graph, dynamic HTML tables with corrected/restated numerical values, and numeric document-verification unit tests (FinQA).

**P2:** train calibrated 15-minute/next-session tail-event baselines using LightGBM, feature ablations, calibrated class probabilities, blackout-window labels, special-event exposure. Add online controls and data staleness monitors.

**P3:** licensed L2/L3 order-flow and derivatives trades/quotes; execution cost calibrated to actual spread/depth profiles; arbitrage-constrained volatility surface; authenticated API producers and market-holiday calendar.

**P4 (only if real data supports it):** event-type temporal Hawkes/graph cascade, verified supplier-credit linkages, posterior predictive regimes and robust optimization with uncertainty sets; demonstrate improvements on frozen unseen history before asserting superiority.

## Final claim discipline

A research-quality solution is one that knows when its evidence is insufficient. This system builds a technically distinct decision layer on top of the mandatory risk engine and both downstream modules. The synthetic tests show that code obeys the intended integrity invariants; **they do not show that AegisSignal Ω detects real crises earlier than S&P Global, RavenPack or HFT companies, or achieves real-money profits.**
