# AegisSignal X v0.4 — Institutional Financial Early-Warning Research, IP Landscape and Critical Audit

**Research snapshot: 10 October 2026.** This is an auditable research prototype, not a representation of internal S&P Global, Citadel, Jane Street, IMC, Tower Research, HRT or Optiver software. Public patents describe claims and selected examples, not production source code or measurable trading alpha. Patent grant status and assignee ownership require independent legal verification. Nothing below establishes a freedom-to-operate opinion.

## Executive decision

The strongest viable product is **an event-time-causal, provenance-aware incident-to-position early-warning system**, not a purported perfect crash detector. The core workflow is: discover credible first-seen issuer events, extract who did what to whom, estimate direction and impact separately, cross-check spreads/depth/flow with market changes, propagate exposure through issuer/supplier/credit graphs, then display model uncertainty, P&L scenarios and estimated execution feasibility. Its defensibility must come from legally acquired data, outcomes matched on receipt timestamps, superior calibration and operational execution, not a claim to proprietary intelligence.

### Patent and competitor map (public disclosures only)

| Holder / organization | Public patent or evidence | Disclosed technical contribution | Independent development application | Caveat |
|---|---|---|---|---|
| S&P Global Inc. (Akshay Sehgal co-inventor) | [US20250265305A1](https://patents.google.com/patent/US20250265305A1/en) | Web HTML graph, meta-path structural clustering, semantic changes | Structural financial filing/table differencing with provenance | Published US **application**, listed pending at research time; no claim this prototype implements its claims |
| S&P Global Inc. | [US20260073020A1](https://patents.google.com/patent/US20260073020A1/en) | Two-stage financial news categorization/pseudo-label retraining | Event taxonomy, human adjudication, weak labels with leakage checks | Published US application; self-training may amplify label error |
| S&P Global Inc. | [US20230367800A1](https://patents.google.com/patent/US20230367800A1/en) / [US12306862B2](https://patents.google.com/patent/US12306862B2/en) | Text extraction from unstructured documents | Extract grounded sections and numeric risk drivers from filings | Examine granted claims separately, don't copy them |
| S&P Global Inc. | [US20230153331A1](https://patents.google.com/patent/US20230153331A1/en) | Entity identity matching and similarity ranking | Ticker/issuer/parent-alias resolution | Incorrect entity joins can invent systemic events |
| Nasdaq | [US12260335B2](https://patents.google.com/patent/US12260335B2/en) | Generate training datasets for retraining market-surveillance ML | Analyst-in-the-loop feedback and anti-manipulation drift monitoring | **Surveillance**, not demonstrated crash prediction |
| Thomson Reuters (historical) | [WO2012012623A1](https://patents.google.com/patent/WO2012012623A1/en) | Credit risk mining from text and financial variables | Cross-modal text + balance-sheet predictive risk | PCT publication; not proof of active granted worldwide rights |
| BlackRock Aladdin Risk | [Official Aladdin Risk](https://www.blackrock.com/aladdin/platforms/products/aladdin-risk) | Integrated positions, cross-asset risk factors, scenarios, governance | Position-first screen, full portfolio reconciliations, factor shocks | Different product objective from first-seen web risk |
| Kensho / S&P Global | [Kensho](https://kensho.com/) | Public-facing NLP, data extraction, enterprise AI | Robust extraction and ID-linked event ingestion | Proprietary internal implementations not public |
| Bloomberg | [BloombergGPT paper](https://arxiv.org/abs/2303.17564) | Domain language modeling for finance | Train/evaluate finance tasks; don't confuse language accuracy with alpha | 50B parameters research model not open replication of trading systems |
| Tower Research connection | [Cont–Kukanov–Stoikov OFI paper](https://arxiv.org/abs/1011.6402) | Order-book imbalance explains short-term changes with inverse-depth price impact | Independently implemented best-level OFI on supplied L1/L2 snapshots | Co-author's affiliation is public, **not** assertion Tower deployed this exact system |

**Who is better than S&P?** There is no established universal ranking. BlackRock Aladdin discloses a deeper integrated portfolio/factor/governance workflow than a pure text-extraction patent. Nasdaq discloses market-surveillance learning techniques. Bloomberg has documented financial NLP research. Specialist HFTs are likely optimized for ultra-low-latency execution, but their internal proprietary mechanisms and comparative live accuracy are not publicly measurable. Judge each along **data latency, licensed coverage, PnL hedging horizon, risk-control quality, recall at false-alert budget, and fees**, rather than a fabricated global ranking.

### Research, models, code and datasets to reproduce comparisons

| Public resource | Application / caution |
|---|---|
| [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | Financial sentiment baseline; train issuer-relative direction and abstentions separately |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | Open financial NLP/data processing implementations; review dataset licensing and date vintage |
| [DeepLOB](https://arxiv.org/abs/1808.03668) | CNN-LSTM LOB architecture; test only with correctly timestamped genuine LOB historical depth |
| [FinBERT original](https://arxiv.org/abs/1908.10063) | Financial pretrained NLP baseline |
| [OFI price-impact](https://arxiv.org/abs/1011.6402) | Derive imbalance and depth-scaled impact, do not infer future shocks without calibration |
| [Hawkes and market microstructure](https://arxiv.org/abs/1502.04592) | Conditional arrival intensity for clusters; self-excitation also from duplicate reporting |
| [SRISK systemic measurement](https://vlab.stern.nyu.edu/srisk) | Estimate systemic-capital shortfalls where equity/volatility/leverage data exist |
| [Arbitrage-free SVI](https://arxiv.org/abs/1204.0646) | Fit actual option quotes with convex, calendar no-arb constraints before highlighting surface anomalies |
| [BIS credit-to-GDP gap data](https://data.bis.org/topics/CREDIT_GAPS) | Macro early-warning risk at quarters/years, distinct from intraday alerts |
| [SEC companyfacts](https://www.sec.gov/edgar/sec-api-documentation) | US structured filings; EDGAR fair-access constraints apply |
| [GDELT](https://www.gdeltproject.org/) | Public global news discovery; news time and publication != machine first-seen time |
| [Hugging Face financial sentiment datasets](https://huggingface.co/datasets?search=financial%20sentiment) | Label reference only; benchmark version, licenses and duplication matter |
| [Kaggle financial sentiment datasets](https://www.kaggle.com/datasets?search=financial+sentiment) | Research training; often no trusted first-seen/stock return labels; never treat random splits as backtest |

### Quantitative research roadmap

**1. First-seen event intelligence.** Preserve `(source, canonical_url, first_seen_utc, published_utc, fetched_utc, payload_hash, issuer_id, revision_id, exchange_session)` and strip boilerplate before extraction. Store each revised disclosure as immutable evidence, and use versioned issuer IDs. Join events to market data **available no later than first_seen_utc**; never use publication timestamps as a substitute for feed capture, and never predict from data revised after decision time.

**2. Entity-conditioned multi-label event modeling.** Jointly predict `(event class, actor, target, beneficiary, negation, materiality, uncertainty, verified provenance, time horizon)`, not just polarity. E.g. a cyber incident can hurt a bank and benefit a cybersecurity contractor. Teacher-student pseudo-labels demand independent human gold labels, abstention, balanced rare-class sampling and minority class stress tests. Let a manually disputed fact stay in review instead of converting misinformation into confidence.

**3. Probabilistic target.** Define for each issuer `i` and horizon `h`, `Y_{i,t,h} = 1[minimum future risk-adjusted abnormal return <= -threshold]` or a similarly precisely specified liquidity/default event. Future returns are **labels only**, not input features. Model `P(Y=1 | information received by t)` plus loss conditional on Y and epistemic/aleatoric uncertainty; calibrate per issuer group, event family, liquidity and regime. Audit ranked alert recall at fixed false alarms per trading day. Early-warning velocity alone is **not** a validated forecast.

**4. Point-process + change detection.** Model true independent incident intensity as `lambda_i(t)=mu_i(t) + sum_{j:t_j<t} a_{type(j),i} exp[-beta(t-t_j)]`. Source-family echo clusters must be downweighted. Use prior-only robust median/MAD, one-sided CUSUM, BOCPD and cooldown with explicit cold-start states. Trigger fused evidence only after independent provenance and persistence tests; bursting unverified stories should increase uncertainty instead of confidence.

**5. Order book / execution.** For bid/ask level-1 prices and displayed sizes, calculate

`OFI_t = 1[b_t>=b_{t-1}]q^b_t - 1[b_t<=b_{t-1}]q^b_{t-1} - 1[a_t<=a_{t-1}]q^a_t + 1[a_t>=a_{t-1}]q^a_{t-1}`.

Short-term price impact often empirically increases with OFI/depth, but calibrate by sector, phase, tick size and quote staleness. For INR notional `Q`, `ADV` (daily **traded value**), a hypothetical flat intraday volume curve, `h` remaining minutes, model `ADV% = 100*|Q|/ADV`, `POV% = 100*|Q|/(ADV*h/375)`. True volume curves are U-shaped/heterogeneous, and a displayed order book is **not** firm executable liquidity. Pre-trade gate refuses to auto-route, marks review if POV, ADV cap or 10bps depth is exceeded. Costs can start with `half_spread + eta*sigma*sqrt(Q/ADV) + urgency + fees`; estimate eta from filled orders, not by intuition.

**6. Volatility surface.** Compute mid/bid/ask options and compare implied volatility in `(log-moneyness, tenor)` space. Fit SVI/SSVI using clean quotes, NSE expiry conventions, forward prices, dividends, day count, holiday calendar and wide/illiquid quote filtering. For call prices, test vertical monotonicity, butterfly convexity, calendar spreads and arbitrage after bid/ask, **not just aesthetics of a plotted surface**. `surface_diagnostics` in this prototype performs only necessary discrete checks and cannot certify no-arbitrage. Model 3D surface in Plotly `go.Surface` and expose smile, skew, term structure and liquidity.

**7. Cross-asset risk.** Compute `NAV_t=sum_i units_i*price_i*FX_i + cash - liabilities`; reconcile every scenario leg with instrument currency, contract multiplier, collateral, corporate actions and valuation source. Price equity loss, bond duration/convexity, credit-spread sensitivity and derivatives Greeks in the correct units; replace small-shock Greeks with full repricing for discontinuous jump/gap scenarios. Model heavy tails using filtered historical returns/Student-t copulas and regime scenarios. Report VaR and expected shortfall `ES_alpha = E[L | L >= VaR_alpha]` under clear confidence and horizon, but avoid VaR-only controls. Model systemic capital stress separately (e.g. SRISK / leverage / credit funding pressure), not as a sum of news-negative scores.

**8. Stress propagation.** Represent issuer-credit-supplier-parent co-movement graph with edge **type** and lag. Learn `h_{i,t} = f(h_{i,t-1}, entity_events_{<=t}, calibrated market states_{<=t}, graph_t)`; prohibit train/test company-link leakage. Test whether graph improves out-of-time alerts after controlling for sector beta. Graph diffusion can easily create false contagion via weak co-mentions.

### Failure-oriented evaluation protocol (must be completed before real use)

- **Data vintage:** record true ingestion first-seen; watermark sequence gaps; audit publish vs wire vs API vs exchange release delays and edits. Prevent timezone/session and corporate-action future knowledge.
- **Label creation:** historical incident + forward liquidity/return target labels **as of historical time**; overlap embargo at least forecast horizon; leakage-free rolling-origin splits by calendar and issuer groups, not random train/test news rows.
- **Class imbalance:** PR-AUC, Brier/calibration, 90/95% conformal interval coverage under drift, precision/recall by bucket, false alarms per issuer/day, time-to-detection and adversarial robustness; compare simple keyword baseline, FinBERT, fusion without LOB, fusion without graph and fully proposed model.
- **Liquidity:** reconcile trades versus audited official ADV/mid/closing prices, timestamp volume curve, L2 queue age/cancel dynamics, market pauses and per-venue lot/price restrictions; conservative fallback to **NO ORDER ROUTING**.
- **Adversarial:** prompt injection in filings, duplicated news, fake website, bots, delayed correction, impersonated issuer, mixed stakeholder direction, ambiguous tickers, English/Indic code-switching, PDF tables, inaccessible JavaScript, blocked/captcha domains, look-ahead and synthetic gap stress.
- **Operational:** broker/OMS exchange memberships and SEBI compliance (if deployed in India), security reviews, PII controls, auth/RBAC, license entitlements, per-source robots/ToS, replayable append-only audit log, circuit breaker, kill switch, isolation between research and trading, alert escalation and human approval.
- **Pilot:** 60–90 trading-day paper-only pilot, independent prospective performance capture and formal review by quant researcher, trader, execution desk, ML risk and compliance before any execution integration.

### Brutal independent code audit

| Original v0.3 failure | v0.4 status | What still fails under real finance conditions |
|---|---|---|
| Hardcoded displayed market headline quotes | Removed hardcoded strip in favor of derived **clearly synthetic** features | Not licensed live Indian market data |
| 390-minute horizon in NSE-like impact | Corrected to 375-minute standard cash session | No official exchange holiday, auction, partial-day calendars; intraday volume curves remain flat |
| Old/new rebalance weights and scenario baseline ambiguous | Weight delta, current/target, `before`, ₹ and % visible in HTML/Streamlit; arithmetic guards tested | True exposures, instruments and contractual notional not verified |
| No data validation for optional structural crawling | New bounded supplied-HTML parser, table and KV diff | Rule-based parsing will fail on dynamic JS/PDF/split tables, cannot claim S&P meta-path implementation |
| Alert cold start and look-ahead risk | New warm-start-safe, prior-only CUSUM with regression prefix test | The alarm counts are **not** calibrated false alarms or proven predictive power |
| Execution represented as executable | Renamed status MODEL PASS; hard liquidity gate requires review | Still not broker quote, cash reconciliation and official lot/tick sizes |
| 3D volatility plot without data quality | Added discrete call monotonicity, convexity, calendar total-variance diagnostics | Simulated grid deliberately smoothed; real smiles require bid/ask-consistent arbitrage-free fitting |
| Few synthetic stories | 2,400 events, 20 symbols, across weekday sessions, reproducible | Generated language templates are low-diversity, neither authentic events nor financial outcomes |
| Untrusted metadata source quality | Client-supplied quality ignored except explicitly synthetic fixtures | Full source authentication, signed notices and provenance graph still missing |
| Query ignored by sample connector | Explicit sample query filter, unit tested | Does not prove externally connected sources are reliable |
| Browser-based view difficult to share | Self-contained HTML snapshot with Plotly 3D, tables and exports | Static, frozen offline view; live Streamlit/API require environment and feeds |

### Test and release scope

- **Unit suite:** 20 tests PASS in local container (older tests plus 8 new tests). Full model GPU transformer inference and Streamlit browser UX **not executed** here.
- **Synthetic 2,400 events / 20 issuers**, 34 *weekday* sessions, 117 option IV points, 10-name tactical basket, ₹5,000,000,000 synthetic NAV. Events and prices are artificial; sample dates may include exchange holidays.
- `scripts/build_desk_snapshot.py` generates offline `outputs/desk_snapshot.html`, `desk_signals.csv`, `desk_iv_surface.csv`, `desk_execution.csv`, `desk_anomaly_monitor.csv`, `desk_stress.json`.
- On the deterministic test fixture: 3 tactical names flagged for liquidity review; 122 CUSUM alarms across a simulated multi-name time series; these values are **not** return-prediction performance metrics.
- No official current exchange prices, no real options chains, no exchange L2/L3 market data, no trading firm proprietary code, no operational alpha backtest, no auditable real-time alert precision or evidence of market crash prediction.

### Priority deployment sequence

1. **P0 data lineage**: only real, licensed ingestion with first-seen, dedupe, identity, historical vintages, secure audit log.
2. **P1 labels and holdouts**: event outcome truth set and prospective benchmark of fixed alarm budget.
3. **P1 correct IV/execution**: actual option chain and full cash-session calendar, broker TCA, order restrictions, kill switches.
4. **P2 trained event graph + LOB**: only after ablations show reliable incremental value over dumb baselines.
5. **P3 governed pilot**: conduct documented trader usability sessions, model approval, staged non-routing dry run then licensed integration with humans in loop.
