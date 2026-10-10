# AegisSignal-X — brutal root-cause review and redesign

This document records what was wrong with the first prototype, why those faults matter in finance, and what changed. It is intentionally more useful than saying “use a bigger transformer.”

## 1. The first fusion equation confused credibility with severity

**Old failure.** A report could describe a catastrophic event yet receive a low `impact_score` because it was the first report and therefore had low cross-source independence. Worse, the old prototype mixed credibility, event magnitude and alert priority into one number.

**Fix.** AegisSignal-X has three different quantities:

- `impact_score`: conditional severity — “if this extracted event is real, how damaging could it be?”
- `early_warning_score`: evidence-aware forward warning — provenance, information acceleration, market/network confirmation.
- `risk_priority_score`: operational routing score combining severity, confidence and early warning.

Rumours can therefore have high conditional impact but low priority. Breaking authoritative reports no longer get an artificially tiny impact merely because a second source has not appeared yet.

## 2. The old NLP fallback was too close to a keyword system

**Old failure.** With transformers disabled by default, sentiment and event recognition were dominated by static lexical rules. Such systems are brittle to negation, numbers, causal wording, multi-entity stories and paraphrases.

**Fix.** MAX mode ensembles multiple finance-domain sentiment models, exposes ensemble disagreement, uses entity-local context, adds event stance/direction, and performs multi-label NLI event recognition. The deterministic ontology is retained only as a reproducible offline fallback. The fallback itself uses phrase boundaries, weighted event terms and duplicate-term damping.

## 3. Event type and event direction were conflated

“Regulator closes investigation” and “regulator opens investigation” have the same topic but opposite implications.

**Fix.** Event classification and harmful/beneficial stance are separate models. `Regulatory/Legal` is the event; `event_direction` determines whether the described state is adverse, beneficial or mixed.

## 4. Syndicated copies could masquerade as independent confirmation

**Old failure.** Distinct domains were counted as independent even when they echoed the same upstream story.

**Fix.** Semantic/lexical story-chain clustering produces a `story_cluster_id` and `duplicate_probability`. Corroboration is counted by independent story/domain groups, not raw URL count. Duplicate bursts are explicitly penalized unless genuinely independent evidence appears.

## 5. Cold-start detectors generated fake “bursts” and “changes”

**Old failure.** A first event could receive a non-trivial burst score. BOCPD short-run mass was naturally high at initialization and was incorrectly interpreted as regime change.

**Fix.** The burst detector requires repeat activity; BOCPD has a warm-up period and subtracts expected short-run mass. Page-Hinkley remains as a complementary fast detector.

## 6. The market feature path had a historical look-ahead hazard

**Old failure.** Pulling “current recent data” for an old article can leak future bars into a historical event.

**Fix.** Market bars are explicitly truncated at `published_at`. The provider falls back from intraday to daily only while preserving the event-time cutoff. Each result exposes `timestamp_aligned`.

## 7. “Market stress” was too shallow

Return z-score alone is not market microstructure.

**Fix.** The market channel now contains benchmark abnormal return, volume shock, GARCH-like volatility expansion, drawdown, Barndorff-Nielsen–Shephard bipower-jump proxy, Amihud-like illiquidity, opening-gap anomaly, EVT downside-tail anomaly and a sticky three-state regime filter. A separate `LimitOrderBookStress` adapter accepts real L2/L3-style snapshots or precomputed exchange features and evaluates spread blow-out, depth collapse, order imbalance, OFI, cancel/trade stress, VPIN-like toxicity and a Kyle-lambda proxy.

The free Yahoo connector is explicitly **not** called Level 2/3 data.

## 8. Linear impact fusion could not represent cascades

**Old failure.** A flat weighted sum misses interactions such as a severe semantic shock arriving during a high-volatility/liquidity-stressed regime.

**Fix.** Nonlinear interactions include semantic × market, network × regime, novelty × burst and tail × contagion. The optional learned impact stack uses gradient-boosted quantile heads, a high-impact classifier and split-conformal calibration on a chronological hold-out slice.

## 9. “Graph risk” was missing or superficial

**Fix.** The live dependency-light path has a dynamic weighted graph with sector, known relation and co-mention edges plus repeated diffusion. The MAX/research path includes a trainable **Temporal Graph Attention + GRU** model (`aegis/risk/temporal_gat.py`) that accepts node feature histories and a dynamic graph. It is deliberately not activated without a trained checkpoint; an untrained GNN would be worse than a transparent graph prior.

## 10. Contagion was not self-exciting

**Fix.** A marked Hawkes-style process models decaying self-, same-sector- and broad-market excitation. This is suitable as an online prior. Production should estimate the excitation matrix and decay constants from timestamped event/trade data rather than treating default couplings as universal.

## 11. The stress module was a static shock calculator

**Fix.** Module B now has deterministic event shocks **and** correlated heavy-tailed Monte Carlo with jump mixtures. It reports VaR/CVaR-style loss statistics, probabilities of large portfolio loss, and reverse stress severity. Bond/loan/derivative approximations remain transparent and are clearly labelled as approximations rather than full revaluation.

## 12. The rebalancer treated sentiment as expected return

**Fix.** The NLP score is now explicitly treated as a ranking signal. Module A optimizes a constrained objective with variance, Student-t scenario CVaR, turnover, transaction-cost penalty and position limits. The implementation falls back to a bounded exponentiated update if SLSQP cannot find a feasible solution.

## 13. No credible performance claim was possible

A demo story set is not a benchmark. “It produced a 9/10 alert” is not evidence of predictive ability.

**Fix.** AegisSignal-X ships a leakage-resistant evaluation plan and `scripts/backtest_alerts.py` with PR-AUC, ROC-AUC, Brier score, recall under a false-alert budget and severity error. `TailImpactEnsemble.train` requires chronologically ordered event-study labels and reserves the final slice for conformal calibration.

## What still requires real institutional data

The code can consume these signals, but the repository does not fabricate them:

- exchange-grade order book / trades with synchronized timestamps;
- licensed low-latency news first-seen metadata;
- source ownership / wire redistribution graph;
- counterparty and beneficial-ownership networks;
- supplier/customer graph with time validity;
- funding/repo/liquidity measures;
- options surface and dealer-gamma estimates;
- maritime/satellite/web-traffic alternative data;
- full portfolio greeks, curves and liquidity horizons.

Without those data, no project should claim “HFT-grade” or “perfect crash detection.”

---

## v0.3 institutional interface / data audit

The second audit found that the risk engine had become much richer than its interface. The UI was still exposing raw dataclass fields through default Streamlit tables, which created several operational defects even when the backend values were present:

- **Field-binding ambiguity:** `old_weight` and stress `before` values were present in backend objects, but generic auto-width tables plus the index made the columns visually ambiguous. v0.3 uses explicitly named, formatted columns (`Current`, `Target`, `Baseline`, `Stressed`) and hides the dataframe index.
- **Unit ambiguity:** previous P&L and percentage values depended on the reader inferring currency/units. v0.3 makes INR, crore/lakh, percent, bps and 1–10 scores explicit.
- **Timestamp waste:** raw UTC ISO strings have been removed from the primary tape; UI timestamps are compact `HH:MM:SS` in IST while machine-readable UTC timestamps remain in the API/output files.
- **No preattentive alerting:** alert text was visually homogeneous. v0.3 uses restrained but high-contrast risk coloring, priority bars and loss/positive signal colors.
- **Poor change encoding:** grouped current/target bars required eye switching. v0.3 adds a signed delta chart centered at zero and keeps current/target values in a compact table.
- **No execution reality check:** the optimizer previously emitted weights without showing whether the required trade was large relative to liquidity. v0.3 adds trade notional, ADV%, participation, spread, square-root impact, slippage/all-in bps, estimated cost and liquidity status.
- **Stress-test readability:** generic columns made baseline exposure easy to miss. v0.3 provides baseline/stressed NAV KPIs, tail-risk metrics and asset loss decomposition.
- **Toy geography / scale:** the default demo used a small US/global book. v0.3 defaults to an explicitly synthetic India-focused ₹500 Cr institutional portfolio and 20-company risk universe.
- **No derivatives-state visualization:** v0.3 adds an interactive synthetic NIFTY-like IV surface, ATM term structure, front-expiry smile and downside skew diagnostics. It does not misrepresent synthetic values as live exchange data.

### Execution model used in the demo

The TCA layer is intentionally transparent. For trade value `Q`, ADV and daily volatility `sigma`, the demo uses a square-root temporary-impact proxy plus half-spread and an urgency/participation term:

`impact ~ eta * sigma * sqrt(Q / ADV)`

The dashboard treats this as an execution **risk control**, not a claim of broker-grade fill prediction. A production implementation should calibrate coefficients by venue, side, time-of-day, order type and realised fills, and should reconcile predictions against post-trade TCA.

### Why the IV surface is synthetic

NSE option-chain data and many institutional volatility feeds carry usage/redistribution restrictions. The packaged surface therefore uses a deterministic NIFTY-like smile/term-structure generator. Its purpose is to demonstrate the correct information architecture and desk workflow. A production deployment should ingest licensed bid/ask option quotes, perform no-arbitrage filtering, derive forwards/dividends and fit a desk-approved representation (for example total-variance splines, SVI or SABR depending on product and governance).
