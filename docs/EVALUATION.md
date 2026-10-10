# Institutional-style evaluation protocol

No synthetic demonstration can establish predictive performance. The project should be judged on **out-of-sample, timestamp-correct, duplicate-aware event studies**.

## 1. Freeze what the target means

For every story first-seen at time `t`, create issuer-specific labels at several horizons: 30 minutes, 2 hours, 1 trading day and 5 trading days. Useful realized outcomes include:

- cumulative abnormal return relative to a pre-event factor/market model;
- volatility expansion;
- spread/depth/illiquidity deterioration if microstructure data are available;
- downside drawdown;
- cross-sectional contagion/breadth.

One robust severity label is a calibrated combination of absolute abnormal return, volatility jump, liquidity jump and cross-sectional spillover. Train on the continuous severity target and separately evaluate the binary tail event.

**Future market data belong only in the label. Never in inference features.**

## 2. Split by time and by story chain

1. Sort by immutable first-seen timestamp.
2. Deduplicate/syndication-cluster **before** splitting.
3. Keep each story chain inside exactly one split.
4. Use expanding or rolling walk-forward train → validation → test.
5. Purge events whose forward label windows overlap a later split.
6. Embargo around split boundaries.
7. Fit all scalers, thresholds, isotonic/conformal calibrators on training/calibration data only.
8. Freeze the final test period before feature/model tuning.

Random row-wise CV is unacceptable here because duplicate stories, temporal dependence and overlapping forward returns create leakage.

## 3. NLP component metrics

### Sentiment
- macro-F1 across positive/neutral/negative;
- negative-class precision/recall/F1;
- Brier score and expected calibration error;
- formal-news, filing and social slices;
- numeric/double-negative/adversarial paraphrase slices.

### Event extraction
- macro-F1 and per-class F1;
- multi-label average precision;
- actor/target/beneficiary role accuracy;
- open-set abstention quality;
- event-status accuracy (rumour/announced/completed/cancelled) where labels exist.

### Entity/relation extraction
- entity precision/recall/F1;
- relation F1;
- ticker ambiguity and multi-entity slice.

## 4. Early-warning metrics

Accuracy is not a useful headline metric for rare stress.

Report:

- PR-AUC;
- ROC-AUC as a secondary measure;
- Brier score / reliability diagram;
- recall at a fixed false-alert budget, e.g. false alerts per 100 events/day;
- precision at operational alert thresholds;
- median lead time to realized stress;
- detection delay after a genuine break;
- false-alert duration;
- crisis vs quiet-regime performance;
- calibration conditional on source tier and market regime.

`scripts/backtest_alerts.py` computes several of these for an already out-of-sample score file.

## 5. Impact severity / tail model

Report:

- MAE and Spearman rank correlation for 1–10 severity;
- pinball loss at q50/q90/q95;
- empirical coverage of conformal intervals;
- coverage conditional on calm/stress regimes;
- calibration of `P(impact>=7)`;
- top-decile lift versus simple baselines.

Baselines must include:

1. sentiment only;
2. event prior only;
3. market-only stress;
4. text + static market features;
5. no graph/Hawkes;
6. no novelty/corroboration;
7. full system.

Without ablations, a large system can hide useless components.

## 6. Adversarial and data-quality suite

Inject:

- 20 syndicated copies from different domains;
- one viral low-trust rumour;
- contradictory authoritative correction;
- old story resurfacing with a new URL;
- paraphrase preserving meaning;
- double negative;
- number whose sign depends on context (e.g. “loss narrows from 10 to 3”);
- ticker/common-word ambiguity;
- source outage;
- clock skew;
- article timestamp after market close;
- stale/revised filing;
- malicious source-provided `source_quality` (do not allow client overrides in hardened production);
- manipulated order-book snapshot / missing levels.

## 7. Market/microstructure validation

Market features must use the last exchange observation **at or before** event time. Test separately for:

- in-hours vs after-hours stories;
- holidays/weekends;
- stock splits/corporate actions;
- missing bars;
- thin securities;
- limit-up/down or halt conditions.

If L2/L3 data are used, synchronize feed clocks and distinguish event time, exchange time, receipt time and processing time.

## 8. Graph and Hawkes validation

For graph models, edges must be time-valid. A supplier relationship learned from a later filing cannot exist in an earlier backtest. Compare:

- sector-only graph;
- known-relations graph;
- dynamic return-dependency graph;
- diffusion baseline;
- Temporal GAT.

For Hawkes, fit excitation parameters on training periods and test event-cluster likelihood, next-event timing, and whether intensity adds incremental PR-AUC after simpler burst features.

## 9. Module A

Report both signal and portfolio statistics:

- gross/net return after realistic costs;
- turnover;
- max drawdown;
- tracking error;
- sector/factor exposures;
- CVaR;
- hit-rate only as secondary context;
- performance by regime.

A backtest must include publication latency, execution latency, spread/fees and delisted names. Do not infer “profitable” from classification accuracy.

## 10. Module B

Validate on historical and hypothetical stress episodes:

- deterministic shock monotonicity;
- VaR/CVaR backtesting where appropriate;
- comparison against full repricing on a representative asset subset;
- sensitivity to correlation, degrees of freedom and jump probability;
- reverse-stress stability;
- liquidity-horizon sensitivity.

## 11. Performance claims allowed after validation

A credible claim looks like:

> “On a locked 2025–2026 test period, the model improved PR-AUC from X to Y and stress-event recall from A% to B% at a fixed Z false alerts per 100 events, with median lead time L.”

Until that test exists, call the system **research-grade architecture**, not “the most accurate crash predictor.”
