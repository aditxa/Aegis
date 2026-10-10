# Failure modes and hardening plan

## 1. Financial language adversarial attacks

Finance transformers can be sensitive to trigger words, numbers without directional context and double negatives. A 2026 Journal of Banking & Finance study reports material attack success against FinBERT/FinGPT-like systems under semantically similar paraphrases.

**Mitigations in project:** model ensemble/disagreement, stance separate from sentiment, source/provenance gating, contradiction score, manipulation score.  
**Production:** adversarial fine-tuning, numerical-reasoning unit tests, abstention/OOD model and continuous red-team corpus.

## 2. Correct topic, wrong issuer

A story may mention an acquirer, target, supplier, rival and regulator. Entity-conditioned context helps but does not fully solve relation roles.

**Production:** supervised financial NER/relation extraction (e.g. FIRE-style relation schema), permanent identifiers such as CIK/LEI/FIGI, event slots and role-aware impact heads.

## 3. Rumour / manipulation is novel

Novelty can increase attention to a false story.

**Mitigation:** novelty never substitutes for credibility; priority is provenance-gated. Duplicate social amplification is discounted. Conditional impact may still be high if the rumour describes a severe event; confidence/priority should remain low.

## 4. Syndication masquerades as corroboration

**Mitigation:** story-chain IDs collapse near duplicates.  
**Residual risk:** wire rewrites can be semantically distant. Production needs source-ownership and citation graphs plus stronger multilingual semantic clustering.

## 5. Corrections and retractions

A later correction can reverse an earlier alert.

**Production requirement:** explicit story state machine linking update/correction/retraction records and generating negative evidence against the prior event.

## 6. Regime shift

A model trained in calm markets can fail in a liquidity crisis.

**Mitigations:** online change detection, volatility/tail/regime features, tail heads, conformal intervals.  
**Production:** rolling champion/challenger retraining, regime-conditional calibration, conservative fallback model.

## 7. Market already priced the news

Text may be true but late.

**Mitigation:** event-time market alignment and abnormal-return context.  
**Production:** first-seen source timestamps, licensed low-latency feeds, pre-event drift and microstructure response modelling.

## 8. Free market data are not HFT data

Yahoo-style bars cannot reveal queue position, hidden liquidity, order cancellations at message granularity or true L3 dynamics.

**Mitigation:** code calls the free path “market context,” not L3. `LimitOrderBookStress` is a separate adapter for real exchange/order-book data.

## 9. LOB spoofing / manipulated liquidity

Displayed depth can disappear or be strategically placed.

**Mitigation:** combine spread, depth, OFI, cancellation intensity, signed flow and realized price response instead of trusting raw depth alone. Production should use venue-specific message data and spoofing-aware filters.

## 10. Hawkes misspecification

Fixed excitation couplings are priors, not fitted truth.

**Production:** MLE/Bayesian estimation with stability constraints (spectral radius below one for stationary models), time-varying kernels where justified, and out-of-sample point-process likelihood.

## 11. Graph contamination

Bad or future-dated edges produce fake contagion.

**Mitigation:** transparent sector/co-mention fallback.  
**Production:** time-valid edge store, provenance for every relationship, uncertainty-weighted edges, dynamic graph evaluation.

## 12. GNN over-smoothing / shortcut learning

A GNN can learn sector labels or market beta rather than causal contagion.

**Production:** temporal edge dropout, relation-type ablations, counterfactual edge tests, shallow message passing, held-out sectors/entities and comparison to diffusion baselines.

## 13. Tail model instability

EVT is unstable with too few exceedances; quantile ML can overfit rare crises.

**Mitigation:** empirical fallback for small samples, minimum data checks, q50/q90 heads.  
**Production:** threshold stability plots, regime-specific tails, bootstrap uncertainty, stress-period oversampling without duplicate leakage.

## 14. “Voice stress detection” from executives

Inferring hidden psychological state from voice is scientifically fragile and can create compliance/ethics problems. The project does **not** claim executive “stress biometrics.” A production multimodal path should prefer observable transcript/prosody features tied to market outcomes and document the limitations.

## 15. Impact score is not a crash probability

`impact_score=8` means high conditional modeled severity, **not** an 80% crash probability. `early_warning_score` is still only probability-like until calibrated on event-study labels.

## 16. Correlation is not causality

Graph propagation, text co-occurrence and abnormal returns can show association without proving an event caused the move. Use “associated with / consistent with,” not causal language, unless the research design supports causal identification.

## 17. Rebalancer procyclicality

Fast negative-news trading can sell into temporary dislocations.

**Mitigation:** turnover, variance/CVaR and transaction-cost penalties.  
**Production:** no-trade bands, minimum holding periods, factor/sector neutrality, liquidity limits, kill switches and human/risk review.

## 18. Stress model basis/nonlinearity

Duration, beta and Greeks are local approximations.

**Production:** full curve/surface repricing, path-dependent options, basis risk, margin/collateral waterfalls, counterparty wrong-way risk and liquidity spirals.

## 19. Multiple testing / backtest overfitting

Trying hundreds of features and keeping the best one creates statistical mirages.

**Mitigation:** locked validation/test windows, ablations, predefined metrics.  
**Production:** model registry, experiment audit trail, multiple-testing-adjusted performance statistics, independent replication.

## 20. Perfect crash detection is impossible to promise

Crashes are heterogeneous, some are reflexive/endogenous, and false alerts matter. The correct target is **better calibrated recall and lead time at a controlled false-alert budget**.
