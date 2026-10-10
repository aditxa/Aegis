# Research and mathematics behind AegisSignal-X

AegisSignal-X is an amalgamation of public financial NLP, market-microstructure, tail-risk, point-process and network-risk ideas. It does **not** claim access to undisclosed proprietary HFT algorithms.

## 1. Financial language layer

The MAX profile ensembles finance-domain sentiment encoders such as FinBERT variants. BloombergGPT is useful evidence that domain-specialized training improves financial NLP, while FinGPT/FinMA show open financial LLM/instruction-tuning directions. We do not redistribute those models; their original licenses apply.

The engine separates three problems:

1. **entity resolution** — which issuer is affected?
2. **event extraction** — what happened?
3. **stance/direction** — is the event harmful, beneficial or mixed for that issuer?

This mirrors the structured-event idea in public financial information-extraction literature and prior art. FIRE (NAACL 2024) provides 13 entity types and 18 financial relation types and is a useful path for replacing heuristic relation extraction with supervised training.

## 2. Story novelty, source independence and adversarial robustness

A source count is not an independence count. Story-chain clustering creates a latent group for near-duplicate narratives. Corroboration is computed across independent groups and direction; contradictory groups increase `contradiction_score` rather than confirmation.

Public news-analytics patent disclosures have long described company-specific sentiment, relevance, volume, novelty/uniqueness, source differentiation and regime/framing as separate features. AegisSignal-X follows that principle but implements original prototype code.

Recent work also shows financial sentiment transformers can be fooled by semantically similar adversarial text, especially with numbers, double negatives and trigger words. That motivates model disagreement, provenance gating, duplicate detection and the `manipulation_score` rather than blindly trusting one classifier.

## 3. Event-rate burst and self-exciting contagion

For event counts, the burst layer compares a short window to a smoothed longer baseline. A marked Hawkes-style contagion prior uses

\[
\lambda_i(t)=\mu_i + \sum_j\sum_{t_k^j<t}\alpha_{ij}m_k\exp[-\beta(t-t_k^j)].
\]

`m_k` is event severity; `alpha_ij` represents self/same-sector/broad-market coupling. The shipped online implementation uses transparent default couplings. Production should fit the excitation matrix on historical event/trade data.

Hawkes processes are well established in high-frequency finance for volatility, order-book dynamics, market stability, contagion and execution.

## 4. Online regime change

AegisSignal-X combines:

- Page-Hinkley for quick mean-shift detection;
- Bayesian online change-point detection (BOCPD) for a posterior over run lengths;
- a sticky three-state Bayesian market regime filter.

BOCPD uses Student-t posterior predictive distributions from a Normal-Inverse-Gamma conjugate model. Cold-start observations are explicitly suppressed so initialization is not mistaken for a regime break.

## 5. Event-time market stress

The market context is restricted to bars at or before the text timestamp.

### Benchmark abnormal return

\[
r_{i,t}=\alpha_i+\beta_i r_{m,t}+\epsilon_{i,t},
\]

and the latest residual is robustly standardized using median/MAD.

### Volatility clustering

A compact GARCH(1,1)-style recursion is used as a volatility expansion feature:

\[
\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta\sigma_{t-1}^2.
\]

### Jump proxy

For high-frequency returns, realized variance and bipower variation are approximated as

\[
RV=\sum r_i^2,\qquad
BPV=\frac{\pi}{2}\sum |r_i||r_{i-1}|,
\]

and positive `RV-BPV` provides a jump component proxy, following the Barndorff-Nielsen/Shephard line of work.

### Extreme-value tail score

For losses above a high threshold `u`, the Peaks-over-Threshold layer fits a Generalized Pareto Distribution:

\[
P(X-u>y\mid X>u)\approx\left(1+\xi y/\sigma\right)^{-1/\xi}.
\]

This estimates how exceptional the current downside move is relative to the fitted tail rather than assuming Gaussian returns.

## 6. Order-book / liquidity stress

With real market-microstructure data, `LimitOrderBookStress` can use:

- quoted spread in basis points;
- displayed depth collapse;
- bid/ask depth imbalance;
- order-flow-imbalance (OFI) proxy;
- cancel-to-trade anomaly;
- VPIN-like signed-flow toxicity;
- Kyle-lambda-style price response per unit signed flow.

A generic depth imbalance is

\[
I_t=\frac{D^b_t-D^a_t}{D^b_t+D^a_t}.
\]

The module is an adapter; the repository does not relabel delayed retail market data as Level 3.

DeepLOB and more recent transformer LOB models such as TLOB motivate learned microstructure encoders, but the project deliberately keeps the core score interpretable unless a properly trained model is supplied.

## 7. Network contagion and temporal graph attention

The transparent live graph iterates

\[
r^{(k+1)}=(1-\alpha)s+\alpha P^\top r^{(k)},
\]

where `s` is current local risk and `P` is the normalized relation graph.

The optional Temporal GAT learns attention weights over a supplied dynamic graph, then applies a GRU to each node's temporal embedding. It should be trained on time-valid supplier, ownership, counterparty, sector, co-mention and/or return-dependency edges; future graph edges must never leak backward.

## 8. Systemic breadth

The systemic channel is conceptually inspired by correlation/breadth-aware stress indices such as the ECB's CISS: simultaneous stress across segments is more concerning than the same average stress concentrated in one isolated name. The implementation combines entity breadth, sector breadth, concentration and mean adverse intensity.

## 9. Dual-score nonlinear fusion

AegisSignal-X intentionally separates **conditional impact** from **evidence strength**.

Let

- `S` = semantic severity,
- `I` = information dynamics (novelty, burst, velocity, change, Hawkes),
- `M` = market/microstructure/tail/regime stress,
- `N` = network/systemic contagion,
- `E` = evidence/provenance gate.

Conditional severity is mapped through

\[
p_{sev}=\sigma(\theta_0+\theta_S S+\theta_I I+\theta_M M+\theta_N N
+\theta_{SM}SM+\theta_{NR}NR+\theta_{TN}TN),
\]

\[
\text{Impact}=1+9p_{sev}.
\]

A separate early-warning equation uses `S*E`, so manipulation, contradiction, source quality and source independence influence operational urgency much more strongly than the event's *conditional* magnitude.

This means “unconfirmed bank-run rumour” can be high-impact-if-true but low-confidence/low-priority, which is the desired representation.

## 10. Learned tail-impact + conformal calibration

When labeled event studies are available, the optional learned stack fits median and 90th-quantile gradient-boosted heads. A chronological calibration slice produces a split-conformal residual correction.

For calibration residuals `R_i`, a finite-sample conformal radius is the empirical quantile

\[
q_{1-\alpha}=\operatorname{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}(R_1,\ldots,R_n).
\]

The repository intentionally does not ship a fabricated pretrained impact model.

## 11. Module A — robust tactical rebalancing

The optimizer treats NLP as a **ranking signal**, not a guaranteed expected return. Its conceptual objective is

\[
\min_w -\mu^\top w + \lambda_v w^\top\Sigma w
+\lambda_c\operatorname{CVaR}_{0.90}(-R w)
+\kappa\|w-w_0\|_1,
\]

subject to

\[
\mathbf{1}^\top w=1,\quad l_i\le w_i\le u_i,\quad
\frac12\|w-w_0\|_1\le \tau.
\]

`R` is generated from a multivariate Student-t scenario distribution to avoid a purely Gaussian risk assumption.

## 12. Module B — event-driven distributional stress

Each event produces a six-factor shock vector: equity, rates, credit spread, FX, volatility and default probability. Bond approximation:

\[
\frac{\Delta P}{P}\approx-D\Delta y+\frac12C(\Delta y)^2-SD\Delta s.
\]

The stochastic stress path uses correlated Student-t shocks plus an event-severity-dependent jump mixture. It reports VaR/CVaR-style losses and large-loss probabilities. This is closer to the requested jump/stochastic-volatility intuition than a single deterministic shock while remaining transparent enough for a hackathon.

## Key public research used

See `REFERENCES.md` for links. Especially relevant are BIS Working Papers 1250 and 1291, BloombergGPT, FinBERT, FIRE, Hawkes-process surveys, BNS jump-variation work, EVT tail-risk literature, CISS, DeepLOB/TLOB, recent graph-risk work and adversarial financial-sentiment research.
