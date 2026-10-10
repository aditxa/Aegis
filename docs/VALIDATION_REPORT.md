# AegisSignal-X v0.3 validation report

Date: 2026-10-02

## Automated checks

- Python compile/compileall: **PASS**
- Pytest: **12/12 PASS**
- API import: **PASS** (`AegisSignal-X Financial Early-Risk Engine 0.3.0`)
- End-to-end institutional demo: **PASS**

## Bundled institutional-demo data

- Synthetic news/event records: **420**
- Distinct issuers: **20**
- Distinct synthetic source domains: **5**
- Alert distribution in deterministic demo run: **268 WATCH / 122 ELEVATED / 30 LOW**
- Maximum conditional impact score: **8.83 / 10**
- Maximum risk-priority score: **6.90 / 10**
- Synthetic NIFTY-like IV surface points: **117**
- Example synthetic microstructure tape: **780 rows** for 10 names × 78 five-minute intervals
- Synthetic portfolio baseline NAV: **₹500 Cr**
- Example generated stress P&L: **-₹34.68 Cr (-6.94%)** for the top demo event
- Example tactical basket gross trade: **₹120.00 Cr**
- Example estimated execution cost: **₹15.63 Lakh** under the transparent demo TCA model

## UI/data defects explicitly addressed

1. Generic raw dataframe rendering was replaced with explicit column mapping, units and hidden row indices.
2. `old_weight` / `before` ambiguity is eliminated through `Current`, `Target`, `Baseline`, `Stressed` display fields.
3. Raw UTC ISO strings are retained in machine outputs but rendered as compact IST `HH:MM:SS` on the primary tape.
4. Currency, percent, score and bps units are explicit.
5. Alert / P&L / sentiment / priority fields use conditional visual encoding for fast scanning.
6. The equal-weight toy portfolio is replaced by a ₹500 Cr India-focused institutional demo book.
7. Rebalance output is augmented with execution notional, %ADV, participation, spread, impact, all-in cost and liquidity status.
8. A 3D IV surface, term structure, skew/smile views and a risk heatmap are included.
9. Stress output includes baseline NAV, stressed NAV, CVaR-style tail metrics and asset-level loss attribution.
10. Model/data-health views expose source quality, model confidence, disagreement, duplicates, contradiction and manipulation proxies.

## Important limitation

The dashboard is an institutional-style **synthetic demo**, not a production OMS/EMS or market-data terminal. Live NSE/BSE order books, licensed option chains, broker fill data, post-trade TCA calibration, entitlements, P&L reconciliation, controls, audit trails and production model-risk governance are deliberately not fabricated.
