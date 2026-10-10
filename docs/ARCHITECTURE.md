# AegisSignal-X architecture

```mermaid
flowchart LR
  N[News: GDELT / RSS] --> C[Canonical TextItem]
  S[Social: Reddit] --> C
  E[SEC EDGAR] --> C
  M[Manual / licensed feeds] --> C

  C --> ER[Entity + role resolution]
  C --> SENT[Finance sentiment ensemble]
  C --> EVT[Multi-label event extraction]
  C --> ST[Impact stance / direction]
  C --> NOV[Story-chain novelty]
  C --> ADV[Manipulation / contradiction]

  ER --> G[Dynamic firm graph]
  EVT --> H[Marked Hawkes contagion]
  NOV --> CORR[Independent-source corroboration]
  EVT --> TEMP[Burst + velocity + PH + BOCPD]

  PX[Event-time market bars] --> MR[Return / volume / vol / jump / EVT / regime]
  LOB[Optional L2/L3 adapter] --> MS[Spread / depth / OFI / toxicity / Kyle]

  SENT --> F[Dual-score nonlinear fusion]
  EVT --> F
  ST --> F
  ADV --> F
  CORR --> F
  TEMP --> F
  H --> F
  G --> F
  MR --> F
  MS --> F

  F --> RS[Structured RiskSignal]
  RS --> A[Module A: robust tactical rebalancer]
  RS --> B[Module B: deterministic + heavy-tail stress]
  RS --> API[FastAPI / JSON / CSV / SQLite]
  RS --> UI[Streamlit dashboard]

  TGAT[Optional trained Temporal GAT] -. replaces/augments .-> G
  IMP[Optional quantile + conformal impact model] -. calibrated blend .-> F
```

## Design rule: independent evidence channels

One model is not asked to do everything. AegisSignal-X separates:

1. semantic understanding;
2. provenance/novelty/corroboration;
3. information arrival dynamics;
4. market and liquidity state;
5. graph/systemic contagion;
6. tail/regime statistics.

This prevents a sentiment classifier from becoming a fake crash oracle.

## Core structured output

Each issuer-level `RiskSignal` contains:

- sentiment and ensemble disagreement;
- primary + secondary event classes;
- harmful/beneficial event direction;
- conditional impact (1–10);
- early-warning score;
- risk-priority score + alert level;
- story cluster / duplicate probability;
- corroboration, contradiction and source independence;
- burst, risk velocity, Page-Hinkley/BOCPD change signal;
- Hawkes contagion;
- graph/systemic contagion;
- market, microstructure, EVT-tail and regime stress;
- manipulation/uncertainty;
- model confidence;
- per-channel contribution trace.

## Dependency profiles

### Core/offline

Runs with deterministic financial-language fallbacks and no model downloads. This mode is for reproducibility, unit tests and demonstrations — not maximum NLP accuracy.

### MAX

`requirements-max.txt` adds PyTorch, Transformers and sentence-transformers. Enable with:

```bash
export AEGIS_USE_TRANSFORMERS=true
export AEGIS_MAX_PROFILE=true
```

MAX uses a finance sentiment ensemble, NLI event classifier and semantic story embeddings.

## Production boundary

The repository contains adapters for stronger signals but deliberately does not pretend to include data it cannot legally/publicly ship: exchange message feeds, proprietary news wires, private counterparty graphs, licensed satellite/maritime data, options/dealer positioning or confidential bank exposures.
