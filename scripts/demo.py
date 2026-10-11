from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis.engine import AegisRiskEngine
from aegis.ingest import SampleConnector
from aegis.downstream import TacticalRebalancer, StressTester
from aegis.institutional import execution_estimate


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    engine = AegisRiskEngine(ROOT / "data" / "india_companies.csv", use_market_context=False)
    items = SampleConnector(ROOT / "data" / "india_sample_articles.jsonl").fetch(limit=2400)
    signals = engine.analyze(items)
    payload = [s.to_dict() for s in signals]

    (out_dir / "signals.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload:
        with open(out_dir / "signals.csv", "w", newline="", encoding="utf-8") as f:
            keys = [k for k in payload[0].keys() if k not in {"evidence", "secondary_events", "contributions"}]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in payload:
                w.writerow({k: row.get(k) for k in keys})

    current = {
        "RELIANCE": .14, "HDFCBANK": .13, "ICICIBANK": .115, "INFY": .10, "TCS": .095,
        "BHARTIARTL": .09, "SBIN": .085, "LT": .08, "ITC": .075, "AXISBANK": .09,
    }
    agg = engine.aggregate_for_rebalancer(list(current))
    reb = TacticalRebalancer(min_w=.04, max_w=.17, turnover_limit=.12).rebalance(current, agg)
    reb_rows = [r.to_dict() for r in reb]
    (out_dir / "rebalance.json").write_text(json.dumps(reb_rows, indent=2), encoding="utf-8")

    aum = 5_000_000_000.0
    execution = []
    for r in reb:
        trade = (r.new_weight - r.old_weight) * aum
        execution.append({**r.to_dict(), **execution_estimate(r.ticker, trade, "BUY" if trade > 0 else "SELL" if trade < 0 else "FLAT")})
    (out_dir / "execution_rebalance.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")

    with open(ROOT / "data" / "india_portfolio.csv", newline="", encoding="utf-8") as f:
        portfolio = list(csv.DictReader(f))
    top = max(signals, key=lambda s: s.impact_score)
    stress = StressTester(ROOT / "data" / "shock_library.json").run(portfolio, top.event_classification, top.impact_score)
    (out_dir / "stress_test.json").write_text(json.dumps(stress, indent=2), encoding="utf-8")

    print("AegisSignal X institutional demo complete")
    print(f"  items: {len(items)}")
    print(f"  signals: {len(signals)}")
    print(f"  elevated+: {sum(s.alert_level in {'ELEVATED','HIGH','CRITICAL'} for s in signals)}")
    print(f"  max priority: {max(s.risk_priority_score for s in signals):.2f}/10")
    print(f"  top impact: {top.ticker} | {top.event_classification} | impact={top.impact_score:.2f}")
    print(f"  portfolio AUM: INR {stress['portfolio_before']/1e7:,.1f} Cr")
    print(f"  stress P&L: INR {stress['portfolio_pnl']/1e7:,.2f} Cr ({stress['portfolio_pnl_pct']:.2f}%)")
    print("  outputs: signals.json/csv, rebalance.json, execution_rebalance.json, stress_test.json")


if __name__ == "__main__":
    main()
