from pathlib import Path
from aegis.downstream import StressTester

ROOT = Path(__file__).resolve().parents[1]


def test_credit_stress_is_loss():
    p = [{"asset_id": "b", "name": "bond", "asset_type": "bond", "market_value": 1000000,
          "duration": 5, "convexity": 30, "spread_duration": 4}]
    r = StressTester(ROOT / "data" / "shock_library.json").run(p, "Credit Event", 9.0)
    assert r["portfolio_pnl"] < 0
    assert r["portfolio_after"] < r["portfolio_before"]
