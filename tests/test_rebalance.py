from aegis.downstream import TacticalRebalancer


def test_rebalance_constraints():
    current = {"A": .2, "B": .2, "C": .2, "D": .2, "E": .2}
    signals = {
        "A": {"sentiment": .8, "impact": 4, "confidence": .9},
        "B": {"sentiment": -.9, "impact": 9, "confidence": .9},
        "C": {"sentiment": .1, "impact": 3, "confidence": .7},
    }
    rows = TacticalRebalancer(min_w=.05, max_w=.35, turnover_limit=.20).rebalance(current, signals)
    assert abs(sum(r.new_weight for r in rows) - 1.0) < 1e-5
    assert all(.05 - 1e-6 <= r.new_weight <= .35 + 1e-6 for r in rows)
    turnover = .5 * sum(abs(r.new_weight - r.old_weight) for r in rows)
    assert turnover <= .205
    assert next(r for r in rows if r.ticker == "A").new_weight > next(r for r in rows if r.ticker == "B").new_weight
