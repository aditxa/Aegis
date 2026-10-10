from aegis.institutional import execution_estimate, synthetic_iv_surface, synthetic_market_tape


def test_execution_cost_monotone_with_trade_size():
    small = execution_estimate("RELIANCE", 10_000_000)
    large = execution_estimate("RELIANCE", 500_000_000)
    assert large["impact_cost_bps"] > small["impact_cost_bps"]
    assert large["adv_pct"] > small["adv_pct"]
    assert execution_estimate("RELIANCE", 0)["all_in_cost_bps"] == 0


def test_surface_and_tape_are_dense_and_finite():
    surf = synthetic_iv_surface()
    assert len(surf) >= 100
    assert surf.iv.between(0.05, 0.80).all()
    tape = synthetic_market_tape(["RELIANCE", "HDFCBANK"], periods=12)
    assert len(tape) == 24
    assert tape.liquidity_stress.between(0, 1).all()
