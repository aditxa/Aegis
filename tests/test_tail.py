from aegis.risk.tail import evt_tail_score,bns_jump_score

def test_extreme_negative_return_has_tail_score():
    r=[0.001*(-1 if i%2 else 1) for i in range(60)] + [-0.08]
    assert evt_tail_score(r) > .3
    assert bns_jump_score(r) > 0
