from aegis.risk.fusion import FusionInputs,fuse_risk

def x(**kw):
    d=dict(sentiment=-.8,sentiment_conf=.9,sentiment_disagreement=.05,event_prior=.9,event_conf=.9,event_direction=-.9,
           novelty=.9,duplicate_probability=.05,corroboration=.6,contradiction=.0,source_quality=.95,source_independence=.8,
           burst=.7,velocity=.6,change=.5,hawkes=.5,graph_contagion=.4,systemic=.4,market_stress=.0,microstructure_stress=.0,tail_risk=.0,regime=.0,
           manipulation=.0,uncertainty=.1)
    d.update(kw); return FusionInputs(**d)

def test_market_confirmation_raises_risk():
    a=fuse_risk(x())
    b=fuse_risk(x(market_stress=.9,tail_risk=.8,regime=.7))
    assert b.impact_score > a.impact_score

def test_manipulation_and_contradiction_reduce_confidence():
    a=fuse_risk(x())
    b=fuse_risk(x(manipulation=.9,contradiction=.8))
    assert b.confidence < a.confidence
