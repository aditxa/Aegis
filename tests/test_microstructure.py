from aegis.risk.microstructure import LimitOrderBookStress

def test_lob_stress_rises_on_liquidity_break():
    rows=[]
    for i in range(15):
        rows.append({"bid":99.99,"ask":100.01,"bid_size":1000,"ask_size":1000,"signed_volume":30,"trade_volume":100,
                     "cancel_count":5,"trade_count":20,"mid_return":0.0001})
    calm=LimitOrderBookStress().analyze(rows)
    stressed=list(rows)
    stressed[-1]={"bid":99.5,"ask":100.5,"bid_size":80,"ask_size":50,"signed_volume":95,"trade_volume":100,
                  "cancel_count":200,"trade_count":10,"mid_return":-0.015}
    hot=LimitOrderBookStress().analyze(stressed)
    assert hot.available and hot.score>calm.score
