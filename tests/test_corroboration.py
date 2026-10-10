from datetime import datetime, timedelta, timezone
from aegis.risk.online import CorroborationTracker


def test_direction_aware_corroboration():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    c = CorroborationTracker()
    a, x, ia = c.score("X", "Credit Event", "one.example", t, -1)
    b, y, ib = c.score("X", "Credit Event", "two.example", t + timedelta(minutes=1), -1)
    d, z, ic = c.score("X", "Credit Event", "three.example", t + timedelta(minutes=2), 1)
    assert b > a
    assert z > 0
    assert d < b
