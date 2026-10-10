from pathlib import Path
from aegis.engine import AegisRiskEngine
from aegis.ingest import SampleConnector

ROOT = Path(__file__).resolve().parents[1]


def test_engine_required_fields():
    engine = AegisRiskEngine(ROOT / "data" / "companies.csv", use_market_context=False)
    items = SampleConnector(ROOT / "data" / "sample_articles.jsonl").fetch(limit=4)
    sigs = engine.analyze(items)
    assert sigs
    for s in sigs:
        assert -1.0 <= s.sentiment_score <= 1.0
        assert 1.0 <= s.impact_score <= 10.0
        assert s.event_classification
        assert s.ticker
        assert s.evidence
