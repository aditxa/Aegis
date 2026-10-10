from aegis.nlp.novelty import NoveltyTracker

def test_story_chain_duplicate_is_discounted():
    n=NoveltyTracker()
    a=n.analyze("AAPL","Apple warns that a supplier outage may reduce iPhone shipments next quarter.")
    b=n.analyze("AAPL","Apple warns that a supplier outage may reduce iPhone shipments next quarter.")
    assert a.novelty > b.novelty
    assert b.duplicate_probability > .5
    assert a.cluster_id == b.cluster_id
