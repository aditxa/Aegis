from aegis.nlp.events import EventClassifier


def test_short_keywords_do_not_match_inside_words():
    c = EventClassifier()
    assert c.classify("Walmart warns margins may remain under pressure after cutting guidance").label == "Earnings"
    assert c.classify("Microsoft reports a suspected cyberattack and major service outage").label == "Cybersecurity"
    assert c.classify("Analysts report strong preorder activity after Apple launch").label == "Product Launch"
