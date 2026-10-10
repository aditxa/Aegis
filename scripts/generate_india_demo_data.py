from __future__ import annotations

from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "india_sample_articles.jsonl"

IST = timezone(timedelta(hours=5, minutes=30))
RNG = random.Random(20261002)

COMPANIES = []
with open(ROOT / "data" / "india_companies.csv", newline="", encoding="utf-8") as f:
    COMPANIES = list(csv.DictReader(f))

SOURCES = [
    ("InstitutionalWireDemo", "tier1-wire.demo", 0.94),
    ("ExchangeNoticeDemo", "exchange-notice.demo", 0.98),
    ("BrokerResearchDemo", "broker-research.demo", 0.82),
    ("BusinessNewsDemo", "business-news.demo", 0.78),
    ("SocialPulseDemo", "social-pulse.demo", 0.40),
]

EVENTS = [
    ("Earnings", [
        ("positive", "{company} beats quarterly profit estimates and raises guidance", "{company} reported strong profit growth, resilient margins and an upgraded outlook after the quarter."),
        ("negative", "{company} misses quarterly expectations as margins weaken", "{company} reported a material profit shortfall and warned that near-term guidance remains weak."),
    ]),
    ("Regulatory/Legal", [
        ("negative", "Regulator opens investigation into controls at {company}", "A regulator opened a probe into {company}. A material fine remains possible while the investigation continues."),
        ("positive", "Regulatory matter involving {company} resolved without major penalty", "The regulator closed a long-running investigation involving {company}; the matter was resolved with no material penalty."),
    ]),
    ("Cybersecurity", [
        ("negative", "{company} investigates cyberattack after systems disruption", "{company} disclosed a cyberattack and data breach that caused a material outage across selected systems."),
    ]),
    ("Supply Chain", [
        ("negative", "Supplier shortage disrupts near-term deliveries at {company}", "A major supplier disruption and logistics shortage may delay shipments for {company} during the current quarter."),
        ("positive", "{company} clears supplier bottleneck and restores shipments", "{company} said a supply chain disruption was resolved and inventory availability has recovered faster than expected."),
    ]),
    ("Operational Incident", [
        ("negative", "Operational outage interrupts services at {company}", "A major outage caused an operational disruption at {company}; recovery teams are restoring normal service."),
    ]),
    ("Product Launch", [
        ("positive", "{company} launches new product line with strong initial demand", "{company} unveiled a major product launch and reported strong early demand indicators."),
    ]),
    ("Management/Governance", [
        ("negative", "Senior executive resignation raises governance questions at {company}", "A senior resignation at {company} prompted fresh governance questions and uncertainty around the leadership transition."),
    ]),
    ("Financing/Liquidity", [
        ("negative", "Funding stress rises as {company} prepares refinancing", "{company} faces refinancing pressure as funding stress and liquidity concerns increase in wholesale markets."),
    ]),
    ("Merger/Acquisition", [
        ("positive", "{company} announces accretive acquisition", "{company} announced an acquisition that management expects to be accretive after integration."),
    ]),
    ("Market/Trading", [
        ("negative", "Volatility spike hits {company} amid heavy selloff", "Shares of {company} faced a sharp selloff and a volume spike as volatility accelerated."),
    ]),
]


def main(n: int = 420) -> None:
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    end = datetime(2026, 10, 1, 15, 29, tzinfo=IST)
    span = int((end - start).total_seconds())
    rows = []
    for i in range(n):
        comp = RNG.choice(COMPANIES)
        event, variants = RNG.choice(EVENTS)
        tone, title_tpl, body_tpl = RNG.choice(variants)
        source, domain, quality = RNG.choices(SOURCES, weights=[26, 10, 18, 28, 18], k=1)[0]
        ts = start + timedelta(seconds=RNG.randrange(max(1, span)))
        company = comp["company"]
        # Use a familiar alias in some articles to exercise entity resolution.
        alias_pool = [a.strip() for a in comp.get("aliases", "").split("|") if a.strip()]
        display = RNG.choice(alias_pool + [company]) if alias_pool else company
        title = title_tpl.format(company=display)
        body = body_tpl.format(company=display)
        # Add mild uncertainty / corroboration language to create richer score distributions.
        if source == "SocialPulseDemo":
            body = "Unconfirmed market chatter: " + body + " Traders cautioned that the claim has not been independently verified."
        elif source == "ExchangeNoticeDemo":
            body = body + " The disclosure was published through an exchange-style filing in this synthetic demo feed."
        elif RNG.random() < 0.20:
            body += " A second source said the development could be significant but details remain uncertain."
        rows.append({
            "item_id": f"india_demo_{i:04d}",
            "source": source,
            "source_domain": domain,
            "published_at": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "title": "[SYNTHETIC] " + title,
            "text": body,
            "url": f"https://{domain}/{comp['ticker'].lower()}/{event.lower().replace('/', '-').replace(' ', '-')}/{i}",
            "metadata": {
                "source_quality": quality,
                "synthetic_fixture": True,
                "demo_market": "NSE/BSE-style India",
                "tone": tone,
            },
        })
    rows.sort(key=lambda x: x["published_at"])
    OUT.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
