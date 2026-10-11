"""Deterministic synthetic decision lab; makes no claim of historical events."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aegis.omega import Evidence, analyze_case, replay_cases

ASOF='2026-10-01T09:00:00+00:00' # 14:30 IST

def load_portfolio():
    with (ROOT/'data'/'india_portfolio.csv').open(newline='') as f:
        result=list(csv.DictReader(f))
    for row in result:
        for k in ('market_value','duration','convexity','spread_duration','beta','delta','gamma','vega','lgd','fx_exposure'):
            row[k]=float(row.get(k,0.) or 0.)
    return result


def sector_map():
    with (ROOT/'data'/'india_companies.csv').open(newline='') as f:
        return {r['ticker']:r['sector'] for r in csv.DictReader(f)}


def fixtures():
    rows=[]
    def add(i,ticker,event,claim,source_type,group,stance,clock,conf,sev,sent,headline):
        rows.append(Evidence(evidence_id=f'OMEGA-{i:03}',ticker=ticker,event_type=event,
            claim_id=claim,headline=headline,publisher=group,publisher_group=group,source_type=source_type,
            stance=stance,published_at=clock,first_seen_at=clock,
            confidence=conf,severity=sev,sentiment=sent,
            url=f'https://example.invalid/simulated/{i}'))
    # Fully synthetic, illustrative issuer event; date is simulation clock only.
    add(1,'RELIANCE','Supply Chain','demo-energy-001','exchange_notice','exchange',
        'support','2026-10-01T04:30:00+00:00',.94,8.9,-.8,'DEMO: energy transport disruption notice')
    add(2,'RELIANCE','Supply Chain','demo-energy-001','established_news','wire-syndicate',
        'support','2026-10-01T04:36:00+00:00',.91,8.2,-.66,'DEMO: shipping constraints reported')
    # Rewrites/syndications must add ZERO independent supporters.
    for i in range(3,17):
        add(i,'RELIANCE','Supply Chain','demo-energy-001','other_news','wire-syndicate',
            'support','2026-10-01T04:36:00+00:00',.91,8.2,-.66,'DEMO: wire-syndicated copied shipping report')
    add(17,'RELIANCE','Supply Chain','demo-energy-001','company_disclosure','issuer-ir',
        'support','2026-10-01T05:30:00+00:00',.89,8.1,-.7,'DEMO: issuer discloses supply interruption')
    # Contradiction must not be merged into positive sentiment.
    add(21,'HDFCBANK','Credit Event','demo-bank-002','established_news','finance-news',
        'support','2026-10-01T04:35:00+00:00',.92,9.,-.95,'DEMO: anonymous report alleged loan irregularities')
    add(22,'HDFCBANK','Credit Event','demo-bank-002','company_disclosure','issuer-ir-bank',
        'contradict','2026-10-01T05:10:00+00:00',.99,9.,-.55,'DEMO: issuer disputes the reported allegation')
    for i in range(23,39):
        add(i,'HDFCBANK','Credit Event','demo-bank-002','social','social-mirror',
            'support','2026-10-01T05:40:00+00:00',.45,9.5,-.9,'DEMO: copied unverified social claim')
    add(41,'INFY','Cybersecurity','demo-tech-003','social','unverified-social',
        'support','2026-10-01T06:00:00+00:00',.55,8.5,-.9,'DEMO: isolated unconfirmed cyber rumor')
    return rows


def build():
    records=fixtures(); portfolio=load_portfolio();sector=sector_map()
    buckets=defaultdict(list)
    for o in records:buckets[o.claim_id].append(o)
    cases=[analyze_case(v,portfolio,sector,ASOF) for _,v in sorted(buckets.items())]
    return dict(data_origin='DETERMINISTIC SYNTHETIC FIXTURE',asof_utc=ASOF,
                model_scope='decision support, not trading/market forecast',
                pipeline=['multi-source observations','issuer-specific extraction',
                          'publisher-group dedup','contradiction arbitration','human-review gate',
                          'conditional portfolio scenario','counterfactual removal'],
                cases=cases,events=[vars(e) for e in records])


if __name__=='__main__':
    obj=build();out=ROOT/'outputs';out.mkdir(exist_ok=True)
    (out/'omega_decision_lab.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False))
    for case in obj['cases']:
        print(case['ticker'],case['decision']['action'],
              'support=',case['state']['support_index'],
              'independent=',case['state']['independent_supporters'],
              'conditional_pnl_pct=',case['portfolio']['conditional_pnl_pct'])
    print('Wrote',out/'omega_decision_lab.json')
