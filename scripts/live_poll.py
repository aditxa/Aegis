from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from aegis.engine import AegisRiskEngine
from aegis.ingest import GDELTConnector,RedditConnector,GoogleNewsRSSConnector,SECEdgarConnector

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("query"); ap.add_argument("--limit",type=int,default=20)
    ap.add_argument("--sources",default="gdelt,rss",help="gdelt,reddit,rss,edgar"); ap.add_argument("--market",action="store_true"); args=ap.parse_args()
    eng=AegisRiskEngine(ROOT/"data"/"companies.csv",ROOT/"outputs"/"aegis.db",use_market_context=args.market)
    cs={"gdelt":GDELTConnector,"reddit":RedditConnector,"rss":GoogleNewsRSSConnector,"edgar":SECEdgarConnector}; items=[]; errors={}
    for name in [x.strip().lower() for x in args.sources.split(",") if x.strip()]:
        try: items.extend(cs[name]().fetch(args.query,args.limit))
        except Exception as e: errors[name]=str(e)
    sigs=eng.analyze(items); print(json.dumps({"items":len(items),"source_errors":errors,"signals":[s.to_dict() for s in sigs]},indent=2))
if __name__=="__main__": main()
