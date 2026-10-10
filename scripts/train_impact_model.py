from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from aegis.risk.impact_model import TailImpactEnsemble, FEATURE_ORDER

def main():
    ap=argparse.ArgumentParser(description="Train quantile + conformal impact heads from leakage-safe event-study labels")
    ap.add_argument("csv_path"); ap.add_argument("--out",default="models/impact_tail.joblib"); args=ap.parse_args()
    raw=[]
    with open(args.csv_path,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f): raw.append(r)
    if raw and "timestamp" in raw[0]:
        raw.sort(key=lambda r:r.get("timestamp",""))
    rows=[{k:float(r.get(k,0.0) or 0.0) for k in FEATURE_ORDER}|{"target_impact":float(r["target_impact"])} for r in raw]
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); TailImpactEnsemble.train(rows,out); print(f"saved {out}")
if __name__=="__main__": main()
