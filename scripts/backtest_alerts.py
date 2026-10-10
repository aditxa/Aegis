from __future__ import annotations

"""Leakage-conscious alert evaluation utility.

Expected CSV columns:
  timestamp, realized_stress, and one or more of
  risk_priority_score, early_warning_score, impact_score
Optional:
  realized_severity, event_id, stress_time

Rows should be generated strictly out-of-sample/walk-forward.  This script evaluates a frozen score;
it does not tune a model on the test set.
"""

import argparse, csv, json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--score",default="risk_priority_score")
    ap.add_argument("--threshold",type=float,default=None)
    ap.add_argument("--false-alerts-per-100",type=float,default=5.0,
                    help="report maximum recall under this false-alert budget")
    args=ap.parse_args()
    rows=list(csv.DictReader(open(args.csv_path,newline="",encoding="utf-8")))
    if not rows: raise SystemExit("empty CSV")
    key=args.score if args.score in rows[0] else ("early_warning_score" if "early_warning_score" in rows[0] else "impact_score")
    s=np.array([float(r[key]) for r in rows],dtype=float)
    y=np.array([int(float(r["realized_stress"])>0) for r in rows],dtype=int)
    # Convert score to [0,1] if it is 1..10.
    p=np.clip((s-1)/9,0,1) if np.nanmax(s)>1.0001 else np.clip(s,0,1)
    out={"rows":len(rows),"positives":int(y.sum()),"score":key}
    if len(np.unique(y))==2:
        out["pr_auc"]=float(average_precision_score(y,p)); out["roc_auc"]=float(roc_auc_score(y,p)); out["brier"]=float(brier_score_loss(y,p))
        prec,rec,thr=precision_recall_curve(y,p)
        best_recall=0.0; best_thr=None; budget=args.false_alerts_per_100/100.0
        for t in np.unique(p):
            pred=p>=t; fp=int(np.sum(pred & (y==0))); nneg=max(1,int(np.sum(y==0)))
            if fp/nneg<=budget:
                rr=float(np.sum(pred & (y==1))/max(1,np.sum(y==1)))
                if rr>best_recall: best_recall,best_thr=rr,float(t)
        out["recall_at_false_alert_budget"]=best_recall; out["budget_threshold_0_1"]=best_thr
    if args.threshold is not None:
        t=(args.threshold-1)/9 if args.threshold>1 else args.threshold
        pred=p>=t; tp=int(np.sum(pred&(y==1))); fp=int(np.sum(pred&(y==0))); fn=int(np.sum((~pred)&(y==1))); tn=int(np.sum((~pred)&(y==0)))
        out["threshold_metrics"]={"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"fpr":fp/max(1,fp+tn)}
    if "realized_severity" in rows[0]:
        r=np.array([float(x["realized_severity"]) for x in rows])
        out["severity_mae"]=float(np.mean(np.abs(s-r)))
        out["severity_spearman"]=float(__import__("scipy").stats.spearmanr(s,r,nan_policy="omit").statistic)
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
