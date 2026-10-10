from __future__ import annotations

import argparse
import numpy as np
from aegis.risk.temporal_gat import TemporalGATSpec, TemporalGraphRiskModel


def main():
    ap=argparse.ArgumentParser(description="Train optional temporal graph-attention contagion head")
    ap.add_argument("npz",help="NPZ with X:[S,T,N,F], A:[N,N] or [S,N,N]/[S,T,N,N], y:[S,N] in [0,1]")
    ap.add_argument("--out",default="models/temporal_gat.pt")
    ap.add_argument("--epochs",type=int,default=80)
    args=ap.parse_args()
    d=np.load(args.npz)
    X=np.asarray(d["X"],dtype=np.float32); A=np.asarray(d["A"],dtype=np.float32); y=np.asarray(d["y"],dtype=np.float32)
    if X.ndim!=4 or y.shape!=(X.shape[0],X.shape[2]): raise ValueError("Expected X[S,T,N,F] and y[S,N]")
    if A.ndim==2: A=np.repeat(A[None,:,:],X.shape[0],axis=0)
    split=max(1,int(0.8*len(X)))
    model=TemporalGraphRiskModel(TemporalGATSpec(input_dim=X.shape[-1]))
    losses=model.fit(X[:split],A[:split],y[:split],epochs=args.epochs)
    model.save(args.out)
    if split<len(X):
        pred=model.predict(X[split:],A[split:]); mae=float(np.mean(np.abs(pred-y[split:])))
        print(f"chronological holdout MAE={mae:.6f}")
    print(f"saved {args.out}; final_train_loss={losses[-1]:.6f}")

if __name__=="__main__": main()
