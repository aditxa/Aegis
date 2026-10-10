from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import numpy as np

FEATURE_ORDER = [
    "sentiment", "sentiment_conf", "sentiment_disagreement", "event_prior", "event_conf", "event_direction",
    "novelty", "duplicate_probability", "corroboration", "contradiction", "source_quality", "source_independence",
    "burst", "velocity", "change", "hawkes", "graph_contagion", "systemic", "market_stress", "microstructure_stress",
    "tail_risk", "regime", "manipulation", "uncertainty",
]


class TailImpactEnsemble:
    """Leakage-conscious learned tail-impact stack.

    Heads:
      * q50 conditional severity
      * q90 conditional tail severity
      * high-impact probability
      * split-conformal correction on the final chronological calibration slice

    The shipped repository deliberately contains no fitted model: a fitted checkpoint without a
    documented event-study label set would only create false confidence.
    """
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.bundle = None
        if model_path and Path(model_path).exists():
            import joblib
            self.bundle = joblib.load(model_path)

    def predict(self, feature_dict: dict[str,float]) -> dict[str,float] | None:
        if self.bundle is None:
            return None
        x = np.array([[float(feature_dict.get(k,0.0)) for k in FEATURE_ORDER]], dtype=float)
        median = float(self.bundle["q50"].predict(x)[0])
        q90 = float(self.bundle["q90"].predict(x)[0])
        q90 = max(q90, median)
        upper_adj = float(self.bundle.get("conformal_upper_adjustment", 0.0))
        abs_adj = float(self.bundle.get("conformal_abs_adjustment", 0.0))
        out = {
            "median": float(np.clip(median,1.0,10.0)),
            "tail90": float(np.clip(q90 + max(0.0, upper_adj),1.0,10.0)),
            "interval_low90": float(np.clip(median-abs_adj,1.0,10.0)),
            "interval_high90": float(np.clip(median+abs_adj,1.0,10.0)),
        }
        clf = self.bundle.get("high_clf")
        if clf is not None:
            p = float(clf.predict_proba(x)[0,1])
            iso = self.bundle.get("high_iso")
            if iso is not None:
                p = float(iso.predict([p])[0])
            out["p_high"] = float(np.clip(p,0.0,1.0))
        return out

    @staticmethod
    def _conformal_quantile(residuals: np.ndarray, coverage: float = 0.90) -> float:
        if len(residuals) == 0:
            return 0.0
        # finite-sample split-conformal quantile
        q = min(1.0, np.ceil((len(residuals)+1)*coverage)/len(residuals))
        return float(np.quantile(residuals, q, method="higher"))

    @staticmethod
    def train(rows: Iterable[dict[str,float]], output: str | Path) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        from sklearn.isotonic import IsotonicRegression
        import joblib

        rows = list(rows)
        if len(rows) < 80:
            raise ValueError("Need at least 80 chronologically ordered labeled events; hundreds/thousands are preferable.")
        X = np.array([[float(r.get(k,0.0)) for k in FEATURE_ORDER] for r in rows], dtype=float)
        y = np.array([float(r["target_impact"]) for r in rows], dtype=float)
        cut = max(50, min(len(rows)-20, int(0.85*len(rows))))
        Xtr,ytr,Xcal,ycal = X[:cut],y[:cut],X[cut:],y[cut:]

        q50 = HistGradientBoostingRegressor(loss="quantile", quantile=0.50, max_iter=350, learning_rate=0.04,
            max_leaf_nodes=24, min_samples_leaf=12, l2_regularization=1.5, random_state=42)
        q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=420, learning_rate=0.035,
            max_leaf_nodes=26, min_samples_leaf=12, l2_regularization=2.0, random_state=43)
        q50.fit(Xtr,ytr); q90.fit(Xtr,ytr)

        pred50=q50.predict(Xcal); pred90=q90.predict(Xcal)
        abs_adj=TailImpactEnsemble._conformal_quantile(np.abs(ycal-pred50),0.90)
        upper_adj=TailImpactEnsemble._conformal_quantile(np.maximum(0.0,ycal-pred90),0.90)

        y_high=(ytr>=7.0).astype(int); high_clf=None; high_iso=None
        if len(np.unique(y_high))==2 and min(np.bincount(y_high))>=8:
            high_clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.04,max_leaf_nodes=24,
                min_samples_leaf=12,l2_regularization=1.5,random_state=44)
            high_clf.fit(Xtr,y_high)
            ycal_high=(ycal>=7.0).astype(int)
            pcal=high_clf.predict_proba(Xcal)[:,1]
            if len(np.unique(ycal_high))==2 and len(ycal)>=20:
                high_iso=IsotonicRegression(out_of_bounds="clip").fit(pcal,ycal_high)

        output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump({"q50":q50,"q90":q90,"high_clf":high_clf,"high_iso":high_iso,
                     "conformal_abs_adjustment":abs_adj,"conformal_upper_adjustment":upper_adj,
                     "features":FEATURE_ORDER,"train_rows":cut,"calibration_rows":len(rows)-cut},output)

    @staticmethod
    def schema_json() -> str:
        return json.dumps({"features":FEATURE_ORDER,"target":"target_impact (1..10 event-study realized severity)",
                           "row_order":"chronological; final 15% is reserved for split-conformal calibration"},indent=2)

# Backward compatible name.
QuantileForestImpactModel = TailImpactEnsemble
