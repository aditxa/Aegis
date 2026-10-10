from __future__ import annotations

import math
import numpy as np


class RegimeFilter:
    """Sticky 3-state Bayesian regime filter (calm / stressed / crisis).

    It is deliberately compact and transparent; production should estimate emissions/transitions
    on a licensed multi-market history rather than treating these standardized defaults as universal.
    """
    def __init__(self) -> None:
        self.p = np.array([0.86, 0.12, 0.02], dtype=float)
        self.T = np.array([[0.97,0.028,0.002],[0.06,0.90,0.04],[0.02,0.10,0.88]], dtype=float)
        self.means = np.array([[0.8,0.4,0.1],[1.5,1.2,0.35],[2.5,2.3,0.70]], dtype=float)
        self.sds = np.array([[0.55,0.65,0.20],[0.65,0.80,0.25],[0.85,1.10,0.25]], dtype=float)

    def update(self, vol_ratio: float, abs_return_z: float, breadth: float = 0.0) -> float:
        obs = np.array([max(0.0, vol_ratio), abs(abs_return_z), max(0.0, min(1.0, breadth))])
        prior = self.p @ self.T
        z = (obs - self.means) / self.sds
        loglik = -0.5 * np.sum(z*z + np.log(2.0*math.pi*self.sds*self.sds), axis=1)
        logpost = np.log(np.maximum(prior, 1e-12)) + loglik
        logpost -= np.max(logpost)
        post = np.exp(logpost); post /= post.sum()
        self.p = post
        return float(post[1] * 0.55 + post[2])
