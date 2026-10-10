from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from scipy.stats import t as student_t


class _BOCPDState:
    def __init__(self, max_run: int, hazard: float):
        self.max_run = max_run
        self.hazard = hazard
        self.R = np.array([1.0])
        self.mu = np.array([0.0])
        self.kappa = np.array([1.0])
        self.alpha = np.array([2.0])
        self.beta = np.array([0.5])
        self.n_total = 0

    def update(self, x: float) -> float:
        self.n_total += 1
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        pred = student_t.pdf(x, df=2.0 * self.alpha, loc=self.mu, scale=np.maximum(scale, 1e-6))
        growth = self.R * pred * (1.0 - self.hazard)
        cp = float(np.sum(self.R * pred * self.hazard))
        newR = np.empty(min(self.max_run + 1, len(growth) + 1))
        newR[0] = cp
        take = len(newR) - 1
        newR[1:] = growth[:take]
        s = float(newR.sum())
        if not math.isfinite(s) or s <= 0:
            newR[:] = 0.0; newR[0] = 1.0
        else:
            newR /= s

        # Posterior sufficient statistics for continuing runs.
        mu_n = (self.kappa * self.mu + x) / (self.kappa + 1.0)
        kappa_n = self.kappa + 1.0
        alpha_n = self.alpha + 0.5
        beta_n = self.beta + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0))
        n = len(newR)
        self.mu = np.concatenate(([0.0], mu_n[:n-1]))
        self.kappa = np.concatenate(([1.0], kappa_n[:n-1]))
        self.alpha = np.concatenate(([2.0], alpha_n[:n-1]))
        self.beta = np.concatenate(([0.5], beta_n[:n-1]))
        self.R = newR
        # Short-run posterior mass is meaningful only after the process had time to establish a
        # baseline. Cold-start run lengths are short by definition and must not be treated as
        # regime changes. Remove the expected short-run mass and rescale the excess.
        if self.n_total < 10:
            return 0.0
        short_mass = float(self.R[:min(4, len(self.R))].sum())
        baseline = 1.0 - (1.0-self.hazard)**4
        return float(max(0.0, min(1.0, (short_mass-baseline) / max(0.12, 1.0-baseline))))


class BOCPDBank:
    def __init__(self, hazard: float = 1/80, max_run: int = 160):
        self.hazard = hazard
        self.max_run = max_run
        self.states = defaultdict(lambda: _BOCPDState(max_run, hazard))

    def update(self, key: str, x: float) -> float:
        return self.states[key].update(float(x))
