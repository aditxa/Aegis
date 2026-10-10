from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import math
import networkx as nx

from aegis.config import SETTINGS


class DynamicRiskGraph:
    """Knowledge/correlation proxy graph with diffusion-style contagion."""
    def __init__(self, companies_csv: str | Path, relations_csv: str | Path | None = None) -> None:
        self.g = nx.Graph()
        self.risk = defaultdict(float)
        sectors = defaultdict(list)
        with open(companies_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                t = r["ticker"].upper(); sec = r.get("sector", "")
                self.g.add_node(t, sector=sec)
                if sec: sectors[sec].append(t)
        for nodes in sectors.values():
            for i, a in enumerate(nodes):
                for b in nodes[i+1:]:
                    self.g.add_edge(a, b, weight=max(self.g.get_edge_data(a,b,{}).get("weight",0.0), 0.18), kind="sector")
        if relations_csv and Path(relations_csv).exists():
            with open(relations_csv, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    a,b = r["source"].upper(), r["target"].upper()
                    if a in self.g and b in self.g:
                        self.g.add_edge(a,b,weight=float(r.get("weight",0.65)),kind=r.get("relation","known"))

    def update_comention(self, tickers: list[str]) -> None:
        ts = sorted(set(t for t in tickers if t in self.g))
        for i,a in enumerate(ts):
            for b in ts[i+1:]:
                old = self.g.get_edge_data(a,b,{}).get("weight",0.0)
                self.g.add_edge(a,b,weight=min(1.0, old + 0.08),kind="comention")

    def update(self, ticker: str, base_risk: float) -> float:
        if ticker not in self.g:
            return 0.0
        self.risk[ticker] = max(0.82 * self.risk[ticker], max(0.0, min(1.0, base_risk)))
        nodes = list(self.g.nodes)
        x = {n: self.risk[n] for n in nodes}
        alpha = max(0.0, min(0.75, SETTINGS.graph_diffusion))
        for _ in range(3):
            nxv = {}
            for n in nodes:
                nbrs = list(self.g[n].items())
                den = sum(float(d.get("weight",1.0)) for _,d in nbrs)
                incoming = sum(float(d.get("weight",1.0))*x[j] for j,d in nbrs) / den if den else 0.0
                nxv[n] = (1-alpha)*self.risk[n] + alpha*incoming
            x = nxv
        propagated = max(0.0, x[ticker] - self.risk[ticker])
        # score captures both received contagion and network-amplified local stress.
        return max(0.0, min(1.0, propagated*2.2 + x[ticker]*0.25))
