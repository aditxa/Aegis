from __future__ import annotations

"""Optional trainable temporal graph-attention risk model.

This is intentionally not required by the dependency-light demo.  With the MAX profile, it can be
trained on a time-indexed firm graph (supplier/customer/ownership/counterparty/sector/return links)
and node features emitted by AegisSignal.  The production engine keeps a transparent diffusion
fallback when no trained graph checkpoint exists.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class TemporalGATSpec:
    input_dim: int
    hidden_dim: int = 48
    heads: int = 4
    gru_dim: int = 64
    dropout: float = 0.12


class TemporalGraphRiskModel:
    def __init__(self, spec: TemporalGATSpec):
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise RuntimeError("TemporalGraphRiskModel requires torch; install requirements-max.txt") from e

        self.torch = torch
        self.nn = nn
        self.spec = spec
        self.model = self._build()

    def _build(self):
        torch, nn, spec = self.torch, self.nn, self.spec

        class DenseGATHead(nn.Module):
            def __init__(self, d_in, d_out, dropout):
                super().__init__()
                self.W = nn.Linear(d_in, d_out, bias=False)
                self.a_src = nn.Linear(d_out, 1, bias=False)
                self.a_dst = nn.Linear(d_out, 1, bias=False)
                self.drop = nn.Dropout(dropout)
                self.act = nn.LeakyReLU(0.2)

            def forward(self, x, adj):
                # x: [B,N,F], adj: [B,N,N] weighted in [0,1]
                h = self.W(x)
                e = self.act(self.a_src(h) + self.a_dst(h).transpose(1,2))
                # Prior graph weights enter attention logits rather than being treated as truth.
                logits = e + torch.log(torch.clamp(adj, min=1e-6))
                mask = adj > 0
                logits = logits.masked_fill(~mask, -1e9)
                alpha = torch.softmax(logits, dim=-1)
                alpha = self.drop(alpha)
                return torch.bmm(alpha, h)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                per = max(8, spec.hidden_dim // spec.heads)
                self.heads = nn.ModuleList([DenseGATHead(spec.input_dim, per, spec.dropout) for _ in range(spec.heads)])
                gout = per * spec.heads
                self.norm = nn.LayerNorm(gout)
                self.gru = nn.GRU(gout, spec.gru_dim, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(spec.gru_dim + spec.input_dim, 64), nn.GELU(), nn.Dropout(spec.dropout),
                    nn.Linear(64, 24), nn.GELU(), nn.Linear(24, 1), nn.Sigmoid(),
                )

            def forward(self, x, adj):
                # x [B,T,N,F], adj [B,N,N] or [B,T,N,N]
                B,T,N,F = x.shape
                zt = []
                for t in range(T):
                    a = adj[:,t] if adj.dim()==4 else adj
                    # Always keep self loops to avoid empty-neighbour softmax.
                    eye = torch.eye(N, device=x.device).unsqueeze(0)
                    a = torch.maximum(a, eye)
                    z = torch.cat([h(x[:,t], a) for h in self.heads], dim=-1)
                    zt.append(self.norm(z))
                z = torch.stack(zt, dim=1)  # [B,T,N,G]
                # each node has its own temporal trajectory
                z = z.permute(0,2,1,3).reshape(B*N,T,-1)
                _, h = self.gru(z)
                h = h[-1].reshape(B,N,-1)
                last = x[:,-1]
                return self.head(torch.cat([h,last],dim=-1)).squeeze(-1)

        return Net()

    def fit(self, X: np.ndarray, A: np.ndarray, y: np.ndarray, epochs: int = 60, lr: float = 2e-3,
            weight_decay: float = 2e-4, device: str | None = None) -> list[float]:
        torch = self.torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        at = torch.as_tensor(A, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        losses=[]
        # Focal-like BCE emphasizes rare severe nodes without discarding calibration completely.
        for _ in range(epochs):
            self.model.train(); opt.zero_grad()
            p = torch.clamp(self.model(xt,at),1e-5,1-1e-5)
            bce = -(yt*torch.log(p)+(1-yt)*torch.log(1-p))
            focal = (1.0 + 2.0*torch.abs(yt-p)) * bce
            loss = focal.mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            opt.step(); losses.append(float(loss.detach().cpu()))
        return losses

    def predict(self, X: np.ndarray, A: np.ndarray, device: str | None = None) -> np.ndarray:
        torch=self.torch; device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device); self.model.eval()
        with torch.no_grad():
            p=self.model(torch.as_tensor(X,dtype=torch.float32,device=device),torch.as_tensor(A,dtype=torch.float32,device=device))
        return p.cpu().numpy()

    def save(self, path: str | Path) -> None:
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
        self.torch.save({"spec":self.spec.__dict__,"state_dict":self.model.state_dict()},p)
        p.with_suffix(p.suffix+".json").write_text(json.dumps(self.spec.__dict__,indent=2))

    @classmethod
    def load(cls, path: str | Path):
        import torch
        b=torch.load(path,map_location="cpu")
        obj=cls(TemporalGATSpec(**b["spec"])); obj.model.load_state_dict(b["state_dict"]); return obj
