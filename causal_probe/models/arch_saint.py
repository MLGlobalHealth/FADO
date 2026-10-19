"""SAINT-style probe: per-column MLP embedder + intersample attention.

Two diffs vs legacy `rowcol`:

1. **Per-column MLP embedder (no weight sharing).** Each of the p+1
   columns gets its own MLP that maps a scalar value to d_model. The
   legacy projection is a single shared `nn.Linear(1, d_model)`; SAINT's
   intuition is that different columns have semantically distinct
   "physics" (treatment vs covariate vs noise) and benefit from
   non-shared input embedders.

   We support variable p_max (the model accepts X with any p in some
   range), so the per-column MLPs are stored as a single embedding-table
   indexed by column position. The same column position always uses the
   same MLP weights regardless of which dataset is being processed; this
   is a soft form of position-conditioned embedding.

2. **Intersample attention.** After the per-cell row attention runs,
   we add an intersample-attention block that lets each row attend to
   every other row's representation of the same column. Implementation:
   reshape (B, n, p+1, d) → (B*(p+1), n, d) and run a transformer block
   over the n axis. This is the workhorse of SAINT.

Reference:
  SAINT (Somepalli et al. 2021): https://arxiv.org/pdf/2106.01342
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_probe.model import _Block


@dataclass
class SAINTConfig:
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6                # paired row + intersample blocks
    p_max: int = 16                  # support up to p_max columns (incl. y)
    embedder_hidden: int = 256       # per-column MLP hidden width
    dropout: float = 0.0
    ff_factor: int = 2


class _PerColumnMLPEmbedder(nn.Module):
    """A bank of p_max small MLPs, one per column index. Each maps a
    scalar to d_model. Weights are NOT shared across column positions."""

    def __init__(self, cfg: SAINTConfig):
        super().__init__()
        # Implementation: store w1 ∈ (p_max, 1, hidden) and b1 ∈ (p_max, hidden),
        # then w2 ∈ (p_max, hidden, d) and b2 ∈ (p_max, d). Per-column gather.
        self.w1 = nn.Parameter(
            torch.randn(cfg.p_max, 1, cfg.embedder_hidden) / 1.0
        )
        self.b1 = nn.Parameter(torch.zeros(cfg.p_max, cfg.embedder_hidden))
        self.w2 = nn.Parameter(
            torch.randn(cfg.p_max, cfg.embedder_hidden, cfg.d_model)
            / (cfg.embedder_hidden ** 0.5)
        )
        self.b2 = nn.Parameter(torch.zeros(cfg.p_max, cfg.d_model))

    def forward(self, Z: Tensor) -> Tensor:
        # Z : (B, n, P). For each column j in 0..P-1, apply MLP_j to Z[:, :, j].
        B, n, P = Z.shape
        # Gather first P MLPs.
        w1 = self.w1[:P]    # (P, 1, hidden)
        b1 = self.b1[:P]    # (P, hidden)
        w2 = self.w2[:P]    # (P, hidden, d)
        b2 = self.b2[:P]    # (P, d)
        # Reshape Z to (B*n, P, 1) for batched per-column matmul via einsum.
        Z_ = Z.reshape(B * n, P, 1)
        h1 = torch.einsum("bpc,pcd->bpd", Z_, w1) + b1   # (B*n, P, hidden)
        h1 = torch.relu(h1)
        h2 = torch.einsum("bph,phd->bpd", h1, w2) + b2   # (B*n, P, d)
        return h2.reshape(B, n, P, -1)


class _IntersampleBlock(nn.Module):
    """Per-column attention over the n_rows samples. Each column's
    representations across rows attend to each other (intersample
    attention from SAINT)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_factor: int):
        super().__init__()
        self.block = _Block(d_model, n_heads, dropout, ff_factor)

    def forward(self, h: Tensor) -> Tensor:
        # h : (B, n, P, d) → reshape so n is the sequence dim, per column.
        B, n, P, d = h.shape
        h_perm = h.permute(0, 2, 1, 3).reshape(B * P, n, d)
        out = self.block(h_perm)
        return out.reshape(B, P, n, d).permute(0, 2, 1, 3)


class SAINTProbe(nn.Module):
    """Per-column MLP embedder → (row attn + intersample attn) × n_layers → readout."""

    def __init__(self, cfg: SAINTConfig):
        super().__init__()
        self.cfg = cfg
        self.embedder = _PerColumnMLPEmbedder(cfg)
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.row_blocks = nn.ModuleList(
            [_Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_layers)]
        )
        self.intersample_blocks = nn.ModuleList(
            [_IntersampleBlock(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_layers)]
        )
        self.head = nn.Linear(cfg.d_model, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        if X.dim() != 3 or y.dim() != 2:
            raise ValueError(f"X must be (B,n,p), y must be (B,n); got {X.shape}, {y.shape}")
        B, n, p = X.shape
        if p + 1 > self.cfg.p_max:
            raise ValueError(f"p+1={p+1} > cfg.p_max={self.cfg.p_max}; rebuild with larger p_max")
        Z = torch.cat([X, y.unsqueeze(-1)], dim=-1)               # (B, n, p+1)
        h = self.embedder(Z)                                       # (B, n, p+1, d)
        type_ids = torch.zeros(p + 1, dtype=torch.long, device=X.device)
        type_ids[p] = 1
        h = h + self.type_emb(type_ids).view(1, 1, p + 1, -1)
        for row_blk, inter_blk in zip(self.row_blocks, self.intersample_blocks):
            # Row attention: each row's p+1 tokens attend to each other.
            h_r = h.reshape(B * n, p + 1, -1)
            h_r = row_blk(h_r).reshape(B, n, p + 1, -1)
            # Intersample attention: each column's n tokens attend to each other.
            h = inter_blk(h_r)
        # Pool rows + readout per feature column.
        col_emb = h.mean(dim=1)                                    # (B, p+1, d)
        feat_emb = col_emb[:, :p, :]                               # (B, p, d)
        return self.head(feat_emb).squeeze(-1)


def build(
    d_model: int = 384,
    n_heads: int = 8,
    n_row_layers: int = 6,
    n_col_layers: int = 6,           # accept-and-ignore: SAINT uses one n_layers
    embedder_hidden: int = 256,
    p_max: int = 16,
    dropout: float = 0.0,
    ff_factor: int = 2,
    n_layers: int | None = None,
    **kwargs,
) -> SAINTProbe:
    if n_layers is None:
        n_layers = max(n_row_layers, n_col_layers)
    cfg = SAINTConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        p_max=p_max, embedder_hidden=embedder_hidden,
        dropout=dropout, ff_factor=ff_factor,
    )
    return SAINTProbe(cfg)
