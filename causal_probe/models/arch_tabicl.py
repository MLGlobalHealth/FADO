"""TabICL-style per-column Set-Transformer column embedder.

In legacy `rowcol`, each cell starts as a scalar projected to d_model.
Row attention then mixes per-row across columns. Here we add an explicit
per-column Set-Transformer embedder *before* the row+col stack: each
column's n scalar values are treated as a set, embedded with a
SetTransformer (induced-set self-attention), and the resulting K
inducing-point summary vectors are mean-pooled into a single d-vector
per column. This per-column summary is then concatenated to the
per-cell representations as a learned global context for that column
before the standard row+col attention.

Reference:
  TabICL: https://arxiv.org/pdf/2502.05564

Notes:
  - The set-attention is per-column (each column is its own input), so it
    is permutation-invariant over rows by construction.
  - We use the same K (inducing points) as arch_settrans for simplicity.
  - This is a "Set Transformer for embeddings", not a replacement for the
    row-attention trunk. The trunk still sees per-cell tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_probe.model import _Block


@dataclass
class TabICLConfig:
    d_model: int = 384
    n_heads: int = 8
    n_row_layers: int = 6
    n_col_layers: int = 6
    n_embed_layers: int = 2          # Set-Transformer depth for column embedder
    k_inducing: int = 8
    dropout: float = 0.0
    ff_factor: int = 2


class _SetTransformerColumnEmbedder(nn.Module):
    """Per-column SetTransformer. Treats the n row values of a column as
    a set; emits K inducing-point summaries; mean-pools to one d-vector
    per column. Weight-shared across columns."""

    def __init__(self, cfg: TabICLConfig):
        super().__init__()
        self.in_proj = nn.Linear(1, cfg.d_model)
        self.queries = nn.Parameter(
            torch.randn(cfg.k_inducing, cfg.d_model) / math.sqrt(cfg.d_model)
        )
        self.cross = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        # Self-attention over inducing points (deepens the embedder).
        self.self_blocks = nn.ModuleList(
            [_Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_embed_layers)]
        )
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, Z: Tensor) -> Tensor:
        # Z : (B, n, P) scalar values. Return per-column embedding (B, P, d).
        B, n, P = Z.shape
        Zp = Z.permute(0, 2, 1).reshape(B * P, n, 1)         # (B*P, n, 1)
        h = self.in_proj(Zp)                                  # (B*P, n, d)
        K = self.queries.shape[0]
        q = self.queries.unsqueeze(0).expand(B * P, -1, -1)   # (B*P, K, d)
        pooled, _ = self.cross(q, h, h, need_weights=False)   # (B*P, K, d)
        pooled = self.ln(pooled)
        for block in self.self_blocks:
            pooled = block(pooled)
        col_emb = pooled.mean(dim=1).reshape(B, P, -1)        # (B, P, d)
        return col_emb


class TabICLProbe(nn.Module):
    """Per-column embedder → row attention with column context → col attention → readout."""

    def __init__(self, cfg: TabICLConfig):
        super().__init__()
        self.cfg = cfg
        self.embedder = _SetTransformerColumnEmbedder(cfg)
        self.value_proj = nn.Linear(1, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.row_blocks = nn.ModuleList(
            [_Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_row_layers)]
        )
        self.col_blocks = nn.ModuleList(
            [_Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_col_layers)]
        )
        self.head = nn.Linear(cfg.d_model, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        if X.dim() != 3 or y.dim() != 2:
            raise ValueError(f"X must be (B,n,p), y must be (B,n); got {X.shape}, {y.shape}")
        B, n, p = X.shape
        Z = torch.cat([X, y.unsqueeze(-1)], dim=-1)             # (B, n, p+1)
        # Per-column Set-Transformer embedding (global column context).
        col_ctx = self.embedder(Z)                              # (B, p+1, d)
        # Per-cell value projection + type embedding.
        h = self.value_proj(Z.unsqueeze(-1))                    # (B, n, p+1, d)
        type_ids = torch.zeros(p + 1, dtype=torch.long, device=X.device)
        type_ids[p] = 1
        h = h + self.type_emb(type_ids).view(1, 1, p + 1, -1)
        # Add the per-column context as a constant offset (broadcast over rows).
        h = h + col_ctx.unsqueeze(1)                            # broadcast over n
        # Row attention.
        h_r = h.reshape(B * n, p + 1, -1)
        for block in self.row_blocks:
            h_r = block(h_r)
        h_r = h_r.reshape(B, n, p + 1, -1)
        # Pool rows + col attention + readout.
        col_emb = h_r.mean(dim=1)                               # (B, p+1, d)
        for block in self.col_blocks:
            col_emb = block(col_emb)
        feat_emb = col_emb[:, :p, :]
        return self.head(feat_emb).squeeze(-1)


def build(
    d_model: int = 384,
    n_heads: int = 8,
    n_row_layers: int = 6,
    n_col_layers: int = 6,
    n_embed_layers: int = 2,
    k_inducing: int = 8,
    dropout: float = 0.0,
    ff_factor: int = 2,
    **kwargs,
) -> TabICLProbe:
    cfg = TabICLConfig(
        d_model=d_model, n_heads=n_heads,
        n_row_layers=n_row_layers, n_col_layers=n_col_layers,
        n_embed_layers=n_embed_layers, k_inducing=k_inducing,
        dropout=dropout, ff_factor=ff_factor,
    )
    return TabICLProbe(cfg)
