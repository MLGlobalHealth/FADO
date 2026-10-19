"""SetTransformer-style row pooling via inducing-point attention (PMA).

The legacy `rowcol` architecture pools n_rows samples into one per-column
representation via simple mean. This loses information about row-level
heterogeneity; SetTransformer's pooling-by-multihead-attention (PMA)
replaces it with a learned set of K "inducing" query tokens that attend
over the n_rows row-tokens, producing K summary tokens.

Here we apply PMA-style pooling at the row dimension: for each column,
we run cross-attention from K learned queries to the n row tokens of
that column, producing K summary tokens per column. The K summaries are
then mean-pooled (or attention-pooled) into a single per-column vector
before col-level attention + readout.

K = 1 collapses to ordinary mean-pooling (modulo the cross-attention
parameters). Default K = 8 gives the model multiple "views" of the rows.

Reference:
  SetTransformer (Lee et al. 2019): https://arxiv.org/pdf/1810.00825
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_probe.model import _Block


@dataclass
class SetTransConfig:
    d_model: int = 384
    n_heads: int = 8
    n_row_layers: int = 6
    n_col_layers: int = 6
    dropout: float = 0.0
    ff_factor: int = 2
    k_inducing: int = 8


class _RowPMA(nn.Module):
    """Pooling-by-Multihead-Attention over the row dimension.

    Each column has n row tokens; K learned queries cross-attend to them,
    producing K summary tokens per column. The K are then mean-pooled.
    Cross-attention is shared across all columns (weight-shared).
    """

    def __init__(self, d_model: int, n_heads: int, k_inducing: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k_inducing, d_model) / math.sqrt(d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, h: Tensor) -> Tensor:
        # h : (B, n, P, d). PMA collapses the n axis to k_inducing.
        B, n, P, d = h.shape
        K = self.queries.shape[0]
        # Reshape so each (B, column) pair becomes its own attention batch.
        # We want queries (B*P, K, d) and keys/values (B*P, n, d).
        h_perm = h.permute(0, 2, 1, 3).reshape(B * P, n, d)
        q = self.queries.unsqueeze(0).expand(B * P, -1, -1)  # (B*P, K, d)
        pooled, _ = self.attn(q, h_perm, h_perm, need_weights=False)
        pooled = self.ln(pooled)
        # Mean over inducing-point dimension to get a single per-column rep.
        return pooled.mean(dim=1).reshape(B, P, d)


class SetTransProbe(nn.Module):
    """Row-attention trunk → PMA pooling → col-attention → readout."""

    def __init__(self, cfg: SetTransConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(1, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.row_blocks = nn.ModuleList(
            [_Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_row_layers)]
        )
        self.pma = _RowPMA(cfg.d_model, cfg.n_heads, cfg.k_inducing)
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
        Z = torch.cat([X, y.unsqueeze(-1)], dim=-1)        # (B, n, p+1)
        h = self.value_proj(Z.unsqueeze(-1))               # (B, n, p+1, d)
        type_ids = torch.zeros(p + 1, dtype=torch.long, device=X.device)
        type_ids[p] = 1
        h = h + self.type_emb(type_ids).view(1, 1, p + 1, -1)
        # Row-level attention.
        h_r = h.reshape(B * n, p + 1, -1)
        for block in self.row_blocks:
            h_r = block(h_r)
        h_r = h_r.reshape(B, n, p + 1, -1)
        # Inducing-point pooling over rows.
        col_emb = self.pma(h_r)                            # (B, p+1, d)
        # Col-level attention.
        for block in self.col_blocks:
            col_emb = block(col_emb)
        feat_emb = col_emb[:, :p, :]                       # (B, p, d)
        return self.head(feat_emb).squeeze(-1)


def build(
    d_model: int = 384,
    n_heads: int = 8,
    n_row_layers: int = 6,
    n_col_layers: int = 6,
    dropout: float = 0.0,
    ff_factor: int = 2,
    k_inducing: int = 8,
    **kwargs,
) -> SetTransProbe:
    cfg = SetTransConfig(
        d_model=d_model, n_heads=n_heads,
        n_row_layers=n_row_layers, n_col_layers=n_col_layers,
        dropout=dropout, ff_factor=ff_factor, k_inducing=k_inducing,
    )
    return SetTransProbe(cfg)
