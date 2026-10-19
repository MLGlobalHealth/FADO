"""Attention-pooled head architecture variant.

Same row-then-column transformer trunk as `arch_rowcol`, but the per-feature
readout is replaced with a *role-codebook* cross-attention head. Each
feature column's representation queries a small learned set of "role"
tokens (acting as a soft codebook) before the linear readout, letting
the model dispatch features to specialised role-vectors based on what
each feature appears to be doing in the SCM (direct cause / mediator /
descendant / irrelevant / etc.).

This preserves column permutation equivariance: the role tokens are
*not* per-feature-position parameters — they are a small fixed-size
codebook shared across all features and all permutations. The per-
feature dispatch is purely a function of the feature's own permutation-
equivariant column embedding.

Motivation: the legacy `arch_rowcol` predicts τ_i via a single shared
`Linear(d, 1)` applied to col_emb[:, i, :]. If different features play
qualitatively different causal roles in OOD motifs, that shared head
must compress the role distinction into a single linear mapping.
attnpool gives the head an extra mechanism to specialise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_probe.model import _Block, ProbeConfig as _RowcolProbeConfig


@dataclass
class AttnPoolConfig:
    d_model: int = 64
    n_heads: int = 4
    n_row_layers: int = 2
    n_col_layers: int = 2
    dropout: float = 0.0
    ff_factor: int = 2
    # attnpool-specific:
    n_roles: int = 4               # size of the learned role codebook
    n_pool_heads: int | None = None  # heads for the cross-attention; defaults to n_heads


class AttnPoolProbe(nn.Module):
    """Row-col transformer trunk + role-codebook attention-pool readout."""

    def __init__(self, cfg: AttnPoolConfig = AttnPoolConfig()):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(1, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)  # feature=0, target=1
        self.row_blocks = nn.ModuleList([
            _Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
            for _ in range(cfg.n_row_layers)
        ])
        self.col_blocks = nn.ModuleList([
            _Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
            for _ in range(cfg.n_col_layers)
        ])
        # Role codebook: K learned tokens that play the role of K abstract
        # "causal roles". Each per-feature query attends to these.
        self.role_tokens = nn.Parameter(torch.randn(cfg.n_roles, cfg.d_model) / math.sqrt(cfg.d_model))
        n_pool_heads = cfg.n_pool_heads or cfg.n_heads
        self.pool_attn = nn.MultiheadAttention(
            cfg.d_model, n_pool_heads, batch_first=True, dropout=cfg.dropout
        )
        self.ln_pool = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        # Small init on the readout so the untrained model predicts ~0.
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        """Return (B, p) tau predictions.

        X : (B, n, p) float
        y : (B, n) float
        """
        if X.dim() != 3 or y.dim() != 2:
            raise ValueError(f"X must be (B,n,p), y must be (B,n); got {X.shape}, {y.shape}")
        B, n, p = X.shape
        if y.shape != (B, n):
            raise ValueError(f"y shape {y.shape} != ({B}, {n})")
        # Combine into (B, n, p+1) with Y as the last column.
        Z = torch.cat([X, y.unsqueeze(-1)], dim=-1)
        h = self.value_proj(Z.unsqueeze(-1))                  # (B, n, p+1, d)
        type_ids = torch.zeros(p + 1, dtype=torch.long, device=X.device)
        type_ids[p] = 1
        te = self.type_emb(type_ids)
        h = h + te.view(1, 1, p + 1, -1)
        # Row-level attention.
        h_r = h.view(B * n, p + 1, -1)
        for block in self.row_blocks:
            h_r = block(h_r)
        h_r = h_r.view(B, n, p + 1, -1)
        # Column pooling: mean over rows.
        col_emb = h_r.mean(dim=1)                              # (B, p+1, d)
        # Column-level attention.
        for block in self.col_blocks:
            col_emb = block(col_emb)
        # Drop target column → per-feature column embeddings.
        feat_emb = col_emb[:, :p, :]                           # (B, p, d)
        # Role-codebook cross-attention: each feature queries the K role
        # tokens. Roles are shared across batch, fixed in count.
        roles = self.role_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        h_pool, _ = self.pool_attn(query=feat_emb, key=roles, value=roles, need_weights=False)
        feat_emb = self.ln_pool(feat_emb + h_pool)
        # Readout.
        tau_hat = self.head(feat_emb).squeeze(-1)              # (B, p)
        return tau_hat


def build(
    d_model: int = 96,
    n_heads: int = 4,
    n_row_layers: int = 3,
    n_col_layers: int = 3,
    dropout: float = 0.0,
    ff_factor: int = 2,
    n_roles: int = 4,
    n_pool_heads: int | None = None,
    # Accept-and-ignore the rowcol ablation flags so the registry caller
    # doesn't need a per-arch arg whitelist; attnpool doesn't expose them.
    no_row_attn: bool = False,
    no_col_attn: bool = False,
    no_type_emb: bool = False,
) -> AttnPoolProbe:
    if no_row_attn or no_col_attn or no_type_emb:
        raise NotImplementedError(
            "arch_attnpool does not support no_row_attn / no_col_attn / no_type_emb"
        )
    cfg = AttnPoolConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_row_layers=n_row_layers,
        n_col_layers=n_col_layers,
        dropout=dropout,
        ff_factor=ff_factor,
        n_roles=n_roles,
        n_pool_heads=n_pool_heads,
    )
    return AttnPoolProbe(cfg)
