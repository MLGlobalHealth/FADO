"""Faithful SAINT intersample attention (Somepalli et al., 2021), for an
apples-to-apples test against our axial `arch_axial`.

SAINT's intersample attention (their Algorithm 1) CONCATENATES a row's feature
embeddings into a single token of dim P*d, then attends over the n samples
(rows), so features from different samples all communicate:

    x : (B, n, P, d)  ->  reshape (B, n, P*d)  ->  attn over n  ->  (B, n, P, d)

This differs from our axial `_IntersampleBlock`, which attends over samples
PER COLUMN independently (feature j only sees feature j in other rows). Every
other component (shared scalar embedder, per-row feature attention, type
embedding, readout) is identical to arch_axial, so a head-to-head
isolates the intersample mechanism (SAINT whole-row vs axial per-column).

Because the intersample token dim is P*d, this arch is far more parameter-hungry
per unit d_model than the axial variant — a direct consequence of SAINT's design.
Inputs are assumed padded to p_max (P = p_max + 1 fixed); we pad internally if not.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_probe.model import _Block


@dataclass
class SaintRealConfig:
    d_model: int = 48
    n_heads: int = 8
    n_layers: int = 6
    p_max: int = 15               # real features; columns P = p_max + 1 (incl. y)
    dropout: float = 0.0
    ff_factor: int = 2


class _SaintIntersampleBlock(nn.Module):
    """SAINT's true intersample attention: concat a row's P feature embeddings
    into one token (dim P*d), attend over the n samples, reshape back."""

    def __init__(self, d_model: int, p_cols: int, n_heads: int, dropout: float, ff_factor: int):
        super().__init__()
        self.p_cols = p_cols
        self.d_model = d_model
        self.token_dim = p_cols * d_model
        # n_heads must divide token_dim; fall back to 1 head if not.
        heads = n_heads if self.token_dim % n_heads == 0 else 1
        self.block = _Block(self.token_dim, heads, dropout, ff_factor)

    def forward(self, h: Tensor) -> Tensor:
        B, n, P, d = h.shape
        if P < self.p_cols:                       # pad columns to p_cols
            pad = h.new_zeros(B, n, self.p_cols - P, d)
            hp = torch.cat([h, pad], dim=2)
        else:
            hp = h
        x = hp.reshape(B, n, self.p_cols * d)     # concat features per row
        x = self.block(x)                         # attention over the n samples
        x = x.reshape(B, n, self.p_cols, d)
        return x[:, :, :P, :]                     # drop padded columns


class SaintRealProbe(nn.Module):
    """Shared scalar embedder -> (per-row feature attn + SAINT intersample) x L."""

    def __init__(self, cfg: SaintRealConfig):
        super().__init__()
        self.cfg = cfg
        p_cols = cfg.p_max + 1
        self.value_proj = nn.Linear(1, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.row_blocks = nn.ModuleList(
            [_Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_layers)]
        )
        self.saint_blocks = nn.ModuleList(
            [_SaintIntersampleBlock(cfg.d_model, p_cols, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_layers)]
        )
        self.head = nn.Linear(cfg.d_model, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        if X.dim() != 3 or y.dim() != 2:
            raise ValueError(f"X must be (B,n,p), y must be (B,n); got {X.shape}, {y.shape}")
        B, n, p = X.shape
        Z = torch.cat([X, y.unsqueeze(-1)], dim=-1)              # (B, n, p+1)
        h = self.value_proj(Z.unsqueeze(-1))                      # (B, n, p+1, d)
        type_ids = torch.zeros(p + 1, dtype=torch.long, device=X.device)
        type_ids[p] = 1
        h = h + self.type_emb(type_ids).view(1, 1, p + 1, -1)
        for row_blk, saint_blk in zip(self.row_blocks, self.saint_blocks):
            h_r = h.reshape(B * n, p + 1, -1)
            h = row_blk(h_r).reshape(B, n, p + 1, -1)
            h = saint_blk(h)
        col_emb = h.mean(dim=1)                                   # (B, p+1, d)
        feat_emb = col_emb[:, :p, :]                              # (B, p, d)
        return self.head(feat_emb).squeeze(-1)


def build(
    d_model: int = 48,
    n_heads: int = 8,
    n_row_layers: int = 6,
    n_col_layers: int = 6,           # accept-and-ignore: uses one n_layers
    dropout: float = 0.0,
    ff_factor: int = 2,
    n_layers: int | None = None,
    p_max: int = 15,
    **kwargs,
) -> SaintRealProbe:
    if n_layers is None:
        n_layers = max(n_row_layers, n_col_layers)
    cfg = SaintRealConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        p_max=p_max, dropout=dropout, ff_factor=ff_factor,
    )
    return SaintRealProbe(cfg)
