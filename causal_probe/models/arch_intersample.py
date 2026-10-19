"""Hybrid: shared scalar embedder + SAINT-style intersample attention.

This isolates the per-column MLP design from SAINT's other innovation
(intersample attention). The intuition: in FADO's training prior, column
identities are permuted every episode, so SAINT's per-column MLPs are
forced to converge to redundant copies of a single shared embedder
(the gradients can't preserve column-specific signal). Replacing the
p_max bank of MLPs with one shared `nn.Linear(1, d_model)` — the same
embedder used by attnpool / rowcol — restores arbitrary-p generalization
without the (likely useless) per-column expressiveness.

If saint's 0.90 mixture pearson came from intersample attention, this
arch matches it. If it came from the per-column MLPs, this arch lands
near attnpool / rowcol.

Identical to arch_saint.py except: (1) the cell embedder is shared
`nn.Linear(1, d_model)`, (2) no p_max ceiling, (3) marginally fewer
params (no per-column MLP bank).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_probe.model import _Block


@dataclass
class IntersampleConfig:
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6              # paired row + intersample blocks
    dropout: float = 0.0
    ff_factor: int = 2
    # Ablation toggles (default off = full model). Intersample's row_blocks
    # attend across the p+1 feature axis; intersample_blocks attend across the
    # n_rows sample axis. Mapped to the shared --no-*-attn flags so the same
    # ablation harness works across archs:
    #   no_row_attn  -> drop cross-sample (intersample) attention
    #   no_col_attn  -> drop cross-feature (row_blocks) attention
    #   no_type_emb  -> drop the feature/target type embedding
    no_row_attn: bool = False
    no_col_attn: bool = False
    no_type_emb: bool = False


class _IntersampleBlock(nn.Module):
    """Per-column attention over the n_rows samples. Each column's
    representations across rows attend to each other (intersample
    attention from SAINT)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_factor: int):
        super().__init__()
        self.block = _Block(d_model, n_heads, dropout, ff_factor)

    def forward(self, h: Tensor) -> Tensor:
        B, n, P, d = h.shape
        h_perm = h.permute(0, 2, 1, 3).reshape(B * P, n, d)
        out = self.block(h_perm)
        return out.reshape(B, P, n, d).permute(0, 2, 1, 3)


class IntersampleProbe(nn.Module):
    """Shared scalar embedder -> (row attn + intersample attn) x n_layers -> readout."""

    def __init__(self, cfg: IntersampleConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(1, cfg.d_model)
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
        Z = torch.cat([X, y.unsqueeze(-1)], dim=-1)              # (B, n, p+1)
        h = self.value_proj(Z.unsqueeze(-1))                      # (B, n, p+1, d)
        if not self.cfg.no_type_emb:
            type_ids = torch.zeros(p + 1, dtype=torch.long, device=X.device)
            type_ids[p] = 1
            h = h + self.type_emb(type_ids).view(1, 1, p + 1, -1)
        for row_blk, inter_blk in zip(self.row_blocks, self.intersample_blocks):
            if not self.cfg.no_col_attn:
                h_r = h.reshape(B * n, p + 1, -1)
                h = row_blk(h_r).reshape(B, n, p + 1, -1)
            if not self.cfg.no_row_attn:
                h = inter_blk(h)
        col_emb = h.mean(dim=1)                                   # (B, p+1, d)
        feat_emb = col_emb[:, :p, :]                              # (B, p, d)
        return self.head(feat_emb).squeeze(-1)


def build(
    d_model: int = 384,
    n_heads: int = 8,
    n_row_layers: int = 6,
    n_col_layers: int = 6,           # accept-and-ignore: uses one n_layers
    dropout: float = 0.0,
    ff_factor: int = 2,
    n_layers: int | None = None,
    no_row_attn: bool = False,
    no_col_attn: bool = False,
    no_type_emb: bool = False,
    **kwargs,
) -> IntersampleProbe:
    if n_layers is None:
        n_layers = max(n_row_layers, n_col_layers)
    cfg = IntersampleConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        dropout=dropout, ff_factor=ff_factor,
        no_row_attn=no_row_attn, no_col_attn=no_col_attn, no_type_emb=no_type_emb,
    )
    return IntersampleProbe(cfg)
