"""Minimal column-permutation-equivariant causal probe model.

Architecture: row-then-column transformer

  1. Each cell (row r, column c) is embedded via a shared Linear(1, d).
  2. A learned 2-element TYPE embedding distinguishes feature columns from
     the target column — this is the ONLY way the model knows which column
     is Y (there is no per-column parameter otherwise).
  3. Row-level attention: for each row r, attend across columns (p+1 tokens
     per row, shared-weight attention) — lets each column's per-row repr
     depend on the target value at that row.
  4. Column pooling: mean across rows gives (B, p+1, d).
  5. Column-level attention: attend across the p+1 column embeddings so the
     prediction for column i can depend on the representations of other
     columns — necessary to distinguish ancestors from descendants.
  6. Feature head: linear readout to a scalar per feature column; drop the
     target column's output.

All linear layers are shared across columns. Column order is permutation-
equivariant (up to the target-column type embedding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ProbeConfig:
    d_model: int = 64
    n_heads: int = 4
    n_row_layers: int = 2
    n_col_layers: int = 2
    dropout: float = 0.0
    ff_factor: int = 2
    # Architecture ablation flags (handoff §9.5).
    no_row_attn: bool = False
    no_col_attn: bool = False
    no_type_emb: bool = False


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_factor: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_factor * d_model),
            nn.GELU(),
            nn.Linear(ff_factor * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x


class CausalProbe(nn.Module):
    """Predict tau_i for every feature column from an observational table."""

    def __init__(self, cfg: ProbeConfig = ProbeConfig()):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(1, cfg.d_model)
        # 2 column types: feature (0) and target (1).
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.row_blocks = nn.ModuleList([
            _Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
            for _ in range(cfg.n_row_layers)
        ])
        self.col_blocks = nn.ModuleList([
            _Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
            for _ in range(cfg.n_col_layers)
        ])
        self.head = nn.Linear(cfg.d_model, 1)
        # Small init on the output projection so the untrained model
        # predicts near-zero tau.
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
        Z = torch.cat([X, y.unsqueeze(-1)], dim=-1)           # (B, n, p+1)
        # Value projection: each cell -> d_model.
        h = self.value_proj(Z.unsqueeze(-1))                  # (B, n, p+1, d)
        # Type embedding (ablation: no_type_emb skips this signal entirely;
        # the model then has no way to identify which column is Y).
        if not self.cfg.no_type_emb:
            type_ids = torch.zeros(p + 1, dtype=torch.long, device=X.device)
            type_ids[p] = 1   # target token is last
            te = self.type_emb(type_ids)                          # (p+1, d)
            h = h + te.view(1, 1, p + 1, -1)
        # Row-level attention: treat each row's p+1 tokens as a sequence.
        # Ablation: no_row_attn replaces this with identity.
        h_r = h.view(B * n, p + 1, -1)
        if not self.cfg.no_row_attn:
            for block in self.row_blocks:
                h_r = block(h_r)
        h_r = h_r.view(B, n, p + 1, -1)
        # Column pooling: mean over rows → (B, p+1, d).
        col_emb = h_r.mean(dim=1)
        # Column-level attention: let each column's repr see other columns.
        # Ablation: no_col_attn replaces this with identity.
        if not self.cfg.no_col_attn:
            for block in self.col_blocks:
                col_emb = block(col_emb)
        # Drop target column, readout per feature.
        feat_emb = col_emb[:, :p, :]                           # (B, p, d)
        tau_hat = self.head(feat_emb).squeeze(-1)              # (B, p)
        return tau_hat


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
