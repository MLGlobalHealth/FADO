"""Two-axis (axial) attention over a tabular dataset.

The model reads a whole dataset — n samples (rows) x p features plus the outcome
y (so p+1 columns) — and returns a per-feature vector tau_hat in one forward pass.
Each scalar cell is embedded by a single shared Linear(1 -> d); a learned type
embedding marks feature-columns vs the target column. Then L layers each apply
two attentions on the (n x (p+1) x d) grid, along orthogonal axes:

  * feature-axis attention: within each row, the p+1 cells attend across columns;
  * sample-axis attention:  within each column, the n cells attend across samples.

Mean-pooling over samples and a linear per-feature head produce tau_hat.

This is standard axial / row-column attention over a table (cf. Axial Transformer;
Set Transformer for sample-permutation invariance; the row/column attention of
in-context tabular transformers such as TabPFN / TabICL). The contribution of the
paper is the task and training (amortized causal attribution over an SCM prior),
not the attention pattern.

Checkpoint compatibility: earlier checkpoints stored these blocks under the
attribute names ``row_blocks`` / ``intersample_blocks`` and the arch key
"intersample"; both are remapped/aliased so old checkpoints load unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_probe.model import _Block


@dataclass
class AxialConfig:
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6              # paired feature-axis + sample-axis blocks
    dropout: float = 0.0
    ff_factor: int = 2
    # Ablation toggles (default off = full model), wired to the shared --no-*-attn
    # flags so one ablation harness works across archs:
    #   no_row_attn  -> drop SAMPLE-axis attention (across rows/samples)
    #   no_col_attn  -> drop FEATURE-axis attention (across columns/features)
    #   no_type_emb  -> drop the feature/target type embedding
    no_row_attn: bool = False
    no_col_attn: bool = False
    no_type_emb: bool = False


class _SampleAxisAttention(nn.Module):
    """Attention across the n samples, applied per column independently: each
    column's n cell-representations attend to each other. Reshapes the grid so a
    standard transformer block attends along the sample axis."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_factor: int):
        super().__init__()
        self.block = _Block(d_model, n_heads, dropout, ff_factor)

    def forward(self, h: Tensor) -> Tensor:
        B, n, P, d = h.shape
        h_perm = h.permute(0, 2, 1, 3).reshape(B * P, n, d)   # (B*P, n, d): per column
        out = self.block(h_perm)
        return out.reshape(B, P, n, d).permute(0, 2, 1, 3)


class AxialTabularProbe(nn.Module):
    """Shared cell embedder -> (feature-axis attn + sample-axis attn) x L -> readout."""

    def __init__(self, cfg: AxialConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(1, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.feat_blocks = nn.ModuleList(
            [_Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_layers)]
        )
        self.sample_blocks = nn.ModuleList(
            [_SampleAxisAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_factor)
             for _ in range(cfg.n_layers)]
        )
        self.head = nn.Linear(cfg.d_model, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Backward-compat: earlier checkpoints named the blocks row_blocks /
        # intersample_blocks. Remap those keys onto feat_blocks / sample_blocks.
        remapped = {
            k.replace("row_blocks.", "feat_blocks.")
             .replace("intersample_blocks.", "sample_blocks."): v
            for k, v in state_dict.items()
        }
        return super().load_state_dict(remapped, *args, **kwargs)

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
        for feat_blk, sample_blk in zip(self.feat_blocks, self.sample_blocks):
            if not self.cfg.no_col_attn:                          # feature-axis attention
                h_r = h.reshape(B * n, p + 1, -1)
                h = feat_blk(h_r).reshape(B, n, p + 1, -1)
            if not self.cfg.no_row_attn:                          # sample-axis attention
                h = sample_blk(h)
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
) -> AxialTabularProbe:
    if n_layers is None:
        n_layers = max(n_row_layers, n_col_layers)
    cfg = AxialConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        dropout=dropout, ff_factor=ff_factor,
        no_row_attn=no_row_attn, no_col_attn=no_col_attn, no_type_emb=no_type_emb,
    )
    return AxialTabularProbe(cfg)
