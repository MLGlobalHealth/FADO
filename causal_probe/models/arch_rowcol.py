"""Row-then-column transformer (the legacy / current arch).

This file wraps `causal_probe.model.CausalProbe` so it can be addressed
through the architecture registry. The implementation itself lives in
`causal_probe/model.py` and is unchanged. Future variants (axial,
attnpool, settrans, ycross, ...) get their own `arch_<name>.py` files.
"""
from __future__ import annotations

from causal_probe.model import CausalProbe, ProbeConfig, count_params  # noqa: F401


def build(
    d_model: int = 96,
    n_heads: int = 4,
    n_row_layers: int = 3,
    n_col_layers: int = 3,
    dropout: float = 0.0,
    ff_factor: int = 2,
    no_row_attn: bool = False,
    no_col_attn: bool = False,
    no_type_emb: bool = False,
) -> CausalProbe:
    cfg = ProbeConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_row_layers=n_row_layers,
        n_col_layers=n_col_layers,
        dropout=dropout,
        ff_factor=ff_factor,
        no_row_attn=no_row_attn,
        no_col_attn=no_col_attn,
        no_type_emb=no_type_emb,
    )
    return CausalProbe(cfg)
