"""Architecture registry for the foundation-scaling sprint.

For backwards compatibility, the legacy `causal_probe.model` module
remains and re-exports `CausalProbe` / `ProbeConfig` / `count_params`
from `causal_probe.models.arch_rowcol`. New architectures live as
`arch_<name>.py` files in this package and register themselves below.

Usage:
    from causal_probe.models import build_arch
    model = build_arch("rowcol", d_model=384, ...)
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from causal_probe.models import (
    arch_attnpool,
    arch_axial,
    arch_parallel,
    arch_rowcol,
    arch_saint,
    arch_saint_real,
    arch_settrans,
    arch_tabicl,
)

ARCH_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "rowcol": arch_rowcol.build,
    "attnpool": arch_attnpool.build,
    "parallel": arch_parallel.build,
    "settrans": arch_settrans.build,
    "tabicl": arch_tabicl.build,
    "saint": arch_saint.build,
    "saint_real": arch_saint_real.build,
    "axial": arch_axial.build,
    "intersample": arch_axial.build,   # deprecated alias for old ckpts/invocations
}


def build_arch(name: str, **kwargs) -> nn.Module:
    if name not in ARCH_REGISTRY:
        raise ValueError(
            f"unknown arch '{name}'; registered: {sorted(ARCH_REGISTRY)}"
        )
    return ARCH_REGISTRY[name](**kwargs)


def list_archs() -> list[str]:
    return sorted(ARCH_REGISTRY.keys())
