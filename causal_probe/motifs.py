"""Motif SCMs for causal-probe stress testing.

Each motif returns an SCM with p=5 features + Y, constructed with an
explicit structural matrix. Unused feature slots are filled with
independent noise features (X_i = eps_i, no edges). This pads the
feature count to the training p without changing semantics.

Labels are computed via A = (I - B)^{-1} like the random generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class MotifSpec:
    name: str
    description: str
    B: np.ndarray             # (N, N) structural matrix, Z = (X_1,...,X_p, Y)
    y_idx_in_Z: int           # index of Y in Z
    feature_to_Z: np.ndarray  # length-p mapping, feature i -> Z-index
    scm_p: int                # feature count
    notes: str = ""


def _blank(p: int) -> np.ndarray:
    """(p+1, p+1) zero matrix."""
    return np.zeros((p + 1, p + 1), dtype=np.float64)


def _w(rng: Optional[np.random.Generator], sign: float = 1.0) -> float:
    """Per-edge weight: sign * U(0.5, 1.5). Falls back to a fixed magnitude
    of 1.0 if no rng is provided, for backward compatibility with callers
    that haven't been updated."""
    if rng is None:
        return float(sign) * 1.0
    return float(sign) * float(rng.uniform(0.5, 1.5))


def _w_strong(rng: Optional[np.random.Generator], sign: float = 1.0) -> float:
    """Strong-cause edge weight: sign * U(1.5, 2.0). High end of the
    training-distribution range (probe trained with weight_hi=2.0).
    Used in motifs B and C to make the dominant predictor (X_1 in B,
    Z in C) clearly dominate tree splits, so the multicollinear decoy
    (X_2 in B, X_1 in C) gets a near-zero residual SHAP — tightens the
    figure's "irrelevant" cell story."""
    if rng is None:
        return float(sign) * 2.0
    return float(sign) * float(rng.uniform(1.5, 2.0))


def motif_direct_cause(p: int = 5, rng: Optional[np.random.Generator] = None) -> MotifSpec:
    """X1 -> Y. Other features are independent noise."""
    B = _blank(p)
    y_idx = p
    B[y_idx, 0] = _w(rng)  # X1 -> Y, sign + (interpretable τ_1 > 0)
    feature_to_Z = np.arange(p, dtype=int)
    return MotifSpec(
        name="A_direct_cause",
        description="X1 -> Y (X2..X_{p} independent)",
        B=B, y_idx_in_Z=y_idx, feature_to_Z=feature_to_Z, scm_p=p,
        notes="tau[0] should be large; tau[1..p-1] ≈ 0",
    )


def motif_proxy(p: int = 5, rng: Optional[np.random.Generator] = None) -> MotifSpec:
    """X1 -> Y and X1 -> X2. X2 is a proxy (correlated with Y, no causal effect).

    Edge weights use _w_strong so X_1 strongly dominates the prediction
    of Y AND X_2 is nearly determined by X_1 — tree-SHAP attributes
    almost everything to X_1, X_2's residual SHAP shrinks toward zero.
    Marginal Corr(X_2, Y) stays high (≈0.75), so the figure still
    contrasts marginal-vs-SHAP behaviour, just with more separation.
    """
    B = _blank(p)
    y_idx = p
    B[y_idx, 0] = _w_strong(rng)  # X1 -> Y, strong (X1 dominates Y)
    B[1, 0]     = _w_strong(rng)  # X1 -> X2, strong (X2 nearly det. by X1)
    feature_to_Z = np.arange(p, dtype=int)
    return MotifSpec(
        name="B_proxy",
        description="X1 -> Y, X1 -> X2 (X2 is descendant-of-cause proxy)",
        B=B, y_idx_in_Z=y_idx, feature_to_Z=feature_to_Z, scm_p=p,
        notes="tau[0] large; tau[1] == 0; both X1 and X2 predict Y marginally",
    )


def motif_observed_confounder(p: int = 5, rng: Optional[np.random.Generator] = None) -> MotifSpec:
    """Z -> X1, Z -> Y, X1 NOT -> Y. X1 is confounded with Y but no causal effect.

    Here Z IS observed (it's feature index 2; X1 is feature 0, Y at the end).
    The Z -> Y edge uses _w_strong so Z clearly dominates Y prediction — tree
    splits on Z absorb the predictive signal, X_1's residual TreeSHAP shrinks
    toward the noise floor, and the probe correctly identifies X_1 as
    causally inert (|τ̂| ≈ 0).
    """
    B = _blank(p)
    y_idx = p
    B[0, 2]     = _w(rng)         # Z -> X1, moderate (X1 stays a moderate proxy of Z)
    B[y_idx, 2] = _w_strong(rng)  # Z -> Y, strong (Z dominates Y prediction)
    feature_to_Z = np.arange(p, dtype=int)
    return MotifSpec(
        name="C_observed_confounder",
        description="Z=X3 -> X1 and Z=X3 -> Y (X1 has no arrow to Y)",
        B=B, y_idx_in_Z=y_idx, feature_to_Z=feature_to_Z, scm_p=p,
        notes="tau[0] == 0 (X1); tau[2] large (Z=X3 direct-causes Y)",
    )


def motif_mediator(p: int = 5, rng: Optional[np.random.Generator] = None) -> MotifSpec:
    """X1 -> X2 -> Y. X2 is mediator. Both have total effect but magnitudes differ."""
    B = _blank(p)
    y_idx = p
    B[1, 0]     = _w(rng)  # X1 -> X2, fixed +
    B[y_idx, 1] = _w(rng)  # X2 -> Y,  fixed +
    feature_to_Z = np.arange(p, dtype=int)
    return MotifSpec(
        name="D_mediator",
        description="X1 -> X2 -> Y (pure chain)",
        B=B, y_idx_in_Z=y_idx, feature_to_Z=feature_to_Z, scm_p=p,
        notes="tau[0] = w(X1->X2)*w(X2->Y) (in raw units); tau[1] = w(X2->Y); both nonzero",
    )


def motif_target_descendant(p: int = 5, rng: Optional[np.random.Generator] = None) -> MotifSpec:
    """Y -> X1. X1 is target-descendant / leakage feature."""
    B = _blank(p)
    # Y must come BEFORE X1 topologically; put Y at Z-index 0.
    y_idx = 0
    B[1, 0] = _w(rng)  # Y -> X1, fixed + (so X1 strongly predicts Y marginally)
    feature_to_Z = np.arange(1, p + 1, dtype=int)
    return MotifSpec(
        name="E_target_descendant",
        description="Y -> X1 (X1 is a descendant of Y / leakage)",
        B=B, y_idx_in_Z=y_idx, feature_to_Z=feature_to_Z, scm_p=p,
        notes="tau[0] == 0 (do(X1) doesn't move Y); but X1 strongly predicts Y",
    )


def motif_collider(p: int = 5, rng: Optional[np.random.Generator] = None) -> MotifSpec:
    """X1 indep of Y; C = X1 + Y + noise is a collider / descendant of both.
    X1 (feature 0) and C (feature 1) are observed. C predicts Y marginally
    (through the common Y path) but do(C) does not change Y.
    """
    B = _blank(p)
    y_idx = p
    B[1, 0]     = _w(rng)  # X1 -> C, fixed +
    B[1, y_idx] = _w(rng)  # Y  -> C, fixed +
    feature_to_Z = np.arange(p, dtype=int)
    return MotifSpec(
        name="F_collider_leakage",
        description="X1 indep Y; C = X1 + Y + noise (C descendant of both)",
        B=B, y_idx_in_Z=y_idx, feature_to_Z=feature_to_Z, scm_p=p,
        notes="tau[0] == 0 (X1 indep Y); tau[1] == 0 (C descendant of Y)",
    )


ALL_MOTIFS: Dict[str, callable] = {
    "A_direct_cause": motif_direct_cause,
    "B_proxy": motif_proxy,
    "C_observed_confounder": motif_observed_confounder,
    "D_mediator": motif_mediator,
    "E_target_descendant": motif_target_descendant,
    "F_collider_leakage": motif_collider,
}


def motif_scm(spec: MotifSpec, rng: np.random.Generator):
    """Build a LinearNonGaussianSCM-like object from an explicit MotifSpec."""
    # Lazy import to keep motifs.py independent for tests.
    from causal_probe.scm import LinearNonGaussianSCM

    # Construct the SCM object but override its sampled structure.
    scm = LinearNonGaussianSCM.__new__(LinearNonGaussianSCM)
    scm.p = spec.scm_p
    scm.noise = "laplace"
    scm.y_idx_in_Z = int(spec.y_idx_in_Z)
    scm.feature_to_Z = spec.feature_to_Z.copy()
    scm.B = spec.B.copy()
    N = spec.B.shape[0]  # may exceed scm_p+1 when the motif has hidden nodes (e.g. hidden confounder)
    scm.eps_var = np.ones(N, dtype=np.float64)
    I = np.eye(N, dtype=np.float64)
    scm.A = np.linalg.solve(I - scm.B, I)
    cov_Z = scm.A @ np.diag(scm.eps_var) @ scm.A.T
    var_Z = np.clip(np.diag(cov_Z), a_min=1e-12, a_max=None)
    scm.std_Z = np.sqrt(var_Z)
    std_y = float(scm.std_Z[scm.y_idx_in_Z])
    tau = np.empty(spec.scm_p, dtype=np.float64)
    for i, zi in enumerate(scm.feature_to_Z.tolist()):
        total = float(scm.A[scm.y_idx_in_Z, int(zi)])
        tau[i] = total * 2.0 * float(scm.std_Z[int(zi)]) / std_y
    scm.tau = tau
    # Fake topo order for sampling: feature_to_Z order + [y_idx] appended.
    # We need actual topological order over ALL N nodes. Do Kahn's algorithm.
    adj = [[] for _ in range(N)]
    indeg = np.zeros(N, dtype=int)
    for child in range(N):
        for parent in range(N):
            if scm.B[child, parent] != 0.0:
                adj[parent].append(child)
                indeg[child] += 1
    order = []
    queue = [i for i in range(N) if indeg[i] == 0]
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    scm._topo_order = np.asarray(order, dtype=int)
    return scm
