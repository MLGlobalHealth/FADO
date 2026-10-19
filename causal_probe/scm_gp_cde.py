"""Gaussian-process conditional-density SCM with exact Kronecker-grid sampling.

Each child node's structural equation is an exact sample from a GP prior
with an RBF kernel, evaluated at the parent vector via d-linear
interpolation on a tensor-product grid:

    Z_c = f_c(z_parents_subset) - E[f_c] + eps_c

where f_c ~ GP(0, k_rbf), with k_rbf separable across parent dimensions.
On the grid the kernel matrix factorises as K = K_1 ⊗ K_2 ⊗ ... ⊗ K_d
with K_j the per-axis Gram matrices, and Cholesky of K equals
L_1 ⊗ L_2 ⊗ ... ⊗ L_d. Exact sampling at the grid is therefore
O(sum n_j^3) per node instead of O(prod n_j^3). For typical d ≤ 3 and
n_j = 6 this is well under 1ms; the dominant cost is the MC tau pass.

Each child caps at ``max_parents_per_node`` parents (default 3) — beyond
that the d-linear interpolation cell count 2^d blows up. If a child's
candidate-parent count exceeds the cap, we sample ``max_parents_per_node``
of them uniformly without replacement; the remaining edges are treated
as absent.

tau labels via MC do(), same pattern as scm_polynomial / scm_spline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from causal_probe.scm import SCMSample
from causal_probe.scm_utils import (
    monte_carlo_tau,
    sample_noise as _sample_noise,
    select_y_and_features,
    standardize_for_sample,
)


# Grid spans [-_GRID_LO, _GRID_HI] per axis, with _N_GRID points.
# 6 points × 6 × 6 = 216 grid cells per d=3 node — Cholesky is 6×6 = trivial.
_GRID_LO = -3.0
_GRID_HI = 3.0
_N_GRID = 6
_KERNEL_LENGTH_SCALE = 1.4


def _rbf_kernel_1d(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    """RBF kernel between two 1D arrays. Returns (len(x1), len(x2))."""
    d = x1.reshape(-1, 1) - x2.reshape(1, -1)
    return np.exp(-0.5 * (d / length_scale) ** 2)


def _kronecker_matvec(Ls: list[np.ndarray], x_flat: np.ndarray) -> np.ndarray:
    """Compute (L_1 ⊗ L_2 ⊗ ... ⊗ L_d) @ x_flat.

    All L_j are square; result is the same length as x_flat. We apply each
    L_j along the corresponding mode of the reshaped tensor — equivalent
    to the Kronecker product matvec but O(sum n_j^2 * prod n_j) instead of
    O((prod n_j)^2).
    """
    shape = [L.shape[0] for L in Ls]
    out = x_flat.reshape(shape)
    for d, L in enumerate(Ls):
        out = np.tensordot(L, out, axes=([1], [d]))  # (n_d, ...other dims)
        out = np.moveaxis(out, 0, d)
    return out.ravel()


class GPGridSCM:
    """GP-prior SCM with exact Kronecker-grid sampling + d-linear interp."""

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.4,
        max_parents_per_node: int = 3,
        amp_scale: float = 0.55,
        n_mc: int = 4096,
        y_position: Optional[int] = None,
        noise: str = "laplace",
    ):
        self.p = p
        self.n_mc = n_mc
        self.noise = noise
        self.max_parents_per_node = int(max_parents_per_node)
        self.amp_scale = float(amp_scale)
        N = p + 1
        if y_position is None:
            self._topo_order, self.y_idx_in_Z, self.feature_to_Z, _ = (
                select_y_and_features(p, n_hidden=0, rng=rng)
            )
        else:
            self._topo_order = rng.permutation(N)
            y_position = int(max(0, min(y_position, N - 1)))
            self.y_idx_in_Z = int(self._topo_order[y_position])
            non_y_nodes = [int(k) for k in self._topo_order if int(k) != self.y_idx_in_Z]
            rank = {int(node): i for i, node in enumerate(self._topo_order.tolist())}
            self.feature_to_Z = np.asarray(
                sorted(non_y_nodes, key=lambda k: rank[int(k)]), dtype=int,
            )

        # Pre-build the shared grid axis and Cholesky factor — same for all
        # parent dimensions (axis-aligned isotropic kernel).
        grid_axis = np.linspace(_GRID_LO, _GRID_HI, _N_GRID)
        K_axis = _rbf_kernel_1d(grid_axis, grid_axis, _KERNEL_LENGTH_SCALE)
        K_axis = K_axis + 1e-6 * np.eye(_N_GRID)
        L_axis = np.linalg.cholesky(K_axis)
        self._grid_axis = grid_axis
        self._L_axis = L_axis
        self._grid_dx = float(grid_axis[1] - grid_axis[0])

        # Pick parents per node, sample grid GP per node.
        # _parents[c] = list of parent Z-indices (len = d, 0 <= d <= max_parents)
        # _grid[c]    = ndarray of shape (n_grid,)*d, sampled GP values
        self._parents: list[list[int]] = [[] for _ in range(N)]
        self._grid: list[np.ndarray] = [np.zeros(()) for _ in range(N)]
        for rank_c, child in enumerate(self._topo_order.tolist()):
            child = int(child)
            parent_candidates = [int(self._topo_order[r]) for r in range(rank_c)]
            kept = [pp for pp in parent_candidates if rng.random() < edge_prob]
            if len(kept) > self.max_parents_per_node:
                # Sample a subset deterministically from rng.
                kept = sorted(
                    rng.choice(kept, size=self.max_parents_per_node, replace=False).tolist()
                )
            if not kept:
                continue
            self._parents[child] = kept
            d = len(kept)
            n_total = _N_GRID ** d
            eps = rng.standard_normal(n_total)
            # (L_axis ⊗ ... ⊗ L_axis) @ eps gives a GP-prior sample on the grid.
            Ls = [L_axis] * d
            f_grid_flat = _kronecker_matvec(Ls, eps) * self.amp_scale
            self._grid[child] = f_grid_flat.reshape((_N_GRID,) * d)

        # Centring: E[f_c(parents)] from a pre-sample.
        self._E_f = np.zeros(N, dtype=np.float64)
        mc_rng = np.random.default_rng(rng.integers(0, 2**31))
        Z_base = self._simulate_raw(n_mc, mc_rng)
        for node in range(N):
            if not self._parents[node]:
                continue
            self._E_f[node] = self._eval_grid(
                node, Z_base[:, self._parents[node]]
            ).mean()

        # Re-simulate with centring.
        Z_base = self._simulate_raw(n_mc, np.random.default_rng(rng.integers(0, 2**31)))
        self._mean_Z = Z_base.mean(axis=0)
        self._std_Z = Z_base.std(axis=0, ddof=0).clip(min=1e-9)

        self.tau = self._compute_tau(rng)

    def _eval_grid(self, child: int, parents_mat: np.ndarray) -> np.ndarray:
        """d-linear interpolation of f_grid at parent values.

        parents_mat: (n, d). Returns (n,) values.
        """
        grid = self._grid[child]
        d = parents_mat.shape[1]
        # Map each parent column to grid index space (continuous).
        idx_cont = (parents_mat - _GRID_LO) / self._grid_dx
        # Clip into [0, n_grid-1] then split into lower-cell and frac.
        idx_cont = np.clip(idx_cont, 0.0, _N_GRID - 1.0)
        lo = np.floor(idx_cont).astype(np.int64)
        lo = np.clip(lo, 0, _N_GRID - 2)
        frac = idx_cont - lo  # (n, d)
        # d-linear interp: weighted sum over 2^d corner values.
        out = np.zeros(parents_mat.shape[0], dtype=np.float64)
        for corner in range(1 << d):
            offsets = np.array([(corner >> j) & 1 for j in range(d)], dtype=np.int64)
            # corner indices (n, d): lo + offsets
            corner_idx = lo + offsets.reshape(1, -1)
            # weight (n,): prod_j (frac if offset else 1-frac)
            w = np.where(offsets.reshape(1, -1) == 1, frac, 1.0 - frac)
            w = np.prod(w, axis=1)
            # Lookup grid values at corner_idx.
            # Build index tuple for fancy indexing.
            grid_vals = grid[tuple(corner_idx[:, j] for j in range(d))]
            out += w * grid_vals
        return out

    def _simulate_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        N = self.p + 1
        eps = _sample_noise((n, N), rng, self.noise)
        Z = np.zeros((n, N), dtype=np.float64)
        for rank, node in enumerate(self._topo_order.tolist()):
            node = int(node)
            parents = self._parents[node]
            if not parents:
                Z[:, node] = eps[:, node]
                continue
            f_vals = self._eval_grid(node, Z[:, parents]) - self._E_f[node]
            Z[:, node] = np.clip(f_vals + eps[:, node], -10.0, 10.0)
        return Z

    def _simulate_intervention(
        self, n: int, rng: np.random.Generator, intervene_idx: int, intervene_val: float
    ) -> np.ndarray:
        N = self.p + 1
        eps = _sample_noise((n, N), rng, self.noise)
        Z = np.zeros((n, N), dtype=np.float64)
        for rank, node in enumerate(self._topo_order.tolist()):
            node = int(node)
            if node == intervene_idx:
                Z[:, node] = intervene_val
                continue
            parents = self._parents[node]
            if not parents:
                Z[:, node] = eps[:, node]
                continue
            f_vals = self._eval_grid(node, Z[:, parents]) - self._E_f[node]
            Z[:, node] = np.clip(f_vals + eps[:, node], -10.0, 10.0)
        return Z

    def simulate_do(self, n: int, rng: np.random.Generator, clamp: dict) -> np.ndarray:
        """do() on a SET of nodes; mirrors _simulate_intervention for coalitions."""
        N = self.p + 1
        eps = _sample_noise((n, N), rng, self.noise)
        Z = np.zeros((n, N), dtype=np.float64)
        for rank, node in enumerate(self._topo_order.tolist()):
            node = int(node)
            if node in clamp:
                Z[:, node] = clamp[node]
                continue
            parents = self._parents[node]
            if not parents:
                Z[:, node] = eps[:, node]
                continue
            f_vals = self._eval_grid(node, Z[:, parents]) - self._E_f[node]
            Z[:, node] = np.clip(f_vals + eps[:, node], -10.0, 10.0)
        return Z

    def _compute_tau(self, rng: np.random.Generator) -> np.ndarray:
        return monte_carlo_tau(
            simulate_intervention=self._simulate_intervention,
            n_mc=self.n_mc, rng=rng,
            mean_Z=self._mean_Z, std_Z=self._std_Z,
            y_idx_in_Z=self.y_idx_in_Z,
            feature_to_Z=self.feature_to_Z,
        )

    def sample(self, n: int, rng: np.random.Generator) -> SCMSample:
        Z_raw = self._simulate_raw(n, rng)
        X, y = standardize_for_sample(
            Z_raw, self._mean_Z, self._std_Z,
            self.y_idx_in_Z, self.feature_to_Z,
        )
        N = self.p + 1
        return SCMSample(
            X=X, y=y, tau=self.tau.copy(),
            y_idx_in_Z=self.y_idx_in_Z,
            feature_to_Z=self.feature_to_Z.copy(),
            B=np.zeros((N, N), dtype=np.float64),
            std_Z=self._std_Z.copy(),
        )
