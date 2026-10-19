"""Additive smooth-function SCM (RBF-basis approximation of splines).

Each child node's structural equation is a SUM of single-variable
functions of its parents:

    Z_j = sum_k f_jk(Z_pa(k)) + eps_j

with each f_jk a random RBF-basis combination

    f_jk(x) = sum_b w_jkb * exp(-(x - c_b)^2 / (2 sigma^2))

This is a smoother nonparametric structural form than the polynomial /
quadratic / MLP variants. Centring constants E[f_jk(Z_pa(k))] are
estimated from a pre-sample so the structural part has zero mean.
eps_j is unit-variance Laplace.

Tau labels via MC do() interventions, same pattern as NonlinearSCM and
PolynomialSCM.
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


# Fixed RBF basis: 5 centres spaced over [-2, 2] with bandwidth 0.7.
_RBF_CENTRES = np.linspace(-2.0, 2.0, 5)
_RBF_SIGMA = 0.7


def _rbf_basis(x: np.ndarray) -> np.ndarray:
    """x: (...,) → (..., n_basis). Gaussian RBF basis evaluated pointwise."""
    diff = x[..., None] - _RBF_CENTRES.reshape((1,) * x.ndim + (-1,))
    return np.exp(-(diff ** 2) / (2.0 * _RBF_SIGMA ** 2))


class SplineAdditiveSCM:
    """Additive RBF-basis-smoothed DAG with MC-derived tau labels."""

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.35,
        weight_scale: float = 0.6,
        n_mc: int = 4096,
        y_position: Optional[int] = None,
        noise: str = "laplace",
    ):
        self.p = p
        self.n_mc = n_mc
        self.noise = noise
        N = p + 1
        n_basis = _RBF_CENTRES.size
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

        # Per-edge weight vector W[child, parent, b]. Zero where there's no edge.
        W = np.zeros((N, N, n_basis), dtype=np.float64)
        edge_mask = np.zeros((N, N), dtype=bool)
        for rank_c, child in enumerate(self._topo_order.tolist()):
            for rank_p in range(rank_c):
                parent = int(self._topo_order[rank_p])
                if rng.random() < edge_prob:
                    edge_mask[int(child), parent] = True
                    W[int(child), parent, :] = (
                        rng.standard_normal(n_basis) * weight_scale
                    )
        self.W = W
        self.edge_mask = edge_mask

        # Centring: estimate E[f_jk(Z_parent)] per (child, parent) edge from a
        # pre-sample so the structural part has zero mean.
        self._E_phi = np.zeros((N, N, n_basis), dtype=np.float64)
        mc_rng = np.random.default_rng(rng.integers(0, 2**31))
        Z_base = self._simulate_raw(n_mc, mc_rng)
        for child in range(N):
            for parent in range(N):
                if edge_mask[child, parent]:
                    self._E_phi[child, parent, :] = _rbf_basis(Z_base[:, parent]).mean(axis=0)

        # Re-simulate with proper centring to get canonical population stats.
        Z_base = self._simulate_raw(n_mc, np.random.default_rng(rng.integers(0, 2**31)))
        self._mean_Z = Z_base.mean(axis=0)
        self._std_Z = Z_base.std(axis=0, ddof=0).clip(min=1e-9)

        self.tau = self._compute_tau(rng)

    def _f_edge(self, child: int, parent: int, x_parent: np.ndarray) -> np.ndarray:
        # Centred RBF expansion: phi(x) @ w - E[phi(Z_parent)] @ w.
        phi = _rbf_basis(x_parent)
        w = self.W[child, parent]
        return phi @ w - float(self._E_phi[child, parent] @ w)

    def _simulate_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        N = self.p + 1
        eps = _sample_noise((n, N), rng, self.noise)
        Z = np.zeros((n, N), dtype=np.float64)
        for rank, node in enumerate(self._topo_order.tolist()):
            node = int(node)
            parents = [int(self._topo_order[r]) for r in range(rank)]
            if not parents:
                Z[:, node] = eps[:, node]
                continue
            contrib = np.zeros(n, dtype=np.float64)
            for parent in parents:
                if not self.edge_mask[node, parent]:
                    continue
                contrib += self._f_edge(node, parent, Z[:, parent])
            Z[:, node] = np.clip(contrib + eps[:, node], -10.0, 10.0)
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
            parents = [int(self._topo_order[r]) for r in range(rank)]
            if not parents:
                Z[:, node] = eps[:, node]
                continue
            contrib = np.zeros(n, dtype=np.float64)
            for parent in parents:
                if not self.edge_mask[node, parent]:
                    continue
                contrib += self._f_edge(node, parent, Z[:, parent])
            Z[:, node] = np.clip(contrib + eps[:, node], -10.0, 10.0)
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
