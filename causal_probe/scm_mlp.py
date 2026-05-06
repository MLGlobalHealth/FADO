"""MLP-nonlinear SCM: each node's conditional is a small random MLP.

Generalizes the polynomial nonlinear SCM. Each node j with parents
pa(j) computes

    Z_j = alpha_j * tanh(W2_j tanh(W1_j z_{pa(j)} + b1_j) + b2_j) + eps_j

with random orthogonal W's and small biases. alpha_j rescales the
structural component to keep Var(Z_j) finite; we calibrate it from a
pre-sample so each Z_j has unit variance in expectation.

Population stats and tau_i labels are computed by MC do()-intervention
identical to scm_nonlinear.NonlinearSCM.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from causal_probe.scm import SCMSample
from causal_probe.scm_utils import (
    monte_carlo_tau,
    sample_laplace_unit as _laplace_unit,
    select_y_and_features,
    standardize_for_sample,
)


def _sample_mlp_params(n_parents: int, rng: np.random.Generator,
                       hidden: int = 8) -> dict:
    if n_parents == 0:
        return None
    scale = 1.0 / np.sqrt(max(1, n_parents))
    W1 = rng.standard_normal((hidden, n_parents)) * scale
    b1 = 0.2 * rng.standard_normal(hidden)
    W2 = rng.standard_normal((1, hidden)) * (1.0 / np.sqrt(hidden))
    b2 = 0.1 * rng.standard_normal(1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def _mlp_forward(params: dict, x: np.ndarray) -> np.ndarray:
    """x: (n, n_parents) → (n,) scalar output."""
    h = np.tanh(x @ params["W1"].T + params["b1"])
    y = np.tanh(h @ params["W2"].T + params["b2"]).squeeze(-1)
    return y


class MLPSCM:
    """Random-MLP nonlinear DAG with MC-derived tau labels."""

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.4,
        mlp_hidden: int = 8,
        n_mc: int = 8192,
        y_position: Optional[int] = None,
    ):
        self.p = p
        self.n_mc = n_mc
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
                sorted(non_y_nodes, key=lambda k: rank[int(k)]), dtype=int
            )

        self._parents = [[] for _ in range(N)]
        for rank_c, child in enumerate(self._topo_order.tolist()):
            for rank_p in range(rank_c):
                parent = int(self._topo_order[rank_p])
                if rng.random() < edge_prob:
                    self._parents[int(child)].append(parent)
        self._mlp_params = [
            _sample_mlp_params(len(self._parents[j]), np.random.default_rng(rng.integers(0, 2**31)),
                               hidden=mlp_hidden) if self._parents[j] else None
            for j in range(N)
        ]
        # alpha_j scales the MLP output; calibrate after one MC sweep so that
        # structural contribution has unit variance. First sweep with alpha=1.
        self._alpha = np.ones(N, dtype=np.float64)
        Z_cal = self._simulate_raw(n_mc, np.random.default_rng(rng.integers(0, 2**31)))
        struct_var = np.clip(np.var(Z_cal, axis=0) - 1.0, a_min=1e-6, a_max=None)
        # Target: total Var(Z_j) = 1 + 1 = 2 (structural + noise). Scale so Var(struct) = 1.
        self._alpha = 1.0 / np.sqrt(struct_var / np.clip(np.var(Z_cal, axis=0), a_min=1e-6, a_max=None) + 1e-9)
        # Clip alpha to [0.1, 5] to avoid degenerate cases.
        self._alpha = np.clip(self._alpha, 0.1, 5.0)

        # Re-sample for population stats after alpha calibration.
        Z_base = self._simulate_raw(n_mc, np.random.default_rng(rng.integers(0, 2**31)))
        self._mean_Z = Z_base.mean(axis=0)
        self._std_Z = Z_base.std(axis=0, ddof=0).clip(min=1e-9)

        self.tau = self._compute_tau(rng)

    def _structural(self, node: int, Z: np.ndarray) -> np.ndarray:
        params = self._mlp_params[node]
        parents = self._parents[node]
        if not parents:
            return np.zeros(Z.shape[0])
        x = Z[:, parents]
        return self._alpha[node] * _mlp_forward(params, x)

    def _simulate_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        N = self.p + 1
        eps = _laplace_unit((n, N), rng)
        Z = np.zeros((n, N), dtype=np.float64)
        for node in self._topo_order.tolist():
            Z[:, int(node)] = self._structural(int(node), Z) + eps[:, int(node)]
        return Z

    def _simulate_intervention(self, n: int, rng: np.random.Generator,
                               intervene_idx: int, intervene_val: float) -> np.ndarray:
        N = self.p + 1
        eps = _laplace_unit((n, N), rng)
        Z = np.zeros((n, N), dtype=np.float64)
        for node in self._topo_order.tolist():
            node = int(node)
            if node == intervene_idx:
                Z[:, node] = intervene_val
            else:
                Z[:, node] = self._structural(node, Z) + eps[:, node]
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
        return SCMSample(
            X=X, y=y, tau=self.tau.copy(),
            y_idx_in_Z=self.y_idx_in_Z,
            feature_to_Z=self.feature_to_Z.copy(),
            B=np.zeros((self.p + 1, self.p + 1)),  # not meaningful for MLP SCM
            std_Z=self._std_Z.copy(),
        )
