"""MLP-nonlinear SCM with latent confounders (truly-realistic setting).

Combines the MLP structural equations of scm_mlp.MLPSCM with the
latent-confounder plumbing of scm_hidden.LinearNonGaussianSCMHidden:
latent nodes precede observed ones in the topo order and parent into
≥ hidden_children observed nodes; the training sample exposes only X
and Y; tau labels come from MC do() over the full (latent+observed)
SCM. The hardest of the sprint's stress tests.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from causal_probe.scm import SCMSample
from causal_probe.scm_mlp import _sample_mlp_params, _mlp_forward
from causal_probe.scm_utils import (
    monte_carlo_tau,
    sample_laplace_unit as _laplace_unit,
    select_y_and_features,
    standardize_for_sample,
)


class MLPSCMHidden:
    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.4,
        mlp_hidden: int = 8,
        n_mc: int = 8192,
        n_hidden: int = 1,
        hidden_children: int = 3,
    ):
        self.p = p
        self.n_mc = n_mc
        self.n_hidden = n_hidden
        N_obs = p + 1
        N = N_obs + n_hidden
        self._topo_order, self.y_idx_in_Z, self.feature_to_Z, self.hidden_indices = (
            select_y_and_features(p, n_hidden=n_hidden, rng=rng)
        )

        # Build parents. Latents parent into hidden_children observed nodes.
        self._parents = [[] for _ in range(N)]
        for h in self.hidden_indices:
            picks = rng.choice(N_obs, size=min(hidden_children, N_obs), replace=False)
            for child in picks.tolist():
                self._parents[int(child)].append(int(h))
        # Observed -> observed DAG edges
        for rank_c, child in enumerate(self._topo_order.tolist()):
            if int(child) in set(self.hidden_indices):
                continue
            for rank_p in range(rank_c):
                parent = int(self._topo_order[rank_p])
                if int(parent) in set(self.hidden_indices):
                    continue
                if rng.random() < edge_prob:
                    self._parents[int(child)].append(int(parent))

        self._mlp_params = [
            _sample_mlp_params(len(self._parents[j]),
                               np.random.default_rng(rng.integers(0, 2**31)),
                               hidden=mlp_hidden) if self._parents[j] else None
            for j in range(N)
        ]
        self._alpha = np.ones(N, dtype=np.float64)
        Z_cal = self._simulate_raw(n_mc, np.random.default_rng(rng.integers(0, 2**31)))
        struct_var = np.clip(np.var(Z_cal, axis=0) - 1.0, a_min=1e-6, a_max=None)
        self._alpha = np.clip(
            1.0 / np.sqrt(struct_var / np.clip(np.var(Z_cal, axis=0), a_min=1e-6, a_max=None) + 1e-9),
            0.1, 5.0,
        )
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
        N = self.p + 1 + self.n_hidden
        eps = _laplace_unit((n, N), rng)
        Z = np.zeros((n, N), dtype=np.float64)
        for node in self._topo_order.tolist():
            Z[:, int(node)] = np.clip(self._structural(int(node), Z) + eps[:, int(node)], -15.0, 15.0)
        return Z

    def _simulate_intervention(self, n: int, rng: np.random.Generator,
                               intervene_idx: int, intervene_val: float) -> np.ndarray:
        N = self.p + 1 + self.n_hidden
        eps = _laplace_unit((n, N), rng)
        Z = np.zeros((n, N), dtype=np.float64)
        for node in self._topo_order.tolist():
            node = int(node)
            if node == intervene_idx:
                Z[:, node] = intervene_val
            else:
                Z[:, node] = np.clip(self._structural(node, Z) + eps[:, node], -15.0, 15.0)
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
            B=np.zeros((self.p + 1 + self.n_hidden,) * 2),
            std_Z=self._std_Z.copy(),
        )
