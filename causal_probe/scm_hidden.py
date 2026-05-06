"""Linear non-Gaussian SCM with latent confounders.

We extend LinearNonGaussianSCM by introducing n_hidden latent nodes that
precede the observed nodes in topological order and parent into ≥2
observed nodes each. The training sample exposes only X (p features) and
Y; latents are marginalized out. tau labels are computed from the FULL
SCM via A = (I - B)^{-1} over the joint (latent + observed) graph.

Latent confounders create systematic observational bias on certain
(feature, Y) pairs; the model must discount those while keeping
genuinely causal effects.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from causal_probe.scm import SCMSample
from causal_probe.scm_utils import (
    sample_laplace_unit as _laplace_unit,
    select_y_and_features,
)


class LinearNonGaussianSCMHidden:
    """Random linear non-Gaussian DAG with n_hidden latent confounders."""

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.35,
        weight_lo: float = 0.5,
        weight_hi: float = 2.0,
        n_hidden: int = 1,
        hidden_children: int = 3,
        noise: str = "laplace",
    ) -> None:
        self.p = p
        self.n_hidden = n_hidden
        self.noise = noise
        N_obs = p + 1              # X_1..X_p + Y
        N = N_obs + n_hidden
        self._topo_order, self.y_idx_in_Z, self.feature_to_Z, self.hidden_indices = (
            select_y_and_features(p, n_hidden=n_hidden, rng=rng)
        )

        B = np.zeros((N, N), dtype=np.float64)
        # Latent -> observed edges: each latent parents into hidden_children
        # random observed nodes (including possibly Y).
        for h in self.hidden_indices:
            picks = rng.choice(N_obs, size=min(hidden_children, N_obs), replace=False)
            for child in picks.tolist():
                sign = 1.0 if rng.random() < 0.5 else -1.0
                B[int(child), int(h)] = sign * rng.uniform(weight_lo, weight_hi)
        # Observed -> observed edges (standard DAG edges).
        for rank_c, child in enumerate(self._topo_order.tolist()):
            if int(child) in set(self.hidden_indices):
                continue
            for rank_p in range(rank_c):
                parent = int(self._topo_order[rank_p])
                if int(parent) in set(self.hidden_indices):
                    continue
                if rng.random() < edge_prob:
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    B[int(child), int(parent)] = sign * rng.uniform(weight_lo, weight_hi)
        self.B = B
        self.eps_var = np.ones(N, dtype=np.float64)
        I = np.eye(N, dtype=np.float64)
        self.A = np.linalg.solve(I - B, I)
        cov_Z = self.A @ np.diag(self.eps_var) @ self.A.T
        self.std_Z = np.sqrt(np.clip(np.diag(cov_Z), a_min=1e-12, a_max=None))

        std_y = float(self.std_Z[self.y_idx_in_Z])
        tau = np.empty(p, dtype=np.float64)
        for i, zi in enumerate(self.feature_to_Z.tolist()):
            total = float(self.A[self.y_idx_in_Z, int(zi)])
            tau[i] = total * 2.0 * float(self.std_Z[int(zi)]) / std_y
        self.tau = tau

    def sample(self, n: int, rng: np.random.Generator) -> SCMSample:
        N = self.p + 1 + self.n_hidden
        eps = _laplace_unit((n, N), rng) if self.noise == "laplace" else rng.standard_normal((n, N))
        Z = np.zeros((n, N), dtype=np.float64)
        for rank, node in enumerate(self._topo_order.tolist()):
            parents = [int(self._topo_order[r]) for r in range(rank)]
            if parents:
                Z[:, int(node)] = Z[:, parents] @ self.B[int(node), parents] + eps[:, int(node)]
            else:
                Z[:, int(node)] = eps[:, int(node)]
        # Standardize observed nodes only.
        Z_std = Z / self.std_Z.reshape(1, -1)
        y = Z_std[:, self.y_idx_in_Z].copy()
        X = Z_std[:, self.feature_to_Z].copy()
        return SCMSample(
            X=X, y=y, tau=self.tau.copy(),
            y_idx_in_Z=self.y_idx_in_Z,
            feature_to_Z=self.feature_to_Z.copy(),
            B=self.B.copy(),
            std_Z=self.std_Z.copy(),
        )
