"""Nonlinear SCM generator with exact Monte-Carlo total-effect labels.

Structural equations mix linear and quadratic terms:

    Z_j = sum_k B_lin[j, k] Z_k + sum_k B_quad[j, k] (Z_k^2 - E[Z_k^2]) + eps_j

The quadratic term is centred so that B_quad does not introduce a
structural mean shift. eps_j is Laplace (non-Gaussian) by default.

Labels tau_i are computed by Monte-Carlo do()-intervention: fix
standardized X_i to +1 (or -1), propagate through descendants under the
SCM, and take the expected Y difference. Standardization uses the
SCM's own population means/stds estimated via a large pre-sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from causal_probe.scm import SCMSample
from causal_probe.scm_utils import (
    monte_carlo_tau,
    sample_laplace_unit as _laplace_unit,
    select_y_and_features,
    standardize_for_sample,
)


class NonlinearSCM:
    """Polynomial-nonlinear DAG with MC-derived population stats + tau.

    Parameters
    ----------
    p : int
        Feature count; outcome Y is an additional node (N = p + 1).
    rng : np.random.Generator
    edge_prob : float
        Probability of a nonzero edge between any topologically-earlier
        parent and later child.
    lin_scale, quad_scale : float
        Magnitudes of linear / quadratic coefficients. Quadratic is set
        smaller than linear by default to keep variances finite.
    n_mc : int
        Sample count used for the population-stat pre-sample and each
        tau_i Monte-Carlo interventional average.
    """

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.35,
        lin_scale: float = 1.2,
        quad_scale: float = 0.6,
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

        B_lin = np.zeros((N, N), dtype=np.float64)
        B_quad = np.zeros((N, N), dtype=np.float64)
        for rank_c, child in enumerate(self._topo_order.tolist()):
            for rank_p in range(rank_c):
                parent = int(self._topo_order[rank_p])
                if rng.random() < edge_prob:
                    sign_l = 1.0 if rng.random() < 0.5 else -1.0
                    B_lin[int(child), parent] = sign_l * rng.uniform(
                        0.5 * lin_scale, lin_scale
                    )
                if rng.random() < edge_prob * 0.7:
                    sign_q = 1.0 if rng.random() < 0.5 else -1.0
                    B_quad[int(child), parent] = sign_q * rng.uniform(
                        0.3 * quad_scale, quad_scale
                    )
        self.B_lin = B_lin
        self.B_quad = B_quad
        self._E_sq = np.zeros(N, dtype=np.float64)  # filled below from MC

        # MC pre-sample to get population means + E[Z_k^2] for centring.
        mc_rng = np.random.default_rng(rng.integers(0, 2**31))
        Z_base = self._simulate_raw(n_mc, mc_rng)
        self._mean_Z = Z_base.mean(axis=0)
        self._E_sq = (Z_base ** 2).mean(axis=0)
        # After the centring of the quadratic term using _E_sq computed here,
        # the distribution shifts. Re-simulate and use THAT as the canonical
        # distribution.
        Z_base = self._simulate_raw(n_mc, np.random.default_rng(rng.integers(0, 2**31)))
        self._mean_Z = Z_base.mean(axis=0)
        self._std_Z = Z_base.std(axis=0, ddof=0).clip(min=1e-9)

        # Compute tau_i: do(X_i = +sigma_i) vs do(X_i = -sigma_i) in raw
        # (un-standardized) space, then convert to standardized contrast.
        self.tau = self._compute_tau(rng)

    def _simulate_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Forward-sim n rows of raw Z without any intervention."""
        N = self.p + 1
        eps = _laplace_unit((n, N), rng)
        Z = np.zeros((n, N), dtype=np.float64)
        for rank, node in enumerate(self._topo_order.tolist()):
            parents = [int(self._topo_order[r]) for r in range(rank)]
            if parents:
                lin = Z[:, parents] @ self.B_lin[int(node), parents]
                quad = (Z[:, parents] ** 2 - self._E_sq[parents].reshape(1, -1)) @ self.B_quad[int(node), parents]
                # Clip each node's contribution to stay bounded under long chains
                # of quadratic amplification (occasional DAGs had Z values
                # blowing up into 1e6+ which NaN'd training).
                Z[:, int(node)] = np.clip(lin + quad + eps[:, int(node)], -15.0, 15.0)
            else:
                Z[:, int(node)] = eps[:, int(node)]
        return Z

    def _simulate_intervention(
        self, n: int, rng: np.random.Generator, intervene_idx: int, intervene_val: float
    ) -> np.ndarray:
        N = self.p + 1
        eps = _laplace_unit((n, N), rng)
        Z = np.zeros((n, N), dtype=np.float64)
        for rank, node in enumerate(self._topo_order.tolist()):
            node = int(node)
            if node == intervene_idx:
                Z[:, node] = intervene_val
                continue
            parents = [int(self._topo_order[r]) for r in range(rank)]
            if parents:
                lin = Z[:, parents] @ self.B_lin[node, parents]
                quad = (Z[:, parents] ** 2 - self._E_sq[parents].reshape(1, -1)) @ self.B_quad[node, parents]
                Z[:, node] = np.clip(lin + quad + eps[:, node], -15.0, 15.0)
            else:
                Z[:, node] = eps[:, node]
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
            B=self.B_lin.copy(),
            std_Z=self._std_Z.copy(),
        )
