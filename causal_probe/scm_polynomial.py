"""Polynomial SCM: structural equations are sums of univariate powers.

Generalizes scm_nonlinear (which mixes linear + quadratic) by drawing a
per-edge degree k from {1, 2, 3}. Each child node's structural equation
is

    Z_j = sum_k a_jk * (Z_pa(k) ^ d_jk - E[Z_pa(k) ^ d_jk]) + eps_j

with d_jk in {1, 2, 3}, biased toward the lower degrees to keep
variances finite. The centring term (E[Z^d]) is estimated from a
pre-sample so the structural part has zero mean. eps is unit-variance
Laplace.

Tau labels are computed by Monte-Carlo do() intervention, same pattern
as NonlinearSCM and MLPSCM.
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


_DEGREE_CHOICES = np.array([1, 2, 3], dtype=int)
# Bias toward degree 1 to keep variances bounded; degree 3 is rare.
_DEGREE_PROBS = np.array([0.55, 0.30, 0.15], dtype=float)


class PolynomialSCM:
    """Polynomial-degree-{1,2,3} DAG with MC-derived tau labels."""

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.35,
        coef_scale: float = 0.7,
        n_mc: int = 4096,
        y_position: Optional[int] = None,
        noise: str = "laplace",
    ):
        self.p = p
        self.n_mc = n_mc
        self.noise = noise
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

        # Per-edge coefficient and degree. Degree-2 and degree-3 coefs are
        # damped to keep moments bounded.
        coef = np.zeros((N, N), dtype=np.float64)
        degree = np.ones((N, N), dtype=int)
        for rank_c, child in enumerate(self._topo_order.tolist()):
            for rank_p in range(rank_c):
                parent = int(self._topo_order[rank_p])
                if rng.random() < edge_prob:
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    d = int(rng.choice(_DEGREE_CHOICES, p=_DEGREE_PROBS))
                    # Damp higher-degree terms: d=1 unscaled, d=2 ×0.5, d=3 ×0.25
                    damp = 1.0 if d == 1 else (0.5 if d == 2 else 0.25)
                    coef[int(child), parent] = sign * rng.uniform(0.5 * coef_scale, coef_scale) * damp
                    degree[int(child), parent] = d
        self.coef = coef
        self.degree = degree

        # Pre-sample to estimate centring constants E[Z_k^d] per parent-degree pair.
        # We only need E[Z_k^d] for the degrees that actually appear as edges into
        # something; cache E[Z_k^1], E[Z_k^2], E[Z_k^3] for every k.
        self._E_pow = np.zeros((N, 4), dtype=np.float64)  # _E_pow[k, d] = E[Z_k^d]
        mc_rng = np.random.default_rng(rng.integers(0, 2**31))
        Z_base = self._simulate_raw(n_mc, mc_rng)
        for d in (1, 2, 3):
            self._E_pow[:, d] = (Z_base ** d).mean(axis=0)

        # After centring with these E[Z^d], the distribution shifts. Re-simulate
        # and use that as canonical.
        Z_base = self._simulate_raw(n_mc, np.random.default_rng(rng.integers(0, 2**31)))
        self._mean_Z = Z_base.mean(axis=0)
        self._std_Z = Z_base.std(axis=0, ddof=0).clip(min=1e-9)

        self.tau = self._compute_tau(rng)

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
                d = int(self.degree[node, parent])
                a = float(self.coef[node, parent])
                if a == 0.0:
                    continue
                centred = Z[:, parent] ** d - self._E_pow[parent, d]
                contrib += a * centred
            # Aggressive clip — degree-3 with Laplace tails can blow up.
            Z[:, node] = np.clip(contrib + eps[:, node], -12.0, 12.0)
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
                d = int(self.degree[node, parent])
                a = float(self.coef[node, parent])
                if a == 0.0:
                    continue
                centred = Z[:, parent] ** d - self._E_pow[parent, d]
                contrib += a * centred
            Z[:, node] = np.clip(contrib + eps[:, node], -12.0, 12.0)
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
            B=self.coef.copy(),
            std_Z=self._std_Z.copy(),
        )
