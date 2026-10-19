"""Binary-outcome SCM family (prior-coverage fix for binary RCT outcomes).

Wraps the linear non-Gaussian SCM but passes the outcome node Y through a
logistic link + Bernoulli draw, giving a BINARY outcome Y in {0,1}. The
per-feature causal label tau_i is the standardized contrast on P(Y=1):

    tau_i = (E[Y_bin | do(X_i=+sigma)] - E[Y_bin | do(X_i=-sigma)]) / std(Y_bin)

estimated by Monte-Carlo through the logistic link (same pattern as the
nonlinear families). Base rate is randomized (incl. rare outcomes) so the
family covers Hillstrom-like visit (~15%) and conversion (~1%) regimes that
the continuous-outcome prior misses.
"""
from __future__ import annotations

import numpy as np

from causal_probe.scm import LinearNonGaussianSCM, SCMSample


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class BinaryOutcomeSCM:
    def __init__(self, p, rng, *, noise="laplace", n_mc=4096, **scm_kwargs):
        self._base = LinearNonGaussianSCM(p=p, rng=rng, noise=noise, **scm_kwargs)
        self.p = p
        self.noise = noise
        self.n_mc = n_mc
        self.y_idx_in_Z = self._base.y_idx_in_Z
        self.feature_to_Z = self._base.feature_to_Z
        self.B = self._base.B
        self._topo_order = self._base._topo_order
        self.std_Z = self._base.std_Z
        # logistic link: P(Y=1) = sigmoid(slope * (Ystd - bias)); bias sets base rate.
        # Rebalanced: 70% moderate base-rate + STRONG effect (so binary effects
        # can be large, like visit +0.22); 30% rare/extreme (conversion-like coverage).
        if rng.random() < 0.7:
            self._slope = float(rng.uniform(2.0, 4.5))   # larger effects
            self._bias = float(rng.uniform(-1.2, 1.2))   # base rate ~0.2-0.8
        else:
            self._slope = float(rng.uniform(1.0, 3.0))
            self._bias = float(rng.uniform(-2.8, 2.8))   # rare/saturated tail
        # MC estimate of Y_bin mean/std and tau on P(Y=1).
        mc_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        Yb0 = self._draw_ybin(self._sim_std(self.n_mc, mc_rng))
        self._ybin_mean = float(Yb0.mean())
        self._ybin_std = float(Yb0.std().clip(min=1e-6))
        self.tau = self._compute_tau(np.random.default_rng(rng.integers(0, 2**32 - 1)))

    # ---- forward sim of standardized latent Z, optionally clamping a node ----
    def _sim_std(self, n, rng, clamp=None):
        N = self.B.shape[0]
        eps = self._base._draw_eps(n, rng)
        Z = np.zeros((n, N))
        for rank, node in enumerate(self._topo_order.tolist()):
            node = int(node)
            if clamp is not None and node in clamp:
                Z[:, node] = clamp[node]
                continue
            parents = [int(self._topo_order[r]) for r in range(rank)]
            Z[:, node] = (Z[:, parents] @ self.B[node, parents] + eps[:, node]) if parents else eps[:, node]
        Zs = Z / self.std_Z.reshape(1, -1)
        return Zs

    def _y_std(self, Zs):
        return Zs[:, self.y_idx_in_Z]

    def _pY(self, Zs):
        return _sigmoid(self._slope * (self._y_std(Zs) - self._bias))

    def _draw_ybin(self, Zs):
        return (np.random.default_rng(0).random(Zs.shape[0]) < self._pY(Zs)).astype(np.float64)

    def _compute_tau(self, rng):
        p = self.p
        tau = np.empty(p)
        for i, zi in enumerate(self.feature_to_Z.tolist()):
            zi = int(zi)
            # clamp X_i at +/- 1 standardized (=> raw +/- std) via std-space clamp value +/-1
            rp = np.random.default_rng(rng.integers(0, 2**31))
            rm = np.random.default_rng(rng.integers(0, 2**31))
            Zp = self._sim_std(self.n_mc, rp, clamp={zi: +1.0 * self.std_Z[zi]})
            Zm = self._sim_std(self.n_mc, rm, clamp={zi: -1.0 * self.std_Z[zi]})
            # note clamp is in raw units in _sim_std (before /std_Z); pass raw +/-std
            Pp, Pm = self._pY(Zp).mean(), self._pY(Zm).mean()
            tau[i] = (Pp - Pm) / self._ybin_std
        return tau

    def sample(self, n, rng):
        Zs = self._sim_std(n, rng)
        pY = self._pY(Zs)
        Yb = (rng.random(n) < pY).astype(np.float64)
        Yb_std = (Yb - self._ybin_mean) / self._ybin_std
        X = Zs[:, self.feature_to_Z].copy()
        return SCMSample(
            X=X.astype(np.float64), y=Yb_std.astype(np.float64), tau=self.tau.copy(),
            y_idx_in_Z=self.y_idx_in_Z, feature_to_Z=self.feature_to_Z.copy(),
            B=self.B.copy(), std_Z=self.std_Z.copy(),
        )
