"""Designated-treatment SCM with imbalanced binary T and ignorability.

Designed to expose FADO at training time to the structural pattern that
shows up in IHDP / RealCauseLalondeCPS / RealCauseLalondePSID: a binary
treatment node with confounder structure on the X's, a propensity that
can produce imbalanced marginal P(T=1), and a linear-additive outcome
model in T and the X's.

Structural equations (sampled per SCM):
  X_1, ..., X_p  : linear non-Gaussian DAG among themselves (no Y, no T edges)
  T = 1{γᵀ X_pre_T + ε_T + b > 0}      ε_T ∼ standard normal
  Y = β_T · T + βᵀ X + ε_Y              ε_Y ∼ Laplace, unit variance

Imbalance comes from b ∼ Uniform(-2, 2). γ is supported only on a
randomly-chosen subset of X's (the confounders). β has full support
plus a treatment coefficient β_T.

Output (at sample() time):
  X[:, t_idx] = T (binary indicator, but observed in standardized space
                   like the rest of the columns — i.e., shifted so column
                   has mean 0 / std 1).
  X[:, j!=t_idx] = standardized continuous covariates from the linear DAG.
  y               = standardized Y.
  tau             = per-feature standardized contrast labels.

Tau via MC, on the standardized scale used by the rest of the SCM family:
  - For i != t_idx: τ_i = (E[Y | do(X_i_std = +1)] - E[Y | do(X_i_std = -1)]) / std(Y)
    propagating through the linear DAG, the binary T, and Y.
  - For t_idx:      τ_T = (E[Y | do(T = 1)] - E[Y | do(T = 0)]) / std(Y)
                          · 2 · std(T_obs)      to match FADO's reporting
                          scale (the 2·std factor is the linear-SCM
                          convention; for a binary intervention it adapts
                          to the observed marginal frequency).

That last factor matters. fado_baseline.py inverts the same formula at
test time:  ate_raw = τ_std · std(Y) / (2 · std(T_obs))   so
training-time τ_std must equal the binary ATE scaled by 2·std_T/std_Y.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from causal_probe.scm import SCMSample
from causal_probe.scm_utils import sample_laplace_unit


class DesignatedTreatmentSCM:
    """Linear-additive SCM with one binary designated-treatment node."""

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.35,
        weight_lo: float = 0.5,
        weight_hi: float = 1.5,
        propensity_density: float = 0.5,   # fraction of X's that are confounders
        outcome_density: float = 0.7,      # fraction of X's that affect Y
        n_mc: int = 4096,
        beta_t_scale: float = 1.5,         # treatment effect magnitude
        b_range: float = 3.0,              # threshold offset range, controls imbalance.
                                           # Originally 5.0 (P(T=1) → 0.005..0.995) but
                                           # extreme imbalance crashed training (NaN
                                           # gradients). 3.0 keeps P(T=1) in roughly
                                           # [0.04, 0.96] — still covers IHDP-like 0.19
                                           # without the LalondeCPS-like 0.01 tail.
        noise: str = "laplace",
    ):
        if p < 2:
            raise ValueError("need p >= 2 (one column for T, plus X's)")
        self.p = p
        self.n_mc = n_mc
        self.noise = noise

        # 1. Pick which feature index is T.
        self.t_idx = int(rng.integers(0, p))
        # X_idxs are all features except t_idx (positions in the output X matrix).
        x_idxs = [j for j in range(p) if j != self.t_idx]

        # 2. Linear DAG among the (p-1) covariates: B_X (lower triangular under
        #    a random topo order). Used only at forward-simulation time.
        n_cov = p - 1
        self._cov_topo_order = rng.permutation(n_cov)
        B_X = np.zeros((n_cov, n_cov), dtype=np.float64)
        for rc, child in enumerate(self._cov_topo_order.tolist()):
            for rp in range(rc):
                parent = int(self._cov_topo_order[rp])
                if rng.random() < edge_prob:
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    B_X[child, parent] = sign * rng.uniform(weight_lo, weight_hi)
        self.B_X = B_X
        # Population stats of X: X = (I - B_X)^{-1} eps, eps unit-variance.
        I_x = np.eye(n_cov)
        self.A_X = np.linalg.solve(I_x - B_X, I_x)
        cov_X = self.A_X @ self.A_X.T  # eps_var = 1
        var_X = np.clip(np.diag(cov_X), a_min=1e-12, a_max=None)
        self.std_X = np.sqrt(var_X)
        self.mean_X = np.zeros(n_cov)  # zero-mean noise & no intercepts

        # 3. Propensity coefficients γ on a random subset of X's.
        gamma = np.zeros(n_cov)
        n_conf = max(1, int(round(propensity_density * n_cov)))
        conf_pos = rng.choice(n_cov, size=n_conf, replace=False)
        gamma[conf_pos] = rng.uniform(0.5, 1.5, size=n_conf) * np.where(
            rng.random(n_conf) < 0.5, -1.0, 1.0
        )
        self.gamma = gamma
        self.b = float(rng.uniform(-b_range, b_range))  # threshold offset → imbalance

        # 4. Outcome coefficients β on a (different) subset of X's, plus β_T.
        beta = np.zeros(n_cov)
        n_out = max(1, int(round(outcome_density * n_cov)))
        out_pos = rng.choice(n_cov, size=n_out, replace=False)
        beta[out_pos] = rng.uniform(0.5, 1.5, size=n_out) * np.where(
            rng.random(n_out) < 0.5, -1.0, 1.0
        )
        self.beta = beta
        self.beta_T = float(rng.uniform(-1.0, 1.0)) * beta_t_scale

        # 5. MC pre-sample to estimate marginal P(T=1), std(Y), std(T_obs).
        Z, T_obs, Y = self._forward_simulate(n_mc, np.random.default_rng(rng.integers(0, 2**31)))
        self.p_t = float(T_obs.mean())
        self.std_T_obs = float(np.sqrt(self.p_t * (1.0 - self.p_t)).clip(min=1e-9))
        self.mean_Y = float(Y.mean())
        self.std_Y = float(Y.std().clip(min=1e-9))

        # 6. Tau labels via MC. tau_T uses the binary-intervention convention
        #    so it matches fado_baseline.py's ate_raw = tau_std · std_Y / (2 · std_T).
        #    For X_i's, use the standardized linear-SCM convention.
        self.tau = self._compute_tau_mc(np.random.default_rng(rng.integers(0, 2**31)))

    def _forward_simulate(
        self, n: int, rng: np.random.Generator,
        force_X: Optional[tuple[int, float]] = None,  # (idx_in_X, value_raw)
        force_T: Optional[float] = None,              # 0.0 or 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward-simulate the SCM. Returns (X_raw, T_obs, Y_raw)."""
        n_cov = self.p - 1

        # Sample X's via topological order; eps is laplace or normal.
        if self.noise == "laplace":
            eps_x = sample_laplace_unit((n, n_cov), rng)
        else:
            eps_x = rng.standard_normal((n, n_cov))
        X = np.zeros((n, n_cov), dtype=np.float64)
        for rank, node in enumerate(self._cov_topo_order.tolist()):
            parents = [int(self._cov_topo_order[r]) for r in range(rank)]
            if parents:
                X[:, node] = X[:, parents] @ self.B_X[node, parents] + eps_x[:, node]
            else:
                X[:, node] = eps_x[:, node]
        if force_X is not None:
            idx, val = force_X
            X[:, idx] = val  # do() on a covariate: clamp and let downstream recompute
            # Re-propagate descendants of `idx` under the topo order.
            rank_idx = int(np.where(self._cov_topo_order == idx)[0][0])
            for rank in range(rank_idx + 1, n_cov):
                node = int(self._cov_topo_order[rank])
                parents = [int(self._cov_topo_order[r]) for r in range(rank)]
                if parents:
                    X[:, node] = X[:, parents] @ self.B_X[node, parents] + eps_x[:, node]

        # Sample T given X (or force).
        eps_t = rng.standard_normal(n)
        if force_T is not None:
            T_obs = np.full(n, float(force_T))
        else:
            logit = X @ self.gamma + eps_t + self.b
            T_obs = (logit > 0).astype(np.float64)

        # Sample Y given T_obs and X.
        if self.noise == "laplace":
            eps_y = sample_laplace_unit((n,), rng)
        else:
            eps_y = rng.standard_normal(n)
        Y = self.beta_T * T_obs + X @ self.beta + eps_y
        return X, T_obs, Y

    def _compute_tau_mc(self, rng: np.random.Generator) -> np.ndarray:
        """Per-feature standardized contrast labels."""
        n_cov = self.p - 1
        tau = np.zeros(self.p, dtype=np.float64)

        # Treatment: do(T=1) vs do(T=0), raw effect / std_Y · 2 · std_T_obs.
        _, _, Y_t1 = self._forward_simulate(
            self.n_mc, np.random.default_rng(rng.integers(0, 2**31)), force_T=1.0
        )
        _, _, Y_t0 = self._forward_simulate(
            self.n_mc, np.random.default_rng(rng.integers(0, 2**31)), force_T=0.0
        )
        ate_raw = float(Y_t1.mean() - Y_t0.mean())
        tau[self.t_idx] = (ate_raw / self.std_Y) * 2.0 * self.std_T_obs

        # Covariates: do(X_i = mean+std) vs do(X_i = mean-std).
        for k, j_out in enumerate([j for j in range(self.p) if j != self.t_idx]):
            xi_mean = self.mean_X[k]
            xi_std = self.std_X[k]
            _, _, Y_p = self._forward_simulate(
                self.n_mc, np.random.default_rng(rng.integers(0, 2**31)),
                force_X=(k, xi_mean + xi_std),
            )
            _, _, Y_m = self._forward_simulate(
                self.n_mc, np.random.default_rng(rng.integers(0, 2**31)),
                force_X=(k, xi_mean - xi_std),
            )
            tau[j_out] = (float(Y_p.mean()) - float(Y_m.mean())) / self.std_Y
        return tau

    def sample(self, n: int, rng: np.random.Generator) -> SCMSample:
        """Draw n observational rows. Output X / y are standardized."""
        X_raw, T_obs, Y_raw = self._forward_simulate(n, rng)

        # Standardize covariate columns.
        n_cov = self.p - 1
        X_std = (X_raw - self.mean_X.reshape(1, -1)) / self.std_X.reshape(1, -1).clip(min=1e-9)
        # Clip extreme z-scores. Cascading linear DAGs with Laplace noise can
        # produce occasional rows tens of stds out; FADO's layer norms break
        # on those. ±8 σ keeps the heavy tail intact while bounding inputs.
        X_std = np.clip(X_std, -8.0, 8.0)
        # Standardize T column too (mean 0, std 1) so FADO's input has uniform
        # scale across columns.
        T_std = (T_obs - self.p_t) / self.std_T_obs
        # Standardize Y.
        y_std = (Y_raw - self.mean_Y) / self.std_Y
        y_std = np.clip(y_std, -8.0, 8.0)

        # Assemble output X with the t_idx column placed in position t_idx.
        X = np.zeros((n, self.p), dtype=np.float64)
        cov_pos = [j for j in range(self.p) if j != self.t_idx]
        for k, j_out in enumerate(cov_pos):
            X[:, j_out] = X_std[:, k]
        X[:, self.t_idx] = T_std

        # Build the SCMSample fields the rest of the pipeline expects.
        # We don't have a clean B / topo / std_Z to expose (the SCM is not a
        # pure linear DAG), so synthesize sentinel values for those fields.
        # Nothing downstream of train.py reads B / std_Z / feature_to_Z in
        # the training loop, only X / y / tau.
        N = self.p + 1
        return SCMSample(
            X=X.astype(np.float64),
            y=y_std.astype(np.float64),
            tau=self.tau.copy(),
            y_idx_in_Z=N - 1,
            feature_to_Z=np.arange(self.p, dtype=int),
            B=np.zeros((N, N), dtype=np.float64),
            std_Z=np.ones(N, dtype=np.float64),
        )
