"""Discrete-mixture SCM: multi-level quantile-binned presentation of features.

Generalises scm_mixed.LinearMixedSCM beyond binary (k=2) to k in {3, 4, 5}.
Each feature column is, with probability p_discrete, replaced at sample()
time with its quantile-binned categorical version: K equally-frequent
bins encoded as {0, 1, ..., K-1} then z-scored to mean 0 std 1 (consistent
with the rest of the X presentation). K is drawn per-feature per-SCM.

Structural equations and tau labels stay continuous (inherited from
LinearNonGaussianSCM) — exactly like scm_mixed. The binarisation /
quantisation is presentation-only; tau corresponds to a do() on the
continuous latent and propagates through the continuous DAG.

Different from scm_mixed: that file only does k=2 (median split). Real
tabular data often has 3- or 4-level categoricals (Likert scales,
discretised continuous covariates, etc.) — and the model should see
those in training so it doesn't catastrophe at inference time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from causal_probe.scm import SCMSample, LinearNonGaussianSCM


class DiscreteMixedSCM(LinearNonGaussianSCM):
    """LinearNonGaussianSCM with multi-level quantile-binned columns.

    Parameters
    ----------
    p_discrete : float
        Per-feature probability of being discretised.
    k_choices : tuple[int, ...]
        Allowed level counts. Default (2, 3, 4, 5) — gives a mix of
        binary, ternary, 4-level, 5-level features.
    k_weights : optional weights matching k_choices. Default uniform.
    """

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.35,
        weight_lo: float = 0.5,
        weight_hi: float = 2.0,
        noise: str = "laplace",
        p_discrete: float = 0.4,
        k_choices: tuple[int, ...] = (2, 3, 4, 5),
        y_position: Optional[int] = None,
    ) -> None:
        super().__init__(
            p=p, rng=rng, edge_prob=edge_prob,
            weight_lo=weight_lo, weight_hi=weight_hi,
            noise=noise, y_position=y_position,
        )
        self.p_discrete = float(p_discrete)
        self.k_choices = tuple(int(k) for k in k_choices)
        # Per-feature: -1 if continuous, else k>=2 for discretised.
        self._k_per_feat = np.full(p, -1, dtype=np.int64)
        for j in range(p):
            if rng.random() < self.p_discrete:
                self._k_per_feat[j] = int(rng.choice(self.k_choices))

    def sample(self, n: int, rng: np.random.Generator) -> SCMSample:
        sample = super().sample(n, rng)
        X = sample.X.copy()
        for j in range(self.p):
            k = int(self._k_per_feat[j])
            if k < 2:
                continue
            col = X[:, j]
            # K equally-frequent bins via quantile thresholds.
            qs = np.quantile(col, np.linspace(0.0, 1.0, k + 1)[1:-1])
            bins = np.zeros(n, dtype=np.int64)
            for q in qs:
                bins += (col > q).astype(np.int64)
            # Re-standardize the integer-coded column to mean 0 std 1
            # so it sits on the same scale as the continuous columns
            # (which the model expects post-z-score).
            bins_f = bins.astype(np.float64)
            mu = bins_f.mean()
            sd = bins_f.std()
            if sd < 1e-9:
                X[:, j] = 0.0
            else:
                X[:, j] = (bins_f - mu) / sd
        return SCMSample(
            X=X, y=sample.y, tau=sample.tau,
            y_idx_in_Z=sample.y_idx_in_Z,
            feature_to_Z=sample.feature_to_Z,
            B=sample.B, std_Z=sample.std_Z,
        )
