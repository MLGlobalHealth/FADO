"""Linear non-Gaussian SCM with a random subset of features observed as binary.

The structural equations and tau labels are identical to LinearNonGaussianSCM.
The only difference is presentation: at sample() time, each feature gets an
independent Bernoulli(p_binary) draw; if selected, that column is replaced
with its sign (binarized at the median of the continuous latent).

Semantics of tau for a binarized feature: we keep the continuous-SCM
contrast tau_i = A[y, i] * 2 * std(X_cont_i) / std(Y) — the
(-1, +1) standardized contrast on the continuous latent X_cont_i.
The binarization (median split → ±1) is a sample-time presentation
step that does NOT change descendants' structural equations, since
descendants in the simulated data depend on the continuous latent.
The model thus receives a binary ±1 observation but learns to predict
the latent-scale effect, and at inference time real binary columns
are z-scored — which sends a balanced indicator to ±1 and aligns
the encoding with training. (Earlier wording of this docstring claimed
the tau corresponds to "pushing X_bin from 0 to 1"; that was wrong:
pushing the binary indicator alone does not propagate to descendants
in this data-generating process. The tau labels reflect a
do-intervention on the continuous latent, not on the binary
indicator.)

Designed to close the gap observed on Hillstrom (binary treatment/outcome
under a continuous-trained model).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from causal_probe.scm import SCMSample, LinearNonGaussianSCM


class LinearMixedSCM(LinearNonGaussianSCM):
    """Same as LinearNonGaussianSCM but a random subset of feature columns
    and optionally Y are observed as binary indicators.

    Parameters
    ----------
    p : int
    rng : np.random.Generator
    p_binary : float
        Per-feature probability that its observed column is binarized.
    binarize_y_prob : float
        Probability that the outcome column is binarized.
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
        p_binary: float = 0.3,
        binarize_y_prob: float = 0.3,
        y_position: Optional[int] = None,
    ) -> None:
        super().__init__(
            p=p, rng=rng, edge_prob=edge_prob,
            weight_lo=weight_lo, weight_hi=weight_hi,
            noise=noise, y_position=y_position,
        )
        self.p_binary = p_binary
        self.binarize_y_prob = binarize_y_prob
        self._binary_mask = rng.random(p) < p_binary
        self._y_is_binary = bool(rng.random() < binarize_y_prob)

    def sample(self, n: int, rng: np.random.Generator) -> SCMSample:
        sample = super().sample(n, rng)  # continuous X, y, tau
        X = sample.X.copy()
        y = sample.y.copy()
        # Binarize at column median (corresponds to 50/50 split).
        for j in range(self.p):
            if self._binary_mask[j]:
                thresh = float(np.median(X[:, j]))
                X[:, j] = np.where(X[:, j] > thresh, 1.0, -1.0)
        if self._y_is_binary:
            thresh = float(np.median(y))
            y = np.where(y > thresh, 1.0, -1.0)
        return SCMSample(
            X=X, y=y, tau=sample.tau,
            y_idx_in_Z=sample.y_idx_in_Z,
            feature_to_Z=sample.feature_to_Z,
            B=sample.B, std_Z=sample.std_Z,
        )
