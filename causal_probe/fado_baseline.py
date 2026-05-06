"""FADO as a CausalPFN-harness BaselineModel.

Lets us run the same evaluation loop as CausalPFN's notebooks/causal_effect.ipynb
with FADO drop-in for ``BaselineModel``: estimate_ate(X, t, y) returns one
ATE on raw outcome scale, ready for the harness's
``rel_err = |hat - true_ate| / |true_ate|`` metric.

FADO is column-symmetric and trained to produce per-feature standardized
contrasts. To answer the harness's "ATE for *this* designated treatment"
question, we:

  1. Prepend t as column 0 of a (n, p+1) matrix X' (FADO is permutation-
     equivariant so position is irrelevant — column 0 chosen for clarity).
  2. Standardize X' and y to mean 0 / std 1 column-wise.
  3. Run FADO -> τ̂[0] on the standardized scale.
  4. Unstandardize using the LaLonde convention
     ``ate_raw = τ̂_std · std_y / (2 · std_T)`` so the harness sees ATE
     in raw outcome units.

CATE is **not** natively supported — FADO returns a population-level
contrast. estimate_cate returns the ATE broadcast to every test row
(degenerate constant CATE), so PEHE reflects the population vs unit-level
gap. Don't read PEHE as a CATE skill claim for FADO; it isn't one.
"""
from __future__ import annotations

import numpy as np
import torch

from causal_probe.eval import _load_model

# We don't subclass benchmarks.baselines.base.BaselineModel because
# importing that pulls in benchmarks.baselines.__init__, which transitively
# imports flaml / econml / catenets / BART deps we don't carry. The
# harness only requires the duck-typed estimate_ate / estimate_cate
# interface.


class FADOBaseline:
    """FADO drop-in for the CausalPFN benchmark harness.

    Args:
        ckpt: path to a .ckpt file produced by causal_probe.train.
        device: torch device string.
        n_rows: optional cap on rows fed to FADO per call (the probe was
            typically trained with n_rows=512; harness datasets like
            LalondeCPS have ~16k rows, so we subsample for inference
            speed and to match the training-time row distribution).
        seed: rng seed for the row subsample (per-call deterministic).
    """

    def __init__(
        self, ckpt: str, *, device: str = "cpu", n_rows: int | None = 512,
        seed: int = 2025,
    ):
        self.ckpt = ckpt
        self.device = device
        self.n_rows = n_rows
        self.seed = seed
        self._model = None

    def _model_lazy(self):
        if self._model is None:
            self._model = _load_model(self.ckpt, device=self.device)
        return self._model

    def _ate_raw(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        """Standardize, run FADO, read off τ̂_T, unstandardize to raw scale."""
        # Prepend t as column 0.
        X_with_T = np.concatenate(
            [t.reshape(-1, 1).astype(np.float64), X.astype(np.float64)], axis=1
        )

        if self.n_rows is not None and X_with_T.shape[0] > self.n_rows:
            rng = np.random.default_rng(self.seed)
            sel = rng.choice(X_with_T.shape[0], size=self.n_rows, replace=False)
            X_sub = X_with_T[sel]
            y_sub = y[sel]
        else:
            X_sub = X_with_T
            y_sub = y

        std_y = float(np.std(y_sub).clip(min=1e-9))
        std_T = float(np.std(t).clip(min=1e-9))

        Xs = (X_sub - X_sub.mean(axis=0)) / X_sub.std(axis=0).clip(min=1e-9)
        ys = (y_sub - y_sub.mean()) / y_sub.std().clip(min=1e-9)

        X_t = torch.from_numpy(Xs.astype(np.float32)).unsqueeze(0).to(self.device)
        y_t = torch.from_numpy(ys.astype(np.float32)).unsqueeze(0).to(self.device)
        model = self._model_lazy()
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        tau_T_std = float(pred[0])

        # FADO scale to raw scale: tau_std = (ate_raw / std_y) * 2 * std_T  =>
        # ate_raw = tau_std * std_y / (2 * std_T).
        return tau_T_std * std_y / (2.0 * max(std_T, 1e-9))

    def estimate_ate(self, X, t, y) -> float:
        return float(self._ate_raw(np.asarray(X), np.asarray(t), np.asarray(y)))

    def estimate_cate(self, X_train, t_train, y_train, X_test) -> np.ndarray:
        """FADO doesn't do unit-level CATE; broadcast ATE to every test row.

        This is intentionally degenerate so PEHE reflects the population-vs-
        unit gap, not a CATE claim.
        """
        ate = self._ate_raw(np.asarray(X_train), np.asarray(t_train), np.asarray(y_train))
        return np.full(len(X_test), ate, dtype=np.float64)
