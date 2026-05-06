"""Baselines for the causal probe.

All baselines take (X, y) as numpy arrays with rows n, columns p (features
only) and return a length-p vector of effect estimates on the standardized
scale (the same scale as tau labels).
"""
from __future__ import annotations

import numpy as np


def baseline_zero(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Trivial: always predict 0."""
    return np.zeros(X.shape[1], dtype=np.float64)


def baseline_marginal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Marginal associational effect: tau_hat_i = 2 * Corr(X_i, y).

    For standardized inputs, the slope of y ~ X_i equals Corr(X_i, y).
    The +1/-1 contrast in standardized units gives 2 * slope.
    """
    p = X.shape[1]
    out = np.empty(p, dtype=np.float64)
    y_var = np.var(y) + 1e-12
    for i in range(p):
        x_i = X[:, i]
        cov = np.cov(x_i, y, ddof=0)[0, 1]
        slope = cov / (np.var(x_i) + 1e-12)
        # Convert to tau-scale: for standardized y, y_std=1. If context y_std differs
        # from label y_std we'd need a correction; we use population-standardized
        # inputs so both are ≈ 1.
        out[i] = 2.0 * slope
    return out


def baseline_multivariate(X: np.ndarray, y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """Multivariate ridge regression: tau_hat_i = 2 * beta_i.

    Ridge stabilizes against near-singular X^T X.
    """
    p = X.shape[1]
    XtX = X.T @ X
    reg = XtX + alpha * np.eye(p) * X.shape[0]
    beta = np.linalg.solve(reg, X.T @ y)
    return 2.0 * beta


BASELINES = {
    "zero": baseline_zero,
    "marginal": baseline_marginal,
    "multivariate": baseline_multivariate,
}
