"""Oracle-graph causal-Shapley baselines for the causal probe.

Two graph-aware attribution methods from the causal-explanation literature,
implemented as baselines that consume the *true* SCM (structural graph, node
stats, feature/target indices) rather than only ``(X, y)``:

  * ``do_shapley``   — do-Shapley values (Jung, Kasiviswanathan, Tian, Janzing,
    Bloebaum, Bareinboim, ICML 2022). Shapley credit assignment whose value
    function is the *interventional* mean of the true outcome,
    ``v(S) = E[Y | do(X_S = x_S)]``.

  * ``causal_shapley`` — Causal Shapley values (Heskes, Sijben, Bucur, Claassen,
    NeurIPS 2020). Same interventional value function, but explaining a *fitted
    predictor* ``f`` (the population OLS of Y on all features) rather than the
    true structural outcome. Isolates the "explain-a-predictor vs. explain-the-
    effect" gap.

Both require the causal graph, which we own in the synthetic setting.

Interventional value function
-----------------------------
The +1/-1 contrast is defined exactly as the tau labels are (see
``scm_utils.monte_carlo_tau``): intervene feature i at ``mean_i ± std_i`` and
hold the coalition features at their mean, so ``m_i(emptyset) == tau_i``.

  * Linear-mechanism SCMs (linear / hidden / mixed / motifs): the interventional
    mean is a clamped zero-mean topological forward pass, computed EXACTLY (no
    Monte-Carlo noise) via ``_interventional_Z_means``.
  * Nonlinear SCMs that expose ``simulate_do(n, rng, clamp)`` (nonlinear / mlp /
    mlp_hidden / gp_cde): the interventional mean is estimated by Monte-Carlo,
    with common random numbers across the +/- contrast to cancel noise.

Exact Shapley enumeration is used for ``p <= max_exact`` features; a
permutation-sampling estimator is used above that. Every returned vector is
length ``p`` on the tau contrast scale, so it drops straight into the eval
harness alongside ``marginal`` / ``multivariate``. On a pure direct-cause graph
both methods return exactly ``tau``; they diverge from ``tau`` (and each other)
where Shapley redistributes credit (mediators, chains) and where a fitted
predictor differs from the structural effect (leakage / confounding).
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np


# --------------------------------------------------------------------------
# Interventional value function (exact for linear, MC for nonlinear)
# --------------------------------------------------------------------------

def _interventional_Z_means(
    B: np.ndarray, topo_order: np.ndarray, clamp: dict[int, float]
) -> np.ndarray:
    """Exact E[Z | do(Z_S = clamp_S)] for a linear SCM with zero-mean noise.

    Forward pass in topological order with all noise means set to 0. Clamped
    nodes take their fixed value (incoming edges cut); every other node takes
    ``B[node] @ Z`` over already-computed ancestors. Returns the full length-N
    latent mean vector (including any hidden confounder nodes).
    """
    N = B.shape[0]
    Z = np.zeros(N, dtype=np.float64)
    for node in topo_order.tolist():
        node = int(node)
        if node in clamp:
            Z[node] = clamp[node]
        else:
            Z[node] = float(B[node] @ Z)
    return Z


def _scm_stats(scm):
    """Return (topo, y_idx, feature_to_Z, mean_Z, std_Z, N) for any SCM family."""
    topo = np.asarray(scm._topo_order, dtype=int)
    y = int(scm.y_idx_in_Z)
    fz = np.asarray(scm.feature_to_Z, dtype=int)
    std = np.asarray(getattr(scm, "std_Z", None)
                     if getattr(scm, "std_Z", None) is not None
                     else scm._std_Z, dtype=np.float64)
    mean = np.asarray(getattr(scm, "_mean_Z", np.zeros_like(std)), dtype=np.float64)
    return topo, y, fz, mean, std, std.shape[0]


def _make_value_fn(scm, n_mc: int):
    """Return (value_fn, is_mc).

    value_fn(clamp, seed) -> full length-N interventional mean vector E[Z|do(clamp)].
    Linear SCMs ignore the seed (deterministic exact pass); MC SCMs use it as a
    common-random-numbers key so a +/- contrast pair shares one noise stream.
    """
    if hasattr(scm, "simulate_do"):
        def value_fn(clamp, seed):
            Z = scm.simulate_do(n_mc, np.random.default_rng(seed), clamp)
            return Z.mean(axis=0)
        return value_fn, True

    B = np.asarray(scm.B, dtype=np.float64)
    topo = np.asarray(scm._topo_order, dtype=int)

    def value_fn(clamp, seed):
        return _interventional_Z_means(B, topo, clamp)
    return value_fn, False


def _predictor_weights(scm, mean, std, fz, y, rng) -> np.ndarray:
    """Population OLS of standardized Y on all standardized features.

    Analytic from the covariance for linear SCMs (exact); estimated from a large
    observational sample for nonlinear SCMs.
    """
    if hasattr(scm, "simulate_do"):
        samp = scm.sample(n=8192, rng=np.random.default_rng(rng.integers(0, 2**31)))
        X, yv = samp.X, samp.y
        p = X.shape[1]
        XtX = X.T @ X + 1e-6 * np.eye(p) * X.shape[0]
        return np.linalg.solve(XtX, X.T @ yv)
    # Linear: analytic standardized covariance.
    B = np.asarray(scm.B, dtype=np.float64)
    eps_var = np.asarray(getattr(scm, "eps_var", np.ones(B.shape[0])), dtype=np.float64)
    A = np.asarray(getattr(scm, "A", None))
    if A is None or A.shape != B.shape:
        A = np.linalg.solve(np.eye(B.shape[0]) - B, np.eye(B.shape[0]))
    cov_raw = A @ np.diag(eps_var) @ A.T
    D_inv = np.diag(1.0 / std)
    cov_std = D_inv @ cov_raw @ D_inv
    Sigma_XX = cov_std[np.ix_(fz, fz)]
    Sigma_XY = cov_std[fz, y]
    return np.linalg.solve(Sigma_XX + 1e-8 * np.eye(len(fz)), Sigma_XY)


# --------------------------------------------------------------------------
# Shapley coalition machinery
# --------------------------------------------------------------------------

def _shapley_weights(p: int) -> list[float]:
    """Shapley coalition weight w(|S|) = |S|! (p-1-|S|)! / p! for |S| in 0..p-1."""
    return [factorial(k) * factorial(p - 1 - k) / factorial(p) for k in range(p)]


def _iter_coalitions(others: list[int], p: int, max_exact: int, n_perm: int, rng):
    """Yield (coalition_tuple, weight). Exact subset enumeration for small p,
    else a permutation-prefix Monte-Carlo estimator with equal weights."""
    if p <= max_exact:
        w = _shapley_weights(p)
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                yield S, w[k]
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        others_arr = np.asarray(others, dtype=int)
        for _ in range(n_perm):
            perm = rng.permutation(others_arr)
            cut = int(rng.integers(0, len(perm) + 1))
            yield tuple(perm[:cut].tolist()), 1.0 / n_perm


def both_shapley(
    scm, *, max_exact: int = 12, n_perm: int = 256, n_mc: int = 3000, rng=None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (do_shapley, causal_shapley) in one coalition pass.

    Sharing the pass matters for the MC families (each interventional mean is an
    expensive simulate_do call). Returns two length-p vectors on the tau scale.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    topo, y, fz, mean, std, N = _scm_stats(scm)
    p = len(fz)
    std_y = float(std[y])
    value_fn, is_mc = _make_value_fn(scm, n_mc)
    w = _predictor_weights(scm, mean, std, fz, y, rng)

    phi_do = np.zeros(p, dtype=np.float64)
    phi_cs = np.zeros(p, dtype=np.float64)
    for i in range(p):
        zi = int(fz[i])
        xi_mean, xi_std = float(mean[zi]), float(std[zi])
        others = [j for j in range(p) if j != i]
        acc_do = acc_cs = 0.0
        for S, wt in _iter_coalitions(others, p, max_exact, n_perm, rng):
            base = {int(fz[j]): float(mean[int(fz[j])]) for j in S}
            seed = int(rng.integers(0, 2**31)) if is_mc else 0
            Zp = value_fn({**base, zi: xi_mean + xi_std}, seed)
            Zm = value_fn({**base, zi: xi_mean - xi_std}, seed)
            # do-Shapley: contrast on the true outcome.
            acc_do += wt * (float(Zp[y]) - float(Zm[y])) / std_y
            # Causal Shapley: contrast on the OLS predictor of standardized Y.
            Xp = (Zp[fz] - mean[fz]) / std[fz]
            Xm = (Zm[fz] - mean[fz]) / std[fz]
            acc_cs += wt * float(w @ (Xp - Xm))
        phi_do[i] = acc_do
        phi_cs[i] = acc_cs
    return phi_do, phi_cs


def do_shapley(scm, **kw) -> np.ndarray:
    """do-Shapley values (Jung et al. 2022). Length-p, tau scale."""
    return both_shapley(scm, **kw)[0]


def causal_shapley(scm, **kw) -> np.ndarray:
    """Causal Shapley values (Heskes et al. 2020) on the oracle OLS predictor.
    Length-p, tau scale."""
    return both_shapley(scm, **kw)[1]


ORACLE_GRAPH_BASELINES = {
    "do_shapley": do_shapley,
    "causal_shapley": causal_shapley,
}
