"""Shared helpers for the SCM family in `causal_probe.scm*`.

Each SCM class (`MLPSCM`, `NonlinearSCM`, `MLPSCMHidden`,
`LinearNonGaussianSCMHidden`) has its own structural equations and
attribute layout, but several mechanical pieces are repeated verbatim:

  * unit-variance Laplace noise sampling
  * topological-order + Y-node selection (with optional latent prefix)
  * Monte-Carlo do-intervention loop for tau labels
  * sample standardization at the boundary of `sample()`

This module collects those four helpers. Numerical behavior is identical
to the inlined originals (same RNG-consumption order). The base class
`LinearNonGaussianSCM` in `scm.py` is intentionally NOT touched.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def sample_laplace_unit(shape, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean unit-variance Laplace noise. Laplace(0, 1/sqrt(2)) has
    variance 2 (1/sqrt(2))^2 = 1."""
    return rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=shape)


def select_y_and_features(
    p: int, n_hidden: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, int, np.ndarray, List[int]]:
    """Build (topo_order, y_idx_in_Z, feature_to_Z, hidden_indices).

    For ``n_hidden == 0``: permutes all N=p+1 nodes, picks Y uniformly,
    sorts the p remaining feature indices by topological rank.

    For ``n_hidden > 0``: latent nodes precede observed nodes in topo
    order, Y is picked uniformly among observed nodes.

    The RNG-consumption order matches the originals in scm_mlp.py,
    scm_nonlinear.py, scm_hidden.py, and scm_mlp_hidden.py exactly.
    """
    N_obs = p + 1
    N = N_obs + n_hidden
    if n_hidden == 0:
        topo_order = rng.permutation(N).astype(int)
        y_position = int(rng.integers(0, N))
        y_idx_in_Z = int(topo_order[y_position])
        non_y = [int(k) for k in topo_order if int(k) != y_idx_in_Z]
        rank = {int(node): i for i, node in enumerate(topo_order.tolist())}
        feature_to_Z = np.asarray(
            sorted(non_y, key=lambda k: rank[int(k)]), dtype=int,
        )
        return topo_order, y_idx_in_Z, feature_to_Z, []

    latent_order = rng.permutation(np.arange(N_obs, N))
    obs_order = rng.permutation(np.arange(N_obs))
    topo_order = np.concatenate([latent_order, obs_order]).astype(int)
    hidden_indices = [int(h) for h in latent_order.tolist()]
    y_idx_in_Z = int(obs_order[rng.integers(0, N_obs)])
    non_y_obs = [int(k) for k in obs_order if int(k) != y_idx_in_Z]
    rank = {int(node): i for i, node in enumerate(topo_order.tolist())}
    feature_to_Z = np.asarray(
        sorted(non_y_obs, key=lambda k: rank[int(k)]), dtype=int,
    )
    return topo_order, y_idx_in_Z, feature_to_Z, hidden_indices


def monte_carlo_tau(
    *,
    simulate_intervention: Callable,
    n_mc: int,
    rng: np.random.Generator,
    mean_Z: np.ndarray,
    std_Z: np.ndarray,
    y_idx_in_Z: int,
    feature_to_Z: np.ndarray,
) -> np.ndarray:
    """Compute tau via do-intervention at xi_mean ± xi_std.

    ``simulate_intervention(n_mc, rng, intervene_idx, intervene_val)``
    must return an (n_mc, N) array of post-intervention Z values.

    Each tau_i = (E[Y | do(X_i = +sigma_i)] - E[Y | do(X_i = -sigma_i)]) / std(Y).
    Two fresh sub-RNGs per feature (matches the originals' RNG order).
    """
    p = int(np.asarray(feature_to_Z).size)
    y_std = float(std_Z[y_idx_in_Z])
    tau = np.empty(p, dtype=np.float64)
    for i, zi in enumerate(np.asarray(feature_to_Z).tolist()):
        xi_mean = float(mean_Z[int(zi)])
        xi_std = float(std_Z[int(zi)])
        rng_plus = np.random.default_rng(rng.integers(0, 2 ** 31))
        rng_minus = np.random.default_rng(rng.integers(0, 2 ** 31))
        Z_plus = simulate_intervention(n_mc, rng_plus, int(zi), xi_mean + xi_std)
        Z_minus = simulate_intervention(n_mc, rng_minus, int(zi), xi_mean - xi_std)
        E_plus = float(Z_plus[:, y_idx_in_Z].mean())
        E_minus = float(Z_minus[:, y_idx_in_Z].mean())
        tau[i] = (E_plus - E_minus) / y_std
    return tau


def standardize_for_sample(
    Z_raw: np.ndarray,
    mean_Z: np.ndarray,
    std_Z: np.ndarray,
    y_idx_in_Z: int,
    feature_to_Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-standardize and split into (X, y) for the SCMSample boundary."""
    Z_std = (Z_raw - mean_Z.reshape(1, -1)) / std_Z.reshape(1, -1)
    y = Z_std[:, y_idx_in_Z].copy()
    X = Z_std[:, feature_to_Z].copy()
    return X, y
