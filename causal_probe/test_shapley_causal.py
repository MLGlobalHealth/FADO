"""Tests for the oracle-graph causal-Shapley baselines."""
from __future__ import annotations

import numpy as np
import pytest

from causal_probe.scm import LinearNonGaussianSCM
from causal_probe.motifs import (
    motif_direct_cause, motif_mediator, motif_observed_confounder,
    motif_target_descendant, motif_collider, motif_chained_mediators_k3,
    motif_scm,
)
from causal_probe.shapley_causal import (
    do_shapley, causal_shapley, _interventional_Z_means, _shapley_weights,
)


def _mc_intervention_mean_Y(scm, clamp, n=400_000, seed=0):
    """Monte-Carlo E[Y | do(Z_S = clamp_S)] by forward-simulating the mutilated SCM."""
    rng = np.random.default_rng(seed)
    N = scm.B.shape[0]
    eps = rng.standard_normal((n, N))  # zero-mean noise; means are noise-free
    Z = np.zeros((n, N))
    for node in scm._topo_order.tolist():
        node = int(node)
        if node in clamp:
            Z[:, node] = clamp[node]
        else:
            parents = np.nonzero(scm.B[node])[0]
            Z[:, node] = Z[:, parents] @ scm.B[node, parents] + eps[:, node]
    return float(Z[:, scm.y_idx_in_Z].mean())


def test_shapley_weights_sum_to_one():
    for p in (2, 3, 5, 8):
        # Sum over coalitions of weight = 1 for a fixed feature i:
        # sum_k C(p-1,k) * w(k) == 1.
        from math import comb
        w = _shapley_weights(p)
        total = sum(comb(p - 1, k) * w[k] for k in range(p))
        assert abs(total - 1.0) < 1e-12


def test_interventional_means_match_monte_carlo():
    rng = np.random.default_rng(3)
    scm = LinearNonGaussianSCM(p=5, rng=rng, noise="gaussian")
    fz = scm.feature_to_Z
    # Clamp features {0, 2} to arbitrary values, check E[Y|do] matches MC.
    clamp = {int(fz[0]): 1.7, int(fz[2]): -0.9}
    exact = _interventional_Z_means(scm.B, scm._topo_order, clamp)[scm.y_idx_in_Z]
    mc = _mc_intervention_mean_Y(scm, clamp)
    assert abs(exact - mc) < 0.02, (exact, mc)


def test_direct_cause_recovers_tau():
    # Pure direct cause: no mediators/confounders -> both methods == tau exactly.
    spec = motif_direct_cause(p=5, rng=np.random.default_rng(1))
    scm = motif_scm(spec, rng=np.random.default_rng(1))
    for phi in (do_shapley(scm), causal_shapley(scm)):
        assert np.allclose(phi, scm.tau, atol=1e-9), (phi, scm.tau)


def test_mediator_do_shapley_splits_credit():
    # X0 -> X1 -> Y. do-Shapley: phi0 = tau0/2 (mediated path split), phi1 = tau1.
    spec = motif_mediator(p=5, rng=np.random.default_rng(2))
    scm = motif_scm(spec, rng=np.random.default_rng(2))
    phi = do_shapley(scm)
    assert abs(phi[0] - scm.tau[0] / 2.0) < 1e-9, (phi[0], scm.tau[0])
    assert abs(phi[1] - scm.tau[1]) < 1e-9, (phi[1], scm.tau[1])
    # Noise features are exactly zero.
    assert np.allclose(phi[2:], 0.0, atol=1e-12)


def test_chained_mediators_attenuate_shapley():
    # X0->X1->X2->X3->Y: earliest cause gets most credit split away.
    spec = motif_chained_mediators_k3(p=5, rng=np.random.default_rng(7))
    scm = motif_scm(spec, rng=np.random.default_rng(7))
    phi = do_shapley(scm)
    # Same total effect sign, but Shapley share of X0 is a fraction of tau0.
    assert np.sign(phi[0]) == np.sign(scm.tau[0])
    assert abs(phi[0]) < abs(scm.tau[0]) + 1e-9
    # X0 is split across more blocking coalitions than X3, so its shrink factor
    # (phi/tau) is smaller than the mediator nearest Y.
    frac0 = phi[0] / scm.tau[0]
    frac3 = phi[3] / scm.tau[3]
    assert frac0 < frac3


@pytest.mark.parametrize("make", [
    motif_observed_confounder, motif_target_descendant, motif_collider,
])
def test_do_shapley_zeros_all_noncausal_features(make):
    # do-Shapley explains the true outcome via do(X_S), so ANY feature with no
    # directed path to Y (confounded, leakage, collider) gets exactly 0.
    spec = make(p=5, rng=np.random.default_rng(5))
    scm = motif_scm(spec, rng=np.random.default_rng(5))
    zero_feats = np.abs(scm.tau) < 1e-9
    phi = do_shapley(scm)
    assert np.allclose(phi[zero_feats], 0.0, atol=1e-6), (phi, scm.tau)


def test_causal_shapley_zeros_observed_confounder():
    # With the confounder OBSERVED, the oracle predictor puts no weight on the
    # confounded proxy X0, so causal-Shapley also zeros it.
    spec = motif_observed_confounder(p=5, rng=np.random.default_rng(5))
    scm = motif_scm(spec, rng=np.random.default_rng(5))
    phi = causal_shapley(scm)
    assert abs(phi[0]) < 1e-6, (phi, scm.tau)


@pytest.mark.parametrize("make", [motif_target_descendant, motif_collider])
def test_causal_shapley_misattributes_leakage(make):
    # Causal-Shapley explains a fitted predictor, which latches onto a strong
    # leakage feature (descendant of Y / collider). Under do() the predictor's
    # output still moves, so it assigns large credit where the true effect is 0.
    # This is the "explain-a-predictor vs explain-the-effect" failure, and the
    # reason do-Shapley and causal-Shapley diverge on leakage graphs.
    spec = make(p=5, rng=np.random.default_rng(5))
    scm = motif_scm(spec, rng=np.random.default_rng(5))
    leak = np.argmax(np.abs(causal_shapley(scm)))
    assert np.abs(scm.tau[leak]) < 1e-9          # true effect there is zero
    assert np.abs(causal_shapley(scm)[leak]) > 0.3  # but the method fires
    assert np.abs(do_shapley(scm)[leak]) < 1e-6     # do-Shapley correctly does not


def _nonlinear_scms():
    from causal_probe.scm_nonlinear import NonlinearSCM
    from causal_probe.scm_mlp import MLPSCM
    from causal_probe.scm_mlp_hidden import MLPSCMHidden
    return [
        ("nonlinear", NonlinearSCM(p=4, rng=np.random.default_rng(1), n_mc=4000)),
        ("mlp", MLPSCM(p=4, rng=np.random.default_rng(2), n_mc=4000)),
        ("mlp_hidden", MLPSCMHidden(p=4, rng=np.random.default_rng(3), n_mc=4000)),
    ]


@pytest.mark.parametrize("name,scm", _nonlinear_scms())
def test_simulate_do_matches_single_node_intervention(name, scm):
    # simulate_do({node: val}) must equal the class's own _simulate_intervention
    # for a singleton clamp, under the same RNG — proves the generalization is
    # a faithful superset of the validated single-node machinery.
    zi = int(scm.feature_to_Z[0])
    val = float(scm._mean_Z[zi] + scm._std_Z[zi])
    a = scm.simulate_do(2000, np.random.default_rng(7), {zi: val})
    b = scm._simulate_intervention(2000, np.random.default_rng(7), zi, val)
    assert np.array_equal(a, b), name


@pytest.mark.parametrize("name,scm", _nonlinear_scms())
def test_mc_empty_coalition_contrast_recovers_tau(name, scm):
    # The single-feature (empty coalition) do-contrast IS tau by construction;
    # check the simulate_do MC estimate matches the SCM's own tau labels.
    fz, y = scm.feature_to_Z, scm.y_idx_in_Z
    std_y = float(scm._std_Z[y])
    for i, zi in enumerate(fz.tolist()):
        m, s = float(scm._mean_Z[zi]), float(scm._std_Z[zi])
        seed = 100 + i
        Zp = scm.simulate_do(20000, np.random.default_rng(seed), {zi: m + s})
        Zm = scm.simulate_do(20000, np.random.default_rng(seed), {zi: m - s})
        tau_i_mc = (Zp[:, y].mean() - Zm[:, y].mean()) / std_y
        assert abs(tau_i_mc - scm.tau[i]) < 0.05, (name, i, tau_i_mc, scm.tau[i])


@pytest.mark.parametrize("name,scm", _nonlinear_scms())
def test_do_shapley_runs_and_signs_track_tau_nonlinear(name, scm):
    from causal_probe.shapley_causal import both_shapley
    phi_do, phi_cs = both_shapley(scm, n_mc=4000, rng=np.random.default_rng(0))
    assert phi_do.shape == scm.tau.shape and np.all(np.isfinite(phi_do))
    assert np.all(np.isfinite(phi_cs))
    # On signal features, do-Shapley sign should agree with tau (causal method).
    signal = np.abs(scm.tau) > 0.15
    if signal.any():
        assert np.mean(np.sign(phi_do[signal]) == np.sign(scm.tau[signal])) >= 0.75, (name, phi_do, scm.tau)


def test_permutation_sampling_approximates_exact():
    rng = np.random.default_rng(11)
    scm = LinearNonGaussianSCM(p=6, rng=rng, noise="laplace")
    exact = do_shapley(scm, max_exact=14)
    approx = do_shapley(scm, max_exact=0, n_perm=20000,
                        rng=np.random.default_rng(0))
    assert np.max(np.abs(exact - approx)) < 0.03, (exact, approx)
