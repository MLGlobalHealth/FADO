"""Unified baseline-driver dispatcher for the causal feature probe.

Replaces the per-method scripts:
  - lingam_baseline.py
  - permutation_baseline.py
  - predictive_baselines.py
  - causal_discovery_baselines.py
  - causalpfn_per_column_baseline.py

Each method exposes a function ``<method>_tau(X, y, **kwargs) -> np.ndarray``
returning a length-p vector of standardized contrast estimates. The
``METHODS`` dict dispatches by name; ``main`` is the unified CLI.

Marginal correlation, multivariate ridge, and zero baselines are imported
under aliases from ``causal_probe.baselines`` so the existing 8 importers
of that module keep working.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from causal_probe.baselines import (
    baseline_marginal as marginal_tau,
    baseline_multivariate as ridge_tau,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# LiNGAM
# ---------------------------------------------------------------------------

def lingam_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """DirectLiNGAM on [X, y]; return standardized contrast vector."""
    from lingam import DirectLiNGAM
    p = X.shape[1]
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    model = DirectLiNGAM()
    model.fit(data)
    B = model.adjacency_matrix_
    N = p + 1
    I = np.eye(N)
    try:
        A = np.linalg.solve(I - B, I)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)
    y_idx = p
    std_X = X.std(axis=0).clip(min=1e-9)
    std_Y = float(y.std().clip(min=1e-9))
    return 2.0 * A[y_idx, :p] * std_X / std_Y


# ---------------------------------------------------------------------------
# Permutation importance (LightGBM + sklearn permutation_importance)
# ---------------------------------------------------------------------------

def permutation_tau(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_repeats: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Signed permutation-importance contrast on the probe's reporting scale."""
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb

    X = X.astype(np.float64)
    y = y.astype(np.float64)
    p = X.shape[1]
    std_y = float(np.std(y).clip(min=1e-9))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed,
    )
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        min_child_samples=10, num_leaves=31, verbose=-1, random_state=seed,
    )
    model.fit(X_tr, y_tr)
    res = permutation_importance(
        model, X_te, y_te, n_repeats=n_repeats, random_state=seed,
        scoring="r2",
    )
    importances = np.asarray(res.importances_mean, dtype=np.float64)

    signs = np.zeros(p, dtype=np.float64)
    for i in range(p):
        sd = float(np.std(X[:, i]).clip(min=1e-9))
        signs[i] = np.sign(np.cov(X[:, i], y, ddof=1)[0, 1] / (sd * std_y))
    signs[signs == 0] = 1.0

    # ΔR² is unitless; do NOT divide by std_y (would give 1/std_y units).
    # Pearson/Spearman are scale-invariant and unaffected; the published
    # tab:predictive-baselines numbers do not move.
    return signs * importances


# ---------------------------------------------------------------------------
# TreeSHAP via LightGBM (predictive attribution)
# ---------------------------------------------------------------------------

def shap_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LightGBM-regressor TreeSHAP, signed by marginal corr, /std(y)."""
    import lightgbm as lgb

    X = X.astype(np.float64)
    y = y.astype(np.float64)
    p = X.shape[1]
    std_y = float(np.std(y).clip(min=1e-9))

    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        min_child_samples=10, num_leaves=31, verbose=-1, random_state=0,
    )
    model.fit(X, y)
    contrib = model.predict(X, pred_contrib=True)
    shap_vals = contrib[:, :-1]
    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    signs = np.zeros(p, dtype=np.float64)
    for i in range(p):
        sd = float(np.std(X[:, i]).clip(min=1e-9))
        signs[i] = np.sign(np.cov(X[:, i], y, ddof=1)[0, 1] / (sd * std_y))
    signs[signs == 0] = 1.0

    return signs * mean_abs / std_y


# ---------------------------------------------------------------------------
# Causal-discovery shared helpers (used by NOTEARS / PC / GES / FCI)
# ---------------------------------------------------------------------------

def _tau_from_W(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convert structural weight matrix W on [X, y] to standardized contrast.

    NOTEARS/PC/GES/FCI all return W in **parent-row** convention
    (W[parent, child] = edge weight). The total-effect matrix
    A = (I - W)^-1 then has A[i, j] = effect of node i on node j (parent
    propagating forward). To get the effect of features (0..p-1) on y
    (index p), we slice column p across rows 0..p-1: A[:p, p].

    LiNGAM, by contrast, uses **child-row** convention (B[child, parent]),
    so its closed-form inverse uses the row-p slice A[p, :p]. The two
    look syntactically similar but pick opposite directions; using the
    row-p slice on a parent-row matrix yields the effect of y on
    features, which is approximately zero in the relevant cases (and
    silently understated the published Pearson values for
    NOTEARS/PC/GES/FCI prior to this fix).
    """
    p = X.shape[1]
    N = p + 1
    I = np.eye(N)
    try:
        A = np.linalg.solve(I - W, I)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)
    std_X = X.std(axis=0).clip(min=1e-9)
    std_y = float(y.std().clip(min=1e-9))
    return 2.0 * A[:p, p] * std_X / std_y


def _ols_refit_weights(Z: np.ndarray, parents_of: dict) -> np.ndarray:
    """OLS-refit edge weights given a parent-set per node."""
    N = Z.shape[1]
    W = np.zeros((N, N), dtype=np.float64)
    for j in range(N):
        pa = parents_of.get(j, [])
        if not pa:
            continue
        X_pa = Z[:, pa]
        try:
            beta, *_ = np.linalg.lstsq(X_pa, Z[:, j], rcond=None)
        except np.linalg.LinAlgError:
            continue
        for k, p_idx in enumerate(pa):
            W[p_idx, j] = beta[k]
    return W


def _cpdag_directed_parents(cg) -> dict:
    """Extract a parents-of-each-node dict from a causal-learn CPDAG."""
    from causallearn.graph.Endpoint import Endpoint
    G = cg.G if hasattr(cg, "G") else cg
    n = G.get_num_nodes()
    nodes = G.get_nodes()
    idx_of = {nd: i for i, nd in enumerate(nodes)}
    parents = {i: [] for i in range(n)}
    for edge in G.get_graph_edges():
        a, b = edge.get_node1(), edge.get_node2()
        ep1 = edge.get_endpoint1()
        ep2 = edge.get_endpoint2()
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            parents[idx_of[b]].append(idx_of[a])
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            parents[idx_of[a]].append(idx_of[b])
    return parents


# ---------------------------------------------------------------------------
# NOTEARS / PC / GES / FCI
# ---------------------------------------------------------------------------

def notears_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from notears.linear import notears_linear
    Z = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    W = notears_linear(Z, lambda1=0.1, loss_type="l2", max_iter=100, w_threshold=0.1)
    return _tau_from_W(W, X, y)


def pc_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from causallearn.search.ConstraintBased.PC import pc
    Z = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    cg = pc(Z, alpha=0.05, indep_test="fisherz", show_progress=False, verbose=False)
    parents = _cpdag_directed_parents(cg)
    W = _ols_refit_weights(Z, parents)
    return _tau_from_W(W, X, y)


def ges_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from causallearn.search.ScoreBased.GES import ges
    Z = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    res = ges(Z, score_func="local_score_BIC")
    parents = _cpdag_directed_parents(res["G"])
    W = _ols_refit_weights(Z, parents)
    return _tau_from_W(W, X, y)


def fci_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.graph.Endpoint import Endpoint

    Z = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    G, _edges = fci(Z, alpha=0.05, independence_test_method="fisherz",
                    verbose=False, show_progress=False)
    n = G.get_num_nodes()
    nodes = G.get_nodes()
    idx_of = {nd: i for i, nd in enumerate(nodes)}
    parents = {i: [] for i in range(n)}
    for edge in G.get_graph_edges():
        a, b = edge.get_node1(), edge.get_node2()
        ep1 = edge.get_endpoint1()
        ep2 = edge.get_endpoint2()
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            parents[idx_of[b]].append(idx_of[a])
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            parents[idx_of[a]].append(idx_of[b])
    W = _ols_refit_weights(Z, parents)
    return _tau_from_W(W, X, y)


# ---------------------------------------------------------------------------
# Per-column DML / Causal Forest / CausalPFN
# ---------------------------------------------------------------------------

def doubleml_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LinearDML per-column."""
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor

    p = X.shape[1]
    std_y = float(y.std().clip(min=1e-9))
    tau = np.full(p, np.nan, dtype=np.float64)
    for i in range(p):
        t = X[:, i]
        if np.std(t) < 1e-9:
            continue
        X_rest = np.delete(X, i, axis=1)
        try:
            est = LinearDML(
                model_y=RandomForestRegressor(n_estimators=50, max_depth=5,
                                              min_samples_leaf=10, random_state=0),
                model_t=RandomForestRegressor(n_estimators=50, max_depth=5,
                                              min_samples_leaf=10, random_state=0),
                discrete_treatment=False, cv=2, random_state=0,
            )
            est.fit(y.astype(np.float64), t.astype(np.float64), W=X_rest.astype(np.float64))
            ate = float(est.ate(X=None))
        except Exception:
            continue
        tau[i] = 2.0 * ate * float(np.std(t)) / std_y
    return tau


def causal_forest_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CausalForestDML per-column."""
    from econml.dml import CausalForestDML
    p = X.shape[1]
    std_y = float(y.std().clip(min=1e-9))
    tau = np.full(p, np.nan, dtype=np.float64)
    for i in range(p):
        t = X[:, i]
        if np.std(t) < 1e-9:
            continue
        X_rest = np.delete(X, i, axis=1)
        try:
            est = CausalForestDML(
                n_estimators=100, min_samples_leaf=10, max_depth=8,
                discrete_treatment=False, cv=2, random_state=0,
            )
            est.fit(y.astype(np.float64), t.astype(np.float64), X=X_rest.astype(np.float64))
            ate = float(est.ate(X=X_rest.astype(np.float64)))
        except Exception:
            continue
        tau[i] = 2.0 * ate * float(np.std(t)) / std_y
    return tau


# ---------------------------------------------------------------------------
# Probe (the trained tabular causal probe itself, callable as a "baseline"
# in the eval driver so probe and other methods can be run paired on the
# same SCM stream)
# ---------------------------------------------------------------------------

_PROBE_MODEL_CACHE: dict = {}


def probe_tau(
    X: np.ndarray, y: np.ndarray, *,
    ckpt_path: str, device: str = "cpu",
) -> np.ndarray:
    """Trained probe forward pass, callable through the same dispatcher.

    Caches the loaded model per ckpt path so repeated calls in a loop
    don't re-deserialize the .ckpt for every SCM.
    """
    import torch
    if ckpt_path not in _PROBE_MODEL_CACHE:
        from causal_probe.eval import _load_model
        _PROBE_MODEL_CACHE[ckpt_path] = _load_model(ckpt_path, device=device)
    model = _PROBE_MODEL_CACHE[ckpt_path]
    X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(X_t, y_t).squeeze(0).cpu().numpy().astype(np.float64)


def causalpfn_tau(
    X: np.ndarray, y: np.ndarray, *,
    device: str = "cpu", cache_dir: str | None = None,
) -> np.ndarray:
    """CausalPFN ATEEstimator once per feature; binarized treatment at median."""
    import os
    from causalpfn import ATEEstimator

    if cache_dir is None:
        cache_dir = os.environ.get("CAUSALPFN_CACHE_DIR")

    p = X.shape[1]
    std_y = float(np.std(y).clip(min=1e-9))
    tau = np.full(p, np.nan, dtype=np.float64)

    for i in range(p):
        x_i = X[:, i]
        med = float(np.median(x_i))
        t = (x_i > med).astype(np.int64)
        if t.sum() < 5 or (1 - t).sum() < 5:
            continue
        # Empirical gap between above-median and below-median X groups, in
        # original-X units. Used to convert binary ATE -> per-unit-std-X
        # effect (under approximate linearity, ate ≈ beta·gap, so
        # ate·std_X/gap ≈ beta·std_X = effect over a 1-std-X span).
        gap = float(x_i[t == 1].mean() - x_i[t == 0].mean())
        if gap < 1e-9:
            continue
        std_X_i = float(np.std(x_i).clip(min=1e-9))
        X_rest = np.delete(X, i, axis=1).astype(np.float64)
        try:
            kwargs = {"device": device}
            if cache_dir is not None:
                kwargs["cache_dir"] = cache_dir
            est = ATEEstimator(**kwargs)
            est.fit(X_rest, t, y.astype(np.float64))
            ate = float(est.estimate_ate())
        except Exception as e:
            print(f"    feature {i}: CausalPFN fit failed: {e}")
            continue
        tau[i] = 2.0 * ate * std_X_i / (gap * std_y)

    return tau


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "marginal":      marginal_tau,
    "ridge":         ridge_tau,
    "permutation":   permutation_tau,
    "shap":          shap_tau,
    "lingam":        lingam_tau,
    "notears":       notears_tau,
    "pc":            pc_tau,
    "ges":           ges_tau,
    "fci":           fci_tau,
    "doubleml":      doubleml_tau,
    "causal_forest": causal_forest_tau,
    "causalpfn":     causalpfn_tau,
    "probe":         probe_tau,
}


# ---------------------------------------------------------------------------
# Shared SCM-sampling and metric helpers
# ---------------------------------------------------------------------------

def _sample_scms(
    scm_type: str, p: int, noise: str, n_scms: int, rng: np.random.Generator,
) -> list:
    """Build n_scms SCMs of the given type. Mirrors the per-driver if-elif chain
    to preserve exact rng-consumption order."""
    scms = []
    for _ in range(n_scms):
        seed = rng.integers(0, 2**31)
        if scm_type == "linear":
            from causal_probe.scm import LinearNonGaussianSCM
            scms.append(LinearNonGaussianSCM(p=p, rng=np.random.default_rng(seed), noise=noise))
        elif scm_type == "nonlinear":
            from causal_probe.scm_nonlinear import NonlinearSCM
            scms.append(NonlinearSCM(p=p, rng=np.random.default_rng(seed), n_mc=2048))
        elif scm_type == "mlp":
            from causal_probe.scm_mlp import MLPSCM
            scms.append(MLPSCM(p=p, rng=np.random.default_rng(seed), n_mc=2048))
        elif scm_type == "hidden":
            from causal_probe.scm_hidden import LinearNonGaussianSCMHidden
            scms.append(LinearNonGaussianSCMHidden(p=p, rng=np.random.default_rng(seed), noise=noise))
        elif scm_type == "mixed":
            from causal_probe.scm_mixed import LinearMixedSCM
            scms.append(LinearMixedSCM(p=p, rng=np.random.default_rng(seed), noise=noise))
    return scms


def _compute_metrics(preds: np.ndarray, truths: np.ndarray) -> dict:
    """Standard Pearson/Spearman/R²/MSE/AUROC/MAE block."""
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import roc_auc_score

    flat_p = preds.reshape(-1)
    flat_t = truths.reshape(-1)
    mask = np.isfinite(flat_p) & np.isfinite(flat_t)
    if not mask.any():
        return {"error": "all predictions are NaN"}
    pe, _ = pearsonr(flat_p[mask], flat_t[mask])
    sp, _ = spearmanr(flat_p[mask], flat_t[mask])
    mse = float(np.mean((flat_p[mask] - flat_t[mask]) ** 2))
    ss_tot = float(np.sum((flat_t[mask] - flat_t[mask].mean()) ** 2))
    r2 = 1.0 - float(np.sum((flat_p[mask] - flat_t[mask]) ** 2)) / max(ss_tot, 1e-12)
    labels_nz = (np.abs(flat_t[mask]) > 0.1).astype(int)
    if 0 < labels_nz.sum() < len(labels_nz):
        auc = float(roc_auc_score(labels_nz, np.abs(flat_p[mask])))
    else:
        auc = float("nan")
    zero_mask = np.abs(flat_t[mask]) <= 0.1
    mae_zero = (
        float(np.mean(np.abs(flat_p[mask][zero_mask] - flat_t[mask][zero_mask])))
        if zero_mask.any() else float("nan")
    )
    return {
        "pearson": float(pe),
        "spearman": float(sp),
        "r2": r2,
        "mse": mse,
        "auroc_nonzero": auc,
        "mae_zero": mae_zero,
        "n_features_finite": int(mask.sum()),
        "n_features_total": int(mask.size),
    }


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

def eval_method_on_scms(
    method: str,
    scms: list,
    n_rows: int,
    rng: np.random.Generator,
    *,
    method_kwargs: dict | None = None,
) -> dict:
    """Run METHODS[method] on a list of SCMs, return metrics dict."""
    if method_kwargs is None:
        method_kwargs = {}
    fn = METHODS[method]

    preds, truths = [], []
    n_failed = 0
    for k, scm in enumerate(scms):
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
        # Permutation needs an extra rng draw for its internal random_state, to
        # match the original driver's rng-consumption order.
        kw = dict(method_kwargs)
        if method == "permutation" and "seed" not in kw:
            kw["seed"] = int(rng.integers(0, 2**31))
        try:
            tau_hat = fn(samp.X.astype(np.float64), samp.y.astype(np.float64), **kw)
        except Exception as e:
            print(f"  [{method}] SCM {k}: {e}")
            n_failed += 1
            continue
        preds.append(tau_hat)
        truths.append(samp.tau)

    if not preds:
        return {"error": f"all {method} evaluations failed", "n_failed": n_failed}

    preds = np.stack(preds)
    truths = np.stack(truths)
    res = _compute_metrics(preds, truths)
    res["method"] = method
    res["n_scms"] = len(preds)
    res["n_failed"] = n_failed
    res["_raw"] = {"pred": preds, "true": truths}  # popped before JSON dump
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS.keys()))
    ap.add_argument(
        "--scm-type", default="linear",
        choices=["linear", "nonlinear", "mlp", "hidden", "mixed"],
    )
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", default="laplace", choices=["laplace", "gaussian"])
    ap.add_argument("--n-scms", type=int, default=100)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-repeats", type=int, default=10,
                    help="permutation: --n-repeats; ignored otherwise")
    ap.add_argument("--device", default="cpu",
                    help="causalpfn / probe: --device; ignored otherwise")
    ap.add_argument("--ckpt", default=None,
                    help="probe: path to a .ckpt file (required for --method probe)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None,
                    help="If set, save per-SCM (n_scms, p) arrays (pred, true) "
                         "for bootstrap CIs.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scms = _sample_scms(args.scm_type, args.p, args.noise, args.n_scms, rng)

    method_kwargs = {}
    if args.method == "permutation":
        method_kwargs["n_repeats"] = args.n_repeats
    elif args.method == "causalpfn":
        method_kwargs["device"] = args.device
    elif args.method == "probe":
        if args.ckpt is None:
            ap.error("--method probe requires --ckpt")
        method_kwargs["ckpt_path"] = args.ckpt
        method_kwargs["device"] = args.device

    print(
        f"Evaluating {args.method} on {len(scms)} SCMs "
        f"({args.scm_type}, p={args.p}, noise={args.noise}, n_rows={args.n_rows})"
    )
    res = eval_method_on_scms(
        args.method, scms, n_rows=args.n_rows, rng=rng,
        method_kwargs=method_kwargs,
    )
    raw = res.pop("_raw", None)
    print(json.dumps(res, indent=2))

    if args.out_npz and raw is not None:
        Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_npz,
            pred=raw["pred"].astype(np.float32),
            true=raw["true"].astype(np.float32),
        )
        print(f"wrote {args.out_npz}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        res["config"] = {
            "method": args.method, "scm_type": args.scm_type, "p": args.p,
            "noise": args.noise, "n_rows": args.n_rows, "seed": args.seed,
            "n_scms": args.n_scms,
        }
        if args.method == "permutation":
            res["config"]["n_repeats"] = args.n_repeats
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
