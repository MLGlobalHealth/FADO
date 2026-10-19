"""Causal-discovery and causal-effect-estimation baselines.

Adds NOTEARS, PC, GES, FCI, and Double Machine Learning to the
existing LiNGAM baseline. All graph-discovery methods share a common
τ-extraction protocol: fit a structural model, extract a weight matrix
$B$, take $A=(I-B)^{-1}$, and report the standardized contrast
$\\tau_i = 2 A_{Y,i}\\,\\sigma_{X_i}/\\sigma_Y$. DoubleML is per-column
like CausalPFN-per-column: for each feature $i$, treat $X_i$ as the
treatment and the rest as covariates.

Methods:
- ``notears`` (Zheng et al. 2018): continuous DAG learning via L-BFGS
  on a smooth acyclicity constraint. Returns a real-valued weight matrix
  directly.
- ``pc`` (Spirtes-Glymour-Scheines): constraint-based, uses CI tests.
  Returns CPDAG; we keep only directed edges, then OLS-refit weights.
- ``ges`` (Greedy Equivalence Search): score-based BIC. Returns CPDAG;
  same OLS refit.
- ``fci`` (Fast Causal Inference): PC variant for hidden confounders.
  Returns PAG; we keep only unambiguously directed edges (tail--head).
- ``doubleml`` (Chernozhukov et al.): per-column. For each $i$, fit
  ``econml.LinearDML`` with $X_i$ as treatment and $X_{\\neq i}$ as
  covariates, take the linear ATE.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tau_from_W(
    W: np.ndarray, X: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """Convert a structural weight matrix $W$ on $[X, y]$ to the
    standardized contrast $\\tau$.

    W is (p+1, p+1) with rows = source, columns = destination.
    Y is the last index (p). $\\tau_i = 2 A_{y,i}\\,\\sigma_{X_i}/\\sigma_y$.
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
    return 2.0 * A[p, :p] * std_X / std_y


def _ols_refit_weights(
    Z: np.ndarray, parents_of: dict[int, list[int]],
) -> np.ndarray:
    """Given a parent-set per node, fit OLS on each node and return
    a directed weight matrix W where W[parent, child] = OLS coef."""
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


# ---------------------------------------------------------------------------
# NOTEARS
# ---------------------------------------------------------------------------

def notears_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from notears.linear import notears_linear
    Z = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    W = notears_linear(Z, lambda1=0.1, loss_type="l2", max_iter=100, w_threshold=0.1)
    return _tau_from_W(W, X, y)


# ---------------------------------------------------------------------------
# PC algorithm (CPDAG output → keep directed edges → OLS refit)
# ---------------------------------------------------------------------------

def _cpdag_directed_parents(cg) -> dict[int, list[int]]:
    """Extract a parents-of-each-node dict from a causal-learn CPDAG.
    Only keeps edges marked as directed (head → tail or i → j)."""
    G = cg.G if hasattr(cg, "G") else cg
    n = G.get_num_nodes()
    nodes = G.get_nodes()
    idx_of = {nd: i for i, nd in enumerate(nodes)}
    parents = {i: [] for i in range(n)}
    for edge in G.get_graph_edges():
        a, b = edge.get_node1(), edge.get_node2()
        from causallearn.graph.Endpoint import Endpoint
        ep1 = edge.get_endpoint1()
        ep2 = edge.get_endpoint2()
        # Endpoint.ARROW means the edge has an arrowhead at that endpoint.
        # A directed edge a -> b has TAIL at a and ARROW at b.
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            parents[idx_of[b]].append(idx_of[a])
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            parents[idx_of[a]].append(idx_of[b])
    return parents


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


# ---------------------------------------------------------------------------
# FCI (PAG output → keep unambiguously directed edges)
# ---------------------------------------------------------------------------

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
        # FCI returns PAG edges with marks {TAIL, ARROW, CIRCLE}.
        # Treat only TAIL→ARROW (i.e., a -> b) as unambiguously directed.
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            parents[idx_of[b]].append(idx_of[a])
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            parents[idx_of[a]].append(idx_of[b])
    W = _ols_refit_weights(Z, parents)
    return _tau_from_W(W, X, y)


# ---------------------------------------------------------------------------
# Double Machine Learning (per-column treatment effect)
# ---------------------------------------------------------------------------

def doubleml_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LinearDML per-column: each X_i serves as the treatment in turn.

    Returns standardized τ̂_i = ATE_i / std(y), so the result is on the
    same reporting scale as the probe and other baselines.
    """
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV
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
            # econml LinearDML expects W=covariates, T=treatment, Y=outcome.
            est.fit(y.astype(np.float64), t.astype(np.float64), W=X_rest.astype(np.float64))
            ate = float(est.ate(X=None))
        except Exception:
            continue
        tau[i] = 2.0 * ate * float(np.std(t)) / std_y
    return tau


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

def causal_forest_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Causal Forest per-column: each X_i serves as the treatment in turn.

    Uses ``econml.dml.CausalForestDML`` with default RandomForest first
    stages. Returns standardized $\\tau_i = 2\\,\\mathrm{ATE}_i\\,
    \\sigma_{X_i}/\\sigma_y$ on the probe's reporting scale.
    """
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


METHODS = {
    "notears": notears_tau,
    "pc": pc_tau,
    "ges": ges_tau,
    "fci": fci_tau,
    "doubleml": doubleml_tau,
    "causal_forest": causal_forest_tau,
}


def eval_method_on_scms(
    method: str, scms: list, n_rows: int, rng: np.random.Generator,
) -> dict:
    from scipy.stats import spearmanr, pearsonr

    fn = METHODS[method]
    preds, truths = [], []
    n_failed = 0
    for k, scm in enumerate(scms):
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
        try:
            tau_hat = fn(samp.X.astype(np.float64), samp.y.astype(np.float64))
        except Exception as e:
            print(f"  [{method}] SCM {k}: crashed: {e}")
            n_failed += 1
            continue
        preds.append(tau_hat)
        truths.append(samp.tau)

    if not preds:
        return {"error": f"all {method} evaluations failed"}

    preds = np.stack(preds)
    truths = np.stack(truths)
    flat_p = preds.reshape(-1)
    flat_t = truths.reshape(-1)
    mask = np.isfinite(flat_p) & np.isfinite(flat_t)
    if not mask.any():
        return {"error": f"all {method} predictions are NaN"}
    pe, _ = pearsonr(flat_p[mask], flat_t[mask])
    sp, _ = spearmanr(flat_p[mask], flat_t[mask])
    mse = float(np.mean((flat_p[mask] - flat_t[mask]) ** 2))
    ss_tot = float(np.sum((flat_t[mask] - flat_t[mask].mean()) ** 2))
    r2 = 1.0 - float(np.sum((flat_p[mask] - flat_t[mask]) ** 2)) / max(ss_tot, 1e-12)
    from sklearn.metrics import roc_auc_score

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
        "method": method, "n_scms": len(preds), "n_failed": n_failed,
        "n_features_finite": int(mask.sum()),
        "n_features_total": int(mask.size),
        "pearson": float(pe), "spearman": float(sp), "r2": r2, "mse": mse,
        "auroc_nonzero": auc, "mae_zero": mae_zero,
    }


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
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scms = []
    for _ in range(args.n_scms):
        seed = rng.integers(0, 2**31)
        if args.scm_type == "linear":
            from causal_probe.scm import LinearNonGaussianSCM
            scms.append(LinearNonGaussianSCM(p=args.p, rng=np.random.default_rng(seed), noise=args.noise))
        elif args.scm_type == "nonlinear":
            from causal_probe.scm_nonlinear import NonlinearSCM
            scms.append(NonlinearSCM(p=args.p, rng=np.random.default_rng(seed), n_mc=2048))
        elif args.scm_type == "mlp":
            from causal_probe.scm_mlp import MLPSCM
            scms.append(MLPSCM(p=args.p, rng=np.random.default_rng(seed), n_mc=2048))
        elif args.scm_type == "hidden":
            from causal_probe.scm_hidden import LinearNonGaussianSCMHidden
            scms.append(LinearNonGaussianSCMHidden(p=args.p, rng=np.random.default_rng(seed), noise=args.noise))
        elif args.scm_type == "mixed":
            from causal_probe.scm_mixed import LinearMixedSCM
            scms.append(LinearMixedSCM(p=args.p, rng=np.random.default_rng(seed), noise=args.noise))

    print(
        f"Evaluating {args.method} on {len(scms)} SCMs "
        f"({args.scm_type}, p={args.p}, noise={args.noise}, n_rows={args.n_rows})"
    )
    res = eval_method_on_scms(args.method, scms, n_rows=args.n_rows, rng=rng)
    print(json.dumps(res, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        res["config"] = {
            "method": args.method, "scm_type": args.scm_type, "p": args.p,
            "noise": args.noise, "n_rows": args.n_rows, "seed": args.seed,
        }
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
