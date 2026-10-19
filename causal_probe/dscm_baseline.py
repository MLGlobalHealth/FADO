"""Deep/learned-mechanism Structural Causal Model (DSCM) baseline.

This fills the missing rung of the oracle ladder for the shapley-comparison
table (a comparison with Deep Structural Causal
Models):

  * do-Shapley / Causal-Shapley  -> TRUE graph AND TRUE mechanisms (full oracle).
  * FADO                         -> nothing (only the observational (X, y)).
  * **DSCM (this module)**        -> TRUE graph, mechanisms LEARNED FROM DATA.

The DSCM is the realistic "you know the DAG but must estimate the mechanisms"
setting (Pawlowski, Castro, Glocker, NeurIPS 2020, "Deep Structural Causal
Models for Tractable Counterfactual Inference"). We are given the true causal
graph over the OBSERVED variables (feature parent sets + topological order,
accessed exactly the way ``shapley_causal.py`` reaches into the SCM), and the
SAME observational sample ``(X, y)`` that FADO sees. For each non-root observed
node we fit a flexible regressor ``E[node | observed parents]`` from that
sample; root observed nodes keep their empirical marginal mean. We then compute
FADO's exact estimand by forward-simulating the conditional means through the
fitted mechanisms under a ±1σ do-intervention.

Estimand (matches ``scm_utils.monte_carlo_tau`` / ``shapley_causal.both_shapley``)
-------------------------------------------------------------------------------
The SCMs hand FADO STANDARDIZED columns: ``X_std_i = (Z_i - mean_i) / std_i`` and
``y_std = (Y - mean_Y) / std_Y``. The tau label is

    tau_i = (E[Y | do(Z_i = mean_i + std_i)] - E[Y | do(Z_i = mean_i - std_i)]) / std_Y.

Because the shift ``mean_i ± std_i`` in raw space is exactly ``±1`` in
standardized space, and dividing the raw-Y contrast by ``std_Y`` is exactly the
standardized-Y contrast (the constant ``mean_Y`` cancels in the difference),

    tau_i = E[y_std | do(X_std_i = +1)] - E[y_std | do(X_std_i = -1)].

So the DSCM can work ENTIRELY in the standardized space of the sample FADO sees:
fit mechanisms on standardized columns, ``do(X_std_i = ±1)``, forward-propagate
conditional means in topological order (non-intervened roots held at their
empirical mean ≈ 0), read off ``E[y_std]``, and take the difference. This is the
identical mean_i / std_i / std_Y contrast with no re-standardization needed and
no reach into the SCM's raw-space stats. Non-intervened observed roots are held
at their sample mean, matching the do-Shapley empty-coalition baseline (features
held at their mean) so the DSCM's tau_i is on the same scale as ``tau`` itself.

Hidden-confounder families (hidden / mlp_hidden)
------------------------------------------------
The realistic DSCM is given the TRUE graph over the OBSERVED variables only — it
does NOT see the latent confounders. We therefore fit each observed node's
mechanism on its OBSERVED parents only (dropping any latent parent), and let the
estimate degrade honestly: observed-parent regression cannot adjust for an
unmeasured common cause, so the fitted ``E[node | observed parents]`` absorbs the
confounding bias. This is the honest, interesting result (DSCM-with-true-graph
is still fooled by latents). We do NOT give the DSCM the latent nodes even
though the synthetic SCM exposes them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# True-graph access over OBSERVED variables (features + Y), latents dropped.
# --------------------------------------------------------------------------

def _observed_parents_and_order(scm):
    """Return (obs_nodes, topo_obs, parents_obs, y_idx) for the OBSERVED graph.

    * ``obs_nodes``  : set of Z-indices that are observed (features + Y).
    * ``topo_obs``   : list of observed Z-indices in topological order.
    * ``parents_obs``: dict Z-index -> list of OBSERVED-parent Z-indices
                       (latent parents dropped — the DSCM never sees latents).
    * ``y_idx``      : Z-index of Y.

    The graph is read the same way ``shapley_causal.py`` reaches into the SCM:
    via ``_topo_order``, ``feature_to_Z``, ``y_idx_in_Z`` and whichever edge
    representation the family exposes (``_parents`` for the MLP families,
    ``B`` / ``B_lin`` for the linear/quadratic families). Hidden nodes
    (``hidden_indices``) are excluded from ``obs_nodes`` and from every parent
    list.
    """
    topo = np.asarray(scm._topo_order, dtype=int)
    y_idx = int(scm.y_idx_in_Z)
    feat = np.asarray(scm.feature_to_Z, dtype=int)
    hidden = set(int(h) for h in getattr(scm, "hidden_indices", []) or [])
    obs_nodes = set(int(f) for f in feat.tolist()) | {y_idx}

    # Rank in the full topo order (latents sit first for hidden families).
    full_rank = {int(n): r for r, n in enumerate(topo.tolist())}

    # Edge extraction: prefer explicit _parents, else a nonzero structural row.
    def raw_parents(node: int) -> list[int]:
        if getattr(scm, "_parents", None) is not None:
            return [int(p) for p in scm._parents[node]]
        B = getattr(scm, "B", None)
        if B is None:
            B = getattr(scm, "B_lin", None)
        if B is not None:
            B = np.asarray(B)
            return [int(j) for j in np.nonzero(B[node])[0]]
        raise AttributeError(
            f"SCM {type(scm).__name__} exposes neither _parents nor B/B_lin; "
            "cannot extract the true observed graph for the DSCM baseline."
        )

    parents_obs: dict[int, list[int]] = {}
    for node in obs_nodes:
        ps = [int(p) for p in raw_parents(node)
              if int(p) in obs_nodes and int(p) not in hidden]
        # keep parents in topological order for a deterministic feature layout
        parents_obs[node] = sorted(set(ps), key=lambda j: full_rank[j])

    topo_obs = [int(n) for n in topo.tolist() if int(n) in obs_nodes]
    return obs_nodes, topo_obs, parents_obs, y_idx


# --------------------------------------------------------------------------
# Mechanism fitting (one regressor per non-root observed node).
# --------------------------------------------------------------------------

def _make_regressor(kind: str, seed: int):
    """Flexible E[node | parents] regressor. ``gbm`` (default) is fast and
    handles the linear and nonlinear mechanisms uniformly; ``mlp`` is a small
    scikit-learn MLP for a neural-mechanism variant closer to the DSCM name."""
    if kind == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=seed,
        )
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=(64, 64), activation="relu",
            alpha=1e-3, max_iter=400, random_state=seed,
        )
    raise ValueError(f"unknown regressor {kind!r}")


def fit_dscm(scm, X, y, *, regressor: str = "gbm", seed: int = 0):
    """Fit the learned-mechanism SCM on the observed sample FADO sees.

    All columns are already standardized by the SCM's ``sample()``. We map the
    observed-graph Z-indices onto the sample's column layout:

      * feature i (sample column i) == Z-index ``feature_to_Z[i]``.
      * Y == sample column "y".

    Returns a dict with everything ``dscm_tau`` needs: the per-node fitted
    regressors (or a constant marginal for roots), the observed topo order,
    the observed parent sets, and the Z<->column maps.
    """
    feat = np.asarray(scm.feature_to_Z, dtype=int)
    obs_nodes, topo_obs, parents_obs, y_idx = _observed_parents_and_order(scm)

    # Column layout in the (X, y) sample: Z-index -> column accessor.
    z_to_col = {int(z): i for i, z in enumerate(feat.tolist())}  # feature cols

    def col_of(z: int) -> np.ndarray:
        if z == y_idx:
            return y
        return X[:, z_to_col[z]]

    mechanisms: dict[int, dict] = {}
    for node in topo_obs:
        ps = parents_obs[node]
        target = col_of(node)
        if not ps:
            # Root observed node: keep its empirical marginal mean (≈0 after
            # standardization, but computed from the actual sample).
            mechanisms[node] = {"kind": "const", "value": float(np.mean(target)),
                                "parents": []}
            continue
        Xp = np.column_stack([col_of(p) for p in ps])
        reg = _make_regressor(regressor, seed + node)
        reg.fit(Xp, target)
        mechanisms[node] = {"kind": "reg", "reg": reg, "parents": ps}

    return {
        "mechanisms": mechanisms,
        "topo_obs": topo_obs,
        "y_idx": y_idx,
        "feat": feat,
        "z_to_col": z_to_col,
        "p": int(feat.size),
    }


# --------------------------------------------------------------------------
# Forward-simulate the conditional MEAN under a do-intervention.
# --------------------------------------------------------------------------

def _forward_mean(fit: dict, clamp: dict[int, float]) -> dict[int, float]:
    """E[Z_obs | do(clamp)] via a topological mean-propagation pass.

    Each non-clamped node takes the mechanism's prediction at the (scalar)
    propagated means of its observed parents; clamped nodes take their fixed
    value; roots take their stored empirical mean. Returns a Z-index -> mean
    dict over the observed nodes. This is the learned-mechanism analogue of
    ``shapley_causal._interventional_Z_means`` (which does the same exact
    zero-mean topological pass for the TRUE linear B)."""
    vals: dict[int, float] = {}
    for node in fit["topo_obs"]:
        if node in clamp:
            vals[node] = float(clamp[node])
            continue
        mech = fit["mechanisms"][node]
        if mech["kind"] == "const":
            vals[node] = mech["value"]
        else:
            pa = np.asarray([[vals[p] for p in mech["parents"]]], dtype=np.float64)
            vals[node] = float(mech["reg"].predict(pa)[0])
    return vals


def dscm_tau(fit: dict) -> np.ndarray:
    """Length-p DSCM estimate of tau on FADO's exact standardized contrast.

    tau_i = E[y_std | do(X_std_i = +1)] - E[y_std | do(X_std_i = -1)].

    Because the sample is standardized, ``do(X_std_i = ±1)`` is exactly the
    ``mean_i ± std_i`` raw contrast, and reading the standardized Y column is
    exactly the ``/ std_Y`` normalization — see the module docstring. Non-
    intervened observed roots stay at their empirical mean (held via the
    forward pass), matching the do-Shapley empty-coalition baseline.
    """
    feat = fit["feat"]
    y_idx = fit["y_idx"]
    p = fit["p"]
    tau = np.empty(p, dtype=np.float64)
    for i in range(p):
        zi = int(feat[i])
        Ep = _forward_mean(fit, {zi: +1.0})
        Em = _forward_mean(fit, {zi: -1.0})
        tau[i] = Ep[y_idx] - Em[y_idx]
    return tau


def dscm_predict(scm, X, y, *, regressor: str = "gbm", seed: int = 0) -> np.ndarray:
    """One-shot: fit the learned-mechanism SCM and return its length-p tau."""
    fit = fit_dscm(scm, X, y, regressor=regressor, seed=seed)
    return dscm_tau(fit)


# --------------------------------------------------------------------------
# Eval loop: same held-out SCMs / families as eval_shapley_comparison.
# --------------------------------------------------------------------------

_MC_FAMILIES = {"nonlinear", "mlp", "mlp_hidden", "gp_cde"}


def _metrics(pred: np.ndarray, tau: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr
    pe = sp = float("nan")
    if pred.size >= 2 and np.std(pred) > 0 and np.std(tau) > 0:
        pe = float(pearsonr(pred, tau)[0])
        sp = float(spearmanr(pred, tau)[0])
    mae = float(np.mean(np.abs(pred - tau)))
    signal = np.abs(tau) > 0.1
    sign_acc = (float(np.mean(np.sign(pred[signal]) == np.sign(tau[signal])))
                if signal.any() else float("nan"))
    return {"pearson": pe, "spearman": sp, "mae": mae, "sign_acc": sign_acc}


def _bootstrap_ci(per_scm_r: list[float], rng, n_boot: int = 2000):
    """Percentile 95% CI of the mean per-SCM Pearson r (NaNs dropped)."""
    arr = np.asarray([r for r in per_scm_r if np.isfinite(r)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = arr[rng.integers(0, arr.size, size=arr.size)].mean()
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_family(scm_type, n_scms, p, n_rows, rng, *, regressor, noise,
                    nonlinear_mc):
    """Fit + score the DSCM on n_scms held-out SCMs of one family.

    Imports ``_make_eval_scm`` / seed bounds from eval.py so the held-out SCM
    draws line up with eval_shapley_comparison's sweep (same family list, same
    disjoint eval-seed range, same per-SCM sample).
    """
    from causal_probe.eval import (_make_eval_scm, _EVAL_SCM_SEED_LO,
                                   _EVAL_SCM_SEED_HI)
    preds, taus, per_scm_r = [], [], []
    for _ in range(n_scms):
        scm = _make_eval_scm(scm_type, p, rng, noise=noise,
                             nonlinear_mc=nonlinear_mc)
        samp = scm.sample(
            n=n_rows,
            rng=np.random.default_rng(rng.integers(_EVAL_SCM_SEED_LO,
                                                   _EVAL_SCM_SEED_HI)),
        )
        pred = dscm_predict(scm, samp.X, samp.y, regressor=regressor,
                            seed=int(rng.integers(0, 2**31)))
        preds.append(pred)
        taus.append(samp.tau)
        per_scm_r.append(_metrics(pred, samp.tau)["pearson"])
    flat_p = np.concatenate(preds)
    flat_t = np.concatenate(taus)
    pooled = _metrics(flat_p, flat_t)
    mean_r, lo, hi = _bootstrap_ci(per_scm_r, rng)
    return {
        "n_scms": n_scms,
        "pearson": pooled["pearson"],            # pooled over all (SCM, feature)
        "spearman": pooled["spearman"],
        "mae": pooled["mae"],
        "sign_acc": pooled["sign_acc"],
        "pearson_mean_per_scm": mean_r,          # mean of per-SCM r (+ CI)
        "pearson_ci95": [lo, hi],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default="linear,nonlinear,mlp,hidden,mlp_hidden",
                    help="comma-separated SCM families (shapley-comparison set)")
    ap.add_argument("--n-scms", type=int, default=100,
                    help="held-out SCMs per family (matches the shapley sweep)")
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512,
                    help="observational sample size — the SAME sample FADO sees")
    ap.add_argument("--noise", default="laplace")
    ap.add_argument("--nonlinear-mc", type=int, default=4096,
                    help="MC rows for the SCMs' own tau labels")
    ap.add_argument("--regressor", choices=["gbm", "mlp"], default="gbm")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="cpu",
                    help="unused by the DSCM (pure sklearn/CPU); kept so the "
                         "CLI matches the FADO side-by-side scripts")
    ap.add_argument("--out", default="causal_probe/results/dscm_baseline.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="fit on 2 linear instances, print per-feature DSCM tau "
                         "vs true tau for one instance (estimand check), exit 0")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    if args.dry_run:
        from causal_probe.eval import (_make_eval_scm, _EVAL_SCM_SEED_LO,
                                       _EVAL_SCM_SEED_HI)
        print(f"[dry-run] regressor={args.regressor}  p={args.p}  "
              f"n_rows={args.n_rows}")
        shown = 0
        k = 0
        # Fit on 2 NON-degenerate linear instances (skip SCMs where Y is a
        # topological source so true tau is all-zero and r is undefined — a
        # real edge case at p=5, not a bug). Print the per-feature tau for the
        # first so the ±1σ estimand match is visible.
        while shown < 2 and k < 40:
            k += 1
            scm = _make_eval_scm("linear", args.p, rng, noise=args.noise)
            samp = scm.sample(
                n=args.n_rows,
                rng=np.random.default_rng(
                    rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI)))
            if np.std(samp.tau) == 0:
                continue  # Y is a source -> all-zero tau, undefined correlation
            pred = dscm_predict(scm, samp.X, samp.y, regressor=args.regressor,
                                seed=args.seed + shown)
            r = _metrics(pred, samp.tau)["pearson"]
            print(f"\n[dry-run] non-degenerate linear SCM  "
                  f"Pearson r(DSCM, true tau) = {r:+.4f}")
            if shown == 0:
                print(f"  {'feat':>4} {'DSCM_tau':>12} {'true_tau':>12} {'abs_err':>10}")
                for i in range(args.p):
                    print(f"  {i:>4} {pred[i]:>+12.4f} {samp.tau[i]:>+12.4f} "
                          f"{abs(pred[i] - samp.tau[i]):>10.4f}")
            shown += 1
        print("\n[dry-run] OK — exiting 0")
        return

    per_family = {}
    for fam in families:
        print(f"[dscm] family={fam}  n_scms={args.n_scms} "
              f"regressor={args.regressor} ...", flush=True)
        per_family[fam] = evaluate_family(
            fam, args.n_scms, args.p, args.n_rows, rng=rng,
            regressor=args.regressor, noise=args.noise,
            nonlinear_mc=args.nonlinear_mc)

    print(f"\n=== DSCM (true graph, learned mechanisms) — Pearson r with true tau "
          f"(p={args.p}, n_rows={args.n_rows}, regressor={args.regressor}) ===")
    print(f"{'family':<14}{'pearson':>12}{'mean_per_scm':>14}"
          f"{'ci95_lo':>10}{'ci95_hi':>10}{'spearman':>12}{'sign_acc':>10}")
    for fam in families:
        d = per_family[fam]
        lo, hi = d["pearson_ci95"]
        print(f"{fam:<14}{d['pearson']:>+12.3f}{d['pearson_mean_per_scm']:>+14.3f}"
              f"{lo:>+10.3f}{hi:>+10.3f}{d['spearman']:>+12.3f}"
              f"{d['sign_acc']:>+10.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"dscm_by_family": per_family, "config": vars(args)}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
