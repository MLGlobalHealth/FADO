"""CausalPFN-per-column baseline for the causal feature probe.

Natural objection: "Why not just run an
amortized causal-effect PFN once per feature, each time moving that
feature into the treatment column?"

This script answers that empirically. Do-PFN itself does not have
public code; CausalPFN (Balazadeh et al., NeurIPS 2025) is a
concurrent in-context causal-effect estimator with the same designated-
treatment API and is on PyPI as `causalpfn`. The concern is
identical between the two: an amortized causal-effect PFN that requires
a designated treatment column.

For each feature i in [p]:
  1. Binarize X_i to t_i = sign(X_i - median(X_i)) ∈ {-1, +1}, mapped
     to {0, 1} for CausalPFN's binary-treatment API.
  2. Treat all other features as covariates X_rest.
  3. ATEEstimator().fit(X_rest, t_i, y); take ATE as τ̂_i^CausalPFN.
  4. Standardize: τ̂_i / std(y).

Compare to:
  - True τ from the SCM (held-out generator label).
  - Our probe's τ̂ (loaded from the same SCMs via causal_probe.eval).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


def causalpfn_tau(
    X: np.ndarray, y: np.ndarray, device: str = "cpu",
    cache_dir: str | None = None,
) -> np.ndarray:
    """Run CausalPFN ATEEstimator once per feature; return τ̂ vector of length p.

    Binarization: t_i = 1 if X_i > median(X_i) else 0. Standardization to
    match the probe's reporting scale: τ̂_i = ATE_i * 2 / std(y), since
    ATE under {-1, +1} treatment encoding equals 2× the {0, 1} ATE
    when the underlying conditional mean is linear in t.

    cache_dir overrides CausalPFN's default ~/.cache/causalpfn (the
    package hard-codes Path.home() rather than honouring XDG_CACHE_HOME).
    """
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
        t = (x_i > med).astype(np.int64)  # binary {0, 1}
        # Skip degenerate splits.
        if t.sum() < 5 or (1 - t).sum() < 5:
            continue
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
        # ATE is the {0,1} contrast (high vs low half). Map to unit-step
        # contrast and standardize like the probe does.
        # Note: we compare directly against the probe's standardized τ;
        # both quantities are dimensionless, so just normalize by std_y.
        tau[i] = ate / std_y

    return tau


def eval_causalpfn_on_scms(
    scms: list, n_rows: int, rng: np.random.Generator, device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Run the per-column CausalPFN baseline on a list of SCMs."""
    from scipy.stats import spearmanr, pearsonr

    preds, truths = [], []
    n_failed = 0
    for k, scm in enumerate(scms):
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
        try:
            tau_hat = causalpfn_tau(
                samp.X.astype(np.float64), samp.y.astype(np.float64), device=device,
            )
        except Exception as e:
            print(f"  SCM {k}: causalpfn_tau crashed: {e}")
            n_failed += 1
            continue
        preds.append(tau_hat)
        truths.append(samp.tau)
        if verbose and (k + 1) % 10 == 0:
            print(f"  SCM {k + 1}/{len(scms)} done")

    if not preds:
        return {"error": "all CausalPFN evaluations failed"}

    preds = np.stack(preds)
    truths = np.stack(truths)
    flat_p = preds.reshape(-1)
    flat_t = truths.reshape(-1)
    mask = np.isfinite(flat_p) & np.isfinite(flat_t)
    if not mask.any():
        return {"error": "all CausalPFN predictions are NaN"}
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
        "n_scms": len(preds),
        "n_failed": n_failed,
        "n_features_total": int(mask.size),
        "n_features_finite": int(mask.sum()),
        "pearson": float(pe),
        "spearman": float(sp),
        "r2": r2,
        "mse": mse,
        "auroc_nonzero": auc,
        "mae_zero": mae_zero,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scm-type", default="linear",
        choices=["linear", "nonlinear", "mlp", "hidden", "mixed"],
    )
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", default="laplace", choices=["laplace", "gaussian"])
    ap.add_argument("--n-scms", type=int, default=30)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--verbose", action="store_true")
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
        f"Evaluating CausalPFN-per-column on {len(scms)} SCMs "
        f"({args.scm_type}, p={args.p}, noise={args.noise}, n_rows={args.n_rows})"
    )
    res = eval_causalpfn_on_scms(
        scms, n_rows=args.n_rows, rng=rng, device=args.device, verbose=args.verbose,
    )
    print(json.dumps(res, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        res["config"] = {
            "scm_type": args.scm_type, "p": args.p, "noise": args.noise,
            "n_rows": args.n_rows, "seed": args.seed,
        }
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
