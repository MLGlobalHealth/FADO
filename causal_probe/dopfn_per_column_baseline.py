"""Do-PFN per-column baseline for the off-label per-feature appendix.

Mirror of `causal_probe/causalpfn_per_column_baseline.py` with the inference
call swapped from CausalPFN's ATEEstimator to Do-PFN's DoPFNRegressor.

For each feature i in [p]:
  1. Binarize X_i to t_i = (X_i > median(X_i)) in {0, 1} (Do-PFN's released
     checkpoint expects a binary treatment, same as CausalPFN's).
  2. Construct D^ob with column 0 = t_i, columns 1..p-1 = X without column i.
     Do-PFN's checkpoint is trained with treatment at col 0 (DoSCM convention).
  3. m.fit(D^ob, y); cate_per_row = m.predict_cate(D^ob); ATE_i = mean(cate).
     This is the canonical Do-PFN-CATE reduction.
  4. Standardize: tau[i] = ATE_i / std(y).

Notes:
  * predict_cate(X) calls predict_cid(X, 1) - predict_cid(X, 0). predict_cid
    mutates X[:, 0] in place, so we pass a fresh clone per call.
  * predict_common_setup expects a torch.Tensor (it calls .cpu().detach()).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _load_dopfn(dopfn_repo: str, device: str = "cuda"):
    """Import DoPFNRegressor from a checked-out Do-PFN repo and set device.

    DoPFNRegressor opens artifact files (`artifacts/dopfn_config.pkl` at init,
    `artifacts/model_submitit_*.cpkt` lazily on first fit) via relative paths.
    The caller must keep the process cwd at `dopfn_repo` for the model's
    lifetime; main() handles that.
    """
    sys.path.insert(0, dopfn_repo)
    from scripts.transformer_prediction_interface.base import DoPFNRegressor

    m = DoPFNRegressor()
    m.device = device
    return m


def dopfn_tau(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Run Do-PFN per-column on (X, y); return tau-hat vector of length p."""
    import torch

    p = X.shape[1]
    std_y = float(np.std(y).clip(min=1e-9))
    tau = np.full(p, np.nan, dtype=np.float64)
    for i in range(p):
        x_i = X[:, i]
        med = float(np.median(x_i))
        t = (x_i > med).astype(np.float32)
        if t.sum() < 5 or (1 - t).sum() < 5:
            continue
        X_rest = np.delete(X, i, axis=1).astype(np.float32)
        D = np.concatenate([t[:, None], X_rest], axis=1).astype(np.float32)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                model.fit(D, y.astype(np.float32))
                cate = model.predict_cate(torch.tensor(D).clone())
            ate = float(np.asarray(cate).mean())
        except Exception as e:
            print(f"    feature {i}: Do-PFN call failed: {e}")
            continue
        tau[i] = ate / std_y
    return tau


def eval_dopfn_on_scms(
    model, scms: list, n_rows: int, rng: np.random.Generator,
    verbose: bool = False,
) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    preds, truths = [], []
    n_failed = 0
    for k, scm in enumerate(scms):
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
        try:
            tau_hat = dopfn_tau(
                model, samp.X.astype(np.float64), samp.y.astype(np.float64),
            )
        except Exception as e:
            print(f"  SCM {k}: dopfn_tau crashed: {e}")
            n_failed += 1
            continue
        preds.append(tau_hat)
        truths.append(samp.tau)
        if verbose and (k + 1) % 10 == 0:
            print(f"  SCM {k + 1}/{len(scms)} done")

    if not preds:
        return {"error": "all Do-PFN evaluations failed"}

    preds = np.stack(preds)
    truths = np.stack(truths)
    flat_p = preds.reshape(-1)
    flat_t = truths.reshape(-1)
    mask = np.isfinite(flat_p) & np.isfinite(flat_t)
    if not mask.any():
        return {"error": "all Do-PFN predictions are NaN"}
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
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scm-type", default="linear",
        choices=["linear", "nonlinear", "mlp", "hidden", "mixed"],
    )
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", default="laplace", choices=["laplace", "gaussian"])
    ap.add_argument("--n-scms", type=int, default=100)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dopfn-repo", required=True,
                    help="Path to a checked-out Do-PFN repository (containing artifacts/).")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Resolve paths relative to the *current* cwd before chdir.
    out_json = Path(args.out_json).resolve() if args.out_json else None
    dopfn_repo = str(Path(args.dopfn_repo).resolve())

    # Build SCMs first, while causal_probe is on the import path.
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

    # Now chdir into Do-PFN (its DoPFNRegressor opens artifact files via
    # relative paths).
    os.chdir(dopfn_repo)

    print(
        f"Evaluating Do-PFN-per-column on {len(scms)} SCMs "
        f"({args.scm_type}, p={args.p}, noise={args.noise}, n_rows={args.n_rows}, "
        f"device={args.device})"
    )
    model = _load_dopfn(dopfn_repo, device=args.device)
    res = eval_dopfn_on_scms(model, scms, n_rows=args.n_rows, rng=rng, verbose=args.verbose)
    print(json.dumps(res, indent=2))

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        res["config"] = {
            "scm_type": args.scm_type, "p": args.p, "noise": args.noise,
            "n_rows": args.n_rows, "seed": args.seed,
        }
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
