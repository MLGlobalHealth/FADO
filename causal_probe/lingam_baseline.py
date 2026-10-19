"""LiNGAM baseline for the causal feature probe.

DirectLiNGAM fits a linear non-Gaussian acyclic model on the concatenated
[X_1, ..., X_p, Y] data, recovers a structural matrix B, and returns the
total-effect matrix A = (I - B)^{-1}. We use the row for the target and
convert to our standardized contrast tau_i = 2 * A[Y, i] * std(X_i) / std(Y).

linear non-Gaussian is LiNGAM's home turf. Direct
comparison on our synthetic benchmark is the natural test.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


def lingam_tau(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Run DirectLiNGAM on [X, y] and return τ̂_i for i = 0..p-1."""
    from lingam import DirectLiNGAM
    p = X.shape[1]
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    model = DirectLiNGAM()
    model.fit(data)
    # model.adjacency_matrix_ is B (destination row, source col) in column convention
    # so for a row ordering equivalent to (I - B) Z = noise, we do:
    B = model.adjacency_matrix_
    N = p + 1
    I = np.eye(N)
    try:
        A = np.linalg.solve(I - B, I)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)
    y_idx = p  # we appended y last
    # τ_i = 2 * A[y, i] * std(X_i) / std(Y)
    std_X = X.std(axis=0).clip(min=1e-9)
    std_Y = float(y.std().clip(min=1e-9))
    tau = 2.0 * A[y_idx, :p] * std_X / std_Y
    return tau


def eval_lingam_on_scms(
    scms: list, n_rows: int, rng: np.random.Generator,
) -> dict:
    """Run LiNGAM on a list of SCMs, return metrics vs true tau."""
    from scipy.stats import spearmanr, pearsonr
    preds, truths = [], []
    for scm in scms:
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
        try:
            tau_hat = lingam_tau(samp.X.astype(np.float64), samp.y.astype(np.float64))
        except Exception as e:
            print(f"  LiNGAM fit failed: {e}")
            continue
        preds.append(tau_hat)
        truths.append(samp.tau)
    preds = np.stack(preds)
    truths = np.stack(truths)
    flat_p = preds.reshape(-1); flat_t = truths.reshape(-1)
    # Drop NaNs
    mask = np.isfinite(flat_p) & np.isfinite(flat_t)
    if not mask.any():
        return {"error": "all LiNGAM fits failed"}
    pe, _ = pearsonr(flat_p[mask], flat_t[mask])
    sp, _ = spearmanr(flat_p[mask], flat_t[mask])
    mse = float(np.mean((flat_p[mask] - flat_t[mask]) ** 2))
    ss_tot = float(np.sum((flat_t[mask] - flat_t[mask].mean()) ** 2))
    r2 = 1.0 - float(np.sum((flat_p[mask] - flat_t[mask]) ** 2)) / max(ss_tot, 1e-12)
    from sklearn.metrics import roc_auc_score
    labels_nz = (np.abs(flat_t[mask]) > 0.1).astype(int)
    if labels_nz.sum() > 0 and labels_nz.sum() < len(labels_nz):
        auc = float(roc_auc_score(labels_nz, np.abs(flat_p[mask])))
    else:
        auc = float("nan")
    zero_mask = np.abs(flat_t[mask]) <= 0.1
    mae_zero = float(np.mean(np.abs(flat_p[mask][zero_mask] - flat_t[mask][zero_mask]))) if zero_mask.any() else float("nan")
    return {
        "n_scms": len(scms), "pearson": float(pe), "spearman": float(sp),
        "r2": r2, "mse": mse, "auroc_nonzero": auc, "mae_zero": mae_zero,
        "n_fits_ok": int(np.sum([np.isfinite(preds[i]).all() for i in range(preds.shape[0])])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scm-type", default="linear",
                    choices=["linear", "nonlinear", "mlp", "hidden", "mixed"])
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

    print(f"Evaluating DirectLiNGAM on {len(scms)} SCMs ({args.scm_type}, p={args.p}, noise={args.noise})")
    res = eval_lingam_on_scms(scms, n_rows=args.n_rows, rng=rng)
    print(json.dumps(res, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        res["config"] = {"scm_type": args.scm_type, "p": args.p, "noise": args.noise,
                         "n_rows": args.n_rows, "seed": args.seed}
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
