"""Ensemble uncertainty: load K checkpoints trained with different seeds,
evaluate each on the same held-out SCMs, report ensemble mean, per-τ̂_i
std, and correlation between uncertainty and error.

Reviewer objection this addresses: "causal effects are not identifiable
from observational data — how does your model communicate that?"
Answer: ensemble std rises under Gaussian-ambiguous generators and under
hidden-confounded generators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.eval import _load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--scm-type", default="linear",
                    choices=["linear", "nonlinear", "mlp", "hidden", "mixed"])
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", default="laplace", choices=["laplace", "gaussian"])
    ap.add_argument("--n-scms", type=int, default=100)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=9999)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from causal_probe.scm import LinearNonGaussianSCM
    from causal_probe.scm_nonlinear import NonlinearSCM
    from causal_probe.scm_mlp import MLPSCM
    from causal_probe.scm_hidden import LinearNonGaussianSCMHidden
    from causal_probe.scm_mixed import LinearMixedSCM

    rng = np.random.default_rng(args.seed)
    scms = []
    for _ in range(args.n_scms):
        seed = rng.integers(0, 2**31)
        if args.scm_type == "linear":
            scms.append(LinearNonGaussianSCM(p=args.p, rng=np.random.default_rng(seed), noise=args.noise))
        elif args.scm_type == "nonlinear":
            scms.append(NonlinearSCM(p=args.p, rng=np.random.default_rng(seed), n_mc=2048))
        elif args.scm_type == "mlp":
            scms.append(MLPSCM(p=args.p, rng=np.random.default_rng(seed), n_mc=2048))
        elif args.scm_type == "hidden":
            scms.append(LinearNonGaussianSCMHidden(p=args.p, rng=np.random.default_rng(seed), noise=args.noise))
        elif args.scm_type == "mixed":
            scms.append(LinearMixedSCM(p=args.p, rng=np.random.default_rng(seed), noise=args.noise))

    # Load all ckpts
    models = [_load_model(p, device=args.device) for p in args.ckpts]
    print(f"Loaded {len(models)} checkpoints")

    all_preds = np.zeros((len(models), args.n_scms, args.p), dtype=np.float64)
    truths = np.zeros((args.n_scms, args.p), dtype=np.float64)
    for s, scm in enumerate(scms):
        samp = scm.sample(n=args.n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
        truths[s] = samp.tau
        X_t = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(args.device)
        y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(args.device)
        for k, m in enumerate(models):
            with torch.no_grad():
                all_preds[k, s] = m(X_t, y_t).squeeze(0).cpu().numpy()

    ensemble_mean = all_preds.mean(axis=0)        # (n_scms, p)
    ensemble_std = all_preds.std(axis=0, ddof=1)  # (n_scms, p)
    abs_error = np.abs(ensemble_mean - truths)

    # Flatten for per-feature metrics
    flat_err = abs_error.reshape(-1)
    flat_std = ensemble_std.reshape(-1)
    from scipy.stats import spearmanr, pearsonr
    sp = float(spearmanr(flat_std, flat_err)[0])
    pe = float(pearsonr(flat_std, flat_err)[0])

    # Coverage of μ ± 2σ
    coverage = float(np.mean(abs_error <= 2 * ensemble_std))

    # Correlation by-bin (monotonicity check)
    n_bins = 5
    q = np.quantile(flat_std, np.linspace(0, 1, n_bins + 1))
    bin_errors = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (flat_std >= q[i]) & (flat_std <= q[i+1])
        else:
            mask = (flat_std >= q[i]) & (flat_std < q[i+1])
        bin_errors.append(float(np.mean(flat_err[mask])) if mask.any() else float("nan"))

    print(f"\n=== Ensemble uncertainty ({args.scm_type}, p={args.p}, noise={args.noise}, K={len(models)}) ===")
    print(f"  Pearson(std, |err|):   {pe:+.3f}")
    print(f"  Spearman(std, |err|):  {sp:+.3f}")
    print(f"  Coverage of μ ± 2σ:    {coverage:.3f}")
    print(f"  Error by std-quintile: {[round(b, 3) for b in bin_errors]}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "ckpts": args.ckpts, "scm_type": args.scm_type, "p": args.p,
                "noise": args.noise, "K": len(models), "n_scms": args.n_scms,
                "pearson_std_err": pe, "spearman_std_err": sp,
                "coverage_2sigma": coverage, "error_by_std_quintile": bin_errors,
            }, f, indent=2)


if __name__ == "__main__":
    main()
