"""Per-output reliability signal from row-bootstrap of a SINGLE model.

Natural objection: "the method offers no 'don't trust this' signal; under
non-identifiable regimes it silently degrades to a prior-weighted average."

This addresses it without an ensemble of checkpoints: take ONE canonical FADO
model, and for each held-out dataset resample its rows with replacement B times,
running the model on each resample. The per-feature spread (std over the B
bootstrap predictions) is a per-output reliability flag — it needs no retraining
and produces as many "members" as you want from one checkpoint, so it is immune
to the "n seeds is too small" critique.

We report, per SCM regime:
  * Pearson/Spearman(bootstrap std, |bootstrap mean - true tau|) — is the signal
    calibrated (does higher spread mean higher error)?
  * mean bootstrap std — is spread ELEVATED in non-identifiable regimes
    (linear+gaussian, hidden-confounded) vs the identifiable one (linear+laplace)?
  * selective-prediction curve: error within each std-quintile (abstaining on the
    top quintile should drop error).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.eval import _load_model, _make_eval_scm, _EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI


def _bootstrap_preds(model, X, y, n_boot, rng, device, chunk=25):
    """Return (n_boot, p) predictions from row-resamples of a single (X, y).

    Rows are resampled with replacement to the original count; the B resampled
    datasets are pushed through the model in chunks to bound memory.
    """
    n, p = X.shape
    Xf = X.astype(np.float32)
    yf = y.astype(np.float32)
    out = np.empty((n_boot, p), dtype=np.float64)
    b = 0
    while b < n_boot:
        k = min(chunk, n_boot - b)
        idx = rng.integers(0, n, size=(k, n))          # (k, n) row indices w/ replacement
        Xb = np.stack([Xf[idx[j]] for j in range(k)])   # (k, n, p)
        yb = np.stack([yf[idx[j]] for j in range(k)])   # (k, n)
        with torch.no_grad():
            pred = model(torch.from_numpy(Xb).to(device),
                         torch.from_numpy(yb).to(device)).cpu().numpy()
        out[b:b + k] = pred
        b += k
    return out


def evaluate_regime(model, scm_type, noise, p, n_rows, n_scms, n_boot, rng, device):
    stds, errs, means, truths = [], [], [], []
    for _ in range(n_scms):
        scm = _make_eval_scm(scm_type, p, rng, noise=noise)
        samp = scm.sample(n=n_rows,
                          rng=np.random.default_rng(rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI)))
        boot = _bootstrap_preds(model, samp.X, samp.y, n_boot,
                                np.random.default_rng(rng.integers(0, 2**31)), device)
        b_mean = boot.mean(axis=0)
        b_std = boot.std(axis=0, ddof=1)
        stds.append(b_std)
        means.append(b_mean)
        truths.append(samp.tau)
        errs.append(np.abs(b_mean - samp.tau))
    std = np.concatenate(stds)
    err = np.concatenate(errs)
    from scipy.stats import spearmanr, pearsonr
    pe = float(pearsonr(std, err)[0]) if std.std() > 0 else float("nan")
    sp = float(spearmanr(std, err)[0]) if std.std() > 0 else float("nan")
    # selective prediction: mean |error| within each std-quintile
    qs = np.quantile(std, np.linspace(0, 1, 6))
    bin_err = []
    for i in range(5):
        m = (std >= qs[i]) & (std <= qs[i + 1]) if i == 4 else (std >= qs[i]) & (std < qs[i + 1])
        bin_err.append(float(err[m].mean()) if m.any() else float("nan"))
    cov2 = float(np.mean(err <= 2 * std))
    return {
        "scm_type": scm_type, "noise": noise, "n_scms": n_scms, "n_boot": n_boot,
        "mean_boot_std": float(std.mean()), "median_boot_std": float(np.median(std)),
        "mean_abs_err": float(err.mean()),
        "pearson_std_err": pe, "spearman_std_err": sp,
        "coverage_2sigma": cov2, "err_by_std_quintile": bin_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--regimes", default="linear:laplace,linear:gaussian,hidden:laplace",
                    help="comma list of scm_type:noise regimes")
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-scms", type=int, default=150)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="causal_probe/results/bootstrap_uncertainty.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = _load_model(args.ckpt, device=args.device)
    print(f"Loaded {args.ckpt}\nrow-bootstrap B={args.n_boot}, {args.n_scms} SCMs/regime\n")

    results = {}
    for reg in args.regimes.split(","):
        scm_type, noise = reg.split(":")
        print(f"[regime] {scm_type} / {noise} ...", flush=True)
        results[reg] = evaluate_regime(model, scm_type, noise, args.p, args.n_rows,
                                       args.n_scms, args.n_boot, rng, args.device)

    ident = {"linear:laplace"}  # the LiNGAM-identifiable reference
    print(f"\n{'regime':<20}{'mean_std':>10}{'mean_|err|':>11}{'r(std,err)':>12}"
          f"{'rho':>7}{'cov2σ':>8}   err-by-std-quintile (low→high)")
    for reg, r in results.items():
        tag = "  <- identifiable" if reg in ident else ""
        q = " ".join(f"{v:.3f}" for v in r["err_by_std_quintile"])
        print(f"{reg:<20}{r['mean_boot_std']:>10.4f}{r['mean_abs_err']:>11.4f}"
              f"{r['pearson_std_err']:>12.3f}{r['spearman_std_err']:>7.2f}"
              f"{r['coverage_2sigma']:>8.2f}   [{q}]{tag}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
