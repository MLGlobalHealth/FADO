"""Head-to-head: single-model BOOTSTRAP sigma vs K-checkpoint ENSEMBLE sigma.

The paper's uncertainty section will report single-model row-bootstrap sigma
(from ONE checkpoint) as its per-feature uncertainty estimator, replacing the
appendix's K-seed ensemble sigma (per-seed std of tau_hat). We need a
head-to-head so the paper can say: "single-model bootstrap sigma tracks error as
well as (or better than) the ensemble, at no K-times cost."

For each held-out synthetic regime and each estimator we report, per feature:
  * Spearman(sigma, |tau_hat - tau_true|)  -- does the flag track error?
  * selective-prediction gain: mean |err| on the low-sigma half vs the high-sigma
    half (bigger gap = more useful abstention signal).

The two estimators share the SAME held-out SCMs and are scored against the same
ground-truth tau, so their Spearman / selective-prediction numbers sit side by
side:
  * ENSEMBLE: run each of K checkpoints on the (X, y); tau_hat = mean over K,
    sigma = per-feature std over K.
  * BOOTSTRAP: run the single --ckpt model on B row-resamples of (X, y);
    tau_hat = mean over B, sigma = per-feature std over B.

Reuses ``_bootstrap_preds`` from ``causal_probe.bootstrap_uncertainty`` and
``_load_model`` / ``_make_eval_scm`` from ``causal_probe.eval``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.eval import (
    _load_model,
    _make_eval_scm,
    _EVAL_SCM_SEED_LO,
    _EVAL_SCM_SEED_HI,
)
from causal_probe.bootstrap_uncertainty import _bootstrap_preds


# Default held-out regimes for the comparison. scm_type:noise.
DEFAULT_REGIMES = "linear:laplace,linear:gaussian,hidden:laplace,mlp:laplace"


def _parse_regimes(regimes: str):
    out = []
    for reg in regimes.split(","):
        reg = reg.strip()
        if not reg:
            continue
        if ":" in reg:
            scm_type, noise = reg.split(":")
        else:
            scm_type, noise = reg, "laplace"
        out.append((reg, scm_type, noise))
    return out


def _sample_p(rng, p_lo, p_hi):
    """Draw a feature count uniformly in the (inclusive) training range."""
    return int(rng.integers(p_lo, p_hi + 1))


def build_regime_scms(scm_type, noise, n_scms, n_rows, p_lo, p_hi, rng):
    """Return a list of (X, y, tau) tuples for one regime.

    p is drawn per-SCM uniformly in [p_lo, p_hi] to stay inside the training
    range. Data-generating seeds are drawn from the held-out eval band.
    """
    out = []
    for _ in range(n_scms):
        p = _sample_p(rng, p_lo, p_hi)
        scm = _make_eval_scm(scm_type, p, rng, noise=noise)
        samp = scm.sample(
            n=n_rows,
            rng=np.random.default_rng(rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI)),
        )
        out.append((samp.X, samp.y, samp.tau))
    return out


def _model_predict(model, X, y, device):
    """Single forward pass -> (p,) tau_hat for one (X, y)."""
    X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(X_t, y_t).squeeze(0).cpu().numpy()


def _estimator_metrics(sigma, err):
    """Spearman(sigma, err) + selective-prediction gain (low vs high half)."""
    from scipy.stats import spearmanr

    sigma = np.asarray(sigma, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    sp = float(spearmanr(sigma, err)[0]) if sigma.std() > 0 else float("nan")

    # Selective prediction: split features at the median sigma.
    med = float(np.median(sigma))
    low = err[sigma <= med]
    high = err[sigma > med]
    low_err = float(low.mean()) if low.size else float("nan")
    high_err = float(high.mean()) if high.size else float("nan")
    # Gain = how much lower the error is on the model-confident (low-sigma) half.
    gain = float(high_err - low_err) if (low.size and high.size) else float("nan")
    return {
        "spearman_sigma_err": sp,
        "mean_sigma": float(sigma.mean()),
        "low_sigma_half_err": low_err,
        "high_sigma_half_err": high_err,
        "selective_gain": gain,
    }


def evaluate_regime(reg_scms, boot_model, ens_models, n_boot, rng, device):
    """Compute bootstrap-sigma and ensemble-sigma metrics on shared SCMs."""
    boot_sigma, boot_err = [], []
    ens_sigma, ens_err = [], []

    for X, y, tau in reg_scms:
        # --- Bootstrap (single model): B row-resamples ---
        boot = _bootstrap_preds(
            boot_model, X, y, n_boot,
            np.random.default_rng(rng.integers(0, 2**31)), device,
        )
        b_mean = boot.mean(axis=0)
        b_std = boot.std(axis=0, ddof=1)
        boot_sigma.append(b_std)
        boot_err.append(np.abs(b_mean - tau))

        # --- Ensemble (K checkpoints): per-feature std across seeds ---
        if ens_models:
            preds = np.stack([_model_predict(m, X, y, device) for m in ens_models])
            e_mean = preds.mean(axis=0)
            e_std = preds.std(axis=0, ddof=1)
            ens_sigma.append(e_std)
            ens_err.append(np.abs(e_mean - tau))

    result = {
        "n_scms": len(reg_scms),
        "n_boot": n_boot,
        "bootstrap": _estimator_metrics(
            np.concatenate(boot_sigma), np.concatenate(boot_err)
        ),
    }
    if ens_models:
        result["K"] = len(ens_models)
        result["ensemble"] = _estimator_metrics(
            np.concatenate(ens_sigma), np.concatenate(ens_err)
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="single model for BOOTSTRAP sigma")
    ap.add_argument("--ckpt-dir",
                    help="dir holding the K ENSEMBLE checkpoints")
    ap.add_argument(
        "--ensemble-ckpts",
        default="probe_foundation_intersample_L_p3_15_intersample_L_seed0.ckpt,"
                "probe_foundation_intersample_L_p3_15_intersample_L_seed1.ckpt,"
                "probe_foundation_intersample_L_p3_15_intersample_L_seed2.ckpt,"
                "probe_foundation_intersample_L_p3_15_intersample_L_seed3.ckpt,"
                "probe_foundation_intersample_L_p3_15_intersample_L_seed4.ckpt",
        help="comma-list of ensemble ckpt filenames (joined to --ckpt-dir)",
    )
    ap.add_argument("--regimes", default=DEFAULT_REGIMES,
                    help="comma list of scm_type[:noise] regimes")
    ap.add_argument("--p-lo", type=int, default=3, help="min p (training range)")
    ap.add_argument("--p-hi", type=int, default=15, help="max p (training range)")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-scms", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="causal_probe/results/uncertainty_estimator_compare.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="build SCMs, print counts, exit 0 (no ckpt / no GPU)")
    args = ap.parse_args()

    regimes = _parse_regimes(args.regimes)
    rng = np.random.default_rng(args.seed)

    # Build the held-out SCMs up front so bootstrap & ensemble see identical data.
    regime_scms = {}
    for reg, scm_type, noise in regimes:
        scms = build_regime_scms(
            scm_type, noise, args.n_scms, args.n_rows,
            args.p_lo, args.p_hi, rng,
        )
        regime_scms[reg] = scms

    if args.dry_run:
        print("=== DRY RUN (no checkpoints loaded, no GPU) ===")
        print(f"seed={args.seed}  n_scms/regime={args.n_scms}  n_rows={args.n_rows}"
              f"  p in [{args.p_lo}, {args.p_hi}]  n_boot={args.n_boot}")
        total_feats = 0
        for reg, scms in regime_scms.items():
            ps = [X.shape[1] for X, _, _ in scms]
            feats = sum(ps)
            total_feats += feats
            print(f"  [{reg:<18}] {len(scms):>4} SCMs  "
                  f"p range built [{min(ps)},{max(ps)}]  mean p={np.mean(ps):.1f}  "
                  f"{feats} feature-rows")
        ens_names = [c.strip() for c in args.ensemble_ckpts.split(",") if c.strip()]
        print(f"total feature-rows across regimes: {total_feats}")
        print(f"ensemble would load K={len(ens_names)} ckpts from "
              f"--ckpt-dir (not loaded in dry-run):")
        for c in ens_names:
            print(f"    {c}")
        print(f"bootstrap ckpt (--ckpt, not loaded): {args.ckpt}")
        print("DRY RUN OK")
        return

    if not args.ckpt:
        ap.error("--ckpt is required unless --dry-run")

    boot_model = _load_model(args.ckpt, device=args.device)
    print(f"Loaded bootstrap model: {args.ckpt}")

    ens_models = []
    if args.ckpt_dir:
        ens_names = [c.strip() for c in args.ensemble_ckpts.split(",") if c.strip()]
        for c in ens_names:
            path = str(Path(args.ckpt_dir) / c)
            ens_models.append(_load_model(path, device=args.device))
        print(f"Loaded ensemble: K={len(ens_models)} from {args.ckpt_dir}")
    else:
        print("No --ckpt-dir: skipping ensemble (bootstrap-only run)")

    results = {}
    for reg, _, _ in regimes:
        print(f"[regime] {reg} ...", flush=True)
        results[reg] = evaluate_regime(
            regime_scms[reg], boot_model, ens_models, args.n_boot, rng, args.device
        )

    # Side-by-side summary table.
    print(f"\n{'regime':<18}"
          f"{'boot rho':>10}{'ens rho':>10}"
          f"{'boot gain':>11}{'ens gain':>11}")
    for reg, r in results.items():
        b = r["bootstrap"]
        e = r.get("ensemble", {})
        print(f"{reg:<18}"
              f"{b['spearman_sigma_err']:>10.3f}"
              f"{e.get('spearman_sigma_err', float('nan')):>10.3f}"
              f"{b['selective_gain']:>11.4f}"
              f"{e.get('selective_gain', float('nan')):>11.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
