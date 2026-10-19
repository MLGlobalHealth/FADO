"""Does FADO's single-model bootstrap credible interval achieve NOMINAL coverage?

The uncertainty section needs to decide whether we can claim FADO's uncertainty
is CALIBRATED, or only that it RANKS error / supports abstention. `bootstrap_uncertainty.py`
answered the ranking question (Spearman(sigma, |err|) and selective-prediction
curves). This script answers the harder, quantitative question:

  For a nominal level (1-alpha) in {50, 80, 90, 95}%, does the central credible
  interval formed from the per-feature bootstrap quantiles contain tau_true a
  fraction (1-alpha) of the time?

We take ONE canonical FADO checkpoint, draw N held-out synthetic SCMs (tau_true
known exactly), and for each SCM row-resample B times (REUSING the resampling
loop `_bootstrap_preds` from bootstrap_uncertainty.py). The per-feature central
interval at level (1-alpha) is [q_{alpha/2}, q_{1-alpha/2}] of the B bootstrap
predictions. COVERAGE = fraction of (SCM, feature) pairs whose interval contains
tau_true.

CRITICAL breakdown, reported per (regime, nominal level):
  (a) all features
  (b) true-cause features only        (|tau_true| > 0.1)
  (c) near-null features              (|tau_true| < 0.01)

The scientifically interesting hypothesis (NOT assumed): the bootstrap measures
SAMPLING variability only, so it should MISS non-identification. In non-
identifiable regimes (linear+gaussian, hidden-confounded) the point estimate is
biased toward a prior-weighted average, the bootstrap spread is narrow, and the
interval should systematically UNDER-cover. In the identifiable regime
(linear+laplace) coverage should be closer to nominal. We report what happens.

Secondary calibration view: mean interval width, and Spearman(sigma, |err|) per
regime (to connect to the ranking claim).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.eval import _load_model, _make_eval_scm, _EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI
from causal_probe.bootstrap_uncertainty import _bootstrap_preds


# Nominal central-interval levels (as fractions of 1).
_NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)

# Default regimes: the four reference regimes + the paper-headline mixture.
# scm_type:noise. (mlp / mixture ignore the noise arg internally but we pass
# laplace for a consistent key.)
_DEFAULT_REGIMES = "linear:laplace,linear:gaussian,hidden:laplace,mlp:laplace,mixture:laplace"

# Feature-class thresholds.
_CAUSE_THRESH = 0.1     # |tau_true| > 0.1  -> true cause
_NULL_THRESH = 0.01     # |tau_true| < 0.01 -> near null


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% by default).

    Robust near 0 and 1 where the normal approximation fails — appropriate for
    coverage fractions that may sit at e.g. 0.98 or 0.30.
    """
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return float(centre - half), float(centre + half)


def _coverage_for_subset(lo: np.ndarray, hi: np.ndarray, tau: np.ndarray,
                         width: np.ndarray, mask: np.ndarray) -> dict:
    """Coverage fraction (+ Wilson CI + mean width) over the masked (SCM,feat) pairs.

    ``lo``/``hi``/``tau``/``width`` are flat arrays over all (SCM, feature) pairs
    for ONE nominal level; ``mask`` selects the feature class.
    """
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "coverage": float("nan"),
                "wilson_lo": float("nan"), "wilson_hi": float("nan"),
                "mean_width": float("nan")}
    contained = (tau[mask] >= lo[mask]) & (tau[mask] <= hi[mask])
    k = int(contained.sum())
    wlo, whi = _wilson_ci(k, n)
    return {"n": n, "coverage": k / n, "wilson_lo": wlo, "wilson_hi": whi,
            "mean_width": float(width[mask].mean())}


def evaluate_regime(model, scm_type, noise, p_min, p_max, n_rows, n_scms,
                    n_boot, rng, device):
    """Draw n_scms held-out SCMs of one regime, bootstrap each, accumulate
    per-(SCM,feature) interval endpoints at every nominal level."""
    # Per (SCM, feature): endpoints for each nominal level, plus tau_true, the
    # bootstrap std, and the point-estimate error.
    endpoints = {lvl: {"lo": [], "hi": []} for lvl in _NOMINAL_LEVELS}
    all_tau, all_std, all_err = [], [], []

    for _ in range(n_scms):
        # p drawn uniformly over the training range so coverage reflects the
        # deployment distribution rather than a single fixed width.
        p = int(rng.integers(p_min, p_max + 1))
        scm = _make_eval_scm(scm_type, p, rng, noise=noise)
        samp = scm.sample(
            n=n_rows,
            rng=np.random.default_rng(rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI)),
        )
        boot = _bootstrap_preds(
            model, samp.X, samp.y, n_boot,
            np.random.default_rng(rng.integers(0, 2**31)), device,
        )                                        # (n_boot, p)
        b_mean = boot.mean(axis=0)
        b_std = boot.std(axis=0, ddof=1)
        for lvl in _NOMINAL_LEVELS:
            alpha = 1.0 - lvl
            qlo = np.quantile(boot, alpha / 2.0, axis=0)
            qhi = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
            endpoints[lvl]["lo"].append(qlo)
            endpoints[lvl]["hi"].append(qhi)
        all_tau.append(samp.tau)
        all_std.append(b_std)
        all_err.append(np.abs(b_mean - samp.tau))

    tau = np.concatenate(all_tau)
    std = np.concatenate(all_std)
    err = np.concatenate(all_err)

    # Feature-class masks (shared across nominal levels).
    mask_all = np.ones_like(tau, dtype=bool)
    mask_cause = np.abs(tau) > _CAUSE_THRESH
    mask_null = np.abs(tau) < _NULL_THRESH

    per_level = {}
    for lvl in _NOMINAL_LEVELS:
        lo = np.concatenate(endpoints[lvl]["lo"])
        hi = np.concatenate(endpoints[lvl]["hi"])
        width = hi - lo
        per_level[f"{int(round(lvl * 100))}"] = {
            "all": _coverage_for_subset(lo, hi, tau, width, mask_all),
            "cause": _coverage_for_subset(lo, hi, tau, width, mask_cause),
            "null": _coverage_for_subset(lo, hi, tau, width, mask_null),
        }

    # Secondary ranking view.
    from scipy.stats import spearmanr
    sp = float(spearmanr(std, err)[0]) if std.std() > 0 else float("nan")

    return {
        "scm_type": scm_type, "noise": noise, "n_scms": n_scms, "n_boot": n_boot,
        "n_features_total": int(tau.size),
        "n_cause": int(mask_cause.sum()), "n_null": int(mask_null.sum()),
        "spearman_std_err": sp,
        "mean_boot_std": float(std.mean()),
        "mean_abs_err": float(err.mean()),
        "coverage": per_level,
    }


def _print_table(results: dict) -> None:
    """Nominal-level x regime x {all/cause/null} coverage table to stdout."""
    print(f"\n{'='*92}")
    print("COVERAGE = fraction of (SCM, feature) pairs whose central credible "
          "interval contains tau_true")
    print(f"{'='*92}")
    header = (f"{'regime':<22}{'nom%':>6}"
             f"{'  all (n)':>18}{'  cause (n)':>18}{'  null (n)':>18}")
    for reg, r in results.items():
        print(f"\n{'-'*92}")
        print(f"regime = {reg}   [rho(sigma,|err|)={r['spearman_std_err']:+.2f}, "
              f"mean_sigma={r['mean_boot_std']:.3f}, mean|err|={r['mean_abs_err']:.3f}]")
        print(header)
        for lvl in _NOMINAL_LEVELS:
            key = f"{int(round(lvl * 100))}"
            cov = r["coverage"][key]

            def cell(d):
                if d["n"] == 0:
                    return f"{'--':>10} ({0})"
                return f"{d['coverage']:>7.3f} (n={d['n']})"
            print(f"{'':<22}{key:>6}"
                  f"{cell(cov['all']):>18}{cell(cov['cause']):>18}{cell(cov['null']):>18}")


def _build_regimes(regimes_str: str):
    out = []
    for reg in regimes_str.split(","):
        reg = reg.strip()
        if not reg:
            continue
        if ":" in reg:
            scm_type, noise = reg.split(":")
        else:
            scm_type, noise = reg, "laplace"
        out.append((reg, scm_type, noise))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None,
                    help="Canonical FADO checkpoint. Not required for --dry-run.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--n-scms", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--p-min", type=int, default=3, help="Training-range p lower bound.")
    ap.add_argument("--p-max", type=int, default=15, help="Training-range p upper bound.")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--regimes", default=_DEFAULT_REGIMES,
                    help="comma list of scm_type:noise regimes")
    ap.add_argument("--out", default="causal_probe/results/bootstrap_coverage.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build SCMs per regime, print per-regime counts, exit 0 "
                         "WITHOUT loading a checkpoint or running the model.")
    args = ap.parse_args()

    regimes = _build_regimes(args.regimes)

    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        print(f"[dry-run] regimes={[r[0] for r in regimes]}")
        print(f"[dry-run] n_scms={args.n_scms}, n_boot={args.n_boot}, "
              f"n_rows={args.n_rows}, p in [{args.p_min},{args.p_max}]")
        print(f"[dry-run] nominal levels={[int(l*100) for l in _NOMINAL_LEVELS]}%")
        for reg, scm_type, noise in regimes:
            # Draw a small sample of SCMs to confirm construction + count features.
            n_feat = 0
            n_cause = 0
            n_null = 0
            n_probe = min(args.n_scms, 20)
            for _ in range(n_probe):
                p = int(rng.integers(args.p_min, args.p_max + 1))
                scm = _make_eval_scm(scm_type, p, rng, noise=noise)
                samp = scm.sample(
                    n=args.n_rows,
                    rng=np.random.default_rng(rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI)),
                )
                tau = samp.tau
                n_feat += tau.size
                n_cause += int((np.abs(tau) > _CAUSE_THRESH).sum())
                n_null += int((np.abs(tau) < _NULL_THRESH).sum())
            print(f"  [{reg:<16}] built {n_probe} SCMs OK: "
                  f"{n_feat} features ({n_cause} cause, {n_null} null); "
                  f"tau shape ok")
        print("[dry-run] OK")
        return

    if args.ckpt is None:
        ap.error("--ckpt is required unless --dry-run is given")

    rng = np.random.default_rng(args.seed)
    model = _load_model(args.ckpt, device=args.device)
    print(f"Loaded {args.ckpt}")
    print(f"coverage: B={args.n_boot} row-resamples, {args.n_scms} SCMs/regime, "
          f"p in [{args.p_min},{args.p_max}], n_rows={args.n_rows}")

    results = {}
    for reg, scm_type, noise in regimes:
        print(f"[regime] {reg} ...", flush=True)
        results[reg] = evaluate_regime(
            model, scm_type, noise, args.p_min, args.p_max, args.n_rows,
            args.n_scms, args.n_boot, rng, args.device,
        )

    _print_table(results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "config": vars(args),
                   "nominal_levels": [int(l * 100) for l in _NOMINAL_LEVELS]},
                  f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
