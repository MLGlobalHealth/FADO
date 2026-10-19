"""Error-vs-abstention (selective prediction) curve on REAL data with known tau_true.

Flagship figure for the paper's uncertainty section. Earlier experiments established that
FADO's single-checkpoint row-bootstrap sigma (causal_probe/bootstrap_uncertainty.py:
resample the context rows B times, sigma_j = sd over bootstrap replicates of
tau_hat_j) is a per-feature reliability signal on held-out SYNTHETIC SCMs, and
causal_probe/decoy_sigma_analysis.py showed sigma flags a known real-data failure
mode via a decoy-mass metric. THIS module makes the clean, quantitative
selective-prediction statement the figure needs: on real benches where tau_true
is known per feature, abstaining on the highest-sigma predictions monotonically
reduces the error on what remains.

Protocol. For each construction (a context, its tau_true, and FADO tau_hat +
bootstrap sigma per feature) we POOL all feature observations within a bench, then
sweep abstention fraction a in {0, 0.1, ..., 0.5}: drop the top-a fraction of
features by sigma and report mean |tau_hat - tau_true| on the retained features.
If sigma is informative the curve DECREASES with a. We report the same curve for a
RANDOM abstention baseline (drop a random a-fraction, averaged over --n-random-reps
draws) so the figure shows sigma-abstention beats random. Scalar summaries per bench:
error at a=0, error at a=0.3, their ratio, and the trapezoidal area under the
error(a) curve (lower = sigma keeps error down across the sweep).

Benches with known per-feature tau_true:
  * chambers: Causal Chambers light-tunnel lt_vis3 + lt_vis1. tau_true built by
    causal_chambers_eval.build_ground_truth (hardware-verified; CPU-only). We reuse
    decoy_sigma_analysis.build_chamber_constructions verbatim.
  * semisynth: the ACIC/IHDP semi-synthetic decoy instances (real covariates,
    synthetic outcome with tau_true known by construction), reused verbatim from
    decoy_sigma_analysis.build_semisynth_constructions.

We deliberately do NOT duplicate the construction / ground-truth code: both bench
builders are imported from decoy_sigma_analysis, and the bootstrap resample loop
from bootstrap_uncertainty._bootstrap_preds.

Usage:
  # CPU dry-run (semisynth only; no ckpt / GPU): build benches, confirm tau_true.
  uv run python -m causal_probe.selective_prediction_curve --dry-run \
      --benches semisynth --covariates ihdp --ihdp-dir data/ihdp

  # Full run (GPU), both benches, canonical intersample-L checkpoint:
  python -m causal_probe.selective_prediction_curve \
      --ckpt <ckpt> --device cuda --benches all \
      --data-dir <chamber data-dir> --covariates acic --acic-path data/acic2016/x.csv \
      --out causal_probe/results/selective_prediction_curve.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from causal_probe.bootstrap_uncertainty import _bootstrap_preds

ABSTAIN_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SUMMARY_A = 0.3  # scalar summary point (error at a=0 vs a=0.3)


# ---------------------------------------------------------------------------
# Bench construction (reuses decoy_sigma_analysis builders verbatim)
# ---------------------------------------------------------------------------


def _builder_args(args) -> SimpleNamespace:
    """Namespace with the attributes the decoy_sigma_analysis builders read.

    build_semisynth_constructions / build_chamber_constructions consume a subset
    of decoy_sigma_analysis's CLI args; we expose exactly those so the shared
    builders run unchanged (same RNG structure -> same constructions).
    """
    return SimpleNamespace(
        seed=args.seed,
        n_scms=args.n_scms,
        n_rows=args.n_rows,
        # semisynth (acic/ihdp)
        families=args.families,
        p_real_cap=args.p_real_cap,
        n_causal=args.n_causal,
        # chambers
        data_dir=args.data_dir,
        chamber_draws=args.chamber_draws,
    )


def build_bench(name: str, args) -> list[dict]:
    """Return the construction list for a bench.

    name == "semisynth" -> build_semisynth_constructions on ACIC/IHDP covariates.
    name == "chambers"  -> build_chamber_constructions on the light-tunnel configs.
    Each construction dict carries X_ctx, y_ctx, tau_true, classes (+ perm for
    chambers) -- identical to decoy_sigma_analysis.
    """
    from causal_probe.decoy_sigma_analysis import (
        build_chamber_constructions, build_semisynth_constructions)

    bargs = _builder_args(args)
    if name == "semisynth":
        X = _load_covariates(args)
        return build_semisynth_constructions(X, bargs)
    if name == "chambers":
        return build_chamber_constructions(bargs)
    raise ValueError(f"unknown bench {name!r}")


def _load_covariates(args) -> np.ndarray:
    """Real covariate matrix for the semisynth bench (ACIC-2016 or IHDP)."""
    if args.covariates == "acic":
        from causal_probe.semisynth_acic_fp import load_acic2016
        X = load_acic2016(args.acic_path)
        print(f"ACIC-2016 covariates: {X.shape[0]} x {X.shape[1]} usable cols")
    else:
        from causal_probe.ihdp_decoy_eval import load_ihdp_covariates
        X, identical = load_ihdp_covariates(args.ihdp_dir)
        print(f"IHDP covariates: {X.shape[0]} x {X.shape[1]} usable cols "
              f"(identical across replicates: {identical})")
    if args.p_real_cap > X.shape[1]:
        print(f"p_real_cap {args.p_real_cap} > p={X.shape[1]}; clipping.")
        args.p_real_cap = X.shape[1]
    return X


# ---------------------------------------------------------------------------
# Model pass: point tau_hat + bootstrap sigma per construction
# ---------------------------------------------------------------------------


def _point_pred(model, X: np.ndarray, y: np.ndarray, device: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32)[None]).to(device),
                     torch.from_numpy(y.astype(np.float32)[None]).to(device)
                     ).squeeze(0).cpu().numpy().astype(np.float64)


def run_constructions(model, constructions: list[dict], args) -> list[dict]:
    """Attach tau_hat (bootstrap mean), sigma to each construction.

    We report tau_hat as the bootstrap MEAN (matches decoy_sigma_analysis's
    boot_mean and keeps tau_hat/sigma from the same resamples); the point
    prediction on the full context is stored too for reference.
    """
    rng = np.random.default_rng(args.seed + 1)
    records = []
    for i, c in enumerate(constructions):
        X, y = c["X_ctx"], c["y_ctx"]
        point = _point_pred(model, X, y, args.device)
        boot = _bootstrap_preds(model, X, y, args.n_boot,
                                np.random.default_rng(rng.integers(0, 2**31)),
                                args.device)
        b_mean, sigma = boot.mean(axis=0), boot.std(axis=0, ddof=1)
        if "perm" in c:  # chamber contexts are column-permuted; undo
            inv = np.empty_like(c["perm"]); inv[c["perm"]] = np.arange(len(c["perm"]))
            point, b_mean, sigma = point[inv], b_mean[inv], sigma[inv]
        records.append({"meta": c["meta"],
                        "tau_true": np.asarray(c["tau_true"], np.float64).tolist(),
                        "tau_hat": b_mean.tolist(),
                        "point": point.tolist(),
                        "sigma": sigma.tolist()})
        if (i + 1) % 10 == 0 or i + 1 == len(constructions):
            print(f"  [{i + 1}/{len(constructions)}] {c['meta']}", flush=True)
    return records


# ---------------------------------------------------------------------------
# Selective-prediction curve
# ---------------------------------------------------------------------------


def _flatten(records: list[dict]):
    th, tt, sg = [], [], []
    for r in records:
        th += r["tau_hat"]
        tt += r["tau_true"]
        sg += r["sigma"]
    return (np.asarray(th, np.float64), np.asarray(tt, np.float64),
            np.asarray(sg, np.float64))


def _retained_error(err: np.ndarray, drop: np.ndarray) -> float:
    keep = ~drop
    return float(err[keep].mean()) if keep.any() else float("nan")


def selective_curve(records: list[dict], n_random_reps: int, seed: int) -> dict:
    """Error-vs-abstention curve for sigma-abstention and a random baseline.

    Pooled over all feature observations in the bench: at each abstention
    fraction a, sigma-abstention drops the top-a by sigma; the random baseline
    drops a random a-fraction (mean +/- sd over n_random_reps draws). Returns
    the two curves plus scalar summaries (err@0, err@SUMMARY_A, ratio, AUC).
    """
    tau_hat, tau_true, sigma = _flatten(records)
    err = np.abs(tau_hat - tau_true)
    n = err.size
    order = np.argsort(-sigma)  # descending sigma; top-a are the first k

    from scipy.stats import spearmanr, pearsonr
    sig_curve, rand_mean, rand_sd = [], [], []
    rng = np.random.default_rng(seed + 7)
    for a in ABSTAIN_FRACTIONS:
        k = int(round(a * n))
        drop_sig = np.zeros(n, dtype=bool); drop_sig[order[:k]] = True
        sig_curve.append(_retained_error(err, drop_sig))
        if k == 0:
            rand_mean.append(float(err.mean())); rand_sd.append(0.0)
            continue
        vals = []
        for _ in range(n_random_reps):
            drop_r = np.zeros(n, dtype=bool)
            drop_r[rng.choice(n, size=k, replace=False)] = True
            vals.append(_retained_error(err, drop_r))
        rand_mean.append(float(np.mean(vals)))
        rand_sd.append(float(np.std(vals)))

    def _auc(y):  # trapezoidal AUC over the abstention sweep, normalized by width
        x = np.asarray(ABSTAIN_FRACTIONS, np.float64)
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy 2 removed trapz
        return float(trap(y, x) / (x[-1] - x[0]))

    i0 = ABSTAIN_FRACTIONS.index(0.0)
    ia = ABSTAIN_FRACTIONS.index(SUMMARY_A)
    e0, ea = sig_curve[i0], sig_curve[ia]
    return {
        "n_features": int(n),
        "abstain_fractions": list(ABSTAIN_FRACTIONS),
        "sigma_abstention_error": sig_curve,
        "random_abstention_error_mean": rand_mean,
        "random_abstention_error_sd": rand_sd,
        "spearman_sigma_err": float(spearmanr(sigma, err)[0]) if sigma.std() > 0 else float("nan"),
        "pearson_sigma_err": float(pearsonr(sigma, err)[0]) if sigma.std() > 0 else float("nan"),
        "summary": {
            "err_at_0": e0,
            f"err_at_{SUMMARY_A}": ea,
            "err_ratio_a_over_0": float(ea / e0) if e0 > 0 else float("nan"),
            "auc_sigma": _auc(sig_curve),
            "auc_random": _auc(rand_mean),
        },
    }


def _print_curve(label: str, res: dict) -> None:
    print(f"\n=== {label} ({res['n_features']} feature observations) ===")
    print(f"  Spearman(sigma, |err|) = {res['spearman_sigma_err']:+.3f}   "
          f"Pearson = {res['pearson_sigma_err']:+.3f}")
    print(f"  {'a':>6}{'sigma-abstain |err|':>22}{'random |err|':>20}")
    for i, a in enumerate(res["abstain_fractions"]):
        rm, rs = res["random_abstention_error_mean"][i], res["random_abstention_error_sd"][i]
        print(f"  {a:>6.1f}{res['sigma_abstention_error'][i]:>22.4f}"
              f"{rm:>13.4f} +/- {rs:.4f}")
    s = res["summary"]
    print(f"  err@0={s['err_at_0']:.4f}  err@{SUMMARY_A}={s[f'err_at_{SUMMARY_A}']:.4f}  "
          f"ratio={s['err_ratio_a_over_0']:.3f}  "
          f"AUC(sigma)={s['auc_sigma']:.4f}  AUC(random)={s['auc_random']:.4f}")


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------


def _dry_run_report(name: str, constructions: list[dict]) -> None:
    n_feat = sum(len(c["tau_true"]) for c in constructions)
    tau_ok = all(np.all(np.isfinite(np.asarray(c["tau_true"], np.float64)))
                 for c in constructions)
    n_nonzero = sum(int(np.count_nonzero(np.abs(np.asarray(c["tau_true"])) > 1e-9))
                    for c in constructions)
    print(f"[dry-run] bench={name}: {len(constructions)} constructions, "
          f"{n_feat} feature observations")
    print(f"    tau_true available: {tau_ok} (all finite); "
          f"{n_nonzero} nonzero-effect features, {n_feat - n_nonzero} null features")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None,
                    help="FADO checkpoint (required unless --dry-run)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-scms", type=int, default=10,
                    help="SCM instances per family x regime (semisynth) and "
                         "context draws per config (chambers, unless "
                         "--chamber-draws)")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="row-bootstrap resamples per construction")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-random-reps", type=int, default=50,
                    help="random-abstention draws to average the baseline over")
    ap.add_argument("--benches", default="all",
                    choices=["chambers", "semisynth", "all"])
    ap.add_argument("--out", default="causal_probe/results/selective_prediction_curve.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="build benches, print feature counts + tau_true "
                         "availability, exit 0 (no checkpoint / GPU)")
    # semisynth (ACIC/IHDP) options
    ap.add_argument("--covariates", default="acic", choices=["acic", "ihdp"],
                    help="covariate source for the semisynth bench")
    ap.add_argument("--acic-path", default="data/acic2016/x.csv")
    ap.add_argument("--ihdp-dir", default="data/ihdp")
    ap.add_argument("--families", default="linear,mixed_nonlinear,interaction")
    ap.add_argument("--p-real-cap", type=int, default=10)
    ap.add_argument("--n-causal", type=int, default=3)
    # chambers options
    ap.add_argument("--data-dir", default="/tmp/causalchamber_data")
    ap.add_argument("--chamber-draws", type=int, default=None,
                    help="context draws per chamber config (default: --n-scms)")
    ap.add_argument("--skip-chamber-download", action="store_true",
                    help="on --dry-run, skip building the chambers bench (which "
                         "downloads the causalchamber dataset) and just print the "
                         "plan; lets a CPU dry-run validate semisynth alone")
    args = ap.parse_args()

    if args.ckpt is None and not args.dry_run:
        ap.error("--ckpt is required unless --dry-run")

    benches = (["chambers", "semisynth"] if args.benches == "all"
               else [args.benches])

    constructions: dict[str, list[dict]] = {}
    for name in benches:
        if (name == "chambers" and args.dry_run and args.skip_chamber_download):
            print("[dry-run] bench=chambers: SKIPPED (--skip-chamber-download). "
                  "Plan: lt_vis3 + lt_vis1 light-tunnel configs, "
                  f"{args.chamber_draws or args.n_scms} context draws each, "
                  "10 features/config; tau_true from build_ground_truth "
                  "(hardware-verified, CPU-only download on the real run).")
            continue
        constructions[name] = build_bench(name, args)

    if args.dry_run:
        for name, cons in constructions.items():
            _dry_run_report(name, cons)
        print("[dry-run] OK -- exiting before model load")
        return

    from causal_probe.eval import _load_model
    model = _load_model(args.ckpt, device=args.device)
    print(f"Loaded {args.ckpt}; B={args.n_boot} row-resamples per construction\n")

    curves: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}
    for name, cons in constructions.items():
        print(f"--- bench {name}: {len(cons)} constructions ---")
        recs = run_constructions(model, cons, args)
        all_records[name] = recs
        curves[name] = selective_curve(recs, args.n_random_reps, args.seed)
        _print_curve(name, curves[name])
    if len(all_records) > 1:
        pooled = [r for recs in all_records.values() for r in recs]
        curves["pooled"] = selective_curve(pooled, args.n_random_reps, args.seed)
        _print_curve("pooled (all benches)", curves["pooled"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()},
                   "curves": curves, "records": all_records}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
