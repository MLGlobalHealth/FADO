"""RCT negative-control battery: can the probe be fooled into "causing the past"?

For each real RCT dataset (Hillstrom, LaLonde/NSW, Criteo uplift) and each
pre-treatment covariate c, build an eval dataset where

  * outcome Y := c (a PRE-treatment covariate),
  * features   := the randomized treatment indicator T plus the OTHER
    pre-treatment covariates (c is dropped from the features; the true
    post-treatment outcomes — visit/conversion/spend, re78 — are dropped
    entirely).

Because T is randomized, it cannot cause the past: the true tau_T is exactly
0 for every such construction. We record tau_hat_T per construction, plus the
repo's marginal baseline (2*corr(T, c) on standardized data, the same scale
as tau_hat) for context — both should be ~0; the point of the table is that
FADO's adjustment machinery is not manufacturing false positives that the
raw randomization balance doesn't have.

Row subsampling follows hillstrom_rct.evaluate(): per construction we draw
`--n-repeats` (default 5) row subsamples of `--n-rows` (default 512) without
replacement from one seeded rng and report mean±sd of tau_hat_T across the
repeats. When the dataset is smaller than n_rows (LaLonde NSW: 445 rows) a
single forward pass is run and sd is 0.

Note on LaLonde: the negative-control logic requires randomized T, so we use
the experimental NSW sample (nswre74 treated + nswre74 control), NOT the
confounded cps3 comparison used by lalonde_rct.py's headline run.

Comparator suite: alongside FADO, every construction is scored by the
graph-free comparators in causal_probe.comparator_suite (predictive
attribution: SHAP, permutation importance; causal: LiNGAM, DoubleML,
causal forest, DR-Learner, CausalPFN, Do-PFN), each reporting its score
AT THE T COLUMN — the known-null. No graph-oracle methods appear here:
no ground-truth causal graph exists for these real datasets. Slow
per-column methods sit behind --skip-slow; --comparators-only runs the
suite without a checkpoint. --exclude-near-constant drops controls whose
covariate has top-value frequency > 0.9 (the near-constant Criteo
pathology); top_freq is
recorded per control either way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.baselines import BASELINES
from causal_probe.comparator_suite import (
    ALL_METHODS,
    default_methods,
    run_comparator_suite,
    summarize_tau_T,
    top_value_frequency,
)
from causal_probe.hillstrom_rct import _load_hillstrom, _standardize
from causal_probe.lalonde_rct import COLUMNS as LALONDE_COLUMNS, load_lalonde_obs

MAX_P = 15  # checkpoint is variable-p (3..15 features incl. T)
DATASETS = ["hillstrom", "lalonde", "criteo"]
# Fixed per-dataset index so construction seeds don't shift when --datasets
# selects a subset.
_DATASET_ORD = {name: i for i, name in enumerate(DATASETS)}

# Post-treatment outcome columns, dropped entirely from every construction.
HILLSTROM_OUTCOMES = ["visit", "conversion", "spend"]
LALONDE_OUTCOME = "re78"

LALONDE_COV_COLS = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]


# ---------------------------------------------------------------------------
# Per-dataset loaders → (T, cov_matrix, cov_names, meta)
# ---------------------------------------------------------------------------

def load_hillstrom_controls(treatment_group: str):
    """Hillstrom pre-treatment covariates: recency/history/mens/womens/newbie
    plus one-hot dummies for zip_code and channel. history_segment is dropped
    (string binning of history)."""
    import pandas as pd

    df = _load_hillstrom()
    df = df[df["segment"].isin([treatment_group, "No E-Mail"])].copy()
    T = (df["segment"] == treatment_group).values.astype(np.float64)

    numeric = ["recency", "history", "mens", "womens", "newbie"]
    dummies = pd.get_dummies(df[["zip_code", "channel"]], prefix=["zip", "channel"])
    cov = pd.concat([df[numeric].reset_index(drop=True),
                     dummies.reset_index(drop=True)], axis=1)
    names = list(cov.columns)
    meta = {"treatment": f"{treatment_group} vs No E-Mail", "n_total": int(len(df))}
    return T, cov.values.astype(np.float64), names, meta


def load_lalonde_controls():
    """Experimental NSW sample (randomized T): nswre74 treated + control."""
    data, cols = load_lalonde_obs(control="nsw")
    col_idx = {c: i for i, c in enumerate(cols)}
    T = data[:, col_idx["T"]].astype(np.float64)
    cov = data[:, [col_idx[c] for c in LALONDE_COV_COLS]].astype(np.float64)
    meta = {"treatment": "NSW training program (experimental sample)",
            "n_total": int(data.shape[0])}
    return T, cov, list(LALONDE_COV_COLS), meta


def load_criteo_controls(subsample: int, seed: int):
    """Criteo uplift f0..f11. The fetched target (visit) is discarded —
    post-treatment outcomes play no role in negative controls."""
    from causal_probe.criteo_rct import _load_criteo_subsample

    X_cov, _y, T, names = _load_criteo_subsample(subsample, seed, "visit")
    meta = {"treatment": "Criteo randomized ad exposure",
            "n_total": int(X_cov.shape[0]), "subsample": int(subsample)}
    return T.astype(np.float64), X_cov.astype(np.float64), names, meta


# ---------------------------------------------------------------------------
# Construction + evaluation
# ---------------------------------------------------------------------------

def build_construction(
    T: np.ndarray,
    cov: np.ndarray,
    cov_names: list[str],
    control_idx: int,
    rng: np.random.Generator,
    max_p: int = MAX_P,
):
    """Y := cov[:, control_idx]; features := T + the other covariates,
    column-shuffled (matching the sibling scripts' hidden-treatment
    convention). If T + others exceed max_p, the other covariates are
    subsampled with the construction rng (noted in the record)."""
    y = cov[:, control_idx].copy()
    other_idx = [i for i in range(cov.shape[1]) if i != control_idx]
    dropped = None
    if 1 + len(other_idx) > max_p:
        keep = sorted(rng.choice(len(other_idx), size=max_p - 1, replace=False).tolist())
        kept_idx = [other_idx[i] for i in keep]
        dropped = [cov_names[i] for i in other_idx if i not in kept_idx]
        other_idx = kept_idx
    parts = [("T", T.reshape(-1, 1))]
    for i in other_idx:
        parts.append((cov_names[i], cov[:, i:i + 1]))
    order = rng.permutation(len(parts))
    feat_names = [parts[i][0] for i in order]
    X = np.concatenate([parts[i][1] for i in order], axis=1)
    T_idx = int(np.where(order == 0)[0][0])
    return X, y, feat_names, T_idx, dropped


def run_construction(
    model,
    Xs: np.ndarray,
    ys: np.ndarray,
    T_idx: int,
    n_rows: int,
    n_repeats: int,
    rng: np.random.Generator,
    device: str,
):
    """n_repeats probe forward passes on seeded row subsamples of the
    standardized construction (mirrors hillstrom_rct.evaluate).
    Returns (mean, sd, n_eff)."""
    import torch

    n_total = Xs.shape[0]
    n_eff = n_repeats if n_total > n_rows else 1  # identical passes otherwise
    taus = []
    for _ in range(n_eff):
        if n_total <= n_rows:
            X_ctx, y_ctx = Xs, ys
        else:
            idx = rng.choice(n_total, size=n_rows, replace=False)
            X_ctx, y_ctx = Xs[idx], ys[idx]
        X_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        taus.append(float(pred[T_idx]))
    taus = np.asarray(taus)
    return float(taus.mean()), float(taus.std()), n_eff


def evaluate_dataset(
    name: str,
    T: np.ndarray,
    cov: np.ndarray,
    cov_names: list[str],
    meta: dict,
    model,
    seed: int,
    n_rows: int,
    n_repeats: int,
    device: str,
    dry_run: bool,
    comp_methods: list[str] | None = None,
    dopfn_repo: str | None = None,
    exclude_near_constant: bool = False,
) -> dict:
    ds_ord = _DATASET_ORD[name]
    records = []
    for c_idx, c_name in enumerate(cov_names):
        rng = np.random.default_rng([seed, ds_ord, c_idx])
        X_raw, y_raw, feat_names, T_idx, dropped = build_construction(
            T, cov, cov_names, c_idx, rng,
        )
        top_freq = top_value_frequency(y_raw)
        rec = {
            "control": c_name,
            "p": int(X_raw.shape[1]),
            "T_idx": T_idx,
            "feature_names": feat_names,
            "cov_dropped_for_p_limit": dropped,
            "top_freq": top_freq,
        }
        if exclude_near_constant and top_freq > 0.9:
            rec["excluded_near_constant"] = True
            if dry_run:
                print(f"    control={c_name:<16s} EXCLUDED (top_freq={top_freq:.3f} > 0.9)")
            records.append(rec)
            continue
        if dry_run:
            print(f"    control={c_name:<16s} X={X_raw.shape} y={y_raw.shape} "
                  f"T at col {T_idx}  top_freq={top_freq:.3f}"
                  + (f"  dropped={dropped}" if dropped else ""))
        else:
            Xs, ys, _, _, _, _ = _standardize(X_raw, y_raw)
            rec["marginal_2corr_T"] = float(BASELINES["marginal"](Xs, ys)[T_idx])
            if model is not None:
                tau_mean, tau_sd, n_eff = run_construction(
                    model, Xs, ys, T_idx, n_rows, n_repeats, rng, device,
                )
                rec.update({
                    "tau_hat_T_mean": tau_mean,
                    "tau_hat_T_sd": tau_sd,
                    "n_repeats_effective": n_eff,
                })
            if comp_methods:
                comp_seed = int(np.random.default_rng(
                    [seed, ds_ord, c_idx, 7]).integers(0, 2**31))
                rec["comparators"] = run_comparator_suite(
                    Xs, ys, T_idx, methods=comp_methods, seed=comp_seed,
                    device=device, dopfn_repo=dopfn_repo,
                )
        records.append(rec)

    out = {**meta, "controls": records}
    if not dry_run:
        evaluated = [r for r in records if not r.get("excluded_near_constant")]
        summary = {
            "n_controls": len(records),
            "n_excluded_near_constant": len(records) - len(evaluated),
        }
        if evaluated:
            abs_marg = np.asarray([abs(r["marginal_2corr_T"]) for r in evaluated])
            summary["mean_abs_marginal_2corr_T"] = float(abs_marg.mean())
            summary["max_abs_marginal_2corr_T"] = float(abs_marg.max())
            if model is not None:
                abs_tau = np.asarray([abs(r["tau_hat_T_mean"]) for r in evaluated])
                summary["mean_abs_tau_hat_T"] = float(abs_tau.mean())
                summary["max_abs_tau_hat_T"] = float(abs_tau.max())
            if comp_methods:
                summary["comparators"] = summarize_tau_T(
                    [r["comparators"] for r in evaluated], comp_methods)
        out["summary"] = summary
    return out


_SHORT = {"marginal": "marg", "permutation": "perm", "doubleml": "dml",
          "causal_forest": "cforest", "drlearner": "drl", "causalpfn": "cpfn"}


def _fmt(v, width: int = 9) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'--':>{width}s}"
    return f"{v:>+{width}.4f}"


def _print_table(name: str, res: dict, comp_methods: list[str] | None) -> None:
    comp_methods = comp_methods or []
    print(f"\n=== {name}: {res['treatment']} (n={res['n_total']}) ===")
    hdr = (f"  {'control outcome':<18s}{'top_frq':>8s}{'tau_hat_T':>12s}{'±sd':>8s}"
           f"{'2corr(T,c)':>11s}")
    for m in comp_methods:
        hdr += f"{_SHORT.get(m, m):>9s}"
    print(hdr)
    for r in res["controls"]:
        line = f"  {r['control']:<18s}{r['top_freq']:>8.3f}"
        if r.get("excluded_near_constant"):
            print(line + "  excluded (near-constant, top_freq > 0.9)")
            continue
        line += _fmt(r.get("tau_hat_T_mean"), 12)
        sd = r.get("tau_hat_T_sd")
        line += f"{sd:>8.4f}" if sd is not None else f"{'--':>8s}"
        line += _fmt(r.get("marginal_2corr_T"), 11)
        comp = r.get("comparators", {})
        for m in comp_methods:
            cr = comp.get(m, {})
            line += f"{'ERR':>9s}" if "error" in cr else _fmt(cr.get("tau_hat_T"))
        print(line)
    s = res["summary"]
    print(f"  {'-' * 58}")
    excl = s.get("n_excluded_near_constant", 0)
    print(f"  n_controls = {s['n_controls']}"
          + (f" ({excl} excluded near-constant)" if excl else ""))
    if "mean_abs_tau_hat_T" in s:
        print(f"  FADO:      mean|tau_hat_T| = {s['mean_abs_tau_hat_T']:.4f}   "
              f"max|tau_hat_T| = {s['max_abs_tau_hat_T']:.4f}")
    if "mean_abs_marginal_2corr_T" in s:
        print(f"  marginal:  mean|2corr| = {s['mean_abs_marginal_2corr_T']:.4f}   "
              f"max|2corr| = {s['max_abs_marginal_2corr_T']:.4f}")
    for m, ms in s.get("comparators", {}).items():
        if ms["n"] == 0:
            print(f"  {m + ':':<11s}all failed ({ms['n_errors']} errors)")
        else:
            print(f"  {m + ':':<11s}mean|tau_hat_T| = {ms['mean_abs_tau_hat_T']:.4f}   "
                  f"max|tau_hat_T| = {ms['max_abs_tau_hat_T']:.4f}"
                  + (f"   ({ms['n_errors']} errors)" if ms["n_errors"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Probe checkpoint (required unless --dry-run).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=5,
                    help="Row-subsample seeds per construction (mean±sd reported).")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--treatment-group", default="Mens E-Mail",
                    choices=["Mens E-Mail", "Womens E-Mail"],
                    help="Hillstrom treatment arm (vs No E-Mail).")
    ap.add_argument("--criteo-subsample", type=int, default=12000)
    ap.add_argument("--out", default="causal_probe/results/rct_negative_controls.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load data + build all constructions, print shapes, "
                         "exit before model load (no checkpoint / GPU needed).")
    ap.add_argument("--comparators-only", action="store_true",
                    help="Run the graph-free comparator suite without the probe "
                         "(no checkpoint / GPU needed).")
    ap.add_argument("--comparators", nargs="+", default=None, choices=ALL_METHODS,
                    help="Comparator subset (default: all cheap methods + slow "
                         "unless --skip-slow; dopfn only with --dopfn-repo).")
    ap.add_argument("--skip-slow", action="store_true",
                    help="Skip the slow per-column comparators "
                         "(doubleml/causal_forest/drlearner/causalpfn/dopfn).")
    ap.add_argument("--no-comparators", action="store_true",
                    help="Probe only, no comparator suite (previous behaviour).")
    ap.add_argument("--dopfn-repo", default=None,
                    help="Checked-out Do-PFN repo (enables the dopfn comparator).")
    ap.add_argument("--exclude-near-constant", action="store_true",
                    help="Drop controls whose covariate has top-value frequency "
                         "> 0.9 (near-constant pathology).")
    args = ap.parse_args()

    if not args.dry_run and not args.comparators_only and args.ckpt is None:
        ap.error("--ckpt is required unless --dry-run or --comparators-only")

    comp_methods = None
    if not args.dry_run and not args.no_comparators:
        comp_methods = args.comparators or default_methods(
            skip_slow=args.skip_slow, dopfn_repo=args.dopfn_repo)
        if args.skip_slow:
            from causal_probe.comparator_suite import SLOW_METHODS
            comp_methods = [m for m in comp_methods if m not in SLOW_METHODS]

    loaders = {
        "hillstrom": lambda: load_hillstrom_controls(args.treatment_group),
        "lalonde": load_lalonde_controls,
        "criteo": lambda: load_criteo_controls(args.criteo_subsample, args.seed),
    }

    model = None
    if not args.dry_run and not args.comparators_only:
        from causal_probe.eval import _load_model
        model = _load_model(args.ckpt, device=args.device)
        print(f"Loaded {args.ckpt}")

    results = {
        "ckpt": args.ckpt, "seed": args.seed,
        "n_rows": args.n_rows, "n_repeats": args.n_repeats,
        "comparator_methods": comp_methods,
        "exclude_near_constant": args.exclude_near_constant,
        "datasets": {},
    }
    for name in args.datasets:
        print(f"\n--- {name} ---" if args.dry_run else f"\nRunning {name} ...")
        T, cov, cov_names, meta = loaders[name]()
        if args.dry_run:
            print(f"  n={len(T)}, covariates={len(cov_names)}: {cov_names}")
        res = evaluate_dataset(
            name, T, cov, cov_names, meta, model,
            args.seed, args.n_rows, args.n_repeats, args.device, args.dry_run,
            comp_methods=comp_methods, dopfn_repo=args.dopfn_repo,
            exclude_near_constant=args.exclude_near_constant,
        )
        results["datasets"][name] = res
        if not args.dry_run:
            _print_table(name, res, comp_methods)

    if args.dry_run:
        print("\ndry run OK — constructions built; exiting before model load.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
