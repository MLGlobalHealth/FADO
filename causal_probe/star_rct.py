"""Tennessee STAR class-size RCT benchmark (kindergarten cross-section).

Project STAR (Student/Teacher Achievement Ratio, 1985-89) randomly
assigned Tennessee kindergarten students and teachers to small classes
(13-17 students), regular classes (22-25), or regular classes with a
full-time aide, *within* each participating school. The small-class
effect on kindergarten test scores is one of the best-replicated
findings in education economics: roughly +0.15 to +0.2 sd on
end-of-kindergarten scores (Krueger, QJE 1999). This is the first
continuous-outcome RCT eval for the probe beyond LaLonde.

Experiment structure (mirrors lalonde_rct.py / hillstrom_rct.py):

  1. Load the AER::STAR kindergarten cross-section from Rdatasets.
  2. Treatment T = 1 if stark == "small"; regular and regular+aide are
     binarized together as T = 0. Rationale: the aide effect is ~0 in
     the literature (Krueger 1999, Table V), and pooling keeps a clean
     binary treatment. Rows missing stark/readk/mathk are dropped.
  3. Outcome Y = readk + mathk (continuous combined K test score).
  4. Features (p = 9, model envelope p <= 15 incl. Y):
       T, T_perm (shuffled-T placebo), girl, afam, birth_time
       (year + quarter/4, continuous), free_lunch, and school
       urbanicity dummies rural/urban/inner-city (suburban = ref).
     All are pre-treatment / at-K-entry. Teacher variables (degree,
     ladder, experience, tethnicity) are excluded: teachers were
     themselves randomized to class types, so teacher columns are
     post-treatment.
  5. Shuffle columns; do NOT tell the model which column is T.
  6. Run the probe -> tau_hat per column; compare tau_hat_T to the
     in-sample diff-in-means on the probe's reporting scale
     (Gelman 2008: tau_RCT = (raw_ATE / std(Y)) * 2 * std(T_raw)).
  7. Placebo: tau_hat at the permuted-T column should be ~0.

Randomization caveat (recorded in the JSON): STAR randomized WITHIN
schools, not across them. School identity is too high-cardinality to
include as a feature (79 schools), so school membership is only
proxied by the urbanicity dummies. Because small-class proportions are
approximately (not exactly) equal across schools, T can carry a small
marginal correlation with school-level covariates (free_lunch, afam).
This slightly contaminates both the diff-in-means target and the
--negative-controls mode; the --largest-school flag restricts to the
single largest school (true unconditional randomization) as a
robustness subset, at the cost of a small n.

--negative-controls mode: regress each individual-level pre-treatment
covariate (girl, afam, birth_time, free_lunch) as the outcome with T
among the features; a causal probe should report tau_hat_T ~ 0 on all
of them (up to the within-school caveat above for afam/free_lunch).

Comparator suite: both modes also report the graph-free comparators from
causal_probe.comparator_suite (SHAP, permutation importance, LiNGAM,
DoubleML, causal forest, DR-Learner, CausalPFN, Do-PFN) at the T column,
on the same standardized construction the probe sees. No graph-oracle
methods appear: no ground-truth causal graph exists for STAR. Slow
per-column methods sit behind --skip-slow; --comparators-only runs the
suite without a checkpoint.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.comparator_suite import (
    ALL_METHODS,
    SLOW_METHODS,
    default_methods,
    run_comparator_suite,
)
from causal_probe.eval import _load_model


URLS_STAR = [
    "https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv",
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/STAR.csv",
]
CACHE_PATH = Path.home() / ".cache" / "causal_probe" / "STAR.csv"

# Individual-level pre-treatment covariates usable as negative-control
# outcomes (school urbanicity dummies excluded: school-level, and the
# within-school randomization caveat applies most strongly there).
NEG_CONTROL_COVS = ["girl", "afam", "birth_time", "free_lunch"]


def _fetch_star_csv() -> str:
    """Return the raw STAR.csv text, using a local cache with URL fallbacks."""
    if CACHE_PATH.exists():
        return CACHE_PATH.read_text()
    last = None
    for u in URLS_STAR:
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(raw)
            return raw
        except Exception as e:
            last = e
    raise RuntimeError(f"STAR fetch failed: {last}")


def load_star(largest_school_only: bool = False):
    """Load the kindergarten cross-section and engineer numeric columns.

    Returns a DataFrame with columns:
      T, girl, afam, birth_time, free_lunch,
      school_rural, school_urban, school_innercity, Y, schoolidk
    plus scalar metadata (n dropped, raw score stats) in a dict.
    """
    import pandas as pd

    df = pd.read_csv(io.StringIO(_fetch_star_csv()))
    n_raw = len(df)

    # K-grade participants with complete K test scores.
    df = df.dropna(subset=["stark", "readk", "mathk"]).copy()
    n_stark = len(df)
    # Small numbers of missing covariates (~22 rows total).
    df = df.dropna(subset=["gender", "ethnicity", "birth", "lunchk", "schoolk"])
    n_kept = len(df)

    out = pd.DataFrame(index=df.index)
    # T = small class; regular and regular+aide binarized to 0 (see docstring).
    out["T"] = (df["stark"] == "small").astype(np.float64)
    out["girl"] = (df["gender"] == "female").astype(np.float64)
    # cauc + afam are >99% of the sample; collapse to an afam indicator.
    out["afam"] = (df["ethnicity"] == "afam").astype(np.float64)
    # birth is "YYYY Qn" -> continuous birth_time = year + (quarter-1)/4.
    yq = df["birth"].str.extract(r"(\d{4})\s*Q(\d)")
    out["birth_time"] = yq[0].astype(float) + (yq[1].astype(float) - 1.0) / 4.0
    out["free_lunch"] = (df["lunchk"] == "free").astype(np.float64)
    out["school_rural"] = (df["schoolk"] == "rural").astype(np.float64)
    out["school_urban"] = (df["schoolk"] == "urban").astype(np.float64)
    out["school_innercity"] = (df["schoolk"] == "inner-city").astype(np.float64)
    out["Y"] = (df["readk"] + df["mathk"]).astype(np.float64)
    out["schoolidk"] = df["schoolidk"].astype(np.float64)

    if largest_school_only:
        top = out["schoolidk"].value_counts().idxmax()
        out = out[out["schoolidk"] == top].copy()

    meta = {
        "n_raw_csv": int(n_raw),
        "n_with_stark_and_k_scores": int(n_stark),
        "n_after_covariate_dropna": int(n_kept),
        "n_analysis": int(len(out)),
        "largest_school_only": bool(largest_school_only),
        "largest_school_id": (float(out["schoolidk"].iloc[0])
                              if largest_school_only else None),
    }
    return out.reset_index(drop=True), meta


COV_COLS = ["girl", "afam", "birth_time", "free_lunch",
            "school_rural", "school_urban", "school_innercity"]


def _build_star_input(df, outcome_col: str, rng: np.random.Generator,
                      drop_cov: str | None = None):
    """Assemble the shuffled feature matrix: T + permuted-T placebo + covariates.

    ``drop_cov`` removes one covariate from the features (used in
    --negative-controls mode when that covariate is the outcome).
    """
    cov_cols = [c for c in COV_COLS if c != drop_cov]
    T = df["T"].values.astype(np.float64)
    y = df[outcome_col].values.astype(np.float64)
    T_perm = rng.permutation(T)  # placebo: same marginal, no causal link

    parts = [("T_true", T.reshape(-1, 1)),
             ("T_perm (placebo)", T_perm.reshape(-1, 1))]
    for nm in cov_cols:
        parts.append((nm, df[nm].values.astype(np.float64).reshape(-1, 1)))
    order = rng.permutation(len(parts))
    names = [parts[i][0] for i in order]
    X = np.concatenate([parts[i][1] for i in order], axis=1)
    T_idx = int(np.where(order == 0)[0][0])
    Tp_idx = int(np.where(order == 1)[0][0])
    return X, y, T, names, T_idx, Tp_idx


def _standardize(X, y):
    Xs = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-9)
    ys = (y - y.mean()) / y.std().clip(min=1e-9)
    return Xs, ys


def _diff_in_means(T: np.ndarray, y: np.ndarray) -> dict:
    """Raw + standardized + probe-scale (Gelman 2008) diff-in-means."""
    ate_raw = float(y[T > 0.5].mean() - y[T < 0.5].mean())
    sd_y = float(y.std())
    sd_T = float(T.std())
    return {
        "ate_raw_points": ate_raw,
        "ate_sd_units": ate_raw / max(sd_y, 1e-9),
        "tau_T_probe_scale": (ate_raw / max(sd_y, 1e-9)) * 2.0 * sd_T,
        "two_std_T": 2.0 * sd_T,
        "n_treated": int((T > 0.5).sum()),
        "n_control": int((T < 0.5).sum()),
    }


def _run_probe(model, Xs, ys, n_rows, n_repeats, rng, device):
    """n_repeats forward passes on seeded row subsamples; mean/std tau_hat."""
    n = Xs.shape[0]
    preds = []
    for _ in range(n_repeats):
        idx = np.arange(n) if n <= n_rows else rng.choice(n, size=n_rows, replace=False)
        X_t = torch.from_numpy(Xs[idx].astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(ys[idx].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            preds.append(model(X_t, y_t).squeeze(0).cpu().numpy())
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def _run_comparators_star(Xs, ys, T_idx, Tp_idx, names, seed, device,
                          comp_methods, dopfn_repo) -> dict:
    """Run the graph-free suite on the standardized STAR construction and
    print a compact table (T column + placebo column where free)."""
    comp = run_comparator_suite(
        Xs, ys, T_idx, methods=comp_methods, seed=seed, device=device,
        dopfn_repo=dopfn_repo, extra_cols={"T_perm": Tp_idx}, return_full=True,
    )
    print(f"\n  comparators ({'/'.join(comp_methods)}) at T | placebo:")
    for m, r in comp.items():
        if "error" in r:
            print(f"    {m:<14s} ERROR: {r['error']}")
            continue
        tT = r.get("tau_hat_T")
        tP = r.get("tau_hat_T_perm")
        tT_s = f"{tT:+.4f}" if tT is not None else "--"
        tP_s = f"{tP:+.4f}" if tP is not None else "--"
        print(f"    {m:<14s} {tT_s:>9s} | {tP_s:>9s}")
    vec = [m for m, r in comp.items() if "tau" in r]
    if vec:
        print(f"\n  {'feature':<20s}" + "".join(f"{m[:9]:>10s}" for m in vec))
        for i, nm in enumerate(names):
            tag = "  <- T" if i == T_idx else ("  <- placebo" if i == Tp_idx else "")
            row = f"  {nm:<20s}"
            for m in vec:
                v = comp[m]["tau"][i]
                row += f"{v:>+10.4f}" if v is not None else f"{'--':>10s}"
            print(row + tag)
    return comp


def evaluate(model, df, meta, n_rows, n_repeats, seed, device,
             comp_methods=None, dopfn_repo=None) -> dict:
    rng = np.random.default_rng(seed)
    X_raw, y_raw, T, names, T_idx, Tp_idx = _build_star_input(df, "Y", rng)
    Xs, ys = _standardize(X_raw, y_raw)
    dim = _diff_in_means(T, y_raw)
    tau_T_ref = dim["tau_T_probe_scale"]

    assoc = BASELINES["marginal"](Xs, ys)
    multi = BASELINES["multivariate"](Xs, ys)

    print(f"\n  reference small-class effect: {dim['ate_raw_points']:+.2f} raw points"
          f" = {dim['ate_sd_units']:+.4f} sd"
          f" -> {tau_T_ref:+.4f} probe scale (2*std(T)={dim['two_std_T']:.3f})")

    res = {
        "dataset": "STAR (AER, kindergarten cross-section)",
        "meta": meta,
        "n": int(len(df)), "p": int(X_raw.shape[1]),
        "n_rows": n_rows, "n_repeats": n_repeats, "seed": seed,
        "diff_in_means": dim,
        "feature_names_in_order": names,
        "T_idx": T_idx, "T_perm_idx": Tp_idx,
        "assoc": assoc.tolist(), "multi": multi.tolist(),
        "caveats": {
            "randomization": "within-school; school id (79 levels) not a feature, "
                             "urbanicity dummies are only a coarse school proxy",
            "treatment_binarization": "regular+aide pooled with regular as T=0",
        },
    }

    if model is not None:
        tau_hat, tau_std = _run_probe(model, Xs, ys, n_rows, n_repeats, rng, device)
        ate_err = abs(float(tau_hat[T_idx]) - tau_T_ref)
        t_rank = 1 + int(np.sum(np.abs(tau_hat) > abs(tau_hat[T_idx])))
        print(f"  {'feature':<20s} {'tau_hat':>9s} {'+/-':>7s} {'assoc':>9s} {'multi':>9s}")
        for i, nm in enumerate(names):
            tag = "  <- T" if i == T_idx else ("  <- placebo" if i == Tp_idx else "")
            print(f"  {nm:<20s} {tau_hat[i]:>+9.4f} {tau_std[i]:>7.4f}"
                  f" {assoc[i]:>+9.4f} {multi[i]:>+9.4f}{tag}")
        print(f"\n  tau_hat_T      = {tau_hat[T_idx]:+.4f}  (ref {tau_T_ref:+.4f}, err {ate_err:.4f})")
        print(f"  tau_hat_T_perm = {tau_hat[Tp_idx]:+.4f}  (placebo, expect ~0)")
        print(f"  T rank by |tau_hat| = {t_rank} / {len(names)}")
        res.update({
            "tau_hat": tau_hat.tolist(), "tau_hat_std": tau_std.tolist(),
            "tau_hat_T": float(tau_hat[T_idx]),
            "tau_hat_T_perm": float(tau_hat[Tp_idx]),
            "success": {
                "sign_correct": bool(tau_hat[T_idx] > 0),
                "ATE_error": ate_err,
                "abs_ratio_vs_diff_in_means": float(abs(tau_hat[T_idx]) / max(abs(tau_T_ref), 1e-9)),
                "placebo_abs": float(abs(tau_hat[Tp_idx])),
                "T_rank": t_rank,
            },
        })

    if comp_methods:
        res["comparators"] = _run_comparators_star(
            Xs, ys, T_idx, Tp_idx, names, seed, device, comp_methods, dopfn_repo)
    return res


def evaluate_negative_controls(model, df, meta, n_rows, n_repeats, seed, device,
                               comp_methods=None, dopfn_repo=None) -> dict:
    """Y := each individual-level pre-treatment covariate; expect tau_hat_T ~ 0."""
    rng = np.random.default_rng(seed)
    per_cov = {}
    comp_hdr = "".join(f"{m[:9]:>10s}" for m in (comp_methods or []))
    print(f"\n  {'outcome':<14s} {'tau_hat_T':>10s} {'assoc_T':>9s} {'dim_probe':>10s}"
          + comp_hdr)
    for cov in NEG_CONTROL_COVS:
        X_raw, y_raw, T, names, T_idx, Tp_idx = _build_star_input(
            df, cov, rng, drop_cov=cov)
        Xs, ys = _standardize(X_raw, y_raw)
        dim = _diff_in_means(T, y_raw)
        assoc = BASELINES["marginal"](Xs, ys)
        rec = {
            "assoc_T": float(assoc[T_idx]),
            "diff_in_means_probe_scale": dim["tau_T_probe_scale"],
        }
        tau_T_s = f"{'--':>10s}"
        if model is not None:
            tau_hat, tau_std = _run_probe(model, Xs, ys, n_rows, n_repeats, rng, device)
            rec.update({
                "tau_hat_T": float(tau_hat[T_idx]),
                "tau_hat_T_std": float(tau_std[T_idx]),
                "tau_hat_T_perm": float(tau_hat[Tp_idx]),
            })
            tau_T_s = f"{tau_hat[T_idx]:>+10.4f}"
        comp_s = ""
        if comp_methods:
            comp = run_comparator_suite(
                Xs, ys, T_idx, methods=comp_methods, seed=seed, device=device,
                dopfn_repo=dopfn_repo, extra_cols={"T_perm": Tp_idx},
            )
            rec["comparators"] = comp
            for m in comp_methods:
                r = comp.get(m, {})
                v = r.get("tau_hat_T")
                comp_s += (f"{'ERR':>10s}" if "error" in r
                           else (f"{v:>+10.4f}" if v is not None else f"{'--':>10s}"))
        per_cov[cov] = rec
        print(f"  {cov:<14s} {tau_T_s} {assoc[T_idx]:>+9.4f}"
              f" {dim['tau_T_probe_scale']:>+10.4f}" + comp_s)
    out_extra = {}
    if model is not None:
        abs_taus = [abs(v["tau_hat_T"]) for v in per_cov.values()]
        out_extra["negative_control_MAE"] = float(np.mean(abs_taus))
        print(f"  negative-control MAE (tau_hat_T) = {out_extra['negative_control_MAE']:.4f}")
    if comp_methods:
        comp_mae = {}
        for m in comp_methods:
            vals = [abs(v["comparators"][m]["tau_hat_T"]) for v in per_cov.values()
                    if v["comparators"].get(m, {}).get("tau_hat_T") is not None]
            comp_mae[m] = float(np.mean(vals)) if vals else None
            if comp_mae[m] is not None:
                print(f"  negative-control MAE ({m}) = {comp_mae[m]:.4f}")
        out_extra["negative_control_MAE_comparators"] = comp_mae
    return {
        "dataset": "STAR (AER, kindergarten cross-section) — negative controls",
        "meta": meta,
        "n": int(len(df)),
        "n_rows": n_rows, "n_repeats": n_repeats, "seed": seed,
        "per_covariate": per_cov,
        **out_extra,
        "caveats": {
            "randomization": "within-school randomization: afam/free_lunch vary "
                             "across schools, so T may carry a small marginal "
                             "association with them even under valid randomization; "
                             "use --largest-school for a clean single-school subset",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="required unless --dry-run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--out", default="causal_probe/results/star_rct.json")
    ap.add_argument("--negative-controls", action="store_true",
                    help="Y := each pre-treatment covariate; expect tau_hat_T ~ 0")
    ap.add_argument("--largest-school", action="store_true",
                    help="restrict to the single largest school (clean "
                         "unconditional randomization; small n)")
    ap.add_argument("--dry-run", action="store_true",
                    help="load data, build matrices, print shapes + "
                         "diff-in-means; no GPU/ckpt needed")
    ap.add_argument("--comparators-only", action="store_true",
                    help="run the graph-free comparator suite without the "
                         "probe (no checkpoint / GPU needed)")
    ap.add_argument("--comparators", nargs="+", default=None, choices=ALL_METHODS,
                    help="comparator subset (default: all cheap + slow unless "
                         "--skip-slow; dopfn only with --dopfn-repo)")
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip the slow per-column comparators")
    ap.add_argument("--no-comparators", action="store_true",
                    help="probe only, no comparator suite (previous behaviour)")
    ap.add_argument("--dopfn-repo", default=None,
                    help="checked-out Do-PFN repo (enables the dopfn comparator)")
    args = ap.parse_args()

    print("=== Tennessee STAR (kindergarten): small class vs regular(+aide) ===")
    df, meta = load_star(largest_school_only=args.largest_school)
    print(f"  n = {meta['n_analysis']} "
          f"(csv {meta['n_raw_csv']} -> K-complete {meta['n_with_stark_and_k_scores']}"
          f" -> covariate-complete {meta['n_after_covariate_dropna']}"
          f"{', largest school only' if args.largest_school else ''})")

    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        X_raw, y_raw, T, names, T_idx, Tp_idx = _build_star_input(df, "Y", rng)
        dim = _diff_in_means(T, y_raw)
        print(f"  X shape = {X_raw.shape}, y shape = {y_raw.shape}"
              f" (p+Y = {X_raw.shape[1] + 1} columns, envelope <= 15)")
        print(f"  features: {names} (T at {T_idx}, T_perm at {Tp_idx})")
        print(f"  treated (small) = {dim['n_treated']}, control = {dim['n_control']}")
        print(f"  diff-in-means: {dim['ate_raw_points']:+.2f} raw points"
              f" = {dim['ate_sd_units']:+.4f} sd of Y"
              f" -> {dim['tau_T_probe_scale']:+.4f} on the probe scale"
              f" (2*std(T) = {dim['two_std_T']:.3f})")
        print("  dry-run OK")
        return

    if not args.ckpt and not args.comparators_only:
        raise SystemExit("--ckpt is required unless --dry-run or --comparators-only")

    comp_methods = None
    if not args.no_comparators:
        comp_methods = args.comparators or default_methods(
            skip_slow=args.skip_slow, dopfn_repo=args.dopfn_repo)
        if args.skip_slow:
            comp_methods = [m for m in comp_methods if m not in SLOW_METHODS]

    model = None
    if not args.comparators_only:
        model = _load_model(args.ckpt, device=args.device)

    if args.negative_controls:
        res = evaluate_negative_controls(
            model, df, meta, args.n_rows, args.n_repeats, args.seed, args.device,
            comp_methods=comp_methods, dopfn_repo=args.dopfn_repo)
    else:
        res = evaluate(
            model, df, meta, args.n_rows, args.n_repeats, args.seed, args.device,
            comp_methods=comp_methods, dopfn_repo=args.dopfn_repo)
    res["ckpt"] = args.ckpt
    res["comparator_methods"] = comp_methods

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
