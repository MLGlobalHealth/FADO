"""Twins benchmark (Louizos et al. 2017, CEVAE): binary-outcome attenuation
diagnostic for the probe.

Data: NCHS linked birth / infant-death records 1989-91, same-sex twin pairs,
as packaged by the CEVAE repo (github.com/AMLab-Amsterdam/CEVAE,
datasets/TWINS — plain CSVs, not gzipped, despite folklore). Treatment
T = "being the heavier twin"; both potential outcomes are observed because
1-year mortality is recorded for BOTH twins in every pair. Idea follows
Almond, Chay & Lee (QJE 2005).

Design choices (every one deliberate):

1.  **Cohort filter** — pairs where BOTH twins weigh < 2000 g, exactly the
    CEVAE Sec. 5.2 restriction ("we focused on twins such that both were
    born weighing less than 2 kg"). This yields 11,984 pairs.

2.  **True ATE** — with both potential outcomes observed, the ground truth
    is exact: ATE = mean(mort_1 - mort_0) over pairs (heavier minus
    lighter). On the <2 kg cohort this is ≈ -0.0252 (lighter-twin mortality
    ≈ 18.9 %, heavier ≈ 16.4 %), matching the literature value ≈ -2.5 %.

3.  **Observational construction (--assignment confounded)** — the standard
    CEVAE protocol: per pair i, select which twin is observed via

        t_i | x_i, z_i ~ Bern( sigmoid( w_o' x_i + w_h * (z_i/10 - 0.1) ) ),
        w_o ~ N(0, 0.1 * I),   w_h ~ N(5, 0.1),

    where z_i = GESTAT10 (gestation-length decile; the file codes it 1..10,
    we shift to 0..9 to match CEVAE's category indexing) and x_i are the
    remaining covariates (CEVAE Sec. 5.2, arXiv:1705.08821). Deviation from
    CEVAE: their w_o acts on all 45 remaining raw features; ours acts on
    the 11 standardized non-gestat covariates of our subset (below), which
    preserves the mechanism (gestation is the dominant, hidden-able
    confounder; the rest add weak covariate-dependent selection) inside our
    <= 15-column envelope. t_i = 1 observes the heavier twin (Y = mort_1,
    T = 1); t_i = 0 the lighter (Y = mort_0, T = 0). Because longer
    gestation lowers mortality and raises P(T=1), the naive diff-in-means
    is biased away from the true ATE. **--assignment random** draws
    t_i ~ Bern(0.5) instead: the clean RCT-like case.

4.  **Covariates** — 12 pair-level (shared) pre-treatment covariates chosen
    by three documented criteria: (i) pair-level only — no _0/_1
    twin-specific columns (those encode the treatment ordering);
    (ii) missingness <= 11 % so complete-case analysis keeps >= 75 % of
    pairs (this excludes tobacco/alcohol/cigar6/drink5/dfageq/feduc6, all
    24-28 % missing); (iii) prognostic relevance per Almond et al. 2005
    (gestation, maternal demographics, prenatal care, risk factors,
    parity). The set: gestat10, csex, mager8, meduc6, dmar, mrace, mpre5,
    nprevistq, adequacy, phyper, preterm, dtotord_min. Rows with any NaN in
    these are dropped (complete case; 9,054 of 11,984 pairs survive).
    **--config with-gestat** feeds the probe T + all 12 (14 columns incl.
    Y); **--config without-gestat** hides gestat10 (13 columns) while the
    confounded assignment still uses it — the hidden-confounding probe.
    Both configs see the *identical* observational sample per repeat, so
    their difference isolates the effect of observing the confounder.

5.  **Outcome handling** — Y is binary 0/1 mortality of the selected twin,
    z-scored exactly like hillstrom_rct.py z-scores the binary `visit`
    outcome (plain (y - mean)/std, no special binary treatment). Columns of
    X likewise z-scored. T sits at column 0 (the probe is
    column-symmetric; ihdp_eval.py uses the same convention, no shuffle,
    no decoy columns — this harness isolates attenuation, not decoys).

6.  **Reporting scale** — identical to ihdp_eval.py / Gelman (2008):
    tau_true_std = (ATE_raw / std_y) * 2 * std_T, with std_y and std_T
    taken from the constructed observational sample of each repeat. The
    probe's tau_hat_T is read off column 0 directly (already on this
    scale). **Headline diagnostic: attenuation ratio
    mean(tau_hat_T) / mean(tau_true_std).**

7.  **Repeats** — n_repeats (default 5). Each repeat redraws the treatment
    assignment (fresh w_o, w_h, Bernoulli draws) AND the seeded n_rows=512
    context subsample, so the reported sd covers both assignment and
    subsampling variance. tau_true_std / naive diff-in-means are computed
    on each repeat's full observational sample (not the 512-row context).
    Baselines (marginal, multivariate from causal_probe.baselines) run on
    the full standardized sample per repeat.

8.  **Rare-outcome caveat** — 1-year mortality in the full same-sex-twin
    cohort is ~2.5-3.9 %, but the <2 kg CEVAE cohort is far less rare
    (~16-19 %). Both base rates land in the JSON
    (base_rate_full_cohort_*, base_rate_lt2kg_*, base_rate_observed);
    still, Y is heavily skewed binary, so std_y ≈ 0.37 and the
    standardized tau_true is small (~ -0.07) — attenuation here probes the
    probe's behaviour on skewed binary outcomes, not just effect recovery.

9.  **Comparator suite** — each repeat's standardized construction is also
    scored by the graph-free comparators in causal_probe.comparator_suite
    (SHAP, permutation importance, LiNGAM, DoubleML, causal forest,
    DR-Learner, CausalPFN, Do-PFN) at the T column (col 0), for both
    assignment modes and both configs; mean±sd across repeats is
    reported per method. No graph-oracle methods appear: no ground-truth
    causal graph exists for the Twins data. Slow per-column methods sit
    behind --skip-slow; --comparators-only runs the suite without a
    checkpoint.

Usage:
    uv run python -m causal_probe.twins_eval --dry-run
    uv run python -m causal_probe.twins_eval --ckpt <probe.ckpt> \
        --assignment confounded --config both
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from causal_probe.baselines import BASELINES
from causal_probe.comparator_suite import (
    ALL_METHODS,
    SLOW_METHODS,
    default_methods,
    run_comparator_suite,
)

BASE_URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS"
FILES = {
    "T": "twin_pairs_T_3years_samesex.csv",
    "X": "twin_pairs_X_3years_samesex.csv",
    "Y": "twin_pairs_Y_3years_samesex.csv",
}

# Pair-level covariates (criteria in module docstring). gestat10 first so
# the with/without-gestat configs are a simple slice.
COVARIATES = [
    "gestat10",     # gestation deciles — THE confounder (CEVAE hidden z)
    "csex",         # sex of the twins (same-sex cohort: pair-level)
    "mager8",       # mother age (8 categories)
    "meduc6",       # mother education (6 categories)
    "dmar",         # mother married
    "mrace",        # mother race
    "mpre5",        # trimester prenatal care begun (4 = none)
    "nprevistq",    # quintile of number of prenatal visits
    "adequacy",     # adequacy-of-care index
    "phyper",       # pregnancy-associated hypertension
    "preterm",      # previous preterm/small birth
    "dtotord_min",  # total births before the twins (parity)
]
CONFIGS = {
    "with-gestat": COVARIATES,
    "without-gestat": [c for c in COVARIATES if c != "gestat10"],
}


def _fetch_csv(name: str, data_dir: Path):
    import pandas as pd
    path = data_dir / name
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{name}"
        print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return pd.read_csv(path, index_col=0)


def load_twins(data_dir: Path):
    """Return (X_cov (n,12), y0, y1, base_rates dict) on the <2kg
    complete-case cohort. y0/y1 = 1-year mortality of lighter/heavier twin."""
    dfT = _fetch_csv(FILES["T"], data_dir)
    dfX = _fetch_csv(FILES["X"], data_dir)
    dfY = _fetch_csv(FILES["Y"], data_dir)

    base_full = {
        "base_rate_full_cohort_lighter": float(dfY["mort_0"].mean()),
        "base_rate_full_cohort_heavier": float(dfY["mort_1"].mean()),
    }
    lt2kg = (dfT["dbirwt_0"] < 2000.0) & (dfT["dbirwt_1"] < 2000.0)
    keep = lt2kg & dfX[COVARIATES].notna().all(axis=1)
    Xc = dfX.loc[keep, COVARIATES].to_numpy(dtype=np.float64)
    y0 = dfY.loc[keep, "mort_0"].to_numpy(dtype=np.float64)
    y1 = dfY.loc[keep, "mort_1"].to_numpy(dtype=np.float64)
    rates = {
        **base_full,
        "n_pairs_lt2kg": int(lt2kg.sum()),
        "n_pairs_complete_case": int(keep.sum()),
        "base_rate_lt2kg_lighter": float(dfY.loc[lt2kg, "mort_0"].mean()),
        "base_rate_lt2kg_heavier": float(dfY.loc[lt2kg, "mort_1"].mean()),
    }
    return Xc, y0, y1, rates


def draw_assignment(X_cov: np.ndarray, assignment: str, rng: np.random.Generator) -> np.ndarray:
    """CEVAE confounded selection or Bern(0.5). Returns t in {0,1}: 1 = observe heavier twin."""
    n = X_cov.shape[0]
    if assignment == "random":
        return (rng.random(n) < 0.5).astype(np.float64)
    if assignment != "confounded":
        raise ValueError(f"unknown assignment {assignment!r}")
    z = X_cov[:, COVARIATES.index("gestat10")] - 1.0          # file codes 1..10 -> 0..9
    others = np.delete(X_cov, COVARIATES.index("gestat10"), axis=1)
    others_std = (others - others.mean(axis=0)) / others.std(axis=0).clip(min=1e-9)
    w_o = rng.normal(0.0, np.sqrt(0.1), size=others_std.shape[1])   # N(0, 0.1*I)
    w_h = rng.normal(5.0, np.sqrt(0.1))                              # N(5, 0.1)
    logits = others_std @ w_o + w_h * (z / 10.0 - 0.1)
    p = 1.0 / (1.0 + np.exp(-logits))
    return (rng.random(n) < p).astype(np.float64)


def build_observational(X_cov, y0, y1, t, cov_names):
    """Assemble [T | covariates] and observed Y for one assignment draw."""
    sel = [COVARIATES.index(c) for c in cov_names]
    X = np.concatenate([t.reshape(-1, 1), X_cov[:, sel]], axis=1)
    y = np.where(t > 0.5, y1, y0)
    names = ["T"] + list(cov_names)
    return X, y, names


def evaluate(args) -> dict:
    import torch
    from causal_probe.eval import _load_model

    X_cov, y0, y1, rates = load_twins(Path(args.data_dir))
    n = X_cov.shape[0]
    true_ate_raw = float((y1 - y0).mean())
    configs = list(CONFIGS) if args.config == "both" else [args.config]

    model = None
    if not args.dry_run and not args.comparators_only:
        model = _load_model(args.ckpt, device=args.device)

    comp_methods = getattr(args, "comp_methods", None) if not args.dry_run else None

    rng = np.random.default_rng(args.seed)
    out = {
        "ckpt": args.ckpt, "assignment": args.assignment, "seed": args.seed,
        "n_rows": args.n_rows, "n_repeats": args.n_repeats,
        "comparator_methods": comp_methods,
        "true_ATE_raw": true_ate_raw, **rates, "configs": {},
    }
    print(f"  cohort: {n} complete-case pairs "
          f"(of {rates['n_pairs_lt2kg']} <2kg pairs)")
    print(f"  base rate lighter/heavier (<2kg): "
          f"{rates['base_rate_lt2kg_lighter']:.4f} / {rates['base_rate_lt2kg_heavier']:.4f}"
          f"   (full cohort: {rates['base_rate_full_cohort_lighter']:.4f} / "
          f"{rates['base_rate_full_cohort_heavier']:.4f})")
    print(f"  true ATE (raw, heavier-lighter) = {true_ate_raw:+.4f}")

    # Per-repeat assignment draws are shared across configs so with/without
    # gestat differ only in what the model sees.
    per_cfg = {c: {"tau_hat_T": [], "tau_true_std": [], "naive_std": [],
                   "assoc_T": [], "multi_T": [], "per_feature": {},
                   "comparators": {m: [] for m in (comp_methods or [])},
                   "comparator_errors": {m: [] for m in (comp_methods or [])}}
               for c in configs}
    naive_raws, treated_fracs, obs_rates = [], [], []

    for r in range(args.n_repeats):
        t = draw_assignment(X_cov, args.assignment, rng)
        treated_fracs.append(float(t.mean()))
        sub_idx = (rng.choice(n, size=args.n_rows, replace=False)
                   if n > args.n_rows else np.arange(n))
        y_obs_full = np.where(t > 0.5, y1, y0)
        obs_rates.append(float(y_obs_full.mean()))
        naive_raw = float(y_obs_full[t > 0.5].mean() - y_obs_full[t < 0.5].mean())
        naive_raws.append(naive_raw)

        for cfg in configs:
            X_raw, y_raw, names = build_observational(X_cov, y0, y1, t, CONFIGS[cfg])
            Xs = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0).clip(min=1e-9)
            ys = (y_raw - y_raw.mean()) / y_raw.std().clip(min=1e-9)
            std_y = float(y_raw.std().clip(min=1e-9))
            std_T = float(t.std().clip(min=1e-9))
            tau_true_std = (true_ate_raw / std_y) * 2.0 * std_T
            naive_std = (naive_raw / std_y) * 2.0 * std_T
            d = per_cfg[cfg]
            d["tau_true_std"].append(tau_true_std)
            d["naive_std"].append(naive_std)
            assoc = BASELINES["marginal"](Xs, ys)
            multi = BASELINES["multivariate"](Xs, ys)
            d["assoc_T"].append(float(assoc[0]))
            d["multi_T"].append(float(multi[0]))
            if model is not None:
                Xc, yc = Xs[sub_idx], ys[sub_idx]
                X_t = torch.from_numpy(Xc.astype(np.float32)).unsqueeze(0).to(args.device)
                y_t = torch.from_numpy(yc.astype(np.float32)).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    pred = model(X_t, y_t).squeeze(0).cpu().numpy()
                d["tau_hat_T"].append(float(pred[0]))
                for i, nm in enumerate(names):
                    d["per_feature"].setdefault(nm, []).append(float(pred[i]))
            if comp_methods:
                comp = run_comparator_suite(
                    Xs, ys, 0, methods=comp_methods,
                    seed=args.seed * 1000 + r, device=args.device,
                    dopfn_repo=args.dopfn_repo,
                )
                for m, cr in comp.items():
                    if "error" in cr:
                        d["comparator_errors"][m].append(cr["error"])
                    elif cr.get("tau_hat_T") is not None:
                        d["comparators"][m].append(cr["tau_hat_T"])

    out["treated_frac_mean"] = float(np.mean(treated_fracs))
    out["base_rate_observed_mean"] = float(np.mean(obs_rates))
    out["naive_diff_in_means_raw_mean"] = float(np.mean(naive_raws))

    for cfg in configs:
        d = per_cfg[cfg]
        tt = np.asarray(d["tau_true_std"])
        nv = np.asarray(d["naive_std"])
        c = {
            "covariates": CONFIGS[cfg],
            "p_total_incl_Y": len(CONFIGS[cfg]) + 2,
            "tau_true_std_mean": float(tt.mean()), "tau_true_std_sd": float(tt.std()),
            "naive_std_mean": float(nv.mean()), "naive_std_sd": float(nv.std()),
            "assoc_T_mean": float(np.mean(d["assoc_T"])),
            "multi_T_mean": float(np.mean(d["multi_T"])),
        }
        if d["tau_hat_T"]:
            th = np.asarray(d["tau_hat_T"])
            c["tau_hat_T_mean"] = float(th.mean())
            c["tau_hat_T_sd"] = float(th.std())
            c["attenuation_ratio"] = float(th.mean() / tt.mean())
            c["ATE_error"] = float(abs(th.mean() - tt.mean()))
            c["per_feature"] = {nm: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                                for nm, v in d["per_feature"].items()}
        if comp_methods:
            comp_agg = {}
            for m in comp_methods:
                vals = d["comparators"][m]
                errs = d["comparator_errors"][m]
                agg = {"n": len(vals), "n_errors": len(errs)}
                if vals:
                    agg["tau_hat_T_mean"] = float(np.mean(vals))
                    agg["tau_hat_T_sd"] = float(np.std(vals))
                    agg["attenuation_ratio"] = float(np.mean(vals) / tt.mean())
                if errs:
                    agg["errors"] = errs[:3]  # first few, enough to diagnose
                comp_agg[m] = agg
            c["comparators"] = comp_agg
        out["configs"][cfg] = c

        print(f"\n  --- config: {cfg} (p={c['p_total_incl_Y']} cols incl. Y, "
              f"assignment={args.assignment}) ---")
        print(f"  tau_true (std scale)    = {c['tau_true_std_mean']:+.4f} "
              f"± {c['tau_true_std_sd']:.4f}")
        print(f"  naive diff-in-means     = {c['naive_std_mean']:+.4f} "
              f"± {c['naive_std_sd']:.4f}  (std scale)")
        print(f"  marginal/multivariate @T= {c['assoc_T_mean']:+.4f} / {c['multi_T_mean']:+.4f}")
        if "tau_hat_T_mean" in c:
            print(f"  tau_hat_T (probe)       = {c['tau_hat_T_mean']:+.4f} "
                  f"± {c['tau_hat_T_sd']:.4f}")
            print(f"  attenuation ratio       = {c['attenuation_ratio']:+.3f}"
                  f"   (ATE err {c['ATE_error']:.4f})")
        for m, agg in c.get("comparators", {}).items():
            if agg["n"] == 0:
                print(f"  tau_hat_T ({m:<13s})  = all failed "
                      f"({agg['n_errors']} errors)")
            else:
                print(f"  tau_hat_T ({m:<13s})  = {agg['tau_hat_T_mean']:+.4f} "
                      f"± {agg['tau_hat_T_sd']:.4f}"
                      f"   (atten {agg['attenuation_ratio']:+.3f}"
                      + (f", {agg['n_errors']} errors" if agg["n_errors"] else "")
                      + ")")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None, help="probe checkpoint (required unless --dry-run)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--data-dir", default="/tmp/twins_cevae",
                    help="cache dir for the CEVAE TWINS csvs (outside git)")
    ap.add_argument("--assignment", default="confounded", choices=["confounded", "random"])
    ap.add_argument("--config", default="both",
                    choices=["with-gestat", "without-gestat", "both"])
    ap.add_argument("--out", default="causal_probe/results/twins_eval.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="download + build matrices + print true ATE/base rates; no model")
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
    if not args.dry_run and not args.comparators_only and not args.ckpt:
        ap.error("--ckpt is required unless --dry-run or --comparators-only")

    args.comp_methods = None
    if not args.no_comparators:
        args.comp_methods = args.comparators or default_methods(
            skip_slow=args.skip_slow, dopfn_repo=args.dopfn_repo)
        if args.skip_slow:
            args.comp_methods = [m for m in args.comp_methods
                                 if m not in SLOW_METHODS]

    print(f"=== Twins (CEVAE) — assignment={args.assignment}, config={args.config} ===")
    res = evaluate(args)

    if args.dry_run:
        for cfg, c in res["configs"].items():
            print(f"  [dry-run] {cfg}: matrix shape "
                  f"({min(res['n_pairs_complete_case'], args.n_rows)}, "
                  f"{c['p_total_incl_Y'] - 1}) + y — OK")
        print("  [dry-run] no model run; exiting 0")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
