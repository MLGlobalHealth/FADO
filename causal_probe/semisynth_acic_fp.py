"""ACIC-2016 semi-synthetic benchmark: FADO + regression baselines + the full
comparator suite (graph-free methods and graph-oracle Shapley skylines),
with per-decoy-kind FP columns (mean |tau_hat| at proxy / leak / noise /
conf-child positions; truth is 0 there by construction).

RNG structure is copied verbatim from causal_probe/semisynth_acic_compare.py
(origin/fix/rct-bench-eval @ c986a48..455cf4a), so the pooled Pearson r per family
MUST reproduce the reference ACIC tables for FADO / marginal / multivariate.
The script checks this and marks each family PASS/FAIL. A FAIL means we are NOT
on the same benchmark as the reference run -- stop and investigate, do not use the FP
numbers.

Beyond the FADO / marginal / multivariate trio, run_regime reports the full
comparator suite (--comparators, default all):
  * graph-free (same standardized (X, y) rows FADO sees, via
    causal_probe.run_baseline.METHODS): shap, permutation, lingam, doubleml,
    causal_forest, causalpfn, dopfn (needs --dopfn-repo). Slow ones are
    droppable with --skip-slow; per-instance errors are recorded and skipped.
  * graph-ORACLE skylines (true decoy-augmented graph + mechanisms via each
    instance's semi_synth_real.DecoyOracleSCM, on a seeded --n-scms-oracle
    subsample): do_shapley, causal_shapley. do-Shapley should score every
    decoy at exactly 0 (no decoy has a path INTO Y) — the skyline FADO is
    compared to — while Causal Shapley (explaining a fitted predictor) loads
    on the leak.
--comparators-only runs the suite without a FADO checkpoint (replication
check disabled).

Run (both regimes, one command):
  PYTHONPATH=<bundle_dir> python semisynth_acic_fp.py \
      --ckpt <path>/probe_intersample_L_seed0.ckpt \
      --acic-path <path>/x.csv --device cuda \
      --out-json acic_fp_results.json
"""
from __future__ import annotations
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ID_COLS = {"sample_id", "idx", "id", "index"}

# reference-run point estimates (rounded to 2 dp there)
EXPECTED = {
    (False, "linear"):          {"fado": 0.61, "marginal": 0.61, "multivariate": 0.69},
    (False, "mixed_nonlinear"): {"fado": 0.41, "marginal": 0.31, "multivariate": 0.08},
    (False, "interaction"):     {"fado": 0.53, "marginal": 0.54, "multivariate": 0.27},
    (True,  "linear"):          {"fado": 0.60, "marginal": 0.53, "multivariate": 0.39},
    (True,  "mixed_nonlinear"): {"fado": 0.38, "marginal": 0.29, "multivariate": 0.06},
    (True,  "interaction"):     {"fado": 0.47, "marginal": 0.43, "multivariate": 0.15},
}
TOL = 0.006  # reference values were rounded to 2 dp

DECOY_KINDS = ["proxy", "leak", "noise", "conf"]
METHODS = ["fado", "marginal", "multivariate"]

# --- full comparator suite -------------------------------------------------
# Graph-free methods see the SAME standardized (X, y) rows FADO sees
# (Xs[idx], ys[idx]); dispatched through causal_probe.run_baseline.METHODS.
# Oracle methods get the TRUE decoy-augmented graph + mechanisms via each
# instance's DecoyOracleSCM (semi_synth_real) and are run on a seeded
# --n-scms-oracle subsample of instances.
GRAPH_FREE_COMPARATORS = ["shap", "shap_interventional", "shap_kernel",
                          "permutation", "lingam", "doubleml",
                          "causal_forest", "causalpfn", "dopfn"]
ORACLE_COMPARATORS = ["do_shapley", "causal_shapley"]
SLOW_COMPARATORS = {"shap_kernel", "doubleml", "causal_forest", "causalpfn", "dopfn"}
DEFAULT_COMPARATORS = ",".join(GRAPH_FREE_COMPARATORS + ORACLE_COMPARATORS)

_DOPFN_CACHE: dict = {}


def _dopfn_tau_xy(X, y, *, dopfn_repo, device):
    """Do-PFN per-column through the existing baseline, with the cwd pinned to
    the Do-PFN checkout for the duration of the call (its artifact paths are
    relative)."""
    from causal_probe.dopfn_per_column_baseline import _load_dopfn, dopfn_tau
    cwd = os.getcwd()
    os.chdir(dopfn_repo)
    try:
        if "model" not in _DOPFN_CACHE:
            _DOPFN_CACHE["model"] = _load_dopfn(dopfn_repo, device=device)
        return dopfn_tau(_DOPFN_CACHE["model"], X, y)
    finally:
        os.chdir(cwd)


def _resolve_comparators(args):
    """Return (graph_free, oracle) comparator name lists from args, applying
    --skip-slow and dropping dopfn when no --dopfn-repo is configured."""
    spec = getattr(args, "comparators", "") or ""
    names = [c.strip() for c in spec.split(",") if c.strip()]
    known = set(GRAPH_FREE_COMPARATORS) | set(ORACLE_COMPARATORS)
    bad = [c for c in names if c not in known]
    if bad:
        raise SystemExit(f"unknown comparators {bad}; known: {sorted(known)}")
    if getattr(args, "skip_slow", False):
        dropped = [c for c in names if c in SLOW_COMPARATORS]
        if dropped:
            print(f"  --skip-slow: dropping {dropped}")
        names = [c for c in names if c not in SLOW_COMPARATORS]
    if "dopfn" in names and not getattr(args, "dopfn_repo", None):
        print("  dopfn requested but no --dopfn-repo given; skipping dopfn")
        names = [c for c in names if c != "dopfn"]
    return ([c for c in names if c in GRAPH_FREE_COMPARATORS],
            [c for c in names if c in ORACLE_COMPARATORS])


def _run_graph_free(name, X, y, s, args):
    """One graph-free comparator on one instance's (X, y). Raises on failure;
    the caller records the error per instance."""
    if name == "dopfn":
        return _dopfn_tau_xy(X.astype(np.float64), y.astype(np.float64),
                             dopfn_repo=args.dopfn_repo,
                             device=getattr(args, "device", "cpu"))
    from causal_probe.run_baseline import METHODS as RB_METHODS
    kw = {}
    if name == "permutation":
        kw["seed"] = 1000 + s
    elif name == "shap_kernel":
        kw["seed"] = 1000 + s
    elif name == "causalpfn":
        kw["device"] = getattr(args, "device", "cpu")
    tau = np.asarray(RB_METHODS[name](X.astype(np.float64),
                                      y.astype(np.float64), **kw), dtype=float)
    if not np.isfinite(tau).any():
        raise RuntimeError("all-NaN prediction")
    return tau


def add_comparator_args(ap):
    """Shared CLI flags for the comparator suite (also used by
    causal_probe/ihdp_decoy_eval.py)."""
    ap.add_argument("--comparators", default=DEFAULT_COMPARATORS,
                    help="comma list from "
                         f"{GRAPH_FREE_COMPARATORS + ORACLE_COMPARATORS}; "
                         "empty string disables the extra suite")
    ap.add_argument("--skip-slow", action="store_true",
                    help=f"drop the slow comparators {sorted(SLOW_COMPARATORS)}")
    ap.add_argument("--comparators-only", action="store_true",
                    help="no FADO checkpoint: run baselines + comparators only "
                         "(replication check disabled)")
    ap.add_argument("--n-scms-oracle", type=int, default=10,
                    help="seeded subsample of SCM instances for the "
                         "do-/Causal-Shapley oracle columns")
    ap.add_argument("--oracle-n-mc", type=int, default=2000,
                    help="MC rows per interventional mean in the oracle Shapley")
    ap.add_argument("--oracle-n-perm", type=int, default=256,
                    help="sampled permutations when p > 12 (exact below)")
    ap.add_argument("--dopfn-repo", default=None,
                    help="path to a Do-PFN checkout (dopfn skipped if unset)")
    return ap


def load_acic2016(path: str) -> np.ndarray:
    # identical to semisynth_acic_compare.py
    df = pd.read_csv(path)
    cols = []
    for c in df.columns:
        if str(c).lower() in _ID_COLS:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype("category").cat.codes.astype(float)
        else:
            s = s.astype(float)
        if s.std() > 0:
            cols.append(s.values)
    return np.column_stack(cols)


def _pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _ci(draws):
    d = np.asarray([x for x in draws if np.isfinite(x)], dtype=float)
    if d.size == 0:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def run_regime(model, Xacic, args, hidden_confound, torch, expected=None):
    """Run all outcome families for one confounding regime.

    ``expected`` maps (hidden_confound, family) -> per-method pooled-r point
    estimates for the replication check. Defaults to the reference ACIC run
    table (EXPECTED); pass {} to disable the check for non-ACIC covariate
    sources (e.g. causal_probe/ihdp_decoy_eval.py).
    """
    from causal_probe.semi_synth_real import make_benchmark
    from causal_probe.baselines import BASELINES

    if expected is None:
        expected = EXPECTED
    if model is None:
        expected = {}  # no FADO forward pass -> no replication check

    graph_free, oracle = _resolve_comparators(args)
    comp_all = graph_free + oracle
    n_oracle = min(getattr(args, "n_scms_oracle", 10), args.n_scms)
    oracle_set = (set(np.random.default_rng(args.seed + 4242)
                      .choice(args.n_scms, size=n_oracle, replace=False).tolist())
                  if oracle else set())
    if oracle:
        from causal_probe.shapley_causal import both_shapley

    def _pool(pred_list, true_list, idx):
        P = np.concatenate([pred_list[i] for i in idx])
        T = np.concatenate([true_list[i] for i in idx])
        return _pearson(P, T)

    def _fp_pool(pred_list, kinds_list, kind, idx):
        vals = []
        for i in idx:
            mask = np.array([k == kind for k in kinds_list[i]])
            if mask.any():
                v = np.abs(np.asarray(pred_list[i])[mask])
                vals.append(v[np.isfinite(v)])
        if not vals:
            return float("nan")
        allv = np.concatenate(vals)
        return float(np.mean(allv)) if allv.size else float("nan")

    out = []
    fam_data = {}
    n_boot = args.n_boot
    for family in args.families.split(","):
        family = family.strip()
        # --- identical RNG structure to semisynth_acic_compare.py ---
        # (the comparator suite below uses only its OWN rngs / the instances'
        # oracle SCM objects, so the fado/marginal/multivariate stream -- and
        # the replication check -- is unchanged by adding comparators)
        fado_p, marg_p, mult_p, true_all, kinds_all = [], [], [], [], []
        comp_pred = {m: [None] * args.n_scms for m in comp_all}
        comp_err = {m: {} for m in comp_all}
        colrng = np.random.default_rng(args.seed)
        for s in range(args.n_scms):
            cols = colrng.choice(Xacic.shape[1], args.p_real_cap, replace=False)
            X = Xacic[:, cols]
            b = make_benchmark(X, np.random.default_rng(1000 * args.seed + s),
                               n_causal=args.n_causal, outcome_family=family,
                               hidden_confound=hidden_confound)
            X_full, y, tau = b["X_full"], b["y"], b["true_tau"]
            n = X_full.shape[0]
            rng = np.random.default_rng(s)
            idx = rng.choice(n, args.n_rows, replace=False) if n > args.n_rows else np.arange(n)
            Xs = (X_full - X_full.mean(0)) / X_full.std(0).clip(min=1e-9)
            ys = (y - y.mean()) / (y.std() or 1)
            if model is None:
                fp = np.full(X_full.shape[1], np.nan)
            else:
                with torch.no_grad():
                    fp = model(torch.tensor(Xs[idx][None], dtype=torch.float32, device=args.device),
                               torch.tensor(ys[idx][None], dtype=torch.float32, device=args.device)
                               ).squeeze(0).cpu().numpy()
            fado_p.append(fp)
            marg_p.append(BASELINES["marginal"](Xs, ys))
            mult_p.append(BASELINES["multivariate"](Xs, ys))
            true_all.append(tau)
            kinds_all.append(b["kinds"])

            # graph-free comparators: same standardized rows FADO sees
            for m in graph_free:
                try:
                    comp_pred[m][s] = _run_graph_free(m, Xs[idx], ys[idx], s, args)
                except Exception as e:
                    comp_err[m][s] = f"{type(e).__name__}: {e}"
            # oracle skylines: true decoy-augmented graph + mechanisms, on a
            # seeded instance subsample (one shared coalition pass for both)
            if oracle and s in oracle_set:
                try:
                    phi_do, phi_cs = both_shapley(
                        b["scm"], n_mc=args.oracle_n_mc,
                        n_perm=getattr(args, "oracle_n_perm", 256),
                        rng=np.random.default_rng(90_000 * (args.seed % 997) + s))
                    if "do_shapley" in oracle:
                        comp_pred["do_shapley"][s] = phi_do
                    if "causal_shapley" in oracle:
                        comp_pred["causal_shapley"][s] = phi_cs
                except Exception as e:
                    for m in oracle:
                        comp_err[m][s] = f"{type(e).__name__}: {e}"

        F = np.concatenate(fado_p)
        Mg = np.concatenate(marg_p)
        Mv = np.concatenate(mult_p)
        T = np.concatenate(true_all)
        fam_data[family] = {"fado": fado_p, "marg": marg_p, "mult": mult_p,
                            "true": true_all, "kinds": kinds_all}

        # per-family bootstrap over SCM instances: r CIs (same brng seed as original)
        # plus FP CIs computed from the SAME bidx draws.
        brng = np.random.default_rng(777)
        dr = {"fado": [], "marg": [], "mult": []}
        dfp = {(m, k): [] for m in METHODS for k in DECOY_KINDS}
        ml = {"fado": fado_p, "marg": marg_p, "mult": mult_p}
        mkey = {"fado": "fado", "marginal": "marg", "multivariate": "mult"}
        for _ in range(n_boot):
            bidx = brng.integers(0, args.n_scms, args.n_scms)
            for m in dr:
                dr[m].append(_pool(ml[m], true_all, bidx))
            for meth in METHODS:
                for k in DECOY_KINDS:
                    dfp[(meth, k)].append(_fp_pool(ml[mkey[meth]], kinds_all, k, bidx))

        full = np.arange(args.n_scms)
        rec = {"source": args.source_label, "family": family, "n_scms": args.n_scms,
               "p_real_cap": args.p_real_cap, "hidden_confound": bool(hidden_confound),
               "fado_pearson": round(_pearson(F, T), 3), "fado_ci": _ci(dr["fado"]),
               "marginal_pearson": round(_pearson(Mg, T), 3), "marginal_ci": _ci(dr["marg"]),
               "multivariate_pearson": round(_pearson(Mv, T), 3), "multivariate_ci": _ci(dr["mult"])}
        for meth in METHODS:
            for k in DECOY_KINDS:
                pt = _fp_pool(ml[mkey[meth]], kinds_all, k, full)
                if np.isfinite(pt):
                    rec[f"{meth}_fp_{k}"] = round(pt, 4)
                    rec[f"{meth}_fp_{k}_ci"] = _ci(dfp[(meth, k)])

        # --- comparator suite: pooled r + per-decoy-kind FP, CIs bootstrapped
        # over each method's AVAILABLE instances (oracle: the seeded subset;
        # graph-free: instances that did not error) ---
        for mi, m in enumerate(comp_all):
            avail = [s for s in range(args.n_scms) if comp_pred[m][s] is not None]
            rec[f"{m}_n_scms"] = len(avail)
            if comp_err[m]:
                rec[f"{m}_errors"] = {str(s): msg for s, msg in comp_err[m].items()}
            if not avail:
                continue
            pl = [comp_pred[m][s] for s in avail]
            tl = [true_all[s] for s in avail]
            kl = [kinds_all[s] for s in avail]
            sub = np.arange(len(avail))
            rec[f"{m}_pearson"] = round(_pool(pl, tl, sub), 3)
            cbrng = np.random.default_rng(778_000 + 131 * mi)
            dr_m, dfp_m = [], {k: [] for k in DECOY_KINDS}
            for _ in range(n_boot):
                bidx = cbrng.integers(0, len(avail), len(avail))
                dr_m.append(_pool(pl, tl, bidx))
                for k in DECOY_KINDS:
                    dfp_m[k].append(_fp_pool(pl, kl, k, bidx))
            rec[f"{m}_ci"] = _ci(dr_m)
            for k in DECOY_KINDS:
                pt = _fp_pool(pl, kl, k, sub)
                if np.isfinite(pt):
                    rec[f"{m}_fp_{k}"] = round(pt, 4)
                    rec[f"{m}_fp_{k}_ci"] = _ci(dfp_m[k])

        # replication check against the reference ACIC run
        exp = expected.get((bool(hidden_confound), family))
        if exp:
            ok = all(abs(rec[f"{m}_pearson"] - exp[m]) <= TOL for m in exp)
            rec["replication_check"] = "PASS" if ok else "FAIL"
        out.append(rec)

        chk = rec.get("replication_check", "n/a")
        print(f"[hidden={hidden_confound}] {family:<16} r: "
              f"FADO {rec['fado_pearson']:.3f} marg {rec['marginal_pearson']:.3f} "
              f"multi {rec['multivariate_pearson']:.3f}  | replication {chk}")
        for meth in METHODS:
            fps = "  ".join(f"{k}={rec.get(f'{meth}_fp_{k}', float('nan')):.3f}"
                            for k in DECOY_KINDS if f"{meth}_fp_{k}" in rec)
            print(f"    FP mean|tau_hat| {meth:<12} {fps}")
        for m in comp_all:
            if f"{m}_pearson" not in rec:
                print(f"    {m:<15} FAILED on all {args.n_scms} instances")
                continue
            fps = "  ".join(f"{k}={rec.get(f'{m}_fp_{k}', float('nan')):.3f}"
                            for k in DECOY_KINDS if f"{m}_fp_{k}" in rec)
            nerr = len(comp_err[m])
            tag = f" ({nerr} errored)" if nerr else ""
            print(f"    {m:<15} r={rec[f'{m}_pearson']:+.3f} "
                  f"[n={rec[f'{m}_n_scms']}]{tag}  FP: {fps}")
        if "do_shapley_fp_noise" in rec and rec["do_shapley_fp_noise"] > 1e-6:
            print("    WARNING: do-Shapley FP on pure-noise decoys should be ~0 "
                  f"(got {rec['do_shapley_fp_noise']}); the augmented-graph "
                  "extension is suspect")
        if "do_shapley_pearson" in rec and rec["do_shapley_pearson"] < 0.9:
            print("    WARNING: do-Shapley r vs true tau should be near 1 on "
                  f"the oracle graph (got {rec['do_shapley_pearson']}); the "
                  "augmented-graph extension is suspect")

    # --- mean over families + FADO-minus-marginal paired diff (same seed as original) ---
    fams = list(fam_data.keys())
    brng = np.random.default_rng(20260731)
    agg = {"fado": [], "marg": [], "mult": []}
    d_fm = []
    for _ in range(n_boot):
        pf = {"fado": [], "marg": [], "mult": []}
        for fam in fams:
            dd = fam_data[fam]
            bidx = brng.integers(0, len(dd["true"]), len(dd["true"]))
            for m in pf:
                pf[m].append(_pool(dd[m], dd["true"], bidx))
        mu = {m: float(np.nanmean(pf[m])) for m in pf}
        for m in agg:
            agg[m].append(mu[m])
        d_fm.append(mu["fado"] - mu["marg"])

    def _pt(key):
        return round(float(np.nanmean([r[key] for r in out])), 3)

    def _excl0(draws):
        d = np.asarray([x for x in draws if np.isfinite(x)], dtype=float)
        return bool(d.size and np.percentile(d, 2.5) > 0)

    summary = {"SUMMARY": True, "source": args.source_label,
               "hidden_confound": bool(hidden_confound),
               "n_scms": args.n_scms, "n_families": len(fams), "n_boot": n_boot,
               "mean_pearson": {"fado": _pt("fado_pearson"),
                                "marginal": _pt("marginal_pearson"),
                                "multivariate": _pt("multivariate_pearson")},
               "mean_ci": {"fado": _ci(agg["fado"]), "marginal": _ci(agg["marg"]),
                           "multivariate": _ci(agg["mult"])},
               "fado_minus_marginal": {"point": round(_pt("fado_pearson") - _pt("marginal_pearson"), 3),
                                       "ci": _ci(d_fm), "excludes_0": _excl0(d_fm)},
               "replication_all_pass": (all(r.get("replication_check") == "PASS" for r in out)
                                        if any("replication_check" in r for r in out) else None)}
    comp_means = {m: round(float(np.nanmean([r[f"{m}_pearson"] for r in out
                                             if f"{m}_pearson" in r])), 3)
                  for m in comp_all
                  if any(f"{m}_pearson" in r for r in out)}
    if comp_means:
        summary["comparator_mean_pearson"] = comp_means
        if oracle:
            summary["n_scms_oracle"] = n_oracle
    out.append(summary)
    if comp_means:
        print("  comparator mean r over families: "
              + "  ".join(f"{m} {v:+.3f}" for m, v in comp_means.items()))
    print(f"  mean r: FADO {summary['mean_pearson']['fado']:.3f}  "
          f"marg {summary['mean_pearson']['marginal']:.3f}  "
          f"multi {summary['mean_pearson']['multivariate']:.3f}  "
          f"| FADO-marg {summary['fado_minus_marginal']['point']:+.3f} "
          f"CI {summary['fado_minus_marginal']['ci']} "
          f"excludes_0={summary['fado_minus_marginal']['excludes_0']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="causal_probe/results/probe_intersample_L_seed0.ckpt")
    ap.add_argument("--acic-path", default="data/acic2016/x.csv")
    ap.add_argument("--source-label", default="acic2016")
    ap.add_argument("--families", default="linear,mixed_nonlinear,interaction")
    ap.add_argument("--n-scms", type=int, default=40)
    ap.add_argument("--p-real-cap", type=int, default=10)
    ap.add_argument("--n-causal", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-json", default="acic_fp_results.json")
    add_comparator_args(ap)
    args = ap.parse_args()

    if args.comparators_only:
        model = torch = None
    else:
        import torch
        from causal_probe.eval import _load_model
        model = _load_model(args.ckpt, device=args.device)

    Xacic = load_acic2016(args.acic_path)
    print(f"ACIC-2016 covariates: {Xacic.shape[0]} subjects x {Xacic.shape[1]} usable cols; "
          f"each SCM uses a random {args.p_real_cap}-col subset\n")

    results = {}
    for hidden in (False, True):
        label = "hidden_confounder" if hidden else "no_confounding"
        print(f"\n=== regime: {label} ===")
        results[label] = run_regime(model, Xacic, args, hidden, torch)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_json}")
    if args.comparators_only:
        print("REPLICATION n/a (--comparators-only: no FADO forward pass)")
    else:
        ok = all(r[-1]["replication_all_pass"] for r in results.values())
        print(f"REPLICATION {'PASS: pooled r matches the reference ACIC run in both regimes' if ok else 'FAIL: DO NOT USE FP numbers — benchmark differs from reference run'}")


if __name__ == "__main__":
    main()
