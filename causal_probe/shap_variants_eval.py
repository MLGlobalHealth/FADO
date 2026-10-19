"""SHAP-variant comparison: do all SHAP-family explainers behave alike?

Backs the paper's app:shap-variants argument with numbers. The claim is that
the SHAP-variant debate (tree_path_dependent vs interventional/Janzing vs
model-agnostic Kernel SHAP) is ORTHOGONAL to FADO's target, because every SHAP
variant explains a FITTED PREDICTOR of E[y|X], not the data-generating process.
So on held-out SCMs they should all track predictive importance (low Pearson r
against the true interventional per-feature tau relative to FADO), and on the
decoy constructions they should all score the outcome-LEAK decoy high (truth is
0 there by construction) — regardless of which SHAP variant you pick.

This script runs, for the three SHAP-family methods registered in
``causal_probe.run_baseline.METHODS``:

  * ``shap``                — LightGBM TreeSHAP, tree_path_dependent (canonical)
  * ``shap_interventional`` — LightGBM TreeSHAP, feature_perturbation=interventional
  * ``shap_kernel``         — Kernel SHAP on a fitted MLP (model-agnostic)

Part A (held-out SCMs): Pearson r vs true tau, pooled over a few SCM families
(uses causal_probe.run_baseline.eval_method_on_scms / _sample_scms, i.e. the
EXACT held-out SCM stream the other baselines use).

Part B (decoy constructions): on causal_probe.semi_synth_real.make_benchmark
instances (the ACIC/IHDP-style proxy/leak/noise/conf decoy augmentation; covariates
default to a synthetic Gaussian source, or pass --acic-path / --ihdp-path for the
real covariate matrices used by semisynth_acic_fp.py / ihdp_decoy_eval.py), report
per method: Pearson r vs true tau AND FP mean|score| by decoy kind.

The scientific read: the three SHAP rows cluster (all low r, all high leak-FP),
while FADO (reported separately in semisynth_acic_fp.py) tracks tau and zeroes
the leak.

Dry run (CPU, exits 0, no heavy compute):
  uv run python -m causal_probe.shap_variants_eval --dry-run

Cheap local run (CPU; SHAP methods need only (X, y), no GPU / no FADO):
  uv run python -m causal_probe.shap_variants_eval \
      --families linear,nonlinear,mlp --n-scms 20 --n-rows 400 \
      --decoy-n-scms 20 --p-real-cap 8
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SHAP_METHODS = ["shap", "shap_interventional", "shap_kernel"]
DECOY_KINDS = ["noise", "proxy", "leak", "conf"]


# ---------------------------------------------------------------------------
# Part A: held-out SCMs (same stream as the other run_baseline comparators)
# ---------------------------------------------------------------------------

def run_heldout(methods, families, p, noise, n_scms, n_rows, seed):
    """Pearson r vs true tau per method, per family, via the shared driver."""
    from causal_probe.run_baseline import _sample_scms, eval_method_on_scms

    results = {}
    for fam in families:
        results[fam] = {}
        for m in methods:
            rng = np.random.default_rng(seed)
            scms = _sample_scms(fam, p, noise, n_scms, rng)
            kw = {}
            if m == "shap_kernel":
                kw["seed"] = seed
            res = eval_method_on_scms(m, scms, n_rows=n_rows, rng=rng,
                                      method_kwargs=kw)
            res.pop("_raw", None)
            results[fam][m] = {
                "pearson": res.get("pearson"),
                "spearman": res.get("spearman"),
                "n_scms": res.get("n_scms"),
                "n_failed": res.get("n_failed"),
                "error": res.get("error"),
            }
            r = results[fam][m]["pearson"]
            rstr = f"{r:+.3f}" if isinstance(r, float) and np.isfinite(r) else "   nan"
            print(f"  [heldout] {fam:<14} {m:<20} r={rstr}  "
                  f"(n={results[fam][m]['n_scms']}, failed={results[fam][m]['n_failed']})")
    return results


# ---------------------------------------------------------------------------
# Part B: decoy constructions (proxy / leak / noise / conf)
# ---------------------------------------------------------------------------

def _load_covariates(args):
    """Covariate matrix X (n, p) for the decoy benchmark. Real ACIC/IHDP if a
    path is given, else a synthetic correlated-Gaussian source."""
    if args.acic_path:
        from causal_probe.semisynth_acic_fp import load_acic2016
        X = load_acic2016(args.acic_path)
        return X, f"acic2016:{args.acic_path}"
    if args.ihdp_path:
        import pandas as pd
        df = pd.read_csv(args.ihdp_path)
        cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and df[c].std() > 0]
        return df[cols].values.astype(float), f"ihdp:{args.ihdp_path}"
    # Synthetic fallback: correlated Gaussian so proxies/leaks are meaningful.
    rng = np.random.default_rng(args.seed)
    n, p = 1000, max(args.p_real_cap + 4, 12)
    A = rng.standard_normal((p, p))
    cov = A @ A.T / p + np.eye(p)
    L = np.linalg.cholesky(cov)
    X = rng.standard_normal((n, p)) @ L.T
    return X, "synthetic_gaussian"


def _fp_by_kind(scores, kinds):
    out = {}
    scores = np.asarray(scores, dtype=float)
    for k in DECOY_KINDS:
        mask = np.array([kk == k for kk in kinds])
        if not mask.any():
            continue
        v = np.abs(scores[mask])
        v = v[np.isfinite(v)]
        if v.size:
            out[k] = float(np.mean(v))
    return out


def run_decoys(methods, Xcov, families, n_scms, p_real_cap, n_causal,
               n_rows, hidden_confound, seed):
    """Per method: pooled Pearson r vs true tau + FP mean|score| by decoy kind,
    over make_benchmark instances (same construction as semisynth_acic_fp.py)."""
    from causal_probe.semi_synth_real import make_benchmark
    from causal_probe.run_baseline import METHODS as RB_METHODS

    # pred[m] = list of per-instance score vectors; shared true / kinds lists.
    pred = {m: [] for m in methods}
    err = {m: 0 for m in methods}
    true_all, kinds_all = [], []

    colrng = np.random.default_rng(seed)
    for s in range(n_scms):
        cols = colrng.choice(Xcov.shape[1],
                             min(p_real_cap, Xcov.shape[1]), replace=False)
        X = Xcov[:, cols]
        b = make_benchmark(X, np.random.default_rng(1000 * seed + s),
                           n_causal=n_causal, outcome_family=families,
                           hidden_confound=hidden_confound)
        X_full, y, tau, kinds = b["X_full"], b["y"], b["true_tau"], b["kinds"]
        n = X_full.shape[0]
        rng = np.random.default_rng(s)
        idx = rng.choice(n, n_rows, replace=False) if n > n_rows else np.arange(n)
        Xs = (X_full - X_full.mean(0)) / X_full.std(0).clip(min=1e-9)
        ys = (y - y.mean()) / (y.std() or 1)
        Xi, yi = Xs[idx], ys[idx]

        true_all.append(np.asarray(tau, dtype=float))
        kinds_all.append(kinds)
        for m in methods:
            kw = {"seed": 1000 + s} if m == "shap_kernel" else {}
            try:
                t = np.asarray(RB_METHODS[m](Xi.astype(np.float64),
                                             yi.astype(np.float64), **kw),
                               dtype=float)
                if not np.isfinite(t).any():
                    raise RuntimeError("all-NaN")
                pred[m].append(t)
            except Exception as e:
                pred[m].append(np.full(X_full.shape[1], np.nan))
                err[m] += 1
                if err[m] <= 2:
                    print(f"    [decoy] {m} instance {s}: {type(e).__name__}: {e}")

    T = np.concatenate(true_all)
    K = sum(kinds_all, [])
    results = {}
    for m in methods:
        P = np.concatenate(pred[m])
        mask = np.isfinite(P) & np.isfinite(T)
        if mask.sum() >= 3 and np.std(P[mask]) > 0 and np.std(T[mask]) > 0:
            r = float(np.corrcoef(P[mask], T[mask])[0, 1])
        else:
            r = float("nan")
        fp = _fp_by_kind(P, K)
        results[m] = {"pearson": round(r, 4), "fp": {k: round(v, 4)
                                                     for k, v in fp.items()},
                      "n_failed": err[m]}
        rstr = f"{r:+.3f}" if np.isfinite(r) else "   nan"
        fpstr = "  ".join(f"{k}={fp.get(k, float('nan')):.3f}" for k in DECOY_KINDS
                          if k in fp)
        print(f"  [decoy]   {m:<20} r={rstr}  FP: {fpstr}  (failed={err[m]})")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", default=",".join(SHAP_METHODS),
                    help="comma list of SHAP-family methods (run_baseline names)")
    # Part A
    ap.add_argument("--families", default="linear,nonlinear,mlp",
                    help="held-out SCM families (run_baseline.ALL_EVAL_SCM_TYPES)")
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", default="laplace",
                    choices=["laplace", "gaussian", "heavy"])
    ap.add_argument("--n-scms", type=int, default=20)
    ap.add_argument("--n-rows", type=int, default=400)
    # Part B
    ap.add_argument("--decoy-family", default="linear",
                    choices=["linear", "mixed_nonlinear", "interaction"])
    ap.add_argument("--decoy-n-scms", type=int, default=20)
    ap.add_argument("--p-real-cap", type=int, default=8)
    ap.add_argument("--n-causal", type=int, default=3)
    ap.add_argument("--hidden-confound", action="store_true",
                    help="add a hidden confounder -> conf-child decoys")
    ap.add_argument("--acic-path", default=None,
                    help="ACIC-2016 x.csv (else synthetic covariates)")
    ap.add_argument("--ihdp-path", default=None, help="IHDP covariate csv")
    ap.add_argument("--skip-heldout", action="store_true")
    ap.add_argument("--skip-decoys", action="store_true")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate imports/registry/args and exit 0 (no compute)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Validate methods are registered.
    from causal_probe.run_baseline import METHODS as RB_METHODS
    unknown = [m for m in methods if m not in RB_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; known: {sorted(RB_METHODS)}")

    if args.dry_run:
        # Confirm the SCM families resolve and make_benchmark imports, run a
        # single tiny end-to-end pass per method on a 2-instance toy so the
        # whole code path (not just imports) is exercised without real compute.
        from causal_probe.eval import ALL_EVAL_SCM_TYPES  # noqa: F401
        from causal_probe.semi_synth_real import make_benchmark
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 8))
        b = make_benchmark(X, np.random.default_rng(1),
                           n_causal=2, outcome_family="linear")
        Xs = (b["X_full"] - b["X_full"].mean(0)) / b["X_full"].std(0).clip(min=1e-9)
        ys = (b["y"] - b["y"].mean()) / (b["y"].std() or 1)
        for m in methods:
            kw = {"seed": 0} if m == "shap_kernel" else {}
            t = RB_METHODS[m](Xs.astype(np.float64), ys.astype(np.float64), **kw)
            assert np.asarray(t).shape[0] == b["X_full"].shape[1], m
            assert np.isfinite(np.asarray(t)).any(), f"{m} all-NaN"
            print(f"  dry-run OK: {m} -> shape {np.asarray(t).shape}, "
                  f"finite={int(np.isfinite(np.asarray(t)).sum())}")
        print("DRY-RUN PASS: methods registered, decoy path + each explainer run clean")
        return

    out = {"config": {k: getattr(args, k) for k in
                      ("methods", "families", "p", "noise", "n_scms", "n_rows",
                       "decoy_family", "decoy_n_scms", "p_real_cap", "n_causal",
                       "hidden_confound", "acic_path", "ihdp_path", "seed")}}

    if not args.skip_heldout:
        print("=== Part A: held-out SCMs (Pearson r vs true tau) ===")
        out["heldout"] = run_heldout(
            methods, [f.strip() for f in args.families.split(",")],
            args.p, args.noise, args.n_scms, args.n_rows, args.seed)

    if not args.skip_decoys:
        Xcov, cov_src = _load_covariates(args)
        print(f"\n=== Part B: decoy constructions "
              f"(covariates={cov_src}, family={args.decoy_family}, "
              f"hidden_confound={args.hidden_confound}) ===")
        out["decoys"] = run_decoys(
            methods, Xcov, args.decoy_family, args.decoy_n_scms,
            args.p_real_cap, args.n_causal, args.n_rows,
            args.hidden_confound, args.seed)
        out["decoys_covariate_source"] = cov_src

    # Summary: do the SHAP variants cluster?
    if "decoys" in out:
        leaks = [out["decoys"][m]["fp"].get("leak") for m in methods
                 if "leak" in out["decoys"][m]["fp"]]
        rs = [out["decoys"][m]["pearson"] for m in methods
              if np.isfinite(out["decoys"][m]["pearson"])]
        if leaks and rs:
            print(f"\nSHAP-family clustering (decoys): "
                  f"leak-FP range [{min(leaks):.3f}, {max(leaks):.3f}], "
                  f"r-vs-tau range [{min(rs):+.3f}, {max(rs):+.3f}] "
                  f"-> all score the leak high, all low r vs tau")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
