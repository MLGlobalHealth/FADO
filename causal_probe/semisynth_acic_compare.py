"""FADO vs CausalPFN on FADO's task with the STANDARD ACIC-2016 real covariates.

Uses the ACIC-2016 Data Analysis Challenge covariate matrix (Hahn/Dorie/Hill;
4802 subjects x 58 real covariates from a real database) as the covariate
source for the semi-synthetic full-per-feature benchmark. Because FADO is
trained on p in [3,15], each SCM draws a random subset of `p_real_cap` real ACIC
covariates (real marginals + real joint structure among the chosen columns),
attaches a simulated outcome via a known f (linear / mixed-nonlinear /
interaction) + proxy/leak/noise decoys -> full per-feature ground truth tau.
Metric = Pearson(tau_hat, tau_true) pooled over SCMs & columns.
"""
from __future__ import annotations
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from causal_probe.semi_synth_real import make_benchmark
from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model


_ID_COLS = {"sample_id", "idx", "id", "index"}


def load_acic2016(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    cols = []
    for c in df.columns:
        if str(c).lower() in _ID_COLS:                         # drop row identifiers (LBIDD sample_id)
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):               # object OR string dtype
            s = s.astype("category").cat.codes.astype(float)   # label-encode letters
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="causal_probe/results/probe_intersample_L_seed0.ckpt")
    ap.add_argument("--acic-path", default="data/acic2016/x.csv")
    ap.add_argument("--source-label", default="acic2016")
    ap.add_argument("--families", default="linear,mixed_nonlinear,interaction")
    ap.add_argument("--n-scms", type=int, default=40)
    ap.add_argument("--p-real-cap", type=int, default=10)
    ap.add_argument("--n-causal", type=int, default=3)
    ap.add_argument("--hidden-confound", action="store_true",
                    help="designate a real column as an unobserved confounder of Y and an "
                         "observed proxy child, then drop it from X_full (unadjustable backdoor)")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples over SCM instances for 95%% CIs")
    ap.add_argument("--out-json", default="causal_probe/results/semisynth_acic_compare.json")
    args = ap.parse_args()
    import torch
    from causal_probe.rct_bench_eval import _patch_causalpfn_small_n
    from causal_probe.causalpfn_per_column_baseline import causalpfn_tau
    _patch_causalpfn_small_n()
    model = _load_model(args.ckpt, device=args.device)

    Xacic = load_acic2016(args.acic_path)
    print(f"ACIC-2016 covariates: {Xacic.shape[0]} subjects x {Xacic.shape[1]} usable cols; "
          f"each SCM uses a random {args.p_real_cap}-col subset\n")
    def _pool(pred_list, true_list, idx):
        P = np.concatenate([pred_list[i] for i in idx])
        T = np.concatenate([true_list[i] for i in idx])
        return _pearson(P, T)

    def _ci(draws):
        d = np.asarray([x for x in draws if np.isfinite(x)], dtype=float)
        if d.size == 0:
            return [float("nan"), float("nan")]
        return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]

    out = []
    fam_data = {}
    n_boot = args.n_boot
    for family in args.families.split(","):
        family = family.strip()
        fado_p, cpfn_p, marg_p, mult_p, true_all = [], [], [], [], []
        colrng = np.random.default_rng(args.seed)
        for s in range(args.n_scms):
            cols = colrng.choice(Xacic.shape[1], args.p_real_cap, replace=False)
            X = Xacic[:, cols]
            b = make_benchmark(X, np.random.default_rng(1000 * args.seed + s),
                               n_causal=args.n_causal, outcome_family=family,
                               hidden_confound=args.hidden_confound)
            X_full, y, tau = b["X_full"], b["y"], b["true_tau"]
            n = X_full.shape[0]
            rng = np.random.default_rng(s)
            idx = rng.choice(n, args.n_rows, replace=False) if n > args.n_rows else np.arange(n)
            Xs = (X_full - X_full.mean(0)) / X_full.std(0).clip(min=1e-9)
            ys = (y - y.mean()) / (y.std() or 1)
            with torch.no_grad():
                fp = model(torch.tensor(Xs[idx][None], dtype=torch.float32, device=args.device),
                           torch.tensor(ys[idx][None], dtype=torch.float32, device=args.device)
                           ).squeeze(0).cpu().numpy()
            cp = causalpfn_tau(X_full[idx].astype(np.float64), y[idx].astype(np.float64),
                               device=args.device)
            fado_p.append(fp); cpfn_p.append(cp)
            marg_p.append(BASELINES["marginal"](Xs, ys))
            mult_p.append(BASELINES["multivariate"](Xs, ys)); true_all.append(tau)
        F = np.concatenate(fado_p); C = np.concatenate(cpfn_p)
        Mg = np.concatenate(marg_p); Mv = np.concatenate(mult_p); T = np.concatenate(true_all)
        fam_data[family] = {"fado": fado_p, "cpfn": cpfn_p, "marg": marg_p,
                            "mult": mult_p, "true": true_all}
        # per-family bootstrap CI over the n_scms instances
        brng = np.random.default_rng(777)
        dr = {"fado": [], "cpfn": [], "marg": [], "mult": []}
        ml = {"fado": fado_p, "cpfn": cpfn_p, "marg": marg_p, "mult": mult_p}
        for _ in range(n_boot):
            bidx = brng.integers(0, args.n_scms, args.n_scms)
            for m in dr:
                dr[m].append(_pool(ml[m], true_all, bidx))
        rec = {"source": args.source_label, "family": family, "n_scms": args.n_scms,
               "p_real_cap": args.p_real_cap,
               "fado_pearson": _pearson(F, T), "fado_ci": _ci(dr["fado"]),
               "causalpfn_pearson": _pearson(C, T), "causalpfn_ci": _ci(dr["cpfn"]),
               "marginal_pearson": _pearson(Mg, T), "marginal_ci": _ci(dr["marg"]),
               "multivariate_pearson": _pearson(Mv, T), "multivariate_ci": _ci(dr["mult"]),
               "causalpfn_valid_frac": float(np.isfinite(C).mean())}
        out.append(rec)
        print(f"{args.source_label} / {family:<16} | "
              f"FADO {rec['fado_pearson']:.2f} {rec['fado_ci']}  "
              f"CausalPFN {rec['causalpfn_pearson']:.2f} {rec['causalpfn_ci']}  "
              f"marg {rec['marginal_pearson']:.2f} {rec['marginal_ci']}  "
              f"multi {rec['multivariate_pearson']:.2f} {rec['multivariate_ci']}")

    # ---- mean over families + paired difference CIs (resample SCMs per family) ----
    fams = list(fam_data.keys())
    brng = np.random.default_rng(20260731)
    agg = {"fado": [], "cpfn": [], "marg": [], "mult": []}
    d_fm, d_fc = [], []
    for _ in range(n_boot):
        pf = {"fado": [], "cpfn": [], "marg": [], "mult": []}
        for fam in fams:
            dd = fam_data[fam]
            bidx = brng.integers(0, len(dd["true"]), len(dd["true"]))
            for m in pf:
                pf[m].append(_pool(dd[m], dd["true"], bidx))
        mu = {m: float(np.nanmean(pf[m])) for m in pf}
        for m in agg:
            agg[m].append(mu[m])
        d_fm.append(mu["fado"] - mu["marg"])
        d_fc.append(mu["fado"] - mu["cpfn"])

    def _pt(key):
        return round(float(np.nanmean([r[key] for r in out])), 3)

    def _excl0(draws):
        d = np.asarray([x for x in draws if np.isfinite(x)], dtype=float)
        return bool(d.size and np.percentile(d, 2.5) > 0)

    summary = {"SUMMARY": True, "source": args.source_label,
               "hidden_confound": bool(args.hidden_confound),
               "n_scms": args.n_scms, "n_families": len(fams), "n_boot": n_boot,
               "mean_pearson": {"fado": _pt("fado_pearson"), "causalpfn": _pt("causalpfn_pearson"),
                                "marginal": _pt("marginal_pearson"),
                                "multivariate": _pt("multivariate_pearson")},
               "mean_ci": {"fado": _ci(agg["fado"]), "causalpfn": _ci(agg["cpfn"]),
                           "marginal": _ci(agg["marg"]), "multivariate": _ci(agg["mult"])},
               "fado_minus_marginal": {"point": round(_pt("fado_pearson") - _pt("marginal_pearson"), 3),
                                       "ci": _ci(d_fm), "excludes_0": _excl0(d_fm)},
               "fado_minus_causalpfn": {"point": round(_pt("fado_pearson") - _pt("causalpfn_pearson"), 3),
                                        "ci": _ci(d_fc), "excludes_0": _excl0(d_fc)}}
    out.append(summary)
    print(f"\n=== MEAN over {len(fams)} families (95% bootstrap CI over {args.n_scms} SCMs) ===")
    for m in ["fado", "causalpfn", "marginal", "multivariate"]:
        print(f"  {m:<12} {summary['mean_pearson'][m]:.3f}  CI {summary['mean_ci'][m]}")
    print(f"  FADO - marginal : {summary['fado_minus_marginal']['point']:+.3f}  "
          f"CI {summary['fado_minus_marginal']['ci']}  excludes_0={summary['fado_minus_marginal']['excludes_0']}")
    print(f"  FADO - CausalPFN: {summary['fado_minus_causalpfn']['point']:+.3f}  "
          f"CI {summary['fado_minus_causalpfn']['ci']}  excludes_0={summary['fado_minus_causalpfn']['excludes_0']}")
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
