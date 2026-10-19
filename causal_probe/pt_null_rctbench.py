"""Non-treatment variable with a KNOWN-null effect, on real RCT_Bench outcomes.

For each RCT trial we build a proxy  P_T = T + noise  and treat it as the
variable of interest. Its TRUE causal effect on Y is exactly 0 -- it does not
cause Y; it is only correlated with Y through the real treatment T (P_T <- T ->
Y). T is thus the confounder of the P_T->Y question.

  * OBSERVED regime:  the confounder T is in the covariate set -> a method that
    adjusts for it should return ~0 (correct: P_T has no causal effect).
  * UNOBSERVED regime: T is removed -> P_T looks like it affects Y (spurious via
    T) -> a method that cannot adjust gets fooled (nonzero).

Ground truth = 0 for every trial, so |estimate| IS the error. Uses real
Hillstrom-free RCT_Bench outcomes; only the proxy column is constructed (the
standard decoy device). Restricted to trials with a non-negligible true T effect
so there is a real signal for P_T to spuriously absorb.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from causal_probe.rct_bench_loader import load_trial, TrialData
from causal_probe.rct_bench_eval import fado_labeled, causalpfn_ate


def ols_coef(V, cov, Y):
    from numpy.linalg import lstsq
    D = np.column_stack([np.ones(len(V)), V.astype(float)] + ([cov] if cov.size else []))
    b, *_ = lstsq(D, Y.astype(float), rcond=None)
    return b[1]


def pick(data_dir, n_min, p_max, min_eff):
    out = []
    for pth in sorted(glob.glob(os.path.join(data_dir, "trial*.csv")),
                      key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))))):
        td = load_trial(pth)
        if td is None or not (2 <= td.p <= p_max) or td.n < n_min:
            continue
        if abs(td.rct_ate_raw() / (np.std(td.Y) or 1)) >= min_eff:  # real signal to absorb
            out.append(td.trial)
    return out


def run(trials, data_dir, sigma, n_seeds, ckpt, device):
    METHODS = ["naive", "ols", "causalpfn", "fado"]
    out = []
    for tid in trials:
        td = load_trial(os.path.join(data_dir, f"{tid}.csv"))
        if td is None:
            continue
        sy = np.std(td.Y) or 1.0
        for regime in ("observed", "unobserved"):
            per = {m: [] for m in METHODS}
            vt_corr = []
            for seed in range(n_seeds):
                rng = np.random.default_rng(9000 + seed)
                P = td.T.astype(float) + sigma * rng.standard_normal(td.n)
                V = (P > np.median(P)).astype(int)
                if V.sum() < 15 or (1 - V).sum() < 15:
                    continue
                vt_corr.append(np.corrcoef(V, td.T)[0, 1])
                # confounder T observed (in covariates) or hidden
                cov = np.column_stack([td.T.astype(float), td.X]) if regime == "observed" else td.X
                sub = TrialData(trial=tid, X=cov, T=V, Y=td.Y,
                                x_names=[f"c{i}" for i in range(cov.shape[1])], primary="Y",
                                primary_type=td.primary_type, control_arm="0", active_arm="1",
                                n_arms_original=2)
                per["naive"].append((td.Y[V == 1].mean() - td.Y[V == 0].mean()) / sy)
                per["ols"].append(ols_coef(V, cov, td.Y) / sy)
                cp = causalpfn_ate(sub, device); per["causalpfn"].append(cp / sy if np.isfinite(cp) else np.nan)
                per["fado"].append(fado_labeled(sub, ckpt, device) / sy)
            if not vt_corr:
                continue
            rec = {"trial": tid, "regime": regime, "corr_VT": float(np.mean(vt_corr)),
                   "gt": 0.0, "n": td.n, "p": td.p}
            for m in METHODS:
                a = np.abs(np.array(per[m], float))     # gt=0 -> |estimate| = error
                rec[f"{m}_abs"] = float(np.nanmean(a))
                rec[f"{m}_seedvals"] = [None if not np.isfinite(v) else float(v) for v in a]
            out.append(rec)
            print(f"{tid:>9} [{regime:>10}] corrVT={rec['corr_VT']:.2f} | " +
                  " ".join(f"{m}={rec[m+'_abs']:.2f}" for m in METHODS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/RCT_Bench/cleaned_data")
    ap.add_argument("--n-min", type=int, default=150)
    ap.add_argument("--p-max", type=int, default=15)
    ap.add_argument("--min-eff", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--ckpt", default="causal_probe/results/probe_intersample_L_seed0.ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="causal_probe/results/pt_null_rctbench.json")
    args = ap.parse_args()
    trials = pick(args.data_dir, args.n_min, args.p_max, args.min_eff)
    print(f"trials ({len(trials)}, |effect|>={args.min_eff}): {trials}\n")
    out = run(trials, args.data_dir, args.sigma, args.n_seeds, args.ckpt, args.device)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
