"""Confounding-injection benchmark built on RCT_Bench.

Turn each randomized trial (T ind X, ATE = diff-in-means) into a *confounded
observational* dataset while keeping the RCT effect recoverable by adjustment,
with a tunable strength knob -- the standard RCT->observational construction
(Gentzel et al. 2021; Kallus). Because selection is on a PRE-TREATMENT
covariate, within-stratum exchangeability is preserved, so adjusting for that
covariate (or all of X, which contains it) still identifies the effect; the
UNADJUSTED diff-in-means becomes biased. This converts RCT_Bench from a weak
discriminator (randomized -> g-estimator trivially valid) into a graded test of
whether a method actually adjusts.

Injection: pick the most prognostic pre-treatment covariate X_c (max |corr(X,Y)|),
standardize it to z. Keep unit i with probability
    p_keep = sigmoid(s * z_i)   if T_i = 1     (treated skew to high X_c)
             sigmoid(-s * z_i)  if T_i = 0     (control skew to low  X_c)
s=0 -> p=0.5 everywhere -> random subsample -> still an RCT (sanity anchor).
s>0 -> corr(T, X_c) > 0 -> confounding, growing with s.

Ground truth = the full-RCT diff-in-means (assumes approx. homogeneous effect;
the OLS-adjusted-on-all-X curve is the built-in oracle check -- if it stays flat
across s, the target is preserved and adjustment works).
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def most_prognostic(td: TrialData) -> int:
    cors = [abs(np.corrcoef(td.X[:, j], td.Y)[0, 1]) if np.std(td.X[:, j]) > 0 else 0.0
            for j in range(td.p)]
    return int(np.nanargmax(cors))


def inject(td: TrialData, c_idx: int, s: float, rng) -> np.ndarray:
    """Return a boolean keep-mask inducing T-X_c confounding of strength s."""
    xc = td.X[:, c_idx].astype(float)
    z = (xc - xc.mean()) / (xc.std() or 1.0)
    p = np.where(td.T == 1, _sigmoid(s * z), _sigmoid(-s * z))
    return rng.random(len(z)) < p


def naive_d(T, Y) -> float:
    return (Y[T == 1].mean() - Y[T == 0].mean()) / (np.std(Y) or 1.0)


def ols_d(T, X, Y) -> float:
    from numpy.linalg import lstsq
    D = np.column_stack([np.ones(len(T)), T.astype(float), X])
    beta, *_ = lstsq(D, Y.astype(float), rcond=None)
    return beta[1] / (np.std(Y) or 1.0)


def validate(trials, strengths, n_seeds, data_dir):
    """Local mechanism check: naive should degrade with s, OLS should stay flat."""
    print(f"{'trial':>9}{'s':>5}{'keep%':>7}{'corrTXc':>9}{'naive_err':>11}{'ols_err':>9}")
    for tid in trials:
        td = load_trial(os.path.join(data_dir, f"{tid}.csv"))
        if td is None:
            continue
        c = most_prognostic(td)
        gt = td.rct_ate_raw() / (np.std(td.Y) or 1.0)
        for s in strengths:
            keeps, corrs, nerr, oerr = [], [], [], []
            for seed in range(n_seeds):
                rng = np.random.default_rng(1000 + seed)
                m = inject(td, c, s, rng)
                if m.sum() < 40 or td.T[m].sum() < 10 or (1 - td.T[m]).sum() < 10:
                    continue
                Ts, Xs, Ys = td.T[m], td.X[m], td.Y[m]
                keeps.append(m.mean())
                corrs.append(np.corrcoef(Ts, Xs[:, c])[0, 1])
                nerr.append(abs(naive_d(Ts, Ys) - gt))
                oerr.append(abs(ols_d(Ts, Xs, Ys) - gt))
            if nerr:
                print(f"{tid:>9}{s:>5.1f}{100*np.mean(keeps):>7.0f}{np.mean(corrs):>+9.2f}"
                      f"{np.mean(nerr):>11.3f}{np.mean(oerr):>9.3f}")
        print()


def pick_trials(data_dir, n_min=150, p_max=15, corr_min=0.10):
    """Auto-select confounding-suitable trials (FADO's p-range, big enough to
    survive subsampling, has a prognostic covariate). Both outcome types."""
    out = []
    for pth in sorted(glob.glob(os.path.join(data_dir, "trial*.csv")),
                      key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))))):
        td = load_trial(pth)
        if td is None or not (3 <= td.p <= p_max) or td.n < n_min:
            continue
        cors = [abs(np.corrcoef(td.X[:, j], td.Y)[0, 1]) if np.std(td.X[:, j]) > 0 else 0
                for j in range(td.p)]
        if np.nanmax(cors) >= corr_min:
            out.append(td.trial)
    return out


def _drop_col(sub: TrialData, c: int) -> TrialData:
    """Return a copy with covariate column c removed (unobserved-confounder)."""
    keep = [j for j in range(sub.p) if j != c]
    return TrialData(trial=sub.trial, X=sub.X[:, keep], T=sub.T, Y=sub.Y,
                     x_names=[sub.x_names[j] for j in keep], primary=sub.primary,
                     primary_type=sub.primary_type, control_arm=sub.control_arm,
                     active_arm=sub.active_arm, n_arms_original=sub.n_arms_original)


def run_sweep(trials, strengths, n_seeds, data_dir, ckpt, device, regime):
    from causal_probe.rct_bench_eval import fado_labeled, causalpfn_ate, lingam_tau_raw
    METHODS = ["naive", "ols", "lingam", "causalpfn", "fado"]
    out = []
    for tid in trials:
        td = load_trial(os.path.join(data_dir, f"{tid}.csv"))
        if td is None:
            continue
        c = most_prognostic(td)
        gt = td.rct_ate_raw() / (np.std(td.Y) or 1.0)
        print(f"\n### [{regime}] {tid} n={td.n} p={td.p} type={td.primary_type} "
              f"gt_d={gt:+.2f} prog_cov={td.x_names[c]}")
        for s in strengths:
            per = {m: [] for m in METHODS}
            corrs, keeps = [], []
            for seed in range(n_seeds):
                rng = np.random.default_rng(1000 + seed)
                m = inject(td, c, s, rng)                       # selection ALWAYS on true X_c
                if m.sum() < 60 or td.T[m].sum() < 15 or (1 - td.T[m]).sum() < 15:
                    continue
                full = TrialData(trial=td.trial, X=td.X[m], T=td.T[m], Y=td.Y[m],
                                 x_names=td.x_names, primary=td.primary,
                                 primary_type=td.primary_type, control_arm=td.control_arm,
                                 active_arm=td.active_arm, n_arms_original=td.n_arms_original)
                corrs.append(np.corrcoef(full.T, full.X[:, c])[0, 1]); keeps.append(m.mean())
                # what the methods GET: full X (observed) or X minus the confounder (unobserved)
                sub = full if regime == "observed" else _drop_col(full, c)
                sy = np.std(sub.Y) or 1.0
                per["naive"].append(naive_d(sub.T, sub.Y))
                per["ols"].append(ols_d(sub.T, sub.X, sub.Y) if sub.p else naive_d(sub.T, sub.Y))
                lg = lingam_tau_raw(sub); per["lingam"].append(lg / sy if np.isfinite(lg) else np.nan)
                cp = causalpfn_ate(sub, device); per["causalpfn"].append(cp / sy if np.isfinite(cp) else np.nan)
                per["fado"].append(fado_labeled(sub, ckpt, device) / sy)
            if not corrs:
                continue
            rec = {"trial": tid, "regime": regime, "type": td.primary_type, "n": td.n, "p": td.p,
                   "s": s, "gt_d": gt, "corr_TXc": float(np.nanmean(corrs)),
                   "keep_frac": float(np.mean(keeps)), "n_seeds": len(corrs)}
            for mth in METHODS:
                arr = np.array(per[mth], float)
                rec[f"{mth}_err"] = float(np.nanmean(np.abs(arr - gt)))
                rec[f"{mth}_d"] = float(np.nanmean(arr))
            out.append(rec)
            print(f"  s={s:>3} corrTXc={rec['corr_TXc']:+.2f} | " +
                  " ".join(f"{m}={rec[m+'_err']:.2f}" for m in METHODS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/RCT_Bench/cleaned_data")
    ap.add_argument("--trials", default="")
    ap.add_argument("--strengths", default="0,0.5,1.0,1.5,2.0")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--ckpt", default="causal_probe/results/probe_intersample_L_seed0.ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--regimes", default="observed,unobserved",
                    help="observed = methods see the confounder; unobserved = it is hidden")
    ap.add_argument("--out-json", default="causal_probe/results/rct_bench_confound_v2.json")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    strengths = [float(x) for x in args.strengths.split(",")]
    trials = args.trials.split(",") if args.trials else pick_trials(args.data_dir)
    print(f"trials ({len(trials)}): {trials}")
    if args.validate:
        validate(trials, strengths, args.n_seeds, args.data_dir)
        return
    out = []
    for regime in args.regimes.split(","):
        out += run_sweep(trials, strengths, args.n_seeds, args.data_dir, args.ckpt,
                         args.device, regime.strip())
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
