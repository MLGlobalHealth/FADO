"""FADO vs CausalPFN on FADO's ACTUAL task, with REAL covariates.

The semi-synthetic real-X benchmark (semi_synth_real.make_benchmark) attaches a
simulated outcome to a REAL covariate matrix, giving full per-feature ground
truth tau over {real causal cols, real non-causal cols, proxy, leak, noise}.
This is the one setting that is both (a) FADO's native task (per-feature profile,
no designated treatment) and (b) on real covariate distributions (escapes the
'evaluated on its own synthetic prior' critique). The paper reports FADO vs
marginal/multivariate here; this adds the missing PFN baselines (CausalPFN
per-column). Metric = Pearson(tau_hat, tau_true) pooled over SCMs & columns.
"""
from __future__ import annotations
import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from causal_probe.semi_synth_real import make_benchmark, _load_covariates
from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model


def _pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


CONFIGS = [
    ("diabetes", "linear"), ("diabetes", "mixed_nonlinear"),
    ("california_housing", "linear"), ("california_housing", "mixed_nonlinear"),
    ("abalone", "interaction"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="causal_probe/results/probe_intersample_L_seed0.ckpt")
    ap.add_argument("--n-scms", type=int, default=20)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-causal", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out-json", default="causal_probe/results/semisynth_pfn_compare.json")
    args = ap.parse_args()
    import torch
    from causal_probe.rct_bench_eval import _patch_causalpfn_small_n
    from causal_probe.causalpfn_per_column_baseline import causalpfn_tau
    _patch_causalpfn_small_n()
    model = _load_model(args.ckpt, device=args.device)

    out = []
    for source, family in CONFIGS:
        Xr, _ = _load_covariates(source)
        fado_p, cpfn_p, marg_p, mult_p, true_all = [], [], [], [], []
        rng = np.random.default_rng(args.seed)
        for s in range(args.n_scms):
            b = make_benchmark(Xr, np.random.default_rng(1000 * args.seed + s),
                               n_causal=args.n_causal, outcome_family=family)
            X_full, y, tau = b["X_full"], b["y"], b["true_tau"]
            n = X_full.shape[0]
            idx = rng.choice(n, args.n_rows, replace=False) if n > args.n_rows else np.arange(n)
            Xs = (X_full - X_full.mean(0)) / X_full.std(0).clip(min=1e-9)
            ys = (y - y.mean()) / (y.std() or 1)
            # FADO: single forward pass -> per-column tau vector
            with torch.no_grad():
                fp = model(torch.tensor(Xs[idx][None], dtype=torch.float32, device=args.device),
                           torch.tensor(ys[idx][None], dtype=torch.float32, device=args.device)
                           ).squeeze(0).cpu().numpy()
            # CausalPFN: per-column sweep on the SAME matrix
            cp = causalpfn_tau(X_full[idx].astype(np.float64), y[idx].astype(np.float64),
                               device=args.device)
            fado_p.append(fp); cpfn_p.append(cp)
            marg_p.append(BASELINES["marginal"](Xs, ys))
            mult_p.append(BASELINES["multivariate"](Xs, ys)); true_all.append(tau)
        F = np.concatenate(fado_p); C = np.concatenate(cpfn_p)
        Mg = np.concatenate(marg_p); Mv = np.concatenate(mult_p); T = np.concatenate(true_all)
        rec = {"source": source, "family": family, "n_scms": args.n_scms,
               "fado_pearson": _pearson(F, T), "causalpfn_pearson": _pearson(C, T),
               "marginal_pearson": _pearson(Mg, T),
               "multivariate_pearson": _pearson(Mv, T),
               "causalpfn_valid_frac": float(np.isfinite(C).mean())}
        out.append(rec)
        print(f"{source:>18} / {family:<16} | FADO r={rec['fado_pearson']:.2f}  "
              f"CausalPFN r={rec['causalpfn_pearson']:.2f}  marginal r={rec['marginal_pearson']:.2f}  "
              f"multivariate r={rec['multivariate_pearson']:.2f}"
              f"  (cpfn valid {100*rec['causalpfn_valid_frac']:.0f}%)")

    print(f"\nMEAN | FADO r={np.nanmean([r['fado_pearson'] for r in out]):.2f}  "
          f"CausalPFN r={np.nanmean([r['causalpfn_pearson'] for r in out]):.2f}  "
          f"marginal r={np.nanmean([r['marginal_pearson'] for r in out]):.2f}  "
          f"multivariate r={np.nanmean([r['multivariate_pearson'] for r in out]):.2f}")
    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
