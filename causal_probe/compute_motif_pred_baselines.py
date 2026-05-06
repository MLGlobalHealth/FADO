"""Compute the four predictive-attribution baselines (marginal corr,
linear regression, permutation importance, TreeSHAP) per-feature for the
motif suite.

Matches the four baselines listed in the paper's
\\cref{tab:predictive-baselines}. Each is computed on the same set of
SCM instances per motif so the resulting (predictive_importance, |tau|)
points are directly comparable across panels of the small-multiples
figure.

Output JSON shape:
  {"config": {...},
   "motifs": {motif_name: {
       "n_repeats": N,
       "rows": [{"feature": i,
                 "marginal_mean": float, "marginal_std": float,
                 "linreg_mean":   float, "linreg_std":   float,
                 "perm_mean":     float, "perm_std":     float,
                 "shap_mean":     float, "shap_std":     float}, ...]
   }}}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.baselines import baseline_marginal, baseline_multivariate
from causal_probe.motifs import ALL_MOTIFS, motif_scm
from causal_probe.run_baseline import permutation_tau, shap_tau


_BASELINES = {
    "marginal": lambda X, y: baseline_marginal(X, y),
    "linreg":   lambda X, y: baseline_multivariate(X, y),
    "perm":     lambda X, y: permutation_tau(X, y, n_repeats=10, seed=0),
    "shap":     lambda X, y: shap_tau(X, y),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out",
                    default="causal_probe/results/motif_pred_baselines_p5.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    motifs_out = {}
    for name, make in ALL_MOTIFS.items():
        per = {k: np.empty((args.n_repeats, args.p), dtype=np.float64)
               for k in _BASELINES}
        for r in range(args.n_repeats):
            spec = make(p=args.p,
                        rng=np.random.default_rng(rng.integers(0, 2**31)))
            scm = motif_scm(spec,
                            rng=np.random.default_rng(rng.integers(0, 2**31)))
            samp = scm.sample(n=args.n_rows,
                              rng=np.random.default_rng(rng.integers(0, 2**31)))
            for k, fn in _BASELINES.items():
                per[k][r] = fn(samp.X, samp.y)
        rows = []
        for i in range(args.p):
            row = {"feature": i}
            for k in _BASELINES:
                row[f"{k}_mean"] = float(per[k][:, i].mean())
                row[f"{k}_std"] = float(per[k][:, i].std())
            rows.append(row)
        motifs_out[name] = {"n_repeats": args.n_repeats, "rows": rows}
        summary = "  ".join(
            f"{k}=" + ",".join(f"{abs(r[f'{k}_mean']):.2f}" for r in rows)
            for k in _BASELINES
        )
        print(f"{name}\n  {summary}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {"p": args.p, "n_rows": args.n_rows,
                       "n_repeats": args.n_repeats, "seed": args.seed},
            "motifs": motifs_out,
        }, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
