"""Compute LightGBM TreeSHAP per-feature attributions for the motif suite.

Matches the methodology used for the Hillstrom SHAP comparison
(`run_baseline.py:shap_tau`): fit a LightGBM regressor to (X, y), take
mean |TreeSHAP| per feature, sign by marginal correlation, divide by
std(y) to put on the same scale as the probe's tau_hat.

Output JSON shape mirrors the motif rows in eval_main_p5_50k.json so
the figure can join on (motif_name, feature_idx).

Usage:
    uv run python causal_probe/compute_motif_shap.py \
        --p 5 --n-rows 512 --n-repeats 50 --seed 100 \
        --out causal_probe/results/motif_shap_p5.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.motifs import ALL_MOTIFS, motif_scm
from causal_probe.run_baseline import shap_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="causal_probe/results/motif_shap_p5.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    motifs_out = {}
    for name, make in ALL_MOTIFS.items():
        per_feat = np.empty((args.n_repeats, args.p), dtype=np.float64)
        for r in range(args.n_repeats):
            spec = make(p=args.p,
                        rng=np.random.default_rng(rng.integers(0, 2**31)))
            scm = motif_scm(spec,
                            rng=np.random.default_rng(rng.integers(0, 2**31)))
            samp = scm.sample(n=args.n_rows,
                              rng=np.random.default_rng(rng.integers(0, 2**31)))
            per_feat[r] = shap_tau(samp.X, samp.y)
        rows = []
        for i in range(args.p):
            rows.append({
                "feature": i,
                "shap_mean": float(per_feat[:, i].mean()),
                "shap_std": float(per_feat[:, i].std()),
            })
        motifs_out[name] = {"n_repeats": args.n_repeats, "rows": rows}
        print(f"{name}: |shap| per-feature = "
              + ", ".join(f"{abs(r['shap_mean']):.3f}" for r in rows))

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
