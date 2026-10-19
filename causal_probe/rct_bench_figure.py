"""Small-multiples scatter: each method's standardized effect estimate d vs the
RCT gold-standard difference-in-means d, across the RCT_Bench trials."""
from __future__ import annotations
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("fado_labeled", "FADO (labeled)"),
    ("fado_hidden", "FADO (hidden + decoys)"),
    ("ols_adjusted", "OLS-adjusted"),
    ("causalpfn", "CausalPFN"),
    ("drlearner", "DR-Learner"),
    ("lingam", "LiNGAM"),
]
LIM = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default="causal_probe/results/rct_bench_eval_full.json")
    ap.add_argument("--out", default="causal_probe/rct_bench_figures/rct_bench_scatter")
    args = ap.parse_args()
    d = json.load(open(args.in_json))
    truth = np.array([r["d"]["rct_unadjusted"] for r in d])

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), sharex=True, sharey=True)
    for ax, (key, name) in zip(axes.ravel(), PANELS):
        est = np.array([r["d"].get(key) if r["d"].get(key) is not None else np.nan for r in d])
        ok = np.isfinite(est) & np.isfinite(truth)
        x, y = truth[ok], np.clip(est[ok], -LIM, LIM)
        sign = (np.sign(est[ok]) == np.sign(truth[ok])).mean() * 100
        corr = np.corrcoef(np.clip(est[ok], -3, 3), truth[ok])[0, 1]
        mae = np.median(np.abs(est[ok] - truth[ok]))
        ax.axline((0, 0), slope=1, color="0.6", lw=1, ls="--", zorder=0)
        ax.axhline(0, color="0.85", lw=0.7, zorder=0); ax.axvline(0, color="0.85", lw=0.7, zorder=0)
        ax.scatter(x, y, s=18, alpha=0.6, edgecolor="none", color="#1f77b4")
        ax.set_title(f"{name}\nn={ok.sum()}  sign {sign:.0f}%  r={corr:.2f}  medAE={mae:.2f}",
                     fontsize=10)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel("RCT effect  d (diff-in-means)")
    for ax in axes[:, 0]:
        ax.set_ylabel("method estimate  d")
    fig.suptitle("Recovery of true randomized effects on RCT_Bench (124 real RCTs)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
