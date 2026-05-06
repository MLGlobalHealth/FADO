"""Cross-dataset summary plots for the anchor audit.

Reads `causal_probe/results/anchor_audit/{dataset}.json` and produces:

  paper/figures/anchor_audit_scatter.{png,pdf}
      one point per (feature, dataset) of |symmetric contrast| vs the
      rank-based curve range; flagged features (large range, near-zero
      contrast) labelled.

  paper/figures/anchor_audit_curves.{png,pdf}
      response curves (Gaussian anchors and rank quantiles) for the
      flagged features.

Usage:
    uv run python -m scripts.plot_anchor_audit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


T_GRID = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
RANK_QS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

DATASET_COLORS = {
    "hillstrom": "#1f77b4",
    "criteo":    "#d62728",
    "lalonde":   "#2ca02c",
    "sachs":     "#9467bd",
    "tubingen":  "#ff7f0e",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="causal_probe/results/anchor_audit")
    ap.add_argument("--out-scatter", default="paper/figures/anchor_audit_scatter.png")
    ap.add_argument("--out-scatter-pdf", default="paper/figures/anchor_audit_scatter.pdf")
    ap.add_argument("--out-curves", default="paper/figures/anchor_audit_curves.png")
    ap.add_argument("--out-curves-pdf", default="paper/figures/anchor_audit_curves.pdf")
    ap.add_argument("--contrast-floor", type=float, default=0.05)
    ap.add_argument("--range-floor", type=float, default=0.20)
    args = ap.parse_args()

    audit_dir = Path(args.audit_dir)
    datasets = []
    flagged_total: list[tuple[str, dict]] = []
    points: list[tuple[str, str, float, float]] = []

    for json_path in sorted(audit_dir.glob("*.json")):
        if json_path.name == "summary.json":
            continue
        with open(json_path) as f:
            d = json.load(f)
        ds_name = d.get("dataset", json_path.stem)
        datasets.append(ds_name)
        rows = d.get("rows", [])
        for r in rows:
            c = r.get("contrast_sym", float("nan"))
            rg = r.get("range_rank", float("nan"))
            if np.isfinite(c) and np.isfinite(rg):
                points.append((ds_name, r["name"], abs(float(c)), float(rg)))
        for fr in d.get("flagged", []):
            flagged_total.append((ds_name, fr))

    # ---- scatter ----
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for ds in datasets:
        xs = [p[2] for p in points if p[0] == ds]
        ys = [p[3] for p in points if p[0] == ds]
        if not xs:
            continue
        ax.scatter(
            xs, ys, s=44, alpha=0.75,
            color=DATASET_COLORS.get(ds, "#666666"),
            edgecolor="black", linewidth=0.5,
            label=f"{ds} (n_features={len(xs)})",
        )
    # Shade the flag region.
    xlim = max(0.5, max((p[2] for p in points), default=0.5) * 1.05)
    ylim = max(1.0, max((p[3] for p in points), default=1.0) * 1.10)
    ax.add_patch(plt.Rectangle(
        (0, args.range_floor), args.contrast_floor, ylim,
        facecolor="#fff8a8", edgecolor="#bbaa44", lw=0.7, alpha=0.45, zorder=0,
    ))
    ax.text(args.contrast_floor / 2, ylim * 0.96, "flag region\n(small symmetric\n contrast,\n large rank range)",
            ha="center", va="top", fontsize=9, color="#665522")

    # Label flagged features.
    for ds, fr in flagged_total:
        x = abs(fr["contrast_sym"])
        y = fr.get("range_rank", fr["range_curve"])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.annotate(
            f"{ds}/{fr['name']}",
            xy=(x, y), xytext=(x + 0.04, y),
            fontsize=9, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        )

    ax.axvline(args.contrast_floor, color="#bbaa44", ls=":", lw=1.0)
    ax.axhline(args.range_floor, color="#bbaa44", ls=":", lw=1.0)
    ax.set_xlim(-0.02, xlim)
    ax.set_ylim(-0.02, ylim)
    ax.set_xlabel(r"$|\hat\Delta_i(-1, +1)|$  (symmetric scalar contrast)")
    ax.set_ylabel(r"rank-based response range  $\max_q \hat r_i(q) - \min_q \hat r_i(q)$")
    ax.set_title("Anchor audit across the paper's real-data benchmarks")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    Path(args.out_scatter).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_scatter, dpi=180, bbox_inches="tight")
    fig.savefig(args.out_scatter_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_scatter}")

    # ---- curves for flagged features ----
    if flagged_total:
        n = len(flagged_total)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
        for ax, (ds, fr) in zip(axes.flatten(), flagged_total):
            r_g = np.array(fr["r_curve"], dtype=float)
            r_r = np.array(fr["rank_curve"], dtype=float)
            ax.plot(T_GRID, r_g, marker="o", color=DATASET_COLORS.get(ds, "#666666"),
                    label=r"Gaussian anchors $\hat r(t)$  on $X_{\rm std}$")
            # Plot rank-based on the same axes by mapping quantile to its
            # standardized X position approximately (norm.ppf would work
            # exactly; for plotting clarity we put it at quantile-positions
            # 0.1..0.9 transformed to roughly -1.28..+1.28).
            from scipy.stats import norm
            r_r_x = norm.ppf(RANK_QS)
            ax.plot(r_r_x, r_r, marker="s", linestyle="--",
                    color="#444444", label=r"rank-based $\hat r(F^{-1}(q))$")
            ax.axhline(0, color="black", lw=0.5, alpha=0.4)
            ax.axvline(-1, color="#bbaaaa", ls=":", lw=0.8)
            ax.axvline(+1, color="#bbaaaa", ls=":", lw=0.8)
            ax.set_xlabel(r"intervention anchor (in $X_{\rm std}$ units)")
            ax.set_ylabel(r"$\hat r_i(t)$")
            ax.set_title(
                f"{ds}: feature {fr['name']}\n"
                rf"$|\hat\Delta_i(-1,+1)|$={abs(fr['contrast_sym']):.3f},  "
                rf"rank range={fr.get('range_rank', float('nan')):.3f}",
                fontsize=10,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle("Flagged features: response curves the symmetric pair would miss",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(args.out_curves, dpi=180, bbox_inches="tight")
        fig.savefig(args.out_curves_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {args.out_curves}")
    else:
        print("no flagged features; curve figure not produced")


if __name__ == "__main__":
    main()
