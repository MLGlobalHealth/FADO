"""Produce the key sprint plots.

  1. Scatter: predicted tau_hat vs true tau on 500 random held-out SCMs,
     with model + marginal baseline for comparison.
  2. Bar plot for proxy motif: X1 (true cause) vs X2 (proxy) showing
     true tau, model tau_hat, marginal baseline.
  3. Bar plot for target-descendant motif: X1 (leakage) showing the
     gap between marginal (big) and model (≈0).
  4. Bar plot comparing random-SCM metrics (Pearson, R², AUROC, MAE-zero).

Outputs PNGs to causal_probe/results/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model
from causal_probe.motifs import ALL_MOTIFS, motif_scm
from causal_probe.scm import LinearNonGaussianSCM


def _scatter_plot(
    model,
    n_scms: int,
    p: int,
    n_rows: int,
    seed: int,
    device: str,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(seed)
    preds, truths, assocs = [], [], []
    for _ in range(n_scms):
        scm = LinearNonGaussianSCM(p=p, rng=np.random.default_rng(rng.integers(0, 2**31)))
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
        X_t = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        preds.append(pred); truths.append(samp.tau); assocs.append(BASELINES["marginal"](samp.X, samp.y))
    preds = np.stack(preds).reshape(-1)
    truths = np.stack(truths).reshape(-1)
    assocs = np.stack(assocs).reshape(-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    lim = max(np.max(np.abs(preds)), np.max(np.abs(truths)), np.max(np.abs(assocs))) * 1.1

    ax1.scatter(truths, preds, alpha=0.5, s=12, color="C0")
    ax1.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.6, label="y=x")
    ax1.set_xlabel("true tau_i"); ax1.set_ylabel("model tau_hat_i")
    rho = float(np.corrcoef(preds, truths)[0, 1])
    ax1.set_title(f"Model (Pearson = {rho:.3f})")
    ax1.set_xlim([-lim, lim]); ax1.set_ylim([-lim, lim])
    ax1.axhline(0, color="gray", linewidth=0.5); ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.legend()

    ax2.scatter(truths, assocs, alpha=0.5, s=12, color="C1")
    ax2.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.6, label="y=x")
    ax2.set_xlabel("true tau_i"); ax2.set_ylabel("marginal assoc baseline")
    rho_a = float(np.corrcoef(assocs, truths)[0, 1])
    ax2.set_title(f"Marginal baseline (Pearson = {rho_a:.3f})")
    ax2.set_xlim([-lim, lim]); ax2.set_ylim([-lim, lim])
    ax2.axhline(0, color="gray", linewidth=0.5); ax2.axvline(0, color="gray", linewidth=0.5)
    ax2.legend()

    fig.suptitle(f"Predicted vs true tau_i over {n_scms} random held-out SCMs (p={p}, n={n_rows})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} and {pdf_path}")


def _motif_bar(motif_name: str, motif_data: dict, out_path: Path) -> None:
    rows = motif_data["rows"]
    feats = [f"X{r['feature']+1}" for r in rows]
    true = [r["tau_true_mean"] for r in rows]
    hat = [r["tau_hat_mean"] for r in rows]
    assoc = [r["assoc_mean"] for r in rows]

    x = np.arange(len(feats))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, true, width=w, color="C2", label="true tau (causal)")
    ax.bar(x, hat, width=w, color="C0", label="model tau_hat")
    ax.bar(x + w, assoc, width=w, color="C1", label="marginal baseline")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(feats)
    ax.set_ylabel("tau (standardized contrast)")
    ax.set_title(f"Motif {motif_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _motifs_combined(eval_data: dict, out_path: Path) -> None:
    """2x3 grid of all six motifs as a single multi-panel figure.

    Used by paper/main.tex (\\cref{fig:motifs-combined}). Each panel is the
    same bar plot as ``_motif_bar`` but the y-axis is shared across panels
    so motif magnitudes can be compared visually.
    """
    motifs = eval_data["motifs"]
    panels = [
        ("A_direct_cause",        "A: direct cause $X_1 \\!\\to\\! Y$"),
        ("B_proxy",               "B: proxy $X_1 \\!\\to\\! Y,\\; X_1 \\!\\to\\! X_2$"),
        ("C_observed_confounder", "C: observed confounder $Z \\!\\to\\! X_1, Y$"),
        ("D_mediator",            "D: mediator $X_1 \\!\\to\\! X_2 \\!\\to\\! Y$"),
        ("E_target_descendant",   "E: target descendant $Y \\!\\to\\! X_1$"),
        ("F_collider_leakage",    "F: collider $X_1 \\!\\to\\! C \\!\\leftarrow\\! Y$"),
    ]
    # Compute global y-axis range so all panels share scale.
    y_max = 0.0
    for key, _ in panels:
        for r in motifs[key]["rows"]:
            y_max = max(y_max, abs(r["tau_true_mean"]),
                        abs(r["tau_hat_mean"]), abs(r["assoc_mean"]))
    y_lim = y_max * 1.15

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharey=True)
    for ax, (key, title) in zip(axes.flat, panels):
        rows = motifs[key]["rows"]
        feats = [f"$X_{{{r['feature']+1}}}$" for r in rows]
        if key == "E_target_descendant":
            # Y is at Z-index 0, features map to Z 1..p; relabel as X_1..X_p.
            feats = [f"$X_{{{r['feature']+1}}}$" for r in rows]
        true = [r["tau_true_mean"] for r in rows]
        hat = [r["tau_hat_mean"] for r in rows]
        assoc = [r["assoc_mean"] for r in rows]

        x = np.arange(len(feats))
        w = 0.27
        ax.bar(x - w, true, width=w, color="C2", label="true $\\tau$")
        ax.bar(x,     hat,  width=w, color="C0", label="probe $\\hat\\tau$")
        ax.bar(x + w, assoc, width=w, color="C1", label="marginal")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(feats)
        ax.set_title(title, fontsize=10)
        ax.set_ylim([-y_lim, y_lim])

    axes[0, 0].set_ylabel("$\\tau$ (standardized)")
    axes[1, 0].set_ylabel("$\\tau$ (standardized)")
    # One legend for the whole figure.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    # Companion PNG next to the PDF, same basename.
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} and {png_path}")


def _metrics_bar(random_metrics: dict, out_path: Path) -> None:
    metrics = [
        ("Pearson", "pearson"),
        ("R^2", "r2"),
        ("AUROC nonzero", "auroc_nonzero"),
        ("MAE zero-effect", "mae_zero_features"),
    ]
    labels = ["model", "marginal", "multivariate"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(metrics))
    w = 0.25
    for i, lab in enumerate(labels):
        vals = []
        for _, key in metrics:
            v = random_metrics.get(key, {}).get(lab, np.nan)
            vals.append(v)
        ax.bar(x + (i - 1) * w, vals, width=w, label=lab)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics])
    ax.set_title("Random held-out SCMs: model vs baselines")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="causal_probe/results/probe_5k.ckpt")
    ap.add_argument("--eval-json", default="causal_probe/results/eval_5k.json")
    ap.add_argument("--out-dir", default="causal_probe/results")
    ap.add_argument("--n-scatter-scms", type=int, default=500)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(args.ckpt, device=args.device)

    with open(args.eval_json) as f:
        eval_data = json.load(f)

    _scatter_plot(model, args.n_scatter_scms, args.p, args.n_rows,
                  args.seed, args.device, out_dir / "scatter_true_vs_pred.png")
    for name, mdata in eval_data["motifs"].items():
        _motif_bar(name, mdata, out_dir / f"motif_{name}.png")
    _motifs_combined(eval_data, out_dir / "motifs_combined.pdf")
    _metrics_bar(eval_data["random"], out_dir / "metrics_bar.png")


if __name__ == "__main__":
    main()
