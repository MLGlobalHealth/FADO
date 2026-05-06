"""Quick mockups of three alternative renderings for Figure 4.

Generates three separate PNGs in /tmp/ for side-by-side comparison:
  /tmp/fig4_alt1_overlay.png       — single panel, probe + marginal overlaid
  /tmp/fig4_alt2_null_hist.png     — histogram of estimator output on null features
  /tmp/fig4_alt3_calibration.png   — binned calibration curve, mean +/- 1 sigma

Reuses the same model + 500-SCM data as causal_probe/plots.py:_scatter_plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model
from causal_probe.scm import LinearNonGaussianSCM


CKPT = "causal_probe/results/probe_main_p5_50k.ckpt"
N_SCMS = 500
P = 5
N_ROWS = 512
SEED = 777
DEVICE = "cpu"
OUT_DIR = Path("/tmp")


def gather():
    rng = np.random.default_rng(SEED)
    model = _load_model(CKPT, device=DEVICE)
    preds, truths, assocs = [], [], []
    for _ in range(N_SCMS):
        scm = LinearNonGaussianSCM(p=P, rng=np.random.default_rng(rng.integers(0, 2**31)))
        samp = scm.sample(n=N_ROWS, rng=np.random.default_rng(rng.integers(0, 2**31)))
        X_t = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(DEVICE)
        y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        preds.append(pred); truths.append(samp.tau)
        assocs.append(BASELINES["marginal"](samp.X, samp.y))
    return (np.stack(preds).reshape(-1),
            np.stack(truths).reshape(-1),
            np.stack(assocs).reshape(-1))


def alt1_overlay(preds, truths, assocs):
    """Single panel: probe + marginal scatter overlaid on same axes, low alpha."""
    lim = max(np.max(np.abs(preds)), np.max(np.abs(truths)),
              np.max(np.abs(assocs))) * 1.1
    fig, ax = plt.subplots(figsize=(6.2, 6))
    rho_p = float(np.corrcoef(preds, truths)[0, 1])
    rho_m = float(np.corrcoef(assocs, truths)[0, 1])
    ax.scatter(truths, assocs, alpha=0.18, s=10, color="C1",
               label=f"marginal baseline (Pearson = {rho_m:.3f})")
    ax.scatter(truths, preds, alpha=0.18, s=10, color="C0",
               label=f"probe $\\hat\\tau$ (Pearson = {rho_p:.3f})")
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.6, label="$y=x$")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("true $\\tau_i$")
    ax.set_ylabel("estimator output")
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim])
    ax.set_title(f"Alt 1: overlay (probe vs marginal, {N_SCMS} held-out SCMs)")
    leg = ax.legend(loc="upper left", framealpha=0.9)
    for lh in leg.legend_handles:
        try:
            lh.set_alpha(0.9)
        except Exception:
            pass
    fig.tight_layout()
    out = OUT_DIR / "fig4_alt1_overlay.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def alt2_null_hist(preds, truths, assocs, eps=0.05):
    """Histogram of estimator output, restricted to features with |true tau| < eps."""
    null_mask = np.abs(truths) < eps
    n_null = int(null_mask.sum())
    p_null = preds[null_mask]
    a_null = assocs[null_mask]
    lim = max(np.max(np.abs(p_null)), np.max(np.abs(a_null))) * 1.05
    bins = np.linspace(-lim, lim, 51)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(a_null, bins=bins, alpha=0.55, color="C1",
            label=f"marginal baseline  (std = {a_null.std():.3f})")
    ax.hist(p_null, bins=bins, alpha=0.55, color="C0",
            label=f"probe $\\hat\\tau$  (std = {p_null.std():.3f})")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("estimator output on null features")
    ax.set_ylabel("count")
    ax.set_title(f"Alt 2: null-feature histogram "
                 f"($|true\\,\\tau| < {eps}$, n = {n_null} of {len(truths)})")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fig4_alt2_null_hist.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def alt3_calibration(preds, truths, assocs, n_bins=12):
    """Binned calibration: mean +/- 1 sigma of estimator within true-tau bins."""
    lo, hi = float(np.quantile(truths, 0.005)), float(np.quantile(truths, 0.995))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def stats(y):
        means, stds = [], []
        for j in range(n_bins):
            m = (truths >= edges[j]) & (truths < edges[j + 1])
            if m.sum() < 3:
                means.append(np.nan); stds.append(np.nan)
            else:
                means.append(y[m].mean()); stds.append(y[m].std())
        return np.asarray(means), np.asarray(stds)

    mp, sp = stats(preds)
    ma, sa = stats(assocs)

    lim = max(np.nanmax(np.abs(mp + sp)), np.nanmax(np.abs(ma + sa)),
              abs(lo), abs(hi)) * 1.05
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.6, label="$y=x$")
    ax.fill_between(centers, ma - sa, ma + sa, color="C1", alpha=0.25)
    ax.plot(centers, ma, "o-", color="C1",
            label=f"marginal baseline (Pearson = {np.corrcoef(assocs, truths)[0,1]:.3f})")
    ax.fill_between(centers, mp - sp, mp + sp, color="C0", alpha=0.25)
    ax.plot(centers, mp, "o-", color="C0",
            label=f"probe $\\hat\\tau$ (Pearson = {np.corrcoef(preds, truths)[0,1]:.3f})")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("true $\\tau_i$ (bin center)")
    ax.set_ylabel("estimator output (mean $\\pm$ 1$\\sigma$ within bin)")
    ax.set_title(f"Alt 3: binned calibration ({n_bins} bins, {N_SCMS} SCMs)")
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim])
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "fig4_alt3_calibration.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    print(f"running probe on {N_SCMS} held-out SCMs (p={P}, n={N_ROWS}) ...")
    preds, truths, assocs = gather()
    print(f"got {len(truths)} feature-level points")
    alt1_overlay(preds, truths, assocs)
    alt2_null_hist(preds, truths, assocs)
    alt3_calibration(preds, truths, assocs)


if __name__ == "__main__":
    main()
