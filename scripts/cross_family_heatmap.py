"""Cross-family generalization sweep: train family x eval family.

Reuses the 7 p=5 checkpoints from tab:held-out-dags and re-evaluates each
on every SCM family (not just the matching one). Produces a 7x6 matrix
for Pearson and AUROC-nonzero, plotted as side-by-side heatmaps.

Cells are cached in cross_family_eval.json so reruns only compute new ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_probe.eval import _load_model, evaluate_random


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "causal_probe" / "results"
CACHE = RESULTS / "cross_family_eval.json"
OUT_PNG = REPO / "notes" / "cross_family_heatmap.png"
OUT_PDF = REPO / "paper" / "figures" / "cross_family_heatmap.pdf"


# (row label, ckpt filename, (train family, train noise))
# The (family, noise) tuple is the "tag" used to identify the diagonal cell.
# The 8th "mixture (foundation)" row uses ("mixture", "laplace") which doesn't
# match any eval-column tag — so no cell gets a diagonal outline for it (correct:
# the mixture probe is never strictly in-distribution for any single eval family).
TRAIN_ROWS: list[tuple[str, str, tuple[str, str]]] = [
    ("linear (Laplace)",   "probe_main_p5_15k.ckpt",         ("linear", "laplace")),
    ("linear (Gaussian)",  "probe_gauss_control_p5.ckpt",    ("linear", "gaussian")),
    ("nonlinear (poly)",   "probe_nonlinear_p5_20k_v2.ckpt", ("nonlinear", "laplace")),
    ("random-MLP",         "probe_mlp_p5_20k.ckpt",          ("mlp", "laplace")),
    ("hidden conf.",       "probe_hidden_p5_20k.ckpt",       ("hidden", "laplace")),
    ("MLP+hidden conf.",   "probe_mlp_hidden_p5_25k.ckpt",   ("mlp_hidden", "laplace")),
    ("mixed bin/cont.",    "probe_mixed_p5_20k.ckpt",        ("mixed", "laplace")),
    ("mixture (foundation)", "probe_main_p5_50k_mixture.ckpt", ("mixture", "laplace")),
]

# (col label, scm_type, noise) passed to evaluate_random.
# Noise only affects linear and hidden families; ignored by the rest.
EVAL_COLS: list[tuple[str, str, str]] = [
    ("linear (Laplace)",  "linear",     "laplace"),
    ("linear (Gaussian)", "linear",     "gaussian"),
    ("nonlinear",         "nonlinear",  "laplace"),
    ("random-MLP",        "mlp",        "laplace"),
    ("hidden",            "hidden",     "laplace"),
    ("MLP+hidden",        "mlp_hidden", "laplace"),
    ("mixed",             "mixed",      "laplace"),
]

P = 5
N_ROWS = 512
N_SCMS = 200
SEED = 100


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=2))


def cell_key(ckpt_name: str, eval_family: str, eval_noise: str) -> str:
    # Old cache entries (pre-noise) used "ckpt::family"; treat those as
    # equivalent to laplace so they stay valid for non-Gaussian columns.
    if eval_noise == "laplace":
        return f"{ckpt_name}::{eval_family}"
    return f"{ckpt_name}::{eval_family}::{eval_noise}"


def run_sweep() -> dict:
    cache = load_cache()
    n_train = len(TRAIN_ROWS)
    n_eval = len(EVAL_COLS)
    pearson = np.full((n_train, n_eval), np.nan)
    auroc = np.full((n_train, n_eval), np.nan)

    for i, (row_label, ckpt_name, _) in enumerate(TRAIN_ROWS):
        ckpt_path = RESULTS / ckpt_name
        model = None
        for j, (col_label, eval_family, eval_noise) in enumerate(EVAL_COLS):
            key = cell_key(ckpt_name, eval_family, eval_noise)
            if key in cache:
                pe = cache[key]["pearson"]
                au = cache[key]["auroc"]
                print(f"[cache] {row_label:>20s} -> {col_label:<18s}  pearson={pe:+.3f}  auroc={au:+.3f}")
            else:
                if model is None:
                    model = _load_model(str(ckpt_path), device="cpu")
                rng = np.random.default_rng(SEED + i * 17 + j)
                res = evaluate_random(
                    model, n_scms=N_SCMS, p=P, n_rows=N_ROWS, rng=rng,
                    device="cpu", scm_type=eval_family, noise=eval_noise,
                )
                pe = res["pearson"]["model"]
                au = res["auroc_nonzero"]["model"]
                cache[key] = {"pearson": pe, "auroc": au}
                save_cache(cache)
                print(f"[run]   {row_label:>20s} -> {col_label:<18s}  pearson={pe:+.3f}  auroc={au:+.3f}")
            pearson[i, j] = pe
            auroc[i, j] = au

    return {"pearson": pearson, "auroc": auroc,
            "row_labels": [r[0] for r in TRAIN_ROWS],
            "col_labels": [c[0] for c in EVAL_COLS],
            "train_tags": [r[2] for r in TRAIN_ROWS],
            "eval_tags": [(c[1], c[2]) for c in EVAL_COLS]}


def plot(data: dict) -> None:
    rows = data["row_labels"]
    cols = data["col_labels"]
    train_tags = data["train_tags"]
    eval_tags = data["eval_tags"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    for ax, mat, title, vmin, vmax in [
        (axes[0], data["pearson"], "Pearson(model, true tau)", 0.0, 1.0),
        (axes[1], data["auroc"],   "AUROC nonzero-feature detection", 0.5, 1.0),
    ]:
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=35, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_xlabel("Eval SCM family")
        ax.set_ylabel("Training SCM family")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                # white text on dark cells, black on bright
                norm = (v - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if norm < 0.55 else "black"
                weight = "bold" if train_tags[i] == eval_tags[j] else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=9, fontweight=weight)
        # outline diagonal (matching train/eval) cells
        for i, tag in enumerate(train_tags):
            if tag in eval_tags:
                j = eval_tags.index(tag)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="red",
                                           linewidth=1.8))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"FADO cross-family generalization (p={P}, n_rows={N_ROWS}, "
        f"{N_SCMS} held-out SCMs/cell). Red outline = matching train/eval; "
        "bold = diagonal cell.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160)
    print(f"\nwrote {OUT_PNG}")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    data = run_sweep()
    plot(data)
