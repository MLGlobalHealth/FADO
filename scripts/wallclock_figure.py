"""Render the wallclock comparison from cached timings.

Reads causal_probe/results/wallclock_baselines.json (filled by
scripts/wallclock_baselines.py on local CPU + cluster GPU runs) and writes:
  - notes/wallclock_baselines.png  — bar chart, log-y per-call latency
  - prints a markdown table to stdout
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "causal_probe" / "results" / "wallclock_baselines.json"
OUT_PNG = REPO / "notes" / "wallclock_baselines.png"

# Pretty names + grouping. CausalPFN is omitted from this comparison
# because its native output is a CATE for a user-designated treatment,
# not the length-p per-feature contrast vector this table times.
PRETTY = {
    "marginal":      ("Marginal corr.",       "predictive"),
    "ridge":         ("Multivariate ridge",   "predictive"),
    "shap":          ("LightGBM-SHAP",        "predictive"),
    "doubleml":      ("DoubleML (LinearDML)", "per-feature CATE"),
    "causal_forest": ("Causal Forest",        "per-feature CATE"),
    "probe":         ("FADO (ours)",          "amortized"),
}

# Color per category
CAT_COLOR = {
    "predictive":       "#5B8DD3",
    "per-feature CATE": "#D38C5B",
    "amortized":        "#5BD37D",
}

# FADO foundation training cost (sacct, job 7553961, probe_main_p5_15k)
FADO_TRAIN_GPU_HOURS = 14.2 / 60  # 14m 12s on 1× L40s


def load_cells():
    if not CACHE.exists():
        raise SystemExit(f"missing cache: {CACHE}")
    return json.loads(CACHE.read_text())


def plot(cells: dict) -> None:
    # Two panels: p=5 and p=13. For each method, prefer the device for
    # which it's natively reported (causalpfn -> cuda if avail, else cpu;
    # probe -> cuda if avail else cpu; everything else -> cpu).
    methods_order = ["marginal", "ridge", "shap", "doubleml", "causal_forest",
                     "probe"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)

    for ax, p in zip(axes, [5, 13]):
        labels, medians, p25, p75, colors, devices = [], [], [], [], [], []
        for m in methods_order:
            label, cat = PRETTY[m]
            # Prefer cuda for amortized methods if cell exists
            preferred = "cuda" if cat == "amortized" else "cpu"
            key_pref = f"{m}::{preferred}::p{p}::n512"
            key_cpu  = f"{m}::cpu::p{p}::n512"
            cell = cells.get(key_pref) or cells.get(key_cpu)
            if not cell or "median_s" not in cell:
                continue
            dev = preferred if (cells.get(key_pref) is not None) else "cpu"
            labels.append(f"{label}\n({dev.upper()})")
            medians.append(cell["median_s"])
            p25.append(cell["p25_s"])
            p75.append(cell["p75_s"])
            colors.append(CAT_COLOR[cat])
            devices.append(dev)

        x = np.arange(len(labels))
        med = np.array(medians)
        err = np.vstack([med - np.array(p25), np.array(p75) - med])
        ax.bar(x, med, color=colors, yerr=err, capsize=3, ecolor="0.3")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Median seconds per SCM (log)")
        ax.set_title(f"p = {p}, n_rows = 512, 20 SCMs")
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)
        # Annotate value above each bar
        for xi, mi, di in zip(x, med, devices):
            unit = "s"
            display = f"{mi:.3f}"
            if mi < 0.001:
                display = f"{mi*1000:.2f}"; unit = "ms"
            elif mi < 1:
                display = f"{mi*1000:.0f}"; unit = "ms"
            ax.text(xi, mi * 1.5, f"{display}{unit}",
                    ha="center", va="bottom", fontsize=8)

    legend_elems = [plt.Rectangle((0, 0), 1, 1, color=c, label=k)
                    for k, c in CAT_COLOR.items()]
    axes[0].legend(handles=legend_elems, loc="upper left", frameon=True,
                   fontsize=9)
    fig.suptitle(
        f"Per-call wallclock at inference time. "
        f"FADO foundation training: {FADO_TRAIN_GPU_HOURS*60:.1f} min "
        f"(0.24 GPU-h) on 1× L40s, paid once.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160)
    print(f"wrote {OUT_PNG}")


def print_table(cells: dict) -> None:
    print("\n## Per-call latency (median seconds per SCM, p=5, n_rows=512)\n")
    print(f"| Method | Category | CPU (s/SCM) | GPU (s/SCM) | n |")
    print(f"|---|---|---:|---:|---:|")
    for m in ["marginal", "ridge", "shap", "doubleml", "causal_forest",
              "probe"]:
        label, cat = PRETTY[m]
        cpu = cells.get(f"{m}::cpu::p5::n512", {})
        gpu = cells.get(f"{m}::cuda::p5::n512", {})
        cpu_s = f"{cpu.get('median_s', float('nan')):.4f}" if cpu else "—"
        gpu_s = f"{gpu.get('median_s', float('nan')):.4f}" if gpu else "—"
        n_cpu = cpu.get("n_completed", 0)
        n_gpu = gpu.get("n_completed", 0)
        n_str = f"{n_cpu}cpu / {n_gpu}gpu" if n_gpu else str(n_cpu)
        print(f"| {label} | {cat} | {cpu_s} | {gpu_s} | {n_str} |")

    print(f"\nFADO foundation training: 14m 12s on 1× L40s GPU "
          f"({FADO_TRAIN_GPU_HOURS:.2f} GPU-h, 449k params, 15k steps).")


if __name__ == "__main__":
    cells = load_cells()
    print_table(cells)
    plot(cells)
