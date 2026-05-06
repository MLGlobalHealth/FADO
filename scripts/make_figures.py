"""Regenerate paper figures from CSVs in results/.

    uv run python scripts/make_figures.py --which all
    uv run python scripts/make_figures.py --which f_speed f_quintet

Each figure is a function `fig_<name>` returning True on success, False if
its input CSVs are missing. Missing inputs are warned, never fatal — Agent E
may run this while Agents B/C/D are still writing outputs.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["font.size"] = 9
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGDIR = REPO / "paper" / "figures"
TABDIR = REPO / "paper" / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

ONECOL = (3.25, 2.4)
TWOCOL = (6.75, 3.5)
BIG8 = (6.75, 8.0)

METHOD_COLORS = {
    "oracle": "C0",
    # kshap / permutation / marginal have two naming conventions in the
    # CSVs: the realdata harness (realdata_greedy.py) writes the short
    # names kernelshap/permutation/marginal_correlation, while the
    # synthetic acquisition bench (tabicl.eval.acquisition) writes the
    # long names kernel_shap/permutation_importance/marginal_correlation.
    # Map both so colours are consistent across figures.
    "kernelshap": "C1",
    "kernel_shap": "C1",
    "permutation": "C2",
    "permutation_importance": "C2",
    "marginal": "C3",
    "marginal_correlation": "C3",
    "random": "C7",
    "expensive_refit": "C4",
}

DATASETS_LOCKED = ["bike_sharing_hourly", "adult_income", "breast_cancer_wi"]
DATASETS_EXTENDED_DEFAULT = [
    "california_housing",
    "wine_quality_red",
    "phoneme",
    "credit_g",
    "magic_telescope",
]


def _warn_missing(name: str, path: str) -> None:
    print(f"[skip] {name}: missing {path}", file=sys.stderr)
    if name == "f_realdata_aufc_table":
        return  # this target is a .tex snippet, not a PDF — placeholder already exists
    out = FIGDIR / f"{name}.pdf"
    if not out.exists():
        fig, ax = plt.subplots(figsize=ONECOL, constrained_layout=True)
        ax.text(
            0.5, 0.5,
            f"{name}\n(awaiting CSV: {Path(path).name})",
            ha="center", va="center", fontsize=9, color="grey",
        )
        ax.set_axis_off()
        _save(fig, out)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO)}")


# ---------- §11.1 label fidelity ----------

def fig_synth_label_fidelity() -> bool:
    """f_synth_label_fidelity: Spearman rho by |S| stratum, one panel per metric row."""
    csv = RESULTS / "eval_heads_s11_1.csv"
    if not csv.exists():
        _warn_missing("f_synth_label_fidelity", str(csv))
        return False
    df = pd.read_csv(csv)
    metric_col = "metric" if "metric" in df.columns else None
    stratum_col = "stratum" if "stratum" in df.columns else ("S_stratum" if "S_stratum" in df.columns else None)
    value_col = "spearman" if "spearman" in df.columns else ("value" if "value" in df.columns else None)
    if stratum_col is None or value_col is None:
        print(f"[skip] f_synth_label_fidelity: expected stratum+spearman columns in {csv.name}", file=sys.stderr)
        return False

    if metric_col is None:
        metrics = [("spearman", df)]
    else:
        metrics = list(df.groupby(metric_col))
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(min(TWOCOL[0], 2.0 * n + 1.5), 2.4), sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    strata_order = sorted(df[stratum_col].unique(), key=str)
    for ax, (mname, mdf) in zip(axes, metrics):
        groups = [mdf.loc[mdf[stratum_col] == s, value_col].dropna().to_numpy() for s in strata_order]
        ax.boxplot(groups, tick_labels=[str(s) for s in strata_order], widths=0.6)
        ax.axhline(0.60, color="k", linestyle="--", linewidth=0.8, label="primary ρ≥0.60")
        ax.axhline(0.50, color="grey", linestyle=":", linewidth=0.8, label="secondary ρ≥0.50")
        ax.set_xlabel("|S| stratum")
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(str(mname), fontsize=9)
    axes[-1].legend(loc="lower right", fontsize=7, frameon=False)
    _save(fig, FIGDIR / "f_synth_label_fidelity.pdf")
    return True


# ---------- §11.2 endpoint calibration ----------

def fig_endpoint_calibration() -> bool:
    """Render per-dataset Spearman distributions for s_i and n_i endpoints.

    tabicl.eval.eval_heads writes summary rows per dataset with columns
    ``spearman_s, pearson_s, mae_s, spearman_n, pearson_n, mae_n`` plus a
    pooled ``calib_slope, calib_intercept`` on the trailing "mean" row
    (blank for per-dataset rows). Raw (oracle, predicted) pairs are not
    exposed, so we plot the distribution of per-dataset Spearman values
    and annotate the pooled calibration slope from the mean row.
    """
    csv = RESULTS / "eval_heads_s11_2.csv"
    if not csv.exists():
        _warn_missing("f_endpoint_calibration", str(csv))
        return False
    df = pd.read_csv(csv)
    needed = {"dataset_id", "spearman_s", "spearman_n"}
    if not needed.issubset(df.columns):
        print(f"[skip] f_endpoint_calibration: expected {needed}, got {list(df.columns)}", file=sys.stderr)
        return False

    # Separate per-dataset rows from the pooled "mean" row.
    per_ds = df[df["dataset_id"] != "mean"].copy()
    mean_row = df[df["dataset_id"] == "mean"]
    pooled_s_slope = float(mean_row["calib_slope"].iloc[0]) if (
        len(mean_row) and "calib_slope" in mean_row.columns
        and not pd.isna(mean_row["calib_slope"].iloc[0])
    ) else None

    fig, axes = plt.subplots(1, 2, figsize=TWOCOL, constrained_layout=True)
    for ax, col, label in [
        (axes[0], "spearman_s", "s_i (sufficiency)"),
        (axes[1], "spearman_n", "n_i (necessity)"),
    ]:
        vals = pd.to_numeric(per_ds[col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            ax.text(0.5, 0.5, f"no finite {col}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=9)
            continue
        ax.hist(vals, bins=min(30, max(5, len(vals) // 4)), color="C0", alpha=0.75,
                label=f"per-dataset ρ (n={len(vals)})")
        mean_rho = float(np.mean(vals))
        ax.axvline(mean_rho, color="C0", linestyle=":", linewidth=1.0,
                   label=f"mean ρ={mean_rho:.3f}")
        ax.axvline(0.60, color="k", linestyle="--", linewidth=0.8, label="primary ρ≥0.60")
        ax.axvline(0.50, color="grey", linestyle=":", linewidth=0.8, label="secondary ρ≥0.50")
        ax.set_xlabel(f"Spearman ρ({label})")
        ax.set_ylabel("dataset count")
        ax.set_title(label, fontsize=9)
        ax.legend(loc="upper left", fontsize=6, frameon=False)
    if pooled_s_slope is not None:
        fig.text(
            0.01, 0.02,
            f"pooled OLS calib_slope(s_i) = {pooled_s_slope:.3f}  "
            f"(preregistered band [0.7, 1.3])",
            fontsize=7, color="grey",
        )
    _save(fig, FIGDIR / "f_endpoint_calibration.pdf")
    return True


# ---------- §11.3 quintet ----------

def fig_quintet() -> bool:
    """Render the 5-panel quintet figure from tabicl.eval.quintet's CSV.

    tabicl.eval.quintet writes rows with columns ``panel, feature, S_state,
    predicted_rms`` plus trailing PASS/FAIL summary rows where ``feature``
    is set to "PASS"/"FAIL" (those aren't plottable and are filtered out).
    """
    csv = RESULTS / "quintet_11_3.csv"
    if not csv.exists():
        _warn_missing("f_quintet", str(csv))
        return False
    df = pd.read_csv(csv)
    if not {"panel", "feature", "S_state", "predicted_rms"}.issubset(df.columns):
        print(f"[skip] f_quintet: expected panel,feature,S_state,predicted_rms columns in {csv.name}", file=sys.stderr)
        return False

    # Drop the trailing per-panel PASS/FAIL rows — feature=="PASS"/"FAIL".
    plot_df = df[~df["feature"].astype(str).isin({"PASS", "FAIL"})].copy()

    fig = plt.figure(figsize=TWOCOL, constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    panel_axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[0, 2]),
        "D": fig.add_subplot(gs[0, 3]),
        "E": fig.add_subplot(gs[1, :]),
    }
    for panel in ["A", "B", "C", "D"]:
        ax = panel_axes[panel]
        sub = plot_df[plot_df["panel"] == panel]
        # Plot the empty-S baseline by default (most common bar chart for
        # the hand-designed panels); if it's absent fall back to any state.
        if "empty" in set(sub["S_state"]):
            sub = sub[sub["S_state"] == "empty"]
        else:
            first_state = sorted(sub["S_state"].unique())[0] if len(sub) else None
            if first_state is not None:
                sub = sub[sub["S_state"] == first_state]
        ax.bar(sub["feature"].astype(str), sub["predicted_rms"], color="C0")
        ax.set_title(f"Panel {panel}", fontsize=9)
        ax.set_ylabel("r̂_{i|S}")
        ax.tick_params(axis="x", labelsize=7)
    ax_e = panel_axes["E"]
    sub_e = plot_df[plot_df["panel"] == "E"]
    if len(sub_e):
        pivot = sub_e.pivot_table(index="S_state", columns="feature", values="predicted_rms", aggfunc="mean")
        im = ax_e.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        ax_e.set_xticks(range(pivot.shape[1]))
        ax_e.set_xticklabels(pivot.columns, fontsize=7)
        ax_e.set_yticks(range(pivot.shape[0]))
        ax_e.set_yticklabels(pivot.index, fontsize=7)
        ax_e.set_title("Panel E: r̂_{i|S} across locked S states", fontsize=9)
        fig.colorbar(im, ax=ax_e, shrink=0.8)
    else:
        ax_e.set_title("Panel E: no rows", fontsize=9)
    _save(fig, FIGDIR / "f_quintet.pdf")
    return True


# ---------- §11.4 synth AUFC ----------

def fig_synth_aufc() -> bool:
    suites_dir = RESULTS / "bench_explainer_eval"
    if not suites_dir.is_dir():
        _warn_missing("f_synth_aufc", str(suites_dir))
        return False
    csvs = sorted(suites_dir.glob("*.csv"))
    if not csvs:
        _warn_missing("f_synth_aufc", f"{suites_dir}/*.csv")
        return False

    frames = []
    for c in csvs:
        d = pd.read_csv(c)
        d["suite"] = c.stem
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    method_col = "method" if "method" in df.columns else None
    aufc_col = "aufc" if "aufc" in df.columns else ("normalized_aufc" if "normalized_aufc" in df.columns else None)
    if method_col is None or aufc_col is None:
        print(f"[skip] f_synth_aufc: expected method,aufc in {csvs[0].name}", file=sys.stderr)
        return False

    suites = sorted(df["suite"].unique())
    # Synthetic bench (tabicl.eval.acquisition) writes method names with
    # underscores; realdata harness uses short names. Prefer the synth-
    # style long names here, but fall back to whatever the CSV has.
    methods = ["oracle", "kernel_shap", "permutation_importance", "marginal_correlation", "random"]
    methods = [m for m in methods if m in set(df[method_col])] or sorted(df[method_col].unique())

    fig, ax = plt.subplots(figsize=TWOCOL, constrained_layout=True)
    x = np.arange(len(suites))
    width = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        vals = [df[(df["suite"] == s) & (df[method_col] == m)][aufc_col].mean() for s in suites]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=m, color=METHOD_COLORS.get(m, f"C{i}"))
    ax.set_xticks(x)
    ax.set_xticklabels(suites)
    ax.set_ylabel("normalized AUFC")
    ax.legend(fontsize=7, frameon=False, ncol=len(methods))
    _save(fig, FIGDIR / "f_synth_aufc.pdf")
    return True


# ---------- §11.5 speed ----------

def fig_speed() -> bool:
    csv = RESULTS / "bench_attribution_latency.csv"
    if not csv.exists():
        _warn_missing("f_speed", str(csv))
        return False
    df = pd.read_csv(csv)
    # bench_attribution_latency.csv schema:
    #   method,n_shap_samples,n_train,p,n_explain,wall_clock_s,timed_out,notes
    # oracle total cost at n_Q=n_explain is tabicl_explainer_fit (includes base.fit)
    # + n_Q * head_c time. KernelSHAP cost is its wall_clock_s at each n_shap_samples.
    df = df.copy()
    df["p"] = df["p"].astype(int)
    df["n_train"] = df["n_train"].astype(int)

    # Build a row per (p, n_train) with oracle_total and kshap_min across budgets.
    grid = df[["p", "n_train"]].drop_duplicates().sort_values(["n_train", "p"]).reset_index(drop=True)
    rows = []
    for _, g in grid.iterrows():
        p, n = int(g["p"]), int(g["n_train"])
        sub = df[(df["p"] == p) & (df["n_train"] == n)]
        fit = sub.loc[sub["method"] == "tabicl_explainer_fit", "wall_clock_s"]
        head = sub.loc[sub["method"] == "head_c_medium", "wall_clock_s"]
        n_q = int(sub["n_explain"].iloc[0]) if len(sub) else 10
        oracle_total = (float(fit.iloc[0]) if len(fit) else np.nan) + n_q * (float(head.iloc[0]) if len(head) else np.nan)
        ks = sub[(sub["method"] == "kernel_shap") & (~sub["timed_out"].fillna(False).astype(bool))]
        kshap_best = float(ks["wall_clock_s"].min()) if len(ks) else np.nan
        ks_timeouts = sub[(sub["method"] == "kernel_shap") & (sub["timed_out"].fillna(False).astype(bool))]
        rows.append(
            dict(
                p=p,
                n_train=n,
                n_q=n_q,
                oracle_total=oracle_total,
                kshap_best=kshap_best,
                kshap_timed_out=len(ks_timeouts) > 0 and not np.isfinite(kshap_best),
            )
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        print("[skip] f_speed: empty summary after processing", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=TWOCOL, constrained_layout=True)
    labels = [f"n={r.n_train}\np={r.p}" for r in summary.itertuples()]
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, summary["oracle_total"], width, label="oracle (fit + 10 queries)", color=METHOD_COLORS["oracle"])
    kshap_plot = summary["kshap_best"].fillna(300.0).to_numpy()
    bars = ax.bar(x + width / 2, kshap_plot, width, label="KernelSHAP (best budget)", color=METHOD_COLORS["kernelshap"])
    for bar, row in zip(bars, summary.itertuples()):
        if row.kshap_timed_out:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "†", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("wall-clock seconds (log)")
    ax.axhline(300.0, color="grey", linestyle=":", linewidth=0.8, label="300s KernelSHAP cap")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.text(
        0.01,
        0.02,
        "† KernelSHAP cap reached at 300s (reported value is attempt time).",
        transform=ax.transAxes,
        fontsize=6,
        color="grey",
    )
    _save(fig, FIGDIR / "f_speed.pdf")
    return True


# ---------- §11.6 realdata paths ----------

def _parse_realdata_folder(name: str) -> tuple[str, int] | None:
    """Parse ``realdata_<dataset>_<jobid>`` or ``realdata_<dataset>_seed<N>_<jobid>``.

    Returns ``(dataset, seed)`` or None if the name is malformed. Seed is 0
    when the folder has no ``_seed<N>`` infix (matches bench_realdata.slurm's
    convention that seed 0 stays on the preregistered path).
    """
    if not name.startswith("realdata_") or name.startswith("realdata_greedy_smoke"):
        return None
    stem = name[len("realdata_"):]
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    ds_raw, _jobid = parts
    m = re.match(r"^(.*)_seed(\d+)$", ds_raw)
    if m:
        return m.group(1), int(m.group(2))
    return ds_raw, 0


def _find_realdata_csvs(suffix: str) -> dict[str, dict[int, Path]]:
    """Map dataset -> {seed -> latest matching CSV under results/realdata_*/}."""
    out: dict[str, dict[int, Path]] = {}
    for d in sorted(RESULTS.glob("realdata_*")):
        if not d.is_dir():
            continue
        parsed = _parse_realdata_folder(d.name)
        if parsed is None:
            continue
        dataset, seed = parsed
        csvs = list(d.glob(f"{dataset}_{suffix}"))
        if not csvs:
            csvs = list(d.glob(f"*_{suffix}"))
        if not csvs:
            continue
        cand = csvs[0]
        seeds = out.setdefault(dataset, {})
        prev = seeds.get(seed)
        if prev is None or cand.stat().st_mtime > prev.stat().st_mtime:
            seeds[seed] = cand
    return out


def _find_realdata_summaries() -> dict[str, dict[int, Path]]:
    return _find_realdata_csvs("summary.csv")


def _find_realdata_steps() -> dict[str, dict[int, Path]]:
    return _find_realdata_csvs("steps.csv")


def _concat_seeds(paths_by_seed: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for seed, p in sorted(paths_by_seed.items()):
        d = pd.read_csv(p)
        d["__seed"] = seed
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_realdata_paths() -> bool:
    steps_map = _find_realdata_steps()
    if not steps_map:
        _warn_missing("f_realdata_paths", str(RESULTS / "realdata_*/<dataset>_steps.csv"))
        return False

    # Locked 3 first, then extended, in discovery order.
    ordered = [d for d in DATASETS_LOCKED if d in steps_map] + [d for d in sorted(steps_map) if d not in DATASETS_LOCKED]
    n = len(ordered)
    ncol = 4
    nrow = max(1, (n + ncol - 1) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(BIG8[0], 2.2 * nrow), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, dataset in zip(axes_flat, ordered):
        df = _concat_seeds(steps_map[dataset])
        method_col = "method" if "method" in df.columns else None
        step_col = next((c for c in ("step", "n_revealed", "t") if c in df.columns), None)
        perf_col = next((c for c in ("roc_auc", "r2", "score", "performance") if c in df.columns), None)
        if method_col is None or step_col is None or perf_col is None:
            ax.set_title(f"{dataset}: bad schema", fontsize=8)
            continue
        n_seeds = df["__seed"].nunique()
        is_locked = dataset in DATASETS_LOCKED
        for method, sub in df.groupby(method_col):
            agg = sub.groupby(step_col)[perf_col].agg(["mean", "std"]).reset_index()
            color = METHOD_COLORS.get(str(method).lower(), None)
            ax.plot(
                agg[step_col],
                agg["mean"],
                linestyle="-" if is_locked else "--",
                marker="o",
                markersize=2,
                linewidth=1.0,
                color=color,
                label=method,
            )
            if n_seeds > 1:
                lo = agg["mean"] - agg["std"].fillna(0)
                hi = agg["mean"] + agg["std"].fillna(0)
                ax.fill_between(agg[step_col], lo, hi, color=color, alpha=0.15, linewidth=0)
        title = f"{dataset}" + (f" (n={n_seeds})" if n_seeds > 1 else "")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("# features revealed")
        ax.set_ylabel(perf_col)
        ax.tick_params(labelsize=7)
    for ax in axes_flat[len(ordered):]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    _save(fig, FIGDIR / "f_realdata_paths.pdf")
    return True


def fig_realdata_aufc_table() -> bool:
    summaries = _find_realdata_summaries()
    if not summaries:
        _warn_missing("f_realdata_aufc_table", str(RESULTS / "realdata_*/<dataset>_summary.csv"))
        return False
    ordered = [d for d in DATASETS_LOCKED if d in summaries] + [d for d in sorted(summaries) if d not in DATASETS_LOCKED]
    # Per (dataset, method): list of per-seed aufc values. Keep stats for
    # mean and std so multi-seed runs show variance in the table.
    stats: dict[str, dict[str, tuple[float, float, int]]] = {}
    method_set: list[str] = []
    for ds in ordered:
        df = _concat_seeds(summaries[ds])
        method_col = "method" if "method" in df.columns else None
        aufc_col = next((c for c in ("normalized_aufc", "aufc_norm", "aufc") if c in df.columns), None)
        if method_col is None or aufc_col is None:
            continue
        per_method: dict[str, tuple[float, float, int]] = {}
        for method, sub in df.groupby(method_col):
            vals = sub[aufc_col].astype(float).dropna().to_numpy()
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            per_method[str(method)] = (mean, std, int(vals.size))
            if str(method) not in method_set:
                method_set.append(str(method))
        if per_method:
            stats[ds] = per_method
    if not stats:
        print("[skip] f_realdata_aufc_table: no parseable summaries", file=sys.stderr)
        return False

    # Realdata writer (src/realdata_greedy.py) uses `marginal_correlation`
    # rather than `marginal`; keep both in the preference list so either
    # naming lands in a stable column order.
    methods_pref = ["oracle", "kernelshap", "permutation", "marginal_correlation", "marginal", "random", "expensive_refit"]
    methods = [m for m in methods_pref if m in method_set] + [m for m in method_set if m not in methods_pref]

    def _fmt(v: tuple[float, float, int] | None, bold: bool) -> str:
        if v is None:
            return "---"
        mean, std, n = v
        if n > 1 and std > 0:
            s = f"{mean:.4f}\\,$\\pm$\\,{std:.4f}"
        else:
            s = f"{mean:.4f}"
        return f"\\textbf{{{s}}}" if bold else s

    lines = [
        "% auto-generated by scripts/make_figures.py — do not hand-edit",
        "\\begin{tabular}{l" + "r" * len(methods) + "}",
        "\\toprule",
        "dataset & " + " & ".join(methods) + " \\\\",
        "\\midrule",
    ]
    for ds in ordered:
        if ds not in stats:
            continue
        per_method = stats[ds]
        means = {m: per_method[m][0] for m in methods if m in per_method}
        best = max(means.values()) if means else None
        cells = []
        for m in methods:
            v = per_method.get(m)
            bold = v is not None and best is not None and abs(v[0] - best) < 1e-6
            cells.append(_fmt(v, bold))
        label = ds.replace("_", r"\_")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = FIGDIR / "f_realdata_aufc_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {out.relative_to(REPO)}")
    return True


# ---------- §11.6 secondary: expensive vs cheap ----------

def fig_expensive_vs_cheap() -> bool:
    """Preregistration §11.6 secondary: cheap oracle AUFC within 0.01 of the
    expensive-refit distribution mean on Breast Cancer.

    Expensive side is a 50-bootstrap distribution over ``expensive_oracle_bc_*.csv``.
    Cheap side is the single oracle AUFC from the Breast Cancer summary. The
    figure is a histogram of the expensive distribution with a vertical line
    at the cheap value and a shaded ±0.01 acceptance band around it.
    """
    expensive = sorted(RESULTS.glob("expensive_oracle_bc_*_s*_c*.csv"))
    if not expensive:
        _warn_missing("f_expensive_vs_cheap", str(RESULTS / "expensive_oracle_bc_*.csv"))
        return False
    exp_df = pd.concat([pd.read_csv(p) for p in expensive], ignore_index=True)
    aufc_col = next((c for c in ("normalized_aufc", "aufc_norm", "aufc") if c in exp_df.columns), None)
    if aufc_col is None:
        print("[skip] f_expensive_vs_cheap: expensive CSVs missing aufc column", file=sys.stderr)
        return False
    exp_vals = exp_df[aufc_col].astype(float).dropna().to_numpy()
    if exp_vals.size == 0:
        print("[skip] f_expensive_vs_cheap: no finite expensive AUFC values", file=sys.stderr)
        return False

    # Cheap side: oracle row of breast_cancer_wi_summary.csv (prefer seed 0).
    summaries = _find_realdata_summaries()
    bc_paths = summaries.get("breast_cancer_wi", {})
    cheap_aufc: float | None = None
    if bc_paths:
        path = bc_paths.get(0) or next(iter(bc_paths.values()))
        cheap_df = pd.read_csv(path)
        method_col = "method" if "method" in cheap_df.columns else None
        cheap_aufc_col = next((c for c in ("normalized_aufc", "aufc_norm", "aufc") if c in cheap_df.columns), None)
        if method_col and cheap_aufc_col:
            oracle = cheap_df[cheap_df[method_col].astype(str).str.lower() == "oracle"]
            if len(oracle):
                cheap_aufc = float(oracle[cheap_aufc_col].iloc[0])

    mean_exp = float(exp_vals.mean())
    std_exp = float(exp_vals.std(ddof=1)) if exp_vals.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=ONECOL, constrained_layout=True)
    ax.hist(
        exp_vals,
        bins=min(20, max(5, exp_vals.size // 2)),
        color=METHOD_COLORS.get("expensive_refit", "C4"),
        alpha=0.7,
        label=f"expensive-refit (n={exp_vals.size})",
    )
    ax.axvline(mean_exp, color=METHOD_COLORS.get("expensive_refit", "C4"), linestyle=":", linewidth=1.0,
               label=f"E[expensive]={mean_exp:.4f}±{std_exp:.4f}")
    if cheap_aufc is not None:
        ax.axvline(cheap_aufc, color=METHOD_COLORS["oracle"], linestyle="-", linewidth=1.4,
                   label=f"cheap oracle={cheap_aufc:.4f}")
        ax.axvspan(cheap_aufc - 0.01, cheap_aufc + 0.01,
                   color=METHOD_COLORS["oracle"], alpha=0.12, label="|Δ|≤0.01 band")
        delta = abs(cheap_aufc - mean_exp)
        verdict = "within" if delta <= 0.01 else "exceeds"
        ax.set_title(f"|cheap − E[expensive]| = {delta:.4f} ({verdict} 0.01)", fontsize=8)
    else:
        ax.set_title("cheap oracle missing — rerun Breast Cancer §11.6", fontsize=8)
    ax.set_xlabel("normalized AUFC")
    ax.set_ylabel("count")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    _save(fig, FIGDIR / "f_expensive_vs_cheap.pdf")
    return True


FIGURES = {
    "f_synth_label_fidelity": fig_synth_label_fidelity,
    "f_endpoint_calibration": fig_endpoint_calibration,
    "f_quintet": fig_quintet,
    "f_synth_aufc": fig_synth_aufc,
    "f_speed": fig_speed,
    "f_realdata_paths": fig_realdata_paths,
    "f_realdata_aufc_table": fig_realdata_aufc_table,
    "f_expensive_vs_cheap": fig_expensive_vs_cheap,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", nargs="+", default=["all"], help="figure names or 'all'")
    args = ap.parse_args()

    requested = list(FIGURES) if args.which == ["all"] else args.which
    unknown = [w for w in requested if w != "all" and w not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {unknown}; available: {list(FIGURES)}", file=sys.stderr)
        return 2

    n_ok = 0
    n_skip = 0
    for name in requested:
        print(f"=== {name} ===")
        try:
            ok = FIGURES[name]()
        except Exception as e:
            print(f"[error] {name}: {e}", file=sys.stderr)
            ok = False
        n_ok += int(ok)
        n_skip += int(not ok)
    print(f"\n{n_ok} produced, {n_skip} skipped (of {len(requested)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
