"""Paired-Δ Pearson CIs for `paper/tables/all_baselines.tex`.

Sweeps every (method, setting) cell in the table and emits both a
markdown paired-Δ table (stdout) and the LaTeX table written to
`paper/tables/all_baselines_paired_ci.tex`. Δ = Pearson(FADO) −
Pearson(baseline), bootstrap CI on the same SCM stream. Reads
`causal_probe/results/bootstrap/{probe,<m>}_<tag>.npz` produced by
`causal_probe.run_baseline --out-npz` at matching --seed.

Run from repo root:

    .venv/bin/python scripts/paired_delta_baselines.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path("causal_probe/results/bootstrap")
TEX_OUT = Path("paper/tables/all_baselines_paired_ci.tex")

SETTINGS = [
    ("gauss_p5",  "LinG p5"),
    ("linear_p5", "LinNG p5"),
    ("linear_p8", "LinNG p8"),
    ("linear_p13","LinNG p13"),
    ("poly_p5",   "Poly p5"),
    ("mlp_p5",    "MLP p5"),
    ("hidden_p5", "Hid p5"),
    ("mixed_p5",  "Mix p5"),
]

METHODS = [
    ("lingam",        "LiNGAM"),
    ("notears",       "NOTEARS"),
    ("pc",            "PC"),
    ("ges",           "GES"),
    ("fci",           "FCI"),
    ("doubleml",      "DML"),
    ("causal_forest", "CausalForest"),
    ("marginal",      "Marginal corr."),
    ("ridge",         "Multivariate regression"),
    ("permutation",   "Permutation"),
    ("shap",          "SHAP"),
]

N_BOOT = 5000
SEED = 0


def _pearson(p, t):
    p = p.reshape(-1); t = t.reshape(-1)
    if p.size < 2 or np.std(p) == 0 or np.std(t) == 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def paired_delta(npz_a: Path, npz_b: Path):
    """Return (point_a, point_b, delta, lo, hi, n_used, n_total).

    Joint NaN-row mask so both methods see the same SCMs in the bootstrap.
    NaN bootstrap samples (zero-variance Pearson) are dropped before
    percentile.
    """
    if not npz_a.exists() or not npz_b.exists():
        return (float("nan"),) * 5 + (0, 0)
    da = np.load(npz_a); db = np.load(npz_b)
    pa, ta = da["pred"], da["true"]
    pb, tb = db["pred"], db["true"]
    if pa.shape != pb.shape:
        return (float("nan"),) * 5 + (0, 0)
    if not np.allclose(ta, tb):
        return (float("nan"),) * 5 + (0, pa.shape[0])
    n_total = pa.shape[0]
    valid = np.array([
        np.all(np.isfinite(pa[i])) and np.all(np.isfinite(pb[i]))
        for i in range(n_total)
    ])
    pa, pb, ta = pa[valid], pb[valid], ta[valid]
    n = pa.shape[0]
    if n < 2:
        return (float("nan"),) * 5 + (n, n_total)
    point_a = _pearson(pa, ta)
    point_b = _pearson(pb, ta)
    delta = point_a - point_b
    rng = np.random.default_rng(SEED)
    diffs = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        diffs[b] = _pearson(pa[idx], ta[idx]) - _pearson(pb[idx], ta[idx])
    finite = np.isfinite(diffs)
    if finite.sum() == 0:
        return point_a, point_b, delta, float("nan"), float("nan"), n, n_total
    lo = float(np.percentile(diffs[finite], 2.5))
    hi = float(np.percentile(diffs[finite], 97.5))
    return point_a, point_b, delta, lo, hi, n, n_total


METHOD_LATEX = {
    "lingam":        r"\quad LiNGAM",
    "notears":       r"\quad NOTEARS",
    "pc":            r"\quad PC",
    "ges":           r"\quad GES",
    "fci":           r"\quad FCI",
    "doubleml":      r"\quad DML",
    "causal_forest": r"\quad CausalForest",
    "marginal":      r"\quad Marginal corr.",
    "ridge":         r"\quad Multivariate regression",
    "permutation":   r"\quad Permutation",
    "shap":          r"\quad SHAP",
}

GROUPS = [
    ("Causal discovery", ["lingam", "notears", "pc", "ges", "fci"]),
    (r"Treatment effect (per-column)$^\dagger$", ["doubleml", "causal_forest"]),
    ("Predictive attribution", ["marginal", "ridge", "permutation", "shap"]),
]


def _fmt_num(x: float) -> str:
    """Two decimal places; drop the leading '+' on non-negative values."""
    s = f"{x:.2f}"
    return s if s.startswith("-") else s.lstrip("+")


def _fmt_latex_cell(delta, lo, hi):
    if np.isnan(delta):
        return "---"
    return (
        f"${_fmt_num(delta)}\\,"
        f"{{\\scriptstyle [{_fmt_num(lo)},{_fmt_num(hi)}]}}$"
    )


def _write_latex(cells: dict[tuple[str, str], tuple[float, float, float]]) -> None:
    lines = [
        "% Paired-Δ Pearson CIs: FADO − each baseline.",
        "% Bootstrap: 5000 resamples of the SCM index, seed=0; same SCM stream",
        "% (--seed 2025, --n-scms 100, --n-rows 512) as tab:all-baselines.",
        "%",
        "% Generator: scripts/paired_delta_baselines.py",
        r"\begin{tabular}{l c c c c c c c c}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Linear} & \multicolumn{2}{c}{Nonlinear} & Hidden & Mixed \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-7}",
        r"& Gauss. & \multicolumn{3}{c}{Non-Gaussian} & Poly. & MLP & conf. & bin/cts \\",
        r"& $p{=}5$ & $p{=}5$ & $p{=}8$ & $p{=}13$ & $p{=}5$ & $p{=}5$ & $p{=}5$ & $p{=}5$ \\",
        r"\midrule",
    ]
    for gi, (group_label, keys) in enumerate(GROUPS):
        if gi > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{group_label}}}}} \\")
        for m_key in keys:
            row_cells = []
            for tag, _ in SETTINGS:
                delta, lo, hi = cells[(m_key, tag)]
                row_cells.append(_fmt_latex_cell(delta, lo, hi))
            lines.append(
                f"{METHOD_LATEX[m_key]} & " + " & ".join(row_cells) + r" \\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    TEX_OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {TEX_OUT}")


def main():
    print("# Paired-Δ Pearson CI: FADO (probe) − baseline")
    print()
    print(f"Bootstrap: {N_BOOT} resamples of SCM index, seed={SEED}.")
    print("Δ = Pearson(FADO) − Pearson(baseline) on the shared SCM stream "
          "(same --seed, --scm-type, --p, --noise; n_scms=100, n_rows=512).")
    print("CI excludes 0 ⇒ FADO significantly better (Δ>0) or worse (Δ<0).")
    print()

    cols = " | ".join(label for _, label in SETTINGS)
    print(f"| Method | {cols} |")
    print("|" + "---|" * (len(SETTINGS) + 1))
    skipped = []
    cells: dict[tuple[str, str], tuple[float, float, float]] = {}
    for m_key, m_label in METHODS:
        delta_cells = []
        ci_cells = []
        for tag, _ in SETTINGS:
            a = ROOT / f"probe_{tag}.npz"
            b = ROOT / f"{m_key}_{tag}.npz"
            _, _, delta, lo, hi, n, n_total = paired_delta(a, b)
            cells[(m_key, tag)] = (delta, lo, hi)
            if np.isnan(delta):
                delta_cells.append("—")
                ci_cells.append("—")
                if not b.exists():
                    skipped.append(f"{m_key}/{tag}: missing {b.name}")
                else:
                    skipped.append(f"{m_key}/{tag}: shape mismatch or all-NaN")
                continue
            sig = "" if (lo <= 0 <= hi) else "*"
            note = f" (n={n}/{n_total})" if n != n_total else ""
            delta_cells.append(f"{delta:+.3f}{sig}")
            ci_cells.append(f"[{lo:+.3f}, {hi:+.3f}]{note}")
        print(f"| **{m_label} Δ** | " + " | ".join(delta_cells) + " |")
        print(f"| 95% CI | " + " | ".join(ci_cells) + " |")
    print()
    print("`*` = CI excludes zero. `(n=k/100)` = k SCMs survived the joint NaN mask.")
    if skipped:
        print()
        print("## Skipped cells")
        for s in skipped:
            print(f"- {s}")
    _write_latex(cells)


if __name__ == "__main__":
    main()
