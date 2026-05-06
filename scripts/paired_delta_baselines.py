"""Paired-Δ Pearson CIs for `paper/tables/all_baselines.tex`.

Sweeps every (method, setting) cell in the table and emits a markdown
paired-Δ table: Δ = Pearson(FADO) − Pearson(baseline), bootstrap CI on
the same SCM stream. Reads `causal_probe/results/bootstrap/{probe,<m>}_<tag>.npz`
produced by `causal_probe.run_baseline --out-npz` at matching --seed.

Run from repo root:

    .venv/bin/python scripts/paired_delta_baselines.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path("causal_probe/results/bootstrap")

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
    ("ridge",         "Lin. regression"),
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
    for m_key, m_label in METHODS:
        delta_cells = []
        ci_cells = []
        for tag, _ in SETTINGS:
            a = ROOT / f"probe_{tag}.npz"
            b = ROOT / f"{m_key}_{tag}.npz"
            _, _, delta, lo, hi, n, n_total = paired_delta(a, b)
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


if __name__ == "__main__":
    main()
