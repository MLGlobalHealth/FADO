"""Does a residual-normality statistic separate the identifiable (linear/Laplace)
from the non-identifiable (linear/Gaussian) regime — where the bootstrap σ could not?

For each dataset we regress Y on X (OLS) and test the residuals for non-Gaussianity.
Linear identifiability hinges on non-Gaussian noise (the LiNGAM result), so the residual
should look Gaussian in the non-identifiable regime and non-Gaussian in the identifiable one.
We report per-regime statistics and the AUROC for classifying regime from the statistic.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from scipy import stats

from causal_probe.scm import LinearNonGaussianSCM


def _resid_stats(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    r = y - Xc @ beta
    r = (r - r.mean()) / (r.std() + 1e-12)
    return {
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True)),   # 0 = Gaussian, ~3 = Laplace
        "dagostino_p": float(stats.normaltest(r).pvalue),           # small p => reject Gaussianity
        "anderson_stat": float(stats.anderson(r, dist="norm").statistic),
    }


def _auroc(scores, labels):
    # labels: 1 = identifiable (Laplace), 0 = non-identifiable (Gaussian). Mann-Whitney AUROC.
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    order = np.argsort(s); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scms", type=int, default=100)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out-json", default="causal_probe/results/residual_normality_regime.json")
    args = ap.parse_args()

    per = {}
    for noise in ["laplace", "gaussian"]:
        rows = []
        for s in range(args.n_scms):
            scm = LinearNonGaussianSCM(p=args.p, rng=np.random.default_rng(1000 * args.seed + s),
                                       noise=noise)
            samp = scm.sample(n=args.n_rows,
                              rng=np.random.default_rng(7 * args.seed + s))
            rows.append(_resid_stats(np.asarray(samp.X, float), np.asarray(samp.y, float)))
        per[noise] = rows

    def col(noise, k):
        return np.array([r[k] for r in per[noise]])

    labels = [1] * args.n_scms + [0] * args.n_scms   # laplace=identifiable, gaussian=non-identifiable
    summary = {"n_scms_per_regime": args.n_scms, "p": args.p, "n_rows": args.n_rows}
    print(f"n={args.n_scms} SCMs/regime, p={args.p}, n_rows={args.n_rows}\n")
    for k, higher_is_identifiable in [("excess_kurtosis", True),
                                      ("anderson_stat", True),
                                      ("dagostino_p", False)]:
        lap, gau = col("laplace", k), col("gaussian", k)
        scores = np.concatenate([lap, gau])
        if not higher_is_identifiable:
            scores = -scores  # smaller p => more identifiable
        auroc = _auroc(scores, labels)
        summary[k] = {
            "laplace_median": round(float(np.median(lap)), 4),
            "laplace_iqr": [round(float(np.percentile(lap, 25)), 4), round(float(np.percentile(lap, 75)), 4)],
            "gaussian_median": round(float(np.median(gau)), 4),
            "gaussian_iqr": [round(float(np.percentile(gau, 25)), 4), round(float(np.percentile(gau, 75)), 4)],
            "auroc_identifiable_vs_not": round(auroc, 3),
        }
        print(f"{k:>16}: identifiable(Laplace) median {np.median(lap):+.3f}  "
              f"non-ident(Gaussian) median {np.median(gau):+.3f}  "
              f"AUROC(sep) {auroc:.3f}")

    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
