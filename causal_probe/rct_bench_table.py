"""Emit the RCT_Bench comparison as a LaTeX booktabs table (+ plaintext).

Ground truth per trial = the RCT unadjusted difference in means on the primary
outcome; every method's estimate is scored as a standardized effect d=ATE/std(Y)
so trials with different raw units are comparable.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

ROWS = [
    ("fado_labeled", "FADO (labeled treatment)"),
    ("fado_hidden", "FADO (hidden + decoys)"),
    ("ols_adjusted", "OLS (covariate-adjusted)"),
    ("drlearner", "DR-Learner"),
    ("causalpfn", "CausalPFN"),
    ("dopfn", "DoPFN"),
    ("lingam", "LiNGAM"),
]


def stats(d, key, truth):
    est = np.array([r["d"].get(key) if r["d"].get(key) is not None else np.nan for r in d])
    ok = np.isfinite(est) & np.isfinite(truth)
    e, t = np.clip(est[ok], -3, 3), truth[ok]
    return dict(n=int(ok.sum()),
                medae=float(np.median(np.abs(est[ok] - t))),
                sign=float((np.sign(est[ok]) == np.sign(t)).mean() * 100),
                corr=float(np.corrcoef(e, t)[0, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default="causal_probe/results/rct_bench_eval_merged.json")
    ap.add_argument("--out-tex", default="causal_probe/rct_bench_figures/rct_bench_table.tex")
    args = ap.parse_args()
    d = json.load(open(args.in_json))
    truth = np.array([r["d"]["rct_unadjusted"] for r in d])
    N = len(d)
    big = np.abs(truth) > 0.5

    print(f"{'method':>26}{'n':>5}{'medAE':>7}{'sign%':>7}{'corr':>6}")
    lines = []
    for key, name in ROWS:
        s = stats(d, key, truth)
        cov = "all" if s["n"] >= N else f"{s['n']}/{N}"
        print(f"{name:>26}{s['n']:>5}{s['medae']:>7.2f}{s['sign']:>7.0f}{s['corr']:>6.2f}")
        lines.append((name, cov, s))

    # LaTeX
    tex = [
        r"\begin{table}[t]", r"  \centering",
        r"  \caption{Recovery of the true randomized effect on RCT\_Bench "
        rf"({N} real RCTs). Each method estimates the ATE of the randomized "
        r"treatment on the primary outcome; ground truth is the trial's "
        r"unadjusted difference in means. Scores use the standardized effect "
        r"$d=\text{ATE}/\mathrm{sd}(Y)$. \emph{med.\ err} is the median "
        r"$|\hat d-d_{\text{RCT}}|$; \emph{sign} is the fraction with matching "
        r"sign; \emph{corr} is the correlation with $d_{\text{RCT}}$. "
        r"\emph{cov} is how many trials the method returns an estimate on.}",
        r"  \label{tab:rct-bench}",
        r"  \begin{tabular}{lrrrr}", r"    \toprule",
        r"    Method & cov & med.\ err & sign\% & corr \\", r"    \midrule",
    ]
    for name, cov, s in lines:
        bold = name.startswith("FADO")
        nm = rf"\textbf{{{name}}}" if bold else name
        covs = "all" if cov == "all" else cov
        tex.append(rf"    {nm} & {covs} & {s['medae']:.2f} & {s['sign']:.0f} & {s['corr']:.2f} \\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    with open(args.out_tex, "w") as f:
        f.write("\n".join(tex) + "\n")

    fado_big = stats([r for i, r in enumerate(d) if big[i]], "fado_labeled",
                     truth[big])
    print(f"\nFADO(labeled) on {int(big.sum())} large effects |d|>0.5: "
          f"sign {fado_big['sign']:.0f}%, corr {fado_big['corr']:.2f}")
    print(f"wrote {args.out_tex}")
    print("\n----- LaTeX -----\n" + "\n".join(tex))


if __name__ == "__main__":
    main()
