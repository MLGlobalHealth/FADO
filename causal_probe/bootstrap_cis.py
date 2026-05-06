"""Compute paired-bootstrap 95% CIs from per-SCM .npz dumps.

Each .npz produced by the eval scripts holds (n, p) arrays for
``pred / true / marginal / multivariate``. Pearson and per-feature mean
tau_hat are recomputed by resampling the n SCMs (or random-coefficient
draws, or motif instances) with replacement; the same resampled indices
are applied to every metric in the same call so CIs are paired.

Usage::

    python -m causal_probe.bootstrap_cis sachs causal_probe/results/bootstrap/sachs_p5.npz
    python -m causal_probe.bootstrap_cis motifs causal_probe/results/bootstrap/eval_main_p5_15k.npz
    python -m causal_probe.bootstrap_cis semisynth causal_probe/results/bootstrap/ss_p13_diabetes_linear.npz

Each command prints a markdown-ready table to stdout.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


_DEFAULT_BOOTSTRAP = 5000
_DEFAULT_SEED = 0


def _pearson_flat(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred.reshape(-1)
    t = true.reshape(-1)
    if p.size < 2 or np.std(p) == 0 or np.std(t) == 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def _bootstrap_ci(
    metric_fn, *, n: int, n_boot: int = _DEFAULT_BOOTSTRAP, seed: int = _DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI: resample 0..n indices with replacement."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = metric_fn(idx)
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return lo, hi


def cmd_sachs(npz_path: str) -> None:
    """Probe / marginal / multivariate Pearson on the same Sachs-DAG draws."""
    d = np.load(npz_path, allow_pickle=True)
    pred, true = d["pred"], d["true"]
    marg, multi = d["marginal"], d["multivariate"]
    n = pred.shape[0]
    feat_names = list(d["feature_names"]) if "feature_names" in d.files else None

    point = {
        "model": _pearson_flat(pred, true),
        "marginal": _pearson_flat(marg, true),
        "multivariate": _pearson_flat(multi, true),
    }
    cis = {}
    for tag, arr in [("model", pred), ("marginal", marg), ("multivariate", multi)]:
        cis[tag] = _bootstrap_ci(
            lambda idx, _arr=arr: _pearson_flat(_arr[idx], true[idx]),
            n=n,
        )
    print(f"# Sachs bootstrap CIs ({npz_path}, n={n})")
    print()
    print("| Method | Pearson | 95% CI |")
    print("|---|---:|---|")
    for tag in ("model", "marginal", "multivariate"):
        lo, hi = cis[tag]
        print(f"| {tag} | {point[tag]:.3f} | [{lo:.3f}, {hi:.3f}] |")
    print()
    # Probe-vs-marginal paired difference
    def _diff(idx):
        p1 = _pearson_flat(pred[idx], true[idx])
        p2 = _pearson_flat(marg[idx], true[idx])
        return p1 - p2
    delta = point["model"] - point["marginal"]
    lo, hi = _bootstrap_ci(_diff, n=n)
    print(f"Δ(model − marginal) = {delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    if lo <= 0 <= hi:
        print("    -> CI straddles zero: probe vs marginal is **not distinguishable**.")
    elif delta > 0:
        print("    -> CI excludes zero from below: probe is significantly **better**.")
    else:
        print("    -> CI excludes zero from above: probe is significantly **worse**.")


def cmd_motifs(npz_path: str) -> None:
    """Per-motif key-feature mean tau_hat with bootstrap CI."""
    d = np.load(npz_path, allow_pickle=True)
    motif_names = sorted({k.rsplit("_", 1)[0] for k in d.files
                          if k.startswith("motif_") and k.endswith("_pred")})

    print(f"# Motif key-feature CIs ({npz_path})")
    print()
    print("| Motif | Key | τ_true mean | τ̂ mean | 95% CI on mean τ̂ | marginal mean |")
    print("|---|---:|---:|---:|---|---:|")
    for mname in motif_names:
        pred = d[f"{mname}_pred"]
        true = d[f"{mname}_true"]
        marg = d[f"{mname}_marginal"]
        n, p = pred.shape

        # Pick "key feature": for B/C/E/F (zero-truth motifs), the column
        # with the largest |marginal| among zero-truth features. For A/D,
        # the column with the largest |truth|.
        true_mean = true.mean(axis=0)
        if "B_proxy" in mname or "confounder" in mname or "target_descendant" in mname or "collider" in mname:
            mask = np.abs(true_mean) < 0.05
            assoc_mean = marg.mean(axis=0)
            scores = np.where(mask, np.abs(assoc_mean), -np.inf)
            key = int(np.argmax(scores))
        else:
            key = int(np.argmax(np.abs(true_mean)))

        tau_hat_key = pred[:, key]
        # Bootstrap mean of tau_hat_key
        rng = np.random.default_rng(_DEFAULT_SEED)
        means = np.empty(_DEFAULT_BOOTSTRAP, dtype=np.float64)
        for b in range(_DEFAULT_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            means[b] = float(tau_hat_key[idx].mean())
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        short = mname.replace("motif_", "")
        print(f"| {short} | {key} | {true_mean[key]:+.3f} | {tau_hat_key.mean():+.3f} | "
              f"[{lo:+.3f}, {hi:+.3f}] | {marg[:, key].mean():+.3f} |")


def cmd_semisynth(npz_path: str) -> None:
    """Pearson and FP-on-decoys with bootstrap CIs."""
    d = np.load(npz_path, allow_pickle=True)
    pred, true = d["pred"], d["true"]
    marg, multi = d["marginal"], d["multivariate"]
    kinds = d["kinds"]  # (n_scms, p_total) of strings
    n = pred.shape[0]

    print(f"# Semisynth bootstrap CIs ({npz_path}, n={n})")
    print()

    # Pearson w/ CI for the three methods
    point = {
        "model": _pearson_flat(pred, true),
        "marginal": _pearson_flat(marg, true),
        "multivariate": _pearson_flat(multi, true),
    }
    print("| Pearson(τ̂, τ) | model | marginal | multivariate |")
    print("|---|---:|---:|---:|")
    print(f"| point | {point['model']:.3f} | {point['marginal']:.3f} | {point['multivariate']:.3f} |")
    cis_row = []
    for tag, arr in [("model", pred), ("marginal", marg), ("multivariate", multi)]:
        lo, hi = _bootstrap_ci(
            lambda idx, _arr=arr: _pearson_flat(_arr[idx], true[idx]), n=n,
        )
        cis_row.append(f"[{lo:.3f}, {hi:.3f}]")
    print(f"| 95% CI | {cis_row[0]} | {cis_row[1]} | {cis_row[2]} |")
    print()

    # FP on decoys: mean |τ̂| restricted to decoy columns
    print("| FP mean \\|τ̂\\| | model | marginal | multivariate |")
    print("|---|---:|---:|---:|")
    for kind_label, kind_set in [
        ("proxy", {"proxy"}), ("leak", {"leak"}), ("noise", {"noise"}),
    ]:
        # Per-SCM mean |τ̂| on columns matching kind_set (NaN if none).
        def _per_scm_mean(arr):
            out = np.empty(n, dtype=np.float64)
            for s in range(n):
                mask = np.array([k in kind_set for k in kinds[s]], dtype=bool)
                out[s] = float(np.mean(np.abs(arr[s][mask]))) if mask.any() else np.nan
            return out

        per_pred = _per_scm_mean(pred)
        per_marg = _per_scm_mean(marg)
        per_multi = _per_scm_mean(multi)
        valid = ~(np.isnan(per_pred) | np.isnan(per_marg) | np.isnan(per_multi))
        if not valid.any():
            print(f"| {kind_label} | — | — | — |")
            continue
        n_valid = int(valid.sum())

        def _ci(per_scm):
            rng = np.random.default_rng(_DEFAULT_SEED)
            vals = np.empty(_DEFAULT_BOOTSTRAP, dtype=np.float64)
            arr = per_scm[valid]
            for b in range(_DEFAULT_BOOTSTRAP):
                idx = rng.integers(0, n_valid, size=n_valid)
                vals[b] = float(arr[idx].mean())
            return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

        lo_p, hi_p = _ci(per_pred); lo_m, hi_m = _ci(per_marg); lo_mu, hi_mu = _ci(per_multi)
        print(f"| {kind_label} mean | {np.nanmean(per_pred):.3f} | "
              f"{np.nanmean(per_marg):.3f} | {np.nanmean(per_multi):.3f} |")
        print(f"| {kind_label} 95% CI | [{lo_p:.3f}, {hi_p:.3f}] | "
              f"[{lo_m:.3f}, {hi_m:.3f}] | [{lo_mu:.3f}, {hi_mu:.3f}] |")


def cmd_paired(npz_a_path: str, npz_b_path: str, label_a: str = "A", label_b: str = "B") -> None:
    """Paired Pearson CI for two methods run on the same SCM stream.

    Both .npz must come from causal_probe.run_baseline at the same
    --seed and --scm-type so 'true' arrays match. Reports each method's
    Pearson with bootstrap CI plus the paired Δ(A − B) CI.
    """
    da = np.load(npz_a_path); db = np.load(npz_b_path)
    pred_a, true_a = da["pred"], da["true"]
    pred_b, true_b = db["pred"], db["true"]
    if not np.allclose(true_a, true_b):
        sys.exit(f"true arrays differ between {npz_a_path} and {npz_b_path} — "
                 f"runs were not paired. Check --seed and --scm-type matched.")
    n = pred_a.shape[0]
    # Mask SCMs where either method returned all-NaN tau (e.g. NOTEARS
    # convergence failure). Bootstrap is over the surviving SCMs.
    valid = np.array([
        np.all(np.isfinite(pred_a[i])) and np.all(np.isfinite(pred_b[i]))
        for i in range(n)
    ])
    if not valid.all():
        print(f"_NOTE: dropping {(~valid).sum()}/{n} SCMs with NaN preds._")
        pred_a = pred_a[valid]; pred_b = pred_b[valid]
        true_a = true_a[valid]; n = pred_a.shape[0]

    point_a = _pearson_flat(pred_a, true_a)
    point_b = _pearson_flat(pred_b, true_a)
    print(f"# Paired Pearson CIs: {label_a} vs {label_b} (n={n}, {npz_a_path.split('/')[-1]} vs {npz_b_path.split('/')[-1]})")
    print()
    print(f"| Method | Pearson | 95% CI |")
    print(f"|---|---:|---|")
    for tag, arr in [(label_a, pred_a), (label_b, pred_b)]:
        lo, hi = _bootstrap_ci(
            lambda idx, _arr=arr: _pearson_flat(_arr[idx], true_a[idx]), n=n,
        )
        print(f"| {tag} | {_pearson_flat(arr, true_a):.3f} | [{lo:.3f}, {hi:.3f}] |")
    delta = point_a - point_b
    def _diff(idx):
        return _pearson_flat(pred_a[idx], true_a[idx]) - _pearson_flat(pred_b[idx], true_a[idx])
    lo, hi = _bootstrap_ci(_diff, n=n)
    print()
    print(f"Δ({label_a} − {label_b}) = {delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    if lo <= 0 <= hi:
        print(f"    -> CI straddles zero: **not distinguishable**.")
    elif delta > 0:
        print(f"    -> CI excludes zero from below: **{label_a} > {label_b}** (significant).")
    else:
        print(f"    -> CI excludes zero from above: **{label_a} < {label_b}** (significant).")


def cmd_random(npz_path: str) -> None:
    """Held-out random DAGs Pearson with bootstrap CI."""
    d = np.load(npz_path, allow_pickle=True)
    pred = d["random_pred"]; true = d["random_true"]
    marg = d["random_marginal"]; multi = d["random_multivariate"]
    n = pred.shape[0]
    print(f"# Held-out random DAGs CIs ({npz_path}, n_scms={n})")
    print()
    print("| Method | Pearson | 95% CI |")
    print("|---|---:|---|")
    for tag, arr in [("probe", pred), ("marginal", marg), ("multivariate", multi)]:
        point_v = _pearson_flat(arr, true)
        lo, hi = _bootstrap_ci(
            lambda idx, _arr=arr: _pearson_flat(_arr[idx], true[idx]), n=n,
        )
        print(f"| {tag} | {point_v:.3f} | [{lo:.3f}, {hi:.3f}] |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["sachs", "motifs", "semisynth", "random", "paired"])
    ap.add_argument("npz_path")
    ap.add_argument("--vs", default=None,
                    help="paired: second .npz (other method on the same SCMs)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()
    if args.kind == "sachs":
        cmd_sachs(args.npz_path)
    elif args.kind == "motifs":
        cmd_motifs(args.npz_path)
    elif args.kind == "semisynth":
        cmd_semisynth(args.npz_path)
    elif args.kind == "random":
        cmd_random(args.npz_path)
    elif args.kind == "paired":
        if args.vs is None:
            sys.exit("paired: --vs <second_npz> required")
        cmd_paired(args.npz_path, args.vs, args.label_a, args.label_b)
    else:
        sys.exit(f"unknown kind: {args.kind}")


if __name__ == "__main__":
    main()
