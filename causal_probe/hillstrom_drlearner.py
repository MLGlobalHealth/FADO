"""DR-Learner baseline on the Hillstrom hidden-treatment + decoy setup.

Uses econml.DRLearner — the canonical doubly-robust meta-learner of
Kennedy 2020 — applied per feature: each feature in turn is treated as
the binary treatment (split at its median to make it categorical), the
remaining features are covariates. Reports per-feature ATE estimate
standardized to the probe's reporting scale.

Output JSON columns mirror hillstrom_rct.py so the comparison plugs
straight into existing aggregator + paper-table code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.hillstrom_rct import _load_hillstrom, _build_probe_input


def drlearner_tau(
    X: np.ndarray, y: np.ndarray, treat_idx: int,
    *, light: bool = False, random_state: int = 0,
) -> float:
    """DR-Learner ATE for treating feature `treat_idx` as the binary
    treatment (split at its median). Returns standardized tau in
    units of std(y).

    With ``light=True``, uses a smaller RandomForest config suitable for
    bootstrapping (each forest 30 trees / max_depth 3 / leaf 50). The
    default heavy config (80 / 5 / 20) is preserved for single-shot
    evaluation matching the published numbers.
    """
    from econml.dr import DRLearner
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    X = X.astype(np.float64)
    y = y.astype(np.float64)
    p = X.shape[1]
    if treat_idx < 0 or treat_idx >= p:
        return float("nan")

    t_raw = X[:, treat_idx]
    if np.std(t_raw) < 1e-9:
        return float("nan")
    uniq = np.unique(t_raw)
    if len(uniq) == 2:
        lo, hi = float(uniq.min()), float(uniq.max())
        t = (t_raw > lo).astype(int)
    else:
        t = (t_raw > np.median(t_raw)).astype(int)
    if t.sum() < 5 or (1 - t).sum() < 5:
        return float("nan")
    X_rest = np.delete(X, treat_idx, axis=1)

    if light:
        n_est, depth, leaf = 30, 3, 50
    else:
        n_est, depth, leaf = 80, 5, 20

    try:
        est = DRLearner(
            model_propensity=RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                                    min_samples_leaf=leaf, random_state=random_state),
            model_regression=RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                                   min_samples_leaf=leaf, random_state=random_state),
            model_final=RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                              min_samples_leaf=leaf, random_state=random_state),
            cv=2, random_state=random_state,
        )
        est.fit(y, t, X=X_rest)
        ate = float(est.ate(X=X_rest))
    except Exception as e:
        print(f"    feat {treat_idx}: DRLearner failed: {e}")
        return float("nan")

    # Probe-convention rescaling (Gelman 2008, "divide by 2 SDs"): the
    # probe outputs τ as response of std-Y to a 2-std swing in raw X.
    # DR-Learner's ATE here is the per-{0,1}-flip effect of the binarized
    # indicator. Convert to the same units as the probe by rescaling with
    # the empirical above-/below-split gap and the original-X std:
    #   tau = 2 * ate * std_X / (gap * std_y).
    # Under approximate linearity (ate ≈ beta * gap), this collapses to
    # 2 * beta * std_X / std_y = the probe's reporting scale. Matches the
    # convention used by causalpfn_tau (PR #16). For a genuinely binary
    # 50/50 feature, std_X=0.5, gap=1, so the factor is `ate/std_y`.
    gap = float(t_raw[t == 1].mean() - t_raw[t == 0].mean())
    if abs(gap) < 1e-9:
        return float("nan")
    std_X_i = float(np.std(t_raw).clip(min=1e-9))
    std_y = float(np.std(y).clip(min=1e-9))
    return 2.0 * ate * std_X_i / (gap * std_y)


def evaluate_bootstrap(
    treatment_group: str, outcome: str, k_noise: int, seed: int,
    bootstrap_B: int,
):
    """Row-bootstrap CI on per-feature DR-Learner tau.

    Uses the lighter RF config (light=True) so each fit takes ~5s vs
    ~17s in the heavy config. Per-feature tau across B resamples is
    aggregated to a percentile 95% CI. Decoys are regenerated per
    resample (matching Hillstrom probe bootstrap).
    """
    import time
    df = _load_hillstrom()
    n_total = len(df)

    # Use the seed=seed assignment to lock down the column order so the
    # leak/proxy column indices line up across resamples (not strictly
    # necessary since we identify by name, but cleaner).
    base_rng = np.random.default_rng(seed)

    # Get a representative column order from the first build.
    X0, y0, names0, T0, P0, L0 = _build_probe_input(
        df, treatment_group, outcome, seed=seed, k_noise=k_noise,
    )
    p = X0.shape[1]
    # Save per-resample (B, p) tau matrix — we'll compute CIs by feature name.
    tau_boot = np.full((bootstrap_B, p), np.nan, dtype=np.float64)
    names_boot = [None] * bootstrap_B  # column ordering may vary per resample

    for b in range(bootstrap_B):
        b_seed = int(base_rng.integers(0, 2**31))
        df_b = df.sample(n=n_total, replace=True, random_state=b_seed).reset_index(drop=True)
        X_b, y_b, names_b, T_b, P_b, L_b = _build_probe_input(
            df_b, treatment_group, outcome, seed=b_seed, k_noise=k_noise,
        )
        names_boot[b] = names_b
        t0 = time.time()
        for i in range(p):
            tau_boot[b, i] = drlearner_tau(X_b, y_b, i, light=True, random_state=b_seed)
        if b < 3 or b % 10 == 0:
            print(f"  resample {b+1}/{bootstrap_B}: {time.time()-t0:.1f}s "
                  f"(features in this resample: {names_b})")

    # Aggregate by feature NAME across resamples (column order shuffled
    # per build_probe_input call).
    all_names = sorted({nm for nm_list in names_boot for nm in nm_list})
    feat_taus = {nm: [] for nm in all_names}
    for b in range(bootstrap_B):
        for i, nm in enumerate(names_boot[b]):
            if np.isfinite(tau_boot[b, i]):
                feat_taus[nm].append(float(tau_boot[b, i]))

    summary = {}
    for nm, vals in feat_taus.items():
        if not vals:
            summary[nm] = {"n": 0}
            continue
        a = np.asarray(vals)
        summary[nm] = {
            "n": int(len(vals)),
            "mean": float(a.mean()),
            "ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        }

    out = {
        "treatment_group": treatment_group, "outcome": outcome,
        "bootstrap_B": bootstrap_B, "config": "DRLearner light=True",
        "per_feature": summary,
    }
    print(f"\n=== DR-Learner bootstrap (B={bootstrap_B}, outcome={outcome}) ===")
    for nm in sorted(summary, key=lambda n: -abs(summary[n].get('mean', 0.0))):
        s = summary[nm]
        if s.get("n", 0) > 0:
            print(f"  {nm:<22} n={s['n']:>4} mean={s['mean']:+.4f}  95% CI=[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]")
        else:
            print(f"  {nm:<22} n=0 (all DRLearner fits failed)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment-group", default="Mens E-Mail",
                    choices=["Mens E-Mail", "Womens E-Mail"])
    ap.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--bootstrap-B", type=int, default=0,
                    help="If >0, run row-bootstrap with B resamples (lighter "
                         "RF config) and report per-feature 95% CIs.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.bootstrap_B > 0:
        res = evaluate_bootstrap(
            args.treatment_group, args.outcome, args.k_noise, args.seed,
            args.bootstrap_B,
        )
        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(res, f, indent=2)
            print(f"wrote {args.out_json}")
        return

    print(f"Loading Hillstrom (treatment={args.treatment_group}, outcome={args.outcome})")
    df = _load_hillstrom()
    X_raw, y_raw, names, T_idx, P_idx, L_idx = _build_probe_input(
        df, args.treatment_group, args.outcome, args.seed, k_noise=args.k_noise,
    )
    p = X_raw.shape[1]
    print(f"  n={X_raw.shape[0]}, p={p}")
    print(f"  T at {T_idx}, proxy at {P_idx}, leak at {L_idx}")
    print(f"  features: {names}")

    tau = np.full(p, np.nan, dtype=float)
    for i in range(p):
        tau[i] = drlearner_tau(X_raw, y_raw, i)
        print(f"  {i:2d}  {names[i]:<22}  tau_DR = {tau[i]:+.4f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "treatment_group": args.treatment_group,
                "outcome": args.outcome,
                "feature_names": names,
                "T_idx": T_idx, "P_idx": P_idx, "L_idx": L_idx,
                "drlearner_tau": tau.tolist(),
            }, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
