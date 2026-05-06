"""SHAP baseline on the Hillstrom hidden-treatment probe setup.

Builds the same feature matrix as causal_probe.hillstrom_rct (real
covariates + T + P_T + L_Y + noise, column-shuffled), fits a
gradient-boosted classifier to the visit outcome, computes TreeSHAP
values, and compares per-feature |mean|SHAP| to the causal probe's
tau_hat on the same shuffle.

The plan's central figure: predictive attribution should highlight the
leak and proxy; the causal probe should zero them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.hillstrom_rct import _build_probe_input, _load_hillstrom


def _fit_and_shap(X: np.ndarray, y: np.ndarray):
    """Fit LightGBM (or fallback to RF), return mean(|SHAP|) per feature."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=-1, num_leaves=31,
            learning_rate=0.05, min_child_samples=20,
            verbosity=-1, random_state=0,
        )
        model.fit(X, y)
        # LightGBM exposes SHAP via pred_contrib
        contrib = model.predict(X, pred_contrib=True)
        # Last column is bias
        shap_vals = contrib[:, :-1]
    except Exception:
        # Fallback: RandomForest + sklearn shap
        from sklearn.ensemble import RandomForestClassifier
        try:
            import shap
        except ImportError:
            raise RuntimeError("Install shap or lightgbm for this baseline")
        model = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X, check_additivity=False)
        if isinstance(sv, list):
            # binary classifier — take positive class
            shap_vals = sv[1] if len(sv) > 1 else sv[0]
        else:
            shap_vals = sv
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    # Also return AUC for sanity
    try:
        from sklearn.metrics import roc_auc_score
        pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        auc = float(roc_auc_score(y, pred_proba))
    except Exception:
        auc = float("nan")
    return mean_abs_shap.astype(np.float64), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment-group", default="Mens E-Mail",
                    choices=["Mens E-Mail", "Womens E-Mail"])
    ap.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--out-json", default=None)
    # Optional: the causal-probe ckpt to read predictions from for side-by-side
    ap.add_argument("--probe-json", default=None,
                    help="hillstrom_*.json from a previous probe run to load model tau_hat from")
    args = ap.parse_args()

    print(f"Loading Hillstrom (treatment={args.treatment_group}, outcome={args.outcome})")
    df = _load_hillstrom()
    X_raw, y_raw, names, T_idx, P_idx, L_idx = _build_probe_input(
        df, args.treatment_group, args.outcome, args.seed, k_noise=args.k_noise,
    )
    print(f"  n={X_raw.shape[0]}, p={X_raw.shape[1]}")
    print(f"  feature order: {names}")
    print(f"  T at {T_idx}, proxy at {P_idx}, leak at {L_idx}")

    # Binary outcome for classifier
    y_bin = (y_raw > 0).astype(int)
    shap_imp, auc = _fit_and_shap(X_raw, y_bin)
    print(f"  classifier AUC = {auc:.3f}")
    print()
    print(f"{'rank':<5}{'feature':<22}{'mean |SHAP|':>14}{'causal_probe':>14}")
    probe_tau = None
    if args.probe_json:
        d = json.load(open(args.probe_json))
        probe_tau = np.asarray(d["tau_hat"])

    order = np.argsort(-shap_imp)
    for i in order:
        row = f"{list(order).index(i)+1:<5}{names[i][:22]:<22}{shap_imp[i]:>14.5f}"
        if probe_tau is not None:
            row += f"{probe_tau[i]:>+14.4f}"
        print(row)

    print()
    print(f"SHAP ranks on key features:")
    for label, idx in [("T_true", T_idx), ("proxy P_T", P_idx), ("leak L_Y", L_idx)]:
        rank = 1 + int(np.sum(shap_imp > shap_imp[idx]))
        print(f"  {label:<12}  rank={rank}/{len(names)}  |SHAP|={shap_imp[idx]:.4f}")

    if args.out_json:
        res = {
            "treatment_group": args.treatment_group, "outcome": args.outcome,
            "auc": auc, "feature_names": names,
            "T_idx": T_idx, "P_idx": P_idx, "L_idx": L_idx,
            "mean_abs_shap": shap_imp.tolist(),
        }
        if probe_tau is not None:
            res["causal_probe_tau_hat"] = probe_tau.tolist()
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
