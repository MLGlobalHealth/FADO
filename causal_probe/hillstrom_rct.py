"""Hillstrom RCT benchmark with hidden treatment + proxy/leak/null decoys.

Implements Experiment A from notes/real_world_causal_probe_experiment_plan.md:

  1. Load Hillstrom (randomized email marketing experiment).
  2. Binary treatment: Mens E-Mail vs No E-Mail.
  3. Outcome: visit (binary).
  4. Feature matrix = pre-treatment covariates + T + treatment proxy
     P_T = T + eps + outcome leak L_Y = Y + eps + k random noise columns.
  5. Shuffle columns; do NOT tell the model which column is T.
  6. Run the causal probe → tau_hat per column.
  7. Compare to randomized ATE on T, zero on decoys.

Headline metrics (per the plan):
  * ATE error: |tau_hat_T - tau_T^{RCT}|
  * null decoy MAE over P_T, L_Y, noise columns
  * rank of T among all features by |tau_hat|
  * placebo: optional pre-treatment outcome run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model


def _load_hillstrom() -> "pd.DataFrame":
    """Fetch Hillstrom via sklift. The ``segment`` column (Mens E-Mail /
    Womens E-Mail / No E-Mail) lives in the treatment array; the target
    array holds visit/conversion/spend by default. We reassemble into a
    single DataFrame with matching semantics to the canonical CSV.
    """
    from sklift.datasets import fetch_hillstrom
    import pandas as pd
    bunch = fetch_hillstrom(target_col="visit", return_X_y_t=False)
    df = bunch.data.copy()
    df["segment"] = bunch.treatment.values
    df["visit"] = bunch.target.values
    # Fetch conversion/spend too.
    conv = fetch_hillstrom(target_col="conversion", return_X_y_t=False)
    spend = fetch_hillstrom(target_col="spend", return_X_y_t=False)
    df["conversion"] = conv.target.values
    df["spend"] = spend.target.values
    return df


def _build_probe_input(
    df,
    treatment_group: str,
    outcome_col: str,
    seed: int,
    k_noise: int = 3,
    proxy_noise_sd: float = 0.3,
    leak_noise_sd: float = 0.3,
):
    import pandas as pd
    # Restrict to treatment vs control.
    df = df[df["segment"].isin([treatment_group, "No E-Mail"])].copy()
    df["T"] = (df["segment"] == treatment_group).astype(np.float64)

    # Continuous pre-treatment covariates only (drop categoricals for
    # the continuous-only probe).
    cov_cols = ["recency", "history", "mens", "womens", "newbie"]
    df = df[cov_cols + ["T", outcome_col]].dropna().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n = len(df)

    X_cov = df[cov_cols].values.astype(np.float64)
    T = df["T"].values.astype(np.float64)
    y = df[outcome_col].values.astype(np.float64)

    # Proxy: P_T = T + eps (post-hoc, not a cause).
    P_T = T + proxy_noise_sd * rng.standard_normal(n)
    # Outcome leak: L_Y = Y + eps (post-hoc, not a cause).
    L_Y = y + leak_noise_sd * rng.standard_normal(n)
    # Noise columns
    N = rng.standard_normal((n, k_noise))

    # Assemble feature matrix (treatment + proxy + leak + real covariates + noise).
    X_parts = [
        ("T_true", T.reshape(-1, 1)),
        ("P_T (proxy)", P_T.reshape(-1, 1)),
        ("L_Y (leak)", L_Y.reshape(-1, 1)),
    ]
    for i, nm in enumerate(cov_cols):
        X_parts.append((nm, X_cov[:, i:i+1]))
    for j in range(k_noise):
        X_parts.append((f"noise_{j+1}", N[:, j:j+1]))

    # Shuffle columns. The name array is shuffled in lockstep.
    order = rng.permutation(len(X_parts))
    names = [X_parts[i][0] for i in order]
    X = np.concatenate([X_parts[i][1] for i in order], axis=1)
    treatment_col_idx = int(np.where(order == 0)[0][0])
    proxy_col_idx = int(np.where(order == 1)[0][0])
    leak_col_idx = int(np.where(order == 2)[0][0])
    return X, y, names, treatment_col_idx, proxy_col_idx, leak_col_idx


def _standardize(X, y):
    mu_X = X.mean(axis=0); sd_X = X.std(axis=0).clip(min=1e-9)
    mu_y = float(y.mean()); sd_y = float(y.std().clip(min=1e-9))
    return (X - mu_X) / sd_X, (y - mu_y) / sd_y, mu_X, sd_X, mu_y, sd_y


def _randomized_ate(
    X_standardized,
    y_standardized,
    T_col: int,
    t_std_raw: float,
) -> float:
    """Randomized ATE on the probe's reporting scale (Gelman 2008).

    The probe outputs τ as the response of std-Y to a 2-std swing in raw X
    (`tau = 2 * std(X) * beta / std(Y)`); see Gelman, "Scaling regression
    inputs by dividing by two standard deviations" (Stat. Med. 2008). To
    compare to a randomised ATE we put the RCT in the same units:

        tau_RCT = (raw_ATE / std(Y)) * 2 * std(T_raw).

    For 50/50 binary T this collapses to raw_ATE/std(Y); for imbalanced T
    the 2*std(T) factor is < 1.
    """
    T_vals = X_standardized[:, T_col]
    uniq = np.unique(T_vals)
    if uniq.size == 2:
        mid = 0.5 * (uniq[0] + uniq[1])
        mask_hi = T_vals > mid
        mask_lo = T_vals <= mid
    else:
        median_T = np.median(T_vals)
        mask_hi = T_vals > median_T
        mask_lo = T_vals < median_T
    if mask_hi.sum() == 0 or mask_lo.sum() == 0:
        return float("nan")
    raw_over_std_y = float(y_standardized[mask_hi].mean() - y_standardized[mask_lo].mean())
    return raw_over_std_y * 2.0 * float(t_std_raw)


def evaluate_bootstrap(
    ckpt: str,
    treatment_group: str,
    outcome_col: str,
    seed: int,
    k_noise: int,
    n_rows: int,
    bootstrap_B: int,
    device: str,
) -> dict:
    """Row-bootstrap CI on tau_hat[T] / [P_T] / [L_Y] / null_MAE.

    Each iteration resamples df rows with replacement, rebuilds the
    decoy columns from that resample (so noise is fresh per bootstrap,
    matching what a from-scratch run would do on a different sample),
    standardizes, and runs one probe forward pass at n_rows=512 context.
    """
    df = _load_hillstrom()
    model = _load_model(ckpt, device=device)
    n_total = len(df)

    boot_T_tau, boot_PT_tau, boot_LY_tau = [], [], []
    boot_null_mae, boot_ate_err, boot_T_rct = [], [], []
    # Per-feature tau across resamples, keyed by feature name (column
    # order is shuffled per build_probe_input call).
    feat_taus: dict[str, list[float]] = {}
    base_rng = np.random.default_rng(seed)
    for b in range(bootstrap_B):
        b_seed = int(base_rng.integers(0, 2**31))
        df_b = df.sample(n=n_total, replace=True, random_state=b_seed).reset_index(drop=True)
        X_raw, y_raw, names, T_idx, P_idx, L_idx = _build_probe_input(
            df_b, treatment_group, outcome_col, seed=b_seed, k_noise=k_noise,
        )
        Xs, ys, _, _, _, _ = _standardize(X_raw, y_raw)
        # One forward pass per bootstrap: take the first n_rows rows
        # (df_b is already shuffled by sample()), or all if smaller.
        X_ctx = Xs[:n_rows] if Xs.shape[0] > n_rows else Xs
        y_ctx = ys[:n_rows] if ys.shape[0] > n_rows else ys
        X_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        # RCT τ from the bootstrap (varies with resample)
        t_std_raw = float(np.std(X_raw[:, T_idx]))
        tau_T_true = _randomized_ate(Xs, ys, T_idx, t_std_raw)
        decoy_idxs = [P_idx, L_idx]
        noise_idxs = [i for i, nm in enumerate(names) if nm.startswith("noise_")]
        null_mae = float(np.mean([abs(pred[i]) for i in decoy_idxs + noise_idxs]))
        boot_T_tau.append(float(pred[T_idx]))
        boot_PT_tau.append(float(pred[P_idx]))
        boot_LY_tau.append(float(pred[L_idx]))
        boot_null_mae.append(null_mae)
        boot_ate_err.append(abs(float(pred[T_idx]) - tau_T_true))
        boot_T_rct.append(tau_T_true)
        for i, nm in enumerate(names):
            feat_taus.setdefault(nm, []).append(float(pred[i]))

    boot_T_tau = np.asarray(boot_T_tau)
    boot_PT_tau = np.asarray(boot_PT_tau)
    boot_LY_tau = np.asarray(boot_LY_tau)
    boot_null_mae = np.asarray(boot_null_mae)
    boot_ate_err = np.asarray(boot_ate_err)
    boot_T_rct = np.asarray(boot_T_rct)

    def _ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    out = {
        "ckpt": ckpt, "treatment_group": treatment_group, "outcome": outcome_col,
        "bootstrap_B": bootstrap_B, "n_total_per_resample": n_total,
        "tau_T_rct_mean":     float(boot_T_rct.mean()),     "tau_T_rct_ci":     _ci(boot_T_rct),
        "tau_hat_T_mean":     float(boot_T_tau.mean()),     "tau_hat_T_ci":     _ci(boot_T_tau),
        "tau_hat_P_T_mean":   float(boot_PT_tau.mean()),    "tau_hat_P_T_ci":   _ci(boot_PT_tau),
        "tau_hat_L_Y_mean":   float(boot_LY_tau.mean()),    "tau_hat_L_Y_ci":   _ci(boot_LY_tau),
        "null_MAE_mean":      float(boot_null_mae.mean()),  "null_MAE_ci":      _ci(boot_null_mae),
        "ATE_error_mean":     float(boot_ate_err.mean()),   "ATE_error_ci":     _ci(boot_ate_err),
        "per_feature": {
            nm: {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "ci": [float(np.percentile(vals, 2.5)),
                       float(np.percentile(vals, 97.5))],
            }
            for nm, vals in feat_taus.items()
        },
    }
    print(f"\n=== Hillstrom bootstrap (B={bootstrap_B}) ===")
    for k, v in out.items():
        if k.endswith("_ci"):
            print(f"  {k:<22s} = [{v[0]:+.4f}, {v[1]:+.4f}]")
        elif isinstance(v, float):
            print(f"  {k:<22s} = {v:+.4f}")
    return out


def evaluate(
    ckpt: str,
    treatment_group: str,
    outcome_col: str,
    seed: int,
    k_noise: int,
    n_rows: int,
    n_repeats: int,
    device: str,
) -> dict:
    df = _load_hillstrom()
    model = _load_model(ckpt, device=device)

    X_raw, y_raw, names, T_idx, P_idx, L_idx = _build_probe_input(
        df, treatment_group, outcome_col, seed, k_noise=k_noise,
    )
    p = X_raw.shape[1]
    print(f"  total features: {p} (T at col {T_idx}, P_T at {P_idx}, L_Y at {L_idx})")
    print(f"  feature names in order: {names}")

    Xs, ys, _, _, _, _ = _standardize(X_raw, y_raw)
    rng = np.random.default_rng(seed)
    n_total = Xs.shape[0]

    preds = []
    for _ in range(n_repeats):
        if n_total <= n_rows:
            X_ctx, y_ctx = Xs, ys
        else:
            idx = rng.choice(n_total, size=n_rows, replace=False)
            X_ctx = Xs[idx]; y_ctx = ys[idx]
        X_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        preds.append(pred)
    preds = np.stack(preds)
    tau_hat = preds.mean(axis=0); tau_hat_std = preds.std(axis=0)

    # Baselines on the full sample
    assoc = BASELINES["marginal"](Xs, ys)
    multi = BASELINES["multivariate"](Xs, ys)

    # True τ_T from randomised ATE on the probe's reporting scale
    # (Convention 1 / Gelman 2008): rescale by 2*std(T_raw) so the RCT
    # ground truth and the probe's tau live in the same units.
    t_std_raw = float(np.std(X_raw[:, T_idx]))
    tau_T_true = _randomized_ate(Xs, ys, T_idx, t_std_raw)

    # True decoys: 0
    decoy_idxs = [P_idx, L_idx]
    # noise columns are anything not T/P_T/L_Y/real-covariate (names start with "noise_")
    noise_idxs = [i for i, nm in enumerate(names) if nm.startswith("noise_")]

    result = {
        "ckpt": ckpt, "treatment_group": treatment_group, "outcome": outcome_col,
        "n_rows": int(n_total), "p": int(p),
        "tau_T_true_rct": float(tau_T_true),
        "T_idx": T_idx, "P_idx": P_idx, "L_idx": L_idx,
        "feature_names_in_order": names,
        "tau_hat": tau_hat.tolist(), "tau_hat_std": tau_hat_std.tolist(),
        "assoc": assoc.tolist(), "multi": multi.tolist(),
    }

    # Headline metrics
    ate_err = abs(float(tau_hat[T_idx]) - tau_T_true)
    null_decoys = [float(abs(tau_hat[i])) for i in decoy_idxs + noise_idxs]
    null_mae = float(np.mean(null_decoys))
    t_rank = 1 + int(np.sum(np.abs(tau_hat) > abs(tau_hat[T_idx])))

    result["ATE_error"] = ate_err
    result["null_MAE"] = null_mae
    result["T_rank"] = t_rank
    result["n_decoys"] = len(decoy_idxs) + len(noise_idxs)

    # Pretty-print
    print(f"\n  τ_T^RCT (standardized) = {tau_T_true:+.4f}")
    print(f"  τ̂_T (model)            = {float(tau_hat[T_idx]):+.4f}  (err {ate_err:.4f})")
    print(f"  τ̂_P_T (proxy)          = {float(tau_hat[P_idx]):+.4f}  (assoc {float(assoc[P_idx]):+.4f})")
    print(f"  τ̂_L_Y (leak)           = {float(tau_hat[L_idx]):+.4f}  (assoc {float(assoc[L_idx]):+.4f})")
    for i in noise_idxs:
        print(f"  τ̂_{names[i]:<12s}   = {float(tau_hat[i]):+.4f}")
    print(f"  null_MAE (decoys+noise) = {null_mae:.4f}")
    print(f"  T rank by |τ̂|           = {t_rank} / {p}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--treatment-group", default="Mens E-Mail",
                    choices=["Mens E-Mail", "Womens E-Mail"])
    ap.add_argument("--outcome", default="visit", choices=["visit", "conversion", "spend"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--bootstrap-B", type=int, default=0,
                    help="If >0, run row-bootstrap with B resamples and "
                         "report 95% CIs instead of the single-shot evaluate.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    print(f"=== Hillstrom RCT: {args.treatment_group} vs No E-Mail, outcome={args.outcome} ===")
    if args.bootstrap_B > 0:
        res = evaluate_bootstrap(
            args.ckpt, args.treatment_group, args.outcome,
            args.seed, args.k_noise, args.n_rows, args.bootstrap_B, args.device,
        )
    else:
        res = evaluate(
            args.ckpt, args.treatment_group, args.outcome,
            args.seed, args.k_noise, args.n_rows, args.n_repeats, args.device,
        )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
