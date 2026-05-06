"""Criteo Uplift RCT — Experiment B.

Randomized incrementality test with 13M rows × 12 features. We subsample
to a manageable context (10k-20k rows) and follow the Hillstrom-style
hidden-treatment + decoys protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model


def _load_criteo_subsample(n: int, seed: int, outcome: str):
    """Use scikit-uplift's fetch_criteo(). We subsample to n rows.

    Override the cache location with SKLIFT_HOME if the default is not writable;
    uses percent10=True (~1M rows) to keep the download tractable.
    """
    import os
    from sklift.datasets import fetch_criteo
    data_home = os.environ.get("SKLIFT_HOME", os.path.expanduser("~/.cache/sklift"))
    os.makedirs(data_home, exist_ok=True)
    bunch = fetch_criteo(target_col=outcome, data_home=data_home, percent10=True)
    X_all = bunch.data.values.astype(np.float64)
    y_all = bunch.target.values.astype(np.float64)
    T_all = bunch.treatment.values.astype(np.float64)
    rng = np.random.default_rng(seed)
    n_use = min(n, len(y_all))
    idx = rng.choice(len(y_all), size=n_use, replace=False)
    names = list(bunch.data.columns)
    return X_all[idx], y_all[idx], T_all[idx], names


def _build_criteo_input(
    X_cov: np.ndarray, T: np.ndarray, y: np.ndarray, names: list[str],
    k_noise: int, rng: np.random.Generator,
):
    """Decoy injection + column shuffle on (possibly resampled) Criteo arrays."""
    n = X_cov.shape[0]
    P_T = T + 0.3 * rng.standard_normal(n)
    L_Y = y + 0.3 * max(y.std(), 1e-3) * rng.standard_normal(n)
    N = rng.standard_normal((n, k_noise))
    parts = [("T_true", T.reshape(-1, 1)),
             ("P_T", P_T.reshape(-1, 1)),
             ("L_Y", L_Y.reshape(-1, 1))]
    for i, nm in enumerate(names):
        parts.append((nm, X_cov[:, i:i+1]))
    for j in range(k_noise):
        parts.append((f"noise_{j+1}", N[:, j:j+1]))
    order = rng.permutation(len(parts))
    feat_names = [parts[i][0] for i in order]
    X = np.concatenate([parts[i][1] for i in order], axis=1)
    T_idx = int(np.where(order == 0)[0][0])
    P_idx = int(np.where(order == 1)[0][0])
    L_idx = int(np.where(order == 2)[0][0])
    return X, feat_names, T_idx, P_idx, L_idx


def evaluate_bootstrap(
    ckpt: str, outcome: str, subsample: int,
    k_noise: int, n_rows: int, bootstrap_B: int, seed: int, device: str,
):
    """Row-bootstrap on Criteo subsample. The randomized RCT reference is
    computed once on the same subsample (without resampling) so its CI
    isn't conflated with probe variance.
    """
    X_cov0, y0, T0, names = _load_criteo_subsample(subsample, seed, outcome)
    n_total = X_cov0.shape[0]

    # Population RCT reference at the subsample level (matches paper N).
    t_std_raw_pop = float(np.std(T0))
    raw_over_std_y_pop = float(
        ((y0 - y0.mean()) / max(y0.std(), 1e-9))[T0 > T0.mean()].mean()
        - ((y0 - y0.mean()) / max(y0.std(), 1e-9))[T0 <= T0.mean()].mean()
    )
    tau_T_rct = raw_over_std_y_pop * 2.0 * t_std_raw_pop

    model = _load_model(ckpt, device=device)
    base_rng = np.random.default_rng(seed)
    boot_T_tau, boot_P_tau, boot_L_tau, boot_null_mae, boot_ate_err = [], [], [], [], []

    for b in range(bootstrap_B):
        b_seed = int(base_rng.integers(0, 2**31))
        b_rng = np.random.default_rng(b_seed)
        idx = b_rng.integers(0, n_total, size=n_total)
        X_cov_b = X_cov0[idx]; T_b = T0[idx]; y_b = y0[idx]
        X_b, feat_names, T_idx, P_idx, L_idx = _build_criteo_input(
            X_cov_b, T_b, y_b, names, k_noise, b_rng,
        )
        Xs = (X_b - X_b.mean(axis=0)) / X_b.std(axis=0).clip(min=1e-9)
        ys = (y_b - y_b.mean()) / y_b.std().clip(min=1e-9)
        if Xs.shape[0] > n_rows:
            sel = b_rng.choice(Xs.shape[0], size=n_rows, replace=False)
            Xs = Xs[sel]; ys = ys[sel]
        X_t = torch.from_numpy(Xs.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(ys.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        noise_idxs = [i for i, nm in enumerate(feat_names) if nm.startswith("noise_")]
        null_mae = float(np.mean([abs(pred[i]) for i in [P_idx, L_idx] + noise_idxs]))
        boot_T_tau.append(float(pred[T_idx]))
        boot_P_tau.append(float(pred[P_idx]))
        boot_L_tau.append(float(pred[L_idx]))
        boot_null_mae.append(null_mae)
        boot_ate_err.append(abs(float(pred[T_idx]) - tau_T_rct))

    arrs = {
        "tau_hat_T": np.asarray(boot_T_tau),
        "tau_hat_P_T": np.asarray(boot_P_tau),
        "tau_hat_L_Y": np.asarray(boot_L_tau),
        "null_MAE": np.asarray(boot_null_mae),
        "ATE_error": np.asarray(boot_ate_err),
    }
    out = {"outcome": outcome, "ckpt": ckpt, "subsample": subsample,
           "bootstrap_B": bootstrap_B, "tau_T_rct": tau_T_rct}
    for k, a in arrs.items():
        out[f"{k}_mean"] = float(a.mean())
        out[f"{k}_ci"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    print(f"\n=== Criteo bootstrap (B={bootstrap_B}, outcome={outcome}, subsample={subsample}) ===")
    print(f"  tau_T_rct = {tau_T_rct:+.4f}")
    for k in arrs:
        print(f"  {k:<14s} mean={out[f'{k}_mean']:+.4f}  95% CI=[{out[f'{k}_ci'][0]:+.4f}, {out[f'{k}_ci'][1]:+.4f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    ap.add_argument("--subsample", type=int, default=12000)
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=1024)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--bootstrap-B", type=int, default=0,
                    help="If >0, run row-bootstrap with B resamples and "
                         "report 95% CIs instead of the single-shot evaluation.")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.bootstrap_B > 0:
        res = evaluate_bootstrap(
            args.ckpt, args.outcome, args.subsample,
            args.k_noise, args.n_rows, args.bootstrap_B, args.seed, args.device,
        )
        if args.out_json:
            from pathlib import Path as _P
            _P(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(res, f, indent=2)
            print(f"\nwrote {args.out_json}")
        return

    print(f"=== Criteo Uplift (outcome={args.outcome}, subsample={args.subsample}) ===")
    X_cov, y, T, names = _load_criteo_subsample(args.subsample, args.seed, args.outcome)
    n = X_cov.shape[0]
    print(f"  n = {n}, p_cov = {X_cov.shape[1]}")
    print(f"  treatment balance: {float(T.mean()):.3f}")

    rng = np.random.default_rng(args.seed)
    P_T = T + 0.3 * rng.standard_normal(n)
    L_Y = y + 0.3 * max(y.std(), 1e-3) * rng.standard_normal(n)
    N = rng.standard_normal((n, args.k_noise))

    cov_names = names
    parts = [("T_true", T.reshape(-1, 1)), ("P_T", P_T.reshape(-1, 1)), ("L_Y", L_Y.reshape(-1, 1))]
    for i, nm in enumerate(cov_names):
        parts.append((nm, X_cov[:, i:i+1]))
    for j in range(args.k_noise):
        parts.append((f"noise_{j+1}", N[:, j:j+1]))
    order = rng.permutation(len(parts))
    feat_names = [parts[i][0] for i in order]
    X = np.concatenate([parts[i][1] for i in order], axis=1)
    T_idx = int(np.where(order == 0)[0][0])
    P_idx = int(np.where(order == 1)[0][0])
    L_idx = int(np.where(order == 2)[0][0])
    print(f"  p total = {X.shape[1]}  (T at {T_idx}, P at {P_idx}, L at {L_idx})")

    Xs = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-9)
    ys = (y - y.mean()) / y.std().clip(min=1e-9)

    # Randomized ATE on the probe's reporting scale (Convention 1 /
    # Gelman 2008): probe τ = response of std-Y to a 2-std swing in raw X.
    # Match by rescaling: tau_RCT = (raw_ATE / std(Y)) * 2 * std(T_raw).
    # For Criteo's imbalanced T (~85/15), 2*std(T) ≈ 0.71. See Gelman,
    # Stat. Med. 2008.
    t_std_raw = float(np.std(T))
    T_vals = Xs[:, T_idx]
    uniq = np.unique(T_vals)
    if uniq.size == 2:
        mid = 0.5 * (uniq[0] + uniq[1])
        raw_over_std_y = float(ys[T_vals > mid].mean() - ys[T_vals <= mid].mean())
        ate = raw_over_std_y * 2.0 * t_std_raw
    else:
        ate = float("nan")
    print(f"  randomised ATE (Conv. 1; 2·std(T)={2*t_std_raw:.3f}) = {ate:+.4f}")

    assoc = BASELINES["marginal"](Xs, ys)
    multi = BASELINES["multivariate"](Xs, ys)

    model = _load_model(args.ckpt, device=args.device)
    preds = []
    for _ in range(args.n_repeats):
        if n <= args.n_rows:
            idx = np.arange(n)
        else:
            idx = rng.choice(n, size=args.n_rows, replace=False)
        Xc = Xs[idx]; yc = ys[idx]
        X_t = torch.from_numpy(Xc.astype(np.float32)).unsqueeze(0).to(args.device)
        y_t = torch.from_numpy(yc.astype(np.float32)).unsqueeze(0).to(args.device)
        with torch.no_grad():
            p = model(X_t, y_t).squeeze(0).cpu().numpy()
        preds.append(p)
    preds = np.stack(preds); tau_hat = preds.mean(axis=0)

    print(f"\n  τ̂_T (model)  = {tau_hat[T_idx]:+.4f}   err vs RCT: {abs(tau_hat[T_idx] - ate):.4f}")
    print(f"  τ̂_P_T        = {tau_hat[P_idx]:+.4f}   assoc = {assoc[P_idx]:+.4f}")
    print(f"  τ̂_L_Y        = {tau_hat[L_idx]:+.4f}   assoc = {assoc[L_idx]:+.4f}")
    noise_idxs = [i for i, nm in enumerate(feat_names) if nm.startswith("noise_")]
    null_mae = float(np.mean([abs(tau_hat[i]) for i in [P_idx, L_idx] + noise_idxs]))
    print(f"  null_MAE (decoys+noise) = {null_mae:.4f}")
    t_rank = 1 + int(np.sum(np.abs(tau_hat) > abs(tau_hat[T_idx])))
    print(f"  T rank by |τ̂| = {t_rank}/{X.shape[1]}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "outcome": args.outcome, "n_rows": n,
                "feature_names": feat_names,
                "T_idx": T_idx, "P_idx": P_idx, "L_idx": L_idx,
                "ate_standardized": ate,
                "tau_hat": tau_hat.tolist(),
                "assoc": assoc.tolist(), "multi": multi.tolist(),
                "null_mae": null_mae, "T_rank": t_rank, "p": X.shape[1],
            }, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
