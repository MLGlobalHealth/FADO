"""Hillstrom binary-T encoding sanity check.

The §3 target functional uses the (-1,+1) standardized contrast for both
continuous and binary features. The Hillstrom inference pipeline z-scores
every column, which sends the balanced binary T to ±1 — matching the
training-time encoding of binarized columns in scm_mixed. This script
confirms that the reported tau_hat_T is invariant to the equivalent
encodings, by running the probe under three input variants:

  1. zscore  — column-wise z-score of every feature (the live pipeline;
               balanced 50/50 → ±1 for T).
  2. raw01   — T passed as raw 0/1, all other columns z-scored.
  3. pm1     — T mapped to {-1, +1} explicitly, all other columns z-scored.

Encodings (1) and (3) should agree to within MC noise. Encoding (2) is
the (0,1) raw convention; for balanced binary it is the same number as
(1) and (3) up to the 1/(2*std(T_raw)) factor — but the model never
saw 0/1 inputs in training, so its forward pass on (2) is undefined
behaviour. We report it for completeness.

Usage:
    python -m causal_probe.hillstrom_encoding_sanity \\
        --ckpt causal_probe/results/probe_p13_20k.ckpt \\
        --seed 2025 \\
        --out-json causal_probe/results/hillstrom_encoding_sanity_seed2025.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.eval import _load_model
from causal_probe.hillstrom_rct import (
    _build_probe_input,
    _load_hillstrom,
    _randomized_ate,
    _standardize,
)


def _encode_T(X_raw: np.ndarray, T_idx: int, mode: str) -> np.ndarray:
    """Build the input matrix under one of three T-column encodings."""
    mu_X = X_raw.mean(axis=0)
    sd_X = X_raw.std(axis=0).clip(min=1e-9)
    X_z = (X_raw - mu_X) / sd_X
    X_out = X_z.copy()
    T_raw = X_raw[:, T_idx]
    if mode == "zscore":
        pass  # T already z-scored in X_z
    elif mode == "raw01":
        X_out[:, T_idx] = T_raw
    elif mode == "pm1":
        X_out[:, T_idx] = 2.0 * T_raw - 1.0
    else:
        raise ValueError(f"unknown encoding mode {mode!r}")
    return X_out


def run(ckpt: str, treatment_group: str, outcome_col: str, seed: int,
        k_noise: int, n_rows: int, n_repeats: int, device: str) -> dict:
    df = _load_hillstrom()
    model = _load_model(ckpt, device=device)

    X_raw, y_raw, names, T_idx, P_idx, L_idx = _build_probe_input(
        df, treatment_group, outcome_col, seed, k_noise=k_noise,
    )
    Xs_full, ys, _, _, _, _ = _standardize(X_raw, y_raw)
    t_std_raw = float(np.std(X_raw[:, T_idx]))
    tau_T_RCT = _randomized_ate(Xs_full, ys, T_idx, t_std_raw)

    n_total = X_raw.shape[0]
    print(f"  n={n_total}, p={X_raw.shape[1]}, T_idx={T_idx}, "
          f"std(T_raw)={t_std_raw:.4f}, tau_T_RCT={tau_T_RCT:+.4f}")

    out = {
        "ckpt": ckpt,
        "treatment_group": treatment_group,
        "outcome": outcome_col,
        "seed": seed,
        "n_rows_total": int(n_total),
        "p": int(X_raw.shape[1]),
        "T_idx": T_idx,
        "feature_names": names,
        "t_std_raw": t_std_raw,
        "tau_T_RCT": float(tau_T_RCT),
        "encodings": {},
    }

    for mode in ["zscore", "raw01", "pm1"]:
        X_enc = _encode_T(X_raw, T_idx, mode)
        rng = np.random.default_rng(seed)  # same row sample across encodings
        preds = []
        for _ in range(n_repeats):
            if n_total <= n_rows:
                X_ctx, y_ctx = X_enc, ys
            else:
                idx = rng.choice(n_total, size=n_rows, replace=False)
                X_ctx, y_ctx = X_enc[idx], ys[idx]
            X_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(device)
            y_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(X_t, y_t).squeeze(0).cpu().numpy()
            preds.append(pred)
        preds = np.stack(preds)
        tau_T_hat = float(preds.mean(axis=0)[T_idx])
        tau_T_std = float(preds.std(axis=0)[T_idx])
        out["encodings"][mode] = {
            "tau_T_hat": tau_T_hat,
            "tau_T_hat_std": tau_T_std,
            "n_repeats": int(n_repeats),
        }
        print(f"  {mode:8s}  tau_T_hat = {tau_T_hat:+.4f}  (std={tau_T_std:.4f})")

    z = out["encodings"]["zscore"]["tau_T_hat"]
    p = out["encodings"]["pm1"]["tau_T_hat"]
    r = out["encodings"]["raw01"]["tau_T_hat"]
    out["zscore_vs_pm1_abs_gap"] = abs(z - p)
    out["zscore_vs_raw01_abs_gap"] = abs(z - r)
    print(f"\n  |zscore - pm1|   = {abs(z - p):.4f}  (expect ~0; same encoding)")
    print(f"  |zscore - raw01| = {abs(z - r):.4f}  (expect != 0; raw01 is OOD)")
    print(f"  tau_T_RCT        = {tau_T_RCT:+.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="causal_probe/results/probe_p13_20k.ckpt")
    ap.add_argument("--treatment-group", default="Mens E-Mail",
                    choices=["Mens E-Mail", "Womens E-Mail"])
    ap.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json",
                    default="causal_probe/results/hillstrom_encoding_sanity.json")
    args = ap.parse_args()
    res = run(args.ckpt, args.treatment_group, args.outcome, args.seed,
              args.k_noise, args.n_rows, args.n_repeats, args.device)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
