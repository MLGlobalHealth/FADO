"""LaLonde / Jobs RCT benchmark (Experiment C).

The NSW experiment randomly assigned disadvantaged workers to a job
training program (treatment T=1) vs control (T=0) in the late 1970s.
Outcome: 1978 earnings. Multiple observational comparison controls
(CPS1/CPS3/PSID) are sold as "confounded" counterparts. Classic
confounding benchmark — Dehejia & Wahba 1999, Smith & Todd 2005.

Experiment structure (per notes/real_world_causal_probe_experiment_plan.md §2):

  1. Observational input = NSW treated + a non-RCT control (CPS3).
  2. Outcome = re78 (earnings in 1978).
  3. Ask the probe τ̂ for every column, including the hidden T column.
  4. Compare τ̂_T to the randomised NSW diff-in-means (~$1800).
  5. Placebo: pre-treatment earnings re75 (should give τ̂ ≈ 0).
"""
from __future__ import annotations

import argparse
import json
import io
import urllib.request
from pathlib import Path

import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model


# Dehejia & Wahba 1999 — nswre74_* includes re74 for a 10-column schema
# that matches the cps3_controls file; plain nsw_* files are 9 cols and
# break the vstack when mixed.
URLS_NSW_TREATED = [
    "http://users.nber.org/~rdehejia/data/nswre74_treated.txt",
    "https://users.nber.org/~rdehejia/data/nswre74_treated.txt",
]
URLS_NSW_CONTROL = [
    "http://users.nber.org/~rdehejia/data/nswre74_control.txt",
    "https://users.nber.org/~rdehejia/data/nswre74_control.txt",
]
URLS_CPS3 = [
    "http://users.nber.org/~rdehejia/data/cps3_controls.txt",
    "https://users.nber.org/~rdehejia/data/cps3_controls.txt",
]

# Dehejia's format: treatment(0/1) age educ black hisp married nodegr re74 re75 re78
COLUMNS = ["T", "age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75", "re78"]


def _fetch_text(urls: list[str]) -> str:
    last = None
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=15) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            last = e
    raise RuntimeError(f"LaLonde fetch failed: {last}")


def _parse_dehejia(raw: str) -> np.ndarray:
    rows = []
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) == len(COLUMNS):
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64)


def load_lalonde_obs(control: str = "cps3"):
    """Return (X, names, T_idx) where X is NSW-treated + observational control."""
    treated = _parse_dehejia(_fetch_text(URLS_NSW_TREATED))
    if control == "nsw":
        ctrl = _parse_dehejia(_fetch_text(URLS_NSW_CONTROL))
    elif control == "cps3":
        ctrl = _parse_dehejia(_fetch_text(URLS_CPS3))
    else:
        raise ValueError(f"Unknown control {control!r}")
    # treated has T=1, ctrl has T=0 (cps3) or T=0 (nsw_control)
    data = np.vstack([treated, ctrl])
    return data, COLUMNS


def _build_lalonde_input(
    data: np.ndarray,
    cols: dict,
    cov_cols: list[str],
    outcome_col: str,
    k_noise: int,
    rng: np.random.Generator,
):
    """Decoy injection + column shuffle on a (possibly resampled) data array.
    Mirrors the inline block in main() so bootstrap and single-shot share logic.
    """
    n = data.shape[0]
    X_cov = data[:, [cols[c] for c in cov_cols]]
    T = data[:, cols["T"]]
    y = data[:, cols[outcome_col]]
    P_T = T + 0.3 * rng.standard_normal(n)
    L_Y = y + 0.3 * y.std() * rng.standard_normal(n)
    N = rng.standard_normal((n, k_noise))
    parts = [("T_true", T.reshape(-1, 1)),
             ("P_T (proxy)", P_T.reshape(-1, 1)),
             ("L_Y (leak)", L_Y.reshape(-1, 1))]
    for i, nm in enumerate(cov_cols):
        parts.append((nm, X_cov[:, i:i+1]))
    for j in range(k_noise):
        parts.append((f"noise_{j+1}", N[:, j:j+1]))
    order = rng.permutation(len(parts))
    feat_names = [parts[i][0] for i in order]
    X = np.concatenate([parts[i][1] for i in order], axis=1)
    T_idx = int(np.where(order == 0)[0][0])
    P_idx = int(np.where(order == 1)[0][0])
    L_idx = int(np.where(order == 2)[0][0])
    return X, y, T, feat_names, T_idx, P_idx, L_idx


def evaluate_bootstrap(
    ckpt: str, control: str, outcome: str,
    k_noise: int, n_rows: int, bootstrap_B: int, seed: int, device: str,
):
    """Row-resample with replacement → fresh decoys → one probe forward
    pass per resample. Reports percentile 95% CIs on tau_hat[T] / [P_T] /
    [L_Y], null_MAE, ATE_error. The randomized-NSW reference tau_T_rct
    is computed once on the experimental NSW subset (not resampled), so
    the CI on ATE_error reflects probe variance only.
    """
    data, names = load_lalonde_obs(control=control)
    cols = {c: i for i, c in enumerate(names)}
    cov_cols = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
    if outcome == "re75":
        cov_cols = [c for c in cov_cols if c != "re75"]
    n_total = data.shape[0]

    # Population-level RCT reference (computed once on experimental NSW).
    if control == "nsw":
        T_full = data[:, cols["T"]]
        y_full = data[:, cols[outcome]]
        ate_raw_pop = float(y_full[T_full > 0.5].mean() - y_full[T_full < 0.5].mean())
        std_y_pop = float(y_full.std())
        std_T_pop = float(T_full.std())
    else:
        nsw_t = _parse_dehejia(_fetch_text(URLS_NSW_TREATED))
        nsw_c = _parse_dehejia(_fetch_text(URLS_NSW_CONTROL))
        re_t = nsw_t[:, cols[outcome]]
        re_c = nsw_c[:, cols[outcome]]
        ate_raw_pop = float(re_t.mean() - re_c.mean())
        std_y_pop = float(data[:, cols[outcome]].std())
        std_T_pop = float(data[:, cols["T"]].std())
    tau_T_rct = (ate_raw_pop / max(std_y_pop, 1e-9)) * 2.0 * std_T_pop

    model = _load_model(ckpt, device=device)
    base_rng = np.random.default_rng(seed)
    boot_T_tau, boot_P_tau, boot_L_tau, boot_null_mae, boot_ate_err = [], [], [], [], []

    for b in range(bootstrap_B):
        b_seed = int(base_rng.integers(0, 2**31))
        b_rng = np.random.default_rng(b_seed)
        idx = b_rng.integers(0, n_total, size=n_total)
        data_b = data[idx]
        X_raw, y_raw, T_raw, feat_names, T_idx, P_idx, L_idx = _build_lalonde_input(
            data_b, cols, cov_cols, outcome, k_noise, b_rng,
        )
        Xs = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0).clip(min=1e-9)
        ys = (y_raw - y_raw.mean()) / y_raw.std().clip(min=1e-9)
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
    out = {"control": control, "outcome": outcome, "ckpt": ckpt,
           "bootstrap_B": bootstrap_B, "tau_T_rct": tau_T_rct}
    for k, a in arrs.items():
        out[f"{k}_mean"] = float(a.mean())
        out[f"{k}_ci"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    print(f"\n=== LaLonde bootstrap (B={bootstrap_B}, control={control}, outcome={outcome}) ===")
    print(f"  tau_T_rct (population) = {tau_T_rct:+.4f}")
    for k in arrs:
        print(f"  {k:<14s} mean={out[f'{k}_mean']:+.4f}  95% CI=[{out[f'{k}_ci'][0]:+.4f}, {out[f'{k}_ci'][1]:+.4f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--control", default="cps3", choices=["cps3", "nsw"],
                    help="cps3 = confounded observational; nsw = RCT control")
    ap.add_argument("--outcome", default="re78", choices=["re78", "re75"])
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=15)
    ap.add_argument("--bootstrap-B", type=int, default=0,
                    help="If >0, run row-bootstrap with B resamples and "
                         "report 95% CIs instead of the single-shot evaluation.")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.bootstrap_B > 0:
        res = evaluate_bootstrap(
            args.ckpt, args.control, args.outcome,
            args.k_noise, args.n_rows, args.bootstrap_B, args.seed, args.device,
        )
        if args.out_json:
            from pathlib import Path as _P
            _P(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(res, f, indent=2)
            print(f"\nwrote {args.out_json}")
        return

    print(f"=== LaLonde: {args.control} controls, outcome={args.outcome} ===")
    data, names = load_lalonde_obs(control=args.control)
    cols = {c: i for i, c in enumerate(names)}
    n = data.shape[0]
    print(f"  n = {n} rows ({int(data[:, cols['T']].sum())} treated, {n - int(data[:, cols['T']].sum())} control)")

    # Build feature matrix: covariates (drop outcomes) + T + proxy + leak + noise
    cov_cols = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
    if args.outcome == "re75":
        cov_cols = [c for c in cov_cols if c != "re75"]

    rng = np.random.default_rng(args.seed)
    X_cov = data[:, [cols[c] for c in cov_cols]]
    T = data[:, cols["T"]]
    y = data[:, cols[args.outcome]]

    P_T = T + 0.3 * rng.standard_normal(n)
    L_Y = y + 0.3 * y.std() * rng.standard_normal(n)
    N = rng.standard_normal((n, args.k_noise))

    parts = [("T_true", T.reshape(-1, 1)),
             ("P_T (proxy)", P_T.reshape(-1, 1)),
             ("L_Y (leak)", L_Y.reshape(-1, 1))]
    for i, nm in enumerate(cov_cols):
        parts.append((nm, X_cov[:, i:i+1]))
    for j in range(args.k_noise):
        parts.append((f"noise_{j+1}", N[:, j:j+1]))
    order = rng.permutation(len(parts))
    feat_names = [parts[i][0] for i in order]
    X = np.concatenate([parts[i][1] for i in order], axis=1)
    T_idx = int(np.where(order == 0)[0][0])
    P_idx = int(np.where(order == 1)[0][0])
    L_idx = int(np.where(order == 2)[0][0])
    print(f"  p total = {X.shape[1]} (T at {T_idx}, P_T at {P_idx}, L_Y at {L_idx})")

    # Standardize and run probe
    Xs = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-9)
    ys = (y - y.mean()) / y.std().clip(min=1e-9)
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
    preds = np.stack(preds)
    tau_hat = preds.mean(axis=0)

    # Randomized ATE on the probe's reporting scale (Convention 1 /
    # Gelman 2008): the probe outputs τ as response of std-Y to a 2-std
    # swing in raw X. Putting the RCT in matching units:
    #   tau_RCT = (raw_ATE / std(Y)) * 2 * std(T_raw)
    # For imbalanced T the 2*std(T) factor is < 1; for 50/50 it is 1.
    # See Gelman, Stat. Med. 2008.
    t_std_raw = float(np.std(T))
    if args.control == "nsw":
        # The nsw_control IS the RCT control, so diff in means on the full
        # sample is the randomized ATE.
        mask_t = T > 0.5; mask_c = T < 0.5
        ate_raw = float(y[mask_t].mean() - y[mask_c].mean())
        ate_std = (ate_raw / y.std()) * 2.0 * t_std_raw
    else:
        # Reference: randomized NSW experimental ATE (Dehejia-Wahba 1999 gives ~$1794)
        # Compute from the NSW experimental subset for this scale, but apply
        # 2*std(T) from the actual analysis sample (treated + cps3 controls).
        nsw_raw_treated = _parse_dehejia(_fetch_text(URLS_NSW_TREATED))
        nsw_raw_control = _parse_dehejia(_fetch_text(URLS_NSW_CONTROL))
        re78_t = nsw_raw_treated[:, cols[args.outcome]]
        re78_c = nsw_raw_control[:, cols[args.outcome]]
        ate_raw = float(re78_t.mean() - re78_c.mean())
        ate_std = (ate_raw / y.std()) * 2.0 * t_std_raw

    print(f"\n  τ_T^RCT  ≈ {ate_raw:.2f} raw / {ate_std:+.4f} (Conv. 1; 2·std(T)={2*t_std_raw:.3f})")
    print(f"  τ̂_T     (model)     = {tau_hat[T_idx]:+.4f}")
    print(f"  τ̂_P_T   (proxy)     = {tau_hat[P_idx]:+.4f}   assoc = {assoc[P_idx]:+.4f}")
    print(f"  τ̂_L_Y   (leak)      = {tau_hat[L_idx]:+.4f}   assoc = {assoc[L_idx]:+.4f}")
    noise_idxs = [i for i, nm in enumerate(feat_names) if nm.startswith("noise_")]
    null_mae = float(np.mean([abs(tau_hat[i]) for i in [P_idx, L_idx] + noise_idxs]))
    print(f"  null_MAE (decoys+noise) = {null_mae:.4f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "control": args.control, "outcome": args.outcome,
                "feature_names": feat_names,
                "T_idx": T_idx, "P_idx": P_idx, "L_idx": L_idx,
                "ate_raw": ate_raw, "ate_std": ate_std,
                "tau_hat": tau_hat.tolist(),
                "assoc": assoc.tolist(), "multi": multi.tolist(),
                "null_mae": null_mae,
                "n_rows": n, "p": X.shape[1],
            }, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
