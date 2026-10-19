"""CausalPFN apples-to-apples baseline on LaLonde.

Counterpart to ``causal_probe.lalonde_rct``: same NSW + CPS3 (or NSW RCT
control) splits, same RCT anchor τ_T = (ATE/std_y)·2·std_T, same row
bootstrap convention. Difference: CausalPFN gets the treatment column
explicitly (via its ATEEstimator) — that is the *easier* problem the
probe is not given.

We run two variants per config:
  * ``with_decoys``: P_T proxy + L_Y leak + k_noise noise columns appended
    as covariates, matching what the probe sees. This is the strict
    apples-to-apples — CausalPFN is forced to adjust for an outcome leak,
    which it isn't built to handle. Expect overadjustment toward zero.
  * ``clean``: only the original 8 LaLonde covariates (minus the outcome
    column). This is CausalPFN's home turf and the closest match to the
    setting in Balazadeh et al. (2025). The probe never sees this input.

Outcome: re78 (post-program earnings); placebo: re75 (pre-program).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.lalonde_rct import (
    URLS_NSW_TREATED,
    URLS_NSW_CONTROL,
    _fetch_text,
    _parse_dehejia,
    load_lalonde_obs,
)


def _build_covariates(
    data: np.ndarray, cols: dict, cov_cols: list[str], with_decoys: bool,
    k_noise: int, rng: np.random.Generator,
) -> np.ndarray:
    """Return the matrix of covariates that CausalPFN will adjust for.

    Treatment T and outcome y are *not* in this matrix — they go to
    ATEEstimator separately. With decoys, mirrors lalonde_rct's input
    minus the T column (P_T, L_Y, the 8 LaLonde covariates, k_noise
    random columns). Clean: only the LaLonde covariates.
    """
    n = data.shape[0]
    X_cov = data[:, [cols[c] for c in cov_cols]]
    if not with_decoys:
        return X_cov
    T = data[:, cols["T"]]
    y_for_leak_col = "re78"
    y_for_leak = data[:, cols[y_for_leak_col]]
    P_T = T + 0.3 * rng.standard_normal(n)
    L_Y = y_for_leak + 0.3 * y_for_leak.std() * rng.standard_normal(n)
    N = rng.standard_normal((n, k_noise))
    return np.concatenate(
        [P_T.reshape(-1, 1), L_Y.reshape(-1, 1), X_cov, N], axis=1,
    )


def _rct_anchor(control: str, outcome: str, data: np.ndarray, cols: dict) -> float:
    """Population RCT-anchored τ_T on the (NSW treated, NSW control) arm,
    expressed on the probe's reporting scale (ATE/std_y)·2·std_T using
    the *combined* split's std_y and std_T.
    """
    if control == "nsw":
        T_full = data[:, cols["T"]]
        y_full = data[:, cols[outcome]]
        ate_raw = float(y_full[T_full > 0.5].mean() - y_full[T_full < 0.5].mean())
        std_y = float(y_full.std())
        std_T = float(T_full.std())
    else:
        nsw_t = _parse_dehejia(_fetch_text(URLS_NSW_TREATED))
        nsw_c = _parse_dehejia(_fetch_text(URLS_NSW_CONTROL))
        re_t = nsw_t[:, cols[outcome]]
        re_c = nsw_c[:, cols[outcome]]
        ate_raw = float(re_t.mean() - re_c.mean())
        std_y = float(data[:, cols[outcome]].std())
        std_T = float(data[:, cols["T"]].std())
    return (ate_raw / max(std_y, 1e-9)) * 2.0 * std_T


def _causalpfn_ate(
    X_cov: np.ndarray, T: np.ndarray, y: np.ndarray,
    *, device: str, cache_dir: str | None,
) -> float | None:
    """Single CausalPFN ATEEstimator fit. Returns raw ATE, or None on failure."""
    from causalpfn import ATEEstimator

    kwargs = {"device": device}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    try:
        est = ATEEstimator(**kwargs)
        est.fit(X_cov.astype(np.float64), T.astype(np.int64), y.astype(np.float64))
        return float(est.estimate_ate())
    except Exception as e:
        print(f"    CausalPFN fit failed: {e}")
        return None


def evaluate(
    *, control: str, outcome: str, with_decoys: bool, k_noise: int,
    bootstrap_B: int, seed: int, device: str, cache_dir: str | None,
    n_subsample: int | None = None,
) -> dict:
    data, names = load_lalonde_obs(control=control)
    cols = {c: i for i, c in enumerate(names)}
    cov_cols = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
    if outcome == "re75":
        cov_cols = [c for c in cov_cols if c != "re75"]

    tau_T_rct = _rct_anchor(control, outcome, data, cols)
    n_total = data.shape[0]
    base_rng = np.random.default_rng(seed)
    n_per_fit = n_subsample if (n_subsample is not None and n_subsample < n_total) else n_total
    print(f"  n_total={n_total}  n_per_fit={n_per_fit}  B={bootstrap_B}")

    boot_tau, boot_err = [], []
    for b in range(bootstrap_B):
        b_seed = int(base_rng.integers(0, 2**31))
        b_rng = np.random.default_rng(b_seed)
        idx = b_rng.integers(0, n_total, size=n_per_fit)
        data_b = data[idx]
        T = data_b[:, cols["T"]].astype(np.int64)
        y = data_b[:, cols[outcome]]
        if T.sum() < 5 or (1 - T).sum() < 5:
            continue
        X_cov = _build_covariates(data_b, cols, cov_cols, with_decoys, k_noise, b_rng)
        ate_raw = _causalpfn_ate(X_cov, T, y, device=device, cache_dir=cache_dir)
        if ate_raw is None:
            continue
        std_y = float(np.std(y).clip(min=1e-9))
        std_T = float(np.std(T).clip(min=1e-9))
        tau_T = (ate_raw / std_y) * 2.0 * std_T
        boot_tau.append(tau_T)
        boot_err.append(abs(tau_T - tau_T_rct))
        if (b + 1) % 25 == 0:
            print(f"  [boot {b+1}/{bootstrap_B}] τ̂_T={tau_T:+.4f}  err={abs(tau_T - tau_T_rct):.4f}")

    boot_tau = np.asarray(boot_tau)
    boot_err = np.asarray(boot_err)

    out = {
        "method": "causalpfn",
        "control": control,
        "outcome": outcome,
        "with_decoys": with_decoys,
        "k_noise": k_noise if with_decoys else 0,
        "n_total": int(n_total),
        "n_per_fit": int(n_per_fit),
        "bootstrap_B": int(boot_tau.size),
        "tau_T_rct": tau_T_rct,
        "tau_hat_T_mean": float(boot_tau.mean()) if boot_tau.size else float("nan"),
        "tau_hat_T_ci": [
            float(np.percentile(boot_tau, 2.5)),
            float(np.percentile(boot_tau, 97.5)),
        ] if boot_tau.size else [float("nan"), float("nan")],
        "ATE_error_mean": float(boot_err.mean()) if boot_err.size else float("nan"),
        "ATE_error_ci": [
            float(np.percentile(boot_err, 2.5)),
            float(np.percentile(boot_err, 97.5)),
        ] if boot_err.size else [float("nan"), float("nan")],
    }
    print(f"\n=== CausalPFN on LaLonde (control={control}, outcome={outcome}, decoys={with_decoys}) ===")
    print(f"  tau_T_rct = {tau_T_rct:+.4f}")
    print(f"  tau_hat_T   = {out['tau_hat_T_mean']:+.4f}  95% CI = [{out['tau_hat_T_ci'][0]:+.4f}, {out['tau_hat_T_ci'][1]:+.4f}]")
    print(f"  ATE_error   = {out['ATE_error_mean']:.4f}    95% CI = [{out['ATE_error_ci'][0]:.4f}, {out['ATE_error_ci'][1]:.4f}]")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", choices=["cps3", "cps1", "nsw"], default="cps3")
    ap.add_argument("--outcome", choices=["re78", "re75"], default="re78")
    ap.add_argument("--with-decoys", action="store_true",
                    help="Append P_T proxy + L_Y leak + k_noise noise to match probe input.")
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--bootstrap-B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-subsample", type=int, default=None,
                    help="Cap rows per bootstrap fit (CPS1 has ~16k rows; "
                         "CausalPFN's ICL context can't handle that).")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import os
    cache_dir = args.cache_dir or os.environ.get("CAUSALPFN_CACHE_DIR")

    res = evaluate(
        control=args.control, outcome=args.outcome,
        with_decoys=args.with_decoys, k_noise=args.k_noise,
        bootstrap_B=args.bootstrap_B, seed=args.seed,
        device=args.device, cache_dir=cache_dir,
        n_subsample=args.n_subsample,
    )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
