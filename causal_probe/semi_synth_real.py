"""Semi-synthetic real-X benchmark with full-vector ground truth.

Takes covariates X from a real tabular dataset (real marginal distributions,
correlations, non-Gaussian structure), picks a random subset C of columns
as the TRUE causal parents, generates a synthetic outcome

    Y = f(X_C) + eps

with known f, then asks the probe to recover (tau_1, ..., tau_p).

Decoys: in addition to the real columns, we append constructed proxy
(P_j = X_j + eps), outcome-leak (L = Y + eps), and noise (N_j) columns.
Ground truth tau for these is zero by construction.

True tau for causal parents is computed by Monte-Carlo do-intervention:
fix X_i at +sigma_i / -sigma_i, resample the "descendant" (Y) n_mc times
via the known f, diff the means, divide by sd(Y). Non-causal real
columns (those not in C) also have tau=0 since Y doesn't depend on them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model


# ---------------------------------------------------------------------------
# Outcome generators
# ---------------------------------------------------------------------------


def _f_linear(XC: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Y = XC @ coeffs (linear in causal parents)."""
    return XC @ coeffs


def _f_mixed_nonlinear(XC: np.ndarray, coeffs_lin: np.ndarray, coeffs_q: np.ndarray) -> np.ndarray:
    """Y = XC @ lin_coeffs + (XC^2) @ quad_coeffs."""
    return XC @ coeffs_lin + (XC ** 2) @ coeffs_q


def _f_interaction(XC: np.ndarray, coeffs_lin: np.ndarray, pair_coef: float) -> np.ndarray:
    """Y = linear + beta * X[:,0] * X[:,1] (first two causal parents interact)."""
    y = XC @ coeffs_lin
    if XC.shape[1] >= 2:
        y = y + pair_coef * XC[:, 0] * XC[:, 1]
    return y


# ---------------------------------------------------------------------------
# Construct the semi-synthetic benchmark
# ---------------------------------------------------------------------------


def make_benchmark(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    n_causal: int = 3,
    outcome_family: str = "linear",
    k_proxy: int = 2,
    k_noise: int = 3,
    add_leak: bool = True,
    n_mc: int = 8192,
    noise_sd: float = 0.3,
):
    """Take real X, attach synthetic outcome + decoys, return everything.

    Returns a dict with:
      X_full:       (n, p_total) feature matrix incl. real + decoys
      y:            (n,) target
      feature_names:list[str]
      true_tau:     (p_total,) ground-truth standardized tau
      causal_idxs:  indices of true causal parents in X_full column space
      proxy_idxs, leak_idx, noise_idxs
    """
    n, p_real = X.shape
    X_std = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-9)

    # Choose causal parents
    causal_in_real = rng.choice(p_real, size=min(n_causal, p_real), replace=False).tolist()
    XC = X_std[:, causal_in_real]

    # Build outcome function
    if outcome_family == "linear":
        coeffs_lin = rng.standard_normal(len(causal_in_real)) * 1.0
        coeffs_q = None
        def _outcome(X_C): return _f_linear(X_C, coeffs_lin)
    elif outcome_family == "mixed_nonlinear":
        coeffs_lin = rng.standard_normal(len(causal_in_real)) * 0.8
        coeffs_q = rng.standard_normal(len(causal_in_real)) * 0.5
        def _outcome(X_C): return _f_mixed_nonlinear(X_C, coeffs_lin, coeffs_q)
    elif outcome_family == "interaction":
        coeffs_lin = rng.standard_normal(len(causal_in_real)) * 0.6
        coeffs_q = None
        pair_coef = float(rng.standard_normal() * 0.8)
        def _outcome(X_C): return _f_interaction(X_C, coeffs_lin, pair_coef)
    else:
        raise ValueError(f"Unknown outcome_family {outcome_family!r}")

    eps_scale = noise_sd
    y_raw = _outcome(XC) + eps_scale * rng.standard_normal(n)

    # Decoys
    proxy_source = rng.choice(causal_in_real, size=min(k_proxy, len(causal_in_real)), replace=False).tolist()
    P = np.zeros((n, k_proxy), dtype=np.float64)
    for j, idx in enumerate(proxy_source):
        P[:, j] = X_std[:, idx] + eps_scale * rng.standard_normal(n)
    L = y_raw + eps_scale * rng.standard_normal(n) if add_leak else None
    N = rng.standard_normal((n, k_noise))

    # Assemble full feature matrix with shuffled order
    parts = []
    for j in range(p_real):
        nm = f"real_{j}" + ("*" if j in causal_in_real else "")
        parts.append((nm, X_std[:, j:j+1], j in causal_in_real, "real"))
    for j in range(k_proxy):
        parts.append((f"proxy_{j+1}_of_real{proxy_source[j]}", P[:, j:j+1], False, "proxy"))
    if add_leak:
        parts.append((f"leak_Y", L.reshape(-1, 1), False, "leak"))
    for j in range(k_noise):
        parts.append((f"noise_{j+1}", N[:, j:j+1], False, "noise"))

    order = rng.permutation(len(parts))
    names = [parts[i][0] for i in order]
    is_causal = [parts[i][2] for i in order]
    kinds = [parts[i][3] for i in order]
    X_full = np.concatenate([parts[i][1] for i in order], axis=1)

    # Compute ground-truth tau via MC do() on the causal parents;
    # non-causal real columns get tau=0 exactly.
    sd_y = float(np.std(y_raw))
    true_tau = np.zeros(X_full.shape[1], dtype=np.float64)
    causal_in_real_set = set(causal_in_real)
    for col_idx in range(X_full.shape[1]):
        if kinds[col_idx] != "real":
            continue  # decoys are 0 by construction
        real_idx = int(names[col_idx].replace("*", "").split("_")[1])
        if real_idx not in causal_in_real_set:
            continue  # non-causal real column → tau=0
        # MC do(X_real_idx = +1) vs do(= -1) on the standardized real
        # distribution. We use a resampling strategy: for each intervention,
        # draw a fresh sample of XC with the target column fixed.
        n_mc_sub = min(n_mc, n)
        sample_idx = rng.choice(n, size=n_mc_sub, replace=True)
        XC_base = X_std[np.ix_(sample_idx, causal_in_real)]
        # Column position of real_idx in XC
        pos = causal_in_real.index(real_idx)
        XC_plus = XC_base.copy(); XC_plus[:, pos] = +1.0
        XC_minus = XC_base.copy(); XC_minus[:, pos] = -1.0
        y_plus = _outcome(XC_plus).mean()
        y_minus = _outcome(XC_minus).mean()
        true_tau[col_idx] = (y_plus - y_minus) / sd_y

    return {
        "X_full": X_full,
        "y": y_raw,
        "feature_names": names,
        "true_tau": true_tau,
        "kinds": kinds,
        "causal_in_real": causal_in_real,
        "outcome_family": outcome_family,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_covariates(name: str) -> tuple[np.ndarray, list[str]]:
    from causal_probe.real_datasets import DATASETS
    X, _, names, _ = DATASETS[name]()
    return X, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--covariate-source", default="diabetes",
                    choices=["diabetes", "california_housing", "wine_red",
                             "abalone", "auto_mpg", "boston"])
    ap.add_argument("--outcome-family", default="linear",
                    choices=["linear", "mixed_nonlinear", "interaction"])
    ap.add_argument("--n-scms", type=int, default=20, help="# SCM instances")
    ap.add_argument("--n-causal", type=int, default=3)
    ap.add_argument("--k-proxy", type=int, default=2)
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None,
                    help="If set, save per-SCM arrays (pred / true / marginal / "
                         "multivariate, each (n_scms, p_total)) plus per-SCM "
                         "kind labels for bootstrap CIs over SCMs.")
    args = ap.parse_args()

    X_real, X_names = _load_covariates(args.covariate_source)
    print(f"Covariates: {args.covariate_source} (n={X_real.shape[0]}, p_real={X_real.shape[1]})")

    model = _load_model(args.ckpt, device=args.device)
    rng = np.random.default_rng(args.seed)

    preds, truths, assocs, multis, kinds_all = [], [], [], [], []
    for s in range(args.n_scms):
        bench = make_benchmark(
            X_real, np.random.default_rng(rng.integers(0, 2**31)),
            n_causal=args.n_causal, outcome_family=args.outcome_family,
            k_proxy=args.k_proxy, k_noise=args.k_noise,
        )
        X_full = bench["X_full"]
        y = bench["y"]
        X_std = (X_full - X_full.mean(axis=0)) / X_full.std(axis=0).clip(min=1e-9)
        y_std = (y - y.mean()) / y.std().clip(min=1e-9)
        n_total = X_std.shape[0]
        episode_preds = []
        for _ in range(args.n_repeats):
            if n_total <= args.n_rows:
                X_ctx, y_ctx = X_std, y_std
            else:
                idx = rng.choice(n_total, size=args.n_rows, replace=False)
                X_ctx = X_std[idx]; y_ctx = y_std[idx]
            X_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(args.device)
            y_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(args.device)
            with torch.no_grad():
                p_hat = model(X_t, y_t).squeeze(0).cpu().numpy()
            episode_preds.append(p_hat)
        preds.append(np.mean(episode_preds, axis=0))
        truths.append(bench["true_tau"])
        assocs.append(BASELINES["marginal"](X_std, y_std))
        multis.append(BASELINES["multivariate"](X_std, y_std))
        kinds_all.append(bench["kinds"])

    preds = np.stack(preds); truths = np.stack(truths)
    assocs = np.stack(assocs); multis = np.stack(multis)

    flat_p = preds.reshape(-1); flat_t = truths.reshape(-1)
    flat_a = assocs.reshape(-1); flat_m = multis.reshape(-1)

    from scipy.stats import spearmanr, pearsonr
    pe_m = float(pearsonr(flat_p, flat_t)[0])
    pe_a = float(pearsonr(flat_a, flat_t)[0])
    pe_r = float(pearsonr(flat_m, flat_t)[0])
    sp_m = float(spearmanr(flat_p, flat_t)[0])

    # False-positive metrics: MAE on decoys (kinds in {proxy, leak, noise})
    def _mean_abs_on_kind(target_kinds: set[str]) -> dict[str, float]:
        out = {}
        for tag, arr in [("model", preds), ("marginal", assocs), ("multivariate", multis)]:
            vals = []
            for s in range(args.n_scms):
                for j, k in enumerate(kinds_all[s]):
                    if k in target_kinds:
                        vals.append(float(abs(arr[s, j])))
            out[tag] = float(np.mean(vals)) if vals else float("nan")
        return out

    # MAE on nonzero causal parents (should be nonzero)
    causal_mse_model = 0.0; causal_mse_assoc = 0.0
    n_causal_obs = 0
    for s in range(args.n_scms):
        for j, k in enumerate(kinds_all[s]):
            if k == "real" and abs(truths[s, j]) > 0.01:  # true causal
                causal_mse_model += (preds[s, j] - truths[s, j]) ** 2
                causal_mse_assoc += (assocs[s, j] - truths[s, j]) ** 2
                n_causal_obs += 1
    causal_mse_model = causal_mse_model / max(n_causal_obs, 1)
    causal_mse_assoc = causal_mse_assoc / max(n_causal_obs, 1)

    mae_proxy = _mean_abs_on_kind({"proxy"})
    mae_leak = _mean_abs_on_kind({"leak"})
    mae_noise = _mean_abs_on_kind({"noise"})

    print(f"\n=== Semi-synthetic real-X ({args.covariate_source} × {args.outcome_family}, {args.n_scms} SCMs) ===")
    print(f"  Pearson(tau_hat, tau)    : model={pe_m:+.3f}  marginal={pe_a:+.3f}  multivariate={pe_r:+.3f}")
    print(f"  Spearman                 : model={sp_m:+.3f}")
    print(f"  True-causal MSE          : model={causal_mse_model:.4f}  marginal={causal_mse_assoc:.4f}")
    print(f"  False-positive |tau_hat| on proxy  : model={mae_proxy['model']:.4f}  marg={mae_proxy['marginal']:.4f}  multi={mae_proxy['multivariate']:.4f}")
    print(f"  False-positive |tau_hat| on leak   : model={mae_leak['model']:.4f}  marg={mae_leak['marginal']:.4f}  multi={mae_leak['multivariate']:.4f}")
    print(f"  False-positive |tau_hat| on noise  : model={mae_noise['model']:.4f}  marg={mae_noise['marginal']:.4f}  multi={mae_noise['multivariate']:.4f}")

    if args.out_npz:
        Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
        # kinds_all is ragged across SCMs only when k_proxy / k_noise vary;
        # under the CLI defaults each SCM has the same column layout, so a
        # 2-D object array is the simplest representation.
        np.savez_compressed(
            args.out_npz,
            pred=preds.astype(np.float32),
            true=truths.astype(np.float32),
            marginal=assocs.astype(np.float32),
            multivariate=multis.astype(np.float32),
            kinds=np.asarray(kinds_all, dtype=object),
        )
        print(f"wrote {args.out_npz}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "covariate_source": args.covariate_source,
                "outcome_family": args.outcome_family, "n_scms": args.n_scms,
                "pearson": {"model": pe_m, "marginal": pe_a, "multivariate": pe_r},
                "spearman_model": sp_m,
                "causal_mse_model": causal_mse_model,
                "causal_mse_marginal": causal_mse_assoc,
                "fp_proxy": mae_proxy, "fp_leak": mae_leak, "fp_noise": mae_noise,
            }, f, indent=2)


if __name__ == "__main__":
    main()
