"""SHAP-vs-causal analysis on TabICL's *identifiable* SCM priors.

Where MLPSCM is non-identifiable (its `simulate(intervene_on=...)` is a
no-op), the identifiable SCM families (LiNGAM, ANM, TreeSCM_Ident) DO
have working interventional simulators.
This module asks the original "what fraction of features does SHAP rate
as important but the causal truth says are irrelevant" question on each
of those families, with our causal probe in the loop as a third column.

For each DGP we run a paired 2x2 SHAP-variant grid (model x variant) so
the comparison shares the same SCM, the same (X, y), and the same true
tau (Janzing 2020; Sundararajan & Najmi 2020 — "you cherry-picked the
SHAP variant"):
  cell 1: (Ridge,    interventional)  KernelSHAP, 40-row data background
  cell 2: (Ridge,    conditional)     LinearExplainer, correlation_dependent
  cell 3: (LightGBM, path-dependent)  LightGBM pred_contrib
  cell 4: (LightGBM, interventional)  TreeExplainer, 40-row data background

For each DGP:
  1. Sample SCM from one of {lingam, anm, tree}.
  2. Sample (X, y) ~ SCM.
  3. Fit Ridge and LightGBM on (X[:N_SHAP_SAMPLES], y[:N_SHAP_SAMPLES]).
  4. Compute |SHAP| per feature for each of the four cells.
  5. True τ via SCM.simulate(intervene_on={i: ±1}) → standardized contrast.
  6. Causal probe forward pass(es) → τ̂ per feature.

Reduce: % of (DGP, feature) pairs where SHAP says important but |τ| < ε,
per cell + probe agreement on those.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import shap

import lightgbm as lgb  # in pyproject.toml's [baselines] extra; install with `uv pip install -e '.[baselines]'`

# Local-source identifiable SCM module (recovered from tabicl b3ad247).
sys.path.insert(0, os.path.dirname(__file__))
from identifiable_scm import LiNGAMSCM, ANMSCM, TreeSCM_Ident

N_SAMPLES      = 600
N_SHAP_SAMPLES = 150
N_SHAP_BG      = 40
N_MC_TAU       = 4000
SHAP_TOPK_FRAC = 0.3
TAU_NEAR_ZERO  = 0.05
PROBE_THRESH   = 0.10
BASE_SEED      = 42

PROBE_CKPTS = {
    "linear_p5_50k": "causal_probe/results/probe_main_p5_50k.ckpt",
    "mlp_p5_20k":    "causal_probe/results/probe_mlp_p5_20k.ckpt",
}


def _load_probe(path):
    if not os.path.exists(path):
        return None
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from causal_probe.eval import _load_model
    return _load_model(path, device="cpu")


PROBES = {name: _load_probe(p) for name, p in PROBE_CKPTS.items()}


def sample_scm(family: str, rng: np.random.Generator):
    p = int(rng.integers(4, 12))
    if family == "lingam":
        scm = LiNGAMSCM(seq_len=N_SAMPLES, num_features=p, rng=rng, device="cpu")
    elif family == "anm":
        scm = ANMSCM(seq_len=N_SAMPLES, num_features=p, rng=rng, device="cpu")
    elif family == "tree":
        scm = TreeSCM_Ident(seq_len=N_SAMPLES, num_features=p, rng=rng, device="cpu")
    else:
        raise ValueError(f"unknown family: {family}")
    return scm, {"family": family, "num_features": p}


def true_tau_via_simulate(scm, p: int, std_y_obs: float) -> np.ndarray:
    """Standardized (do(X_i=+1) − do(X_i=−1))/std(Y_obs), MC over fresh noise."""
    tau = np.full(p, np.nan, dtype=float)
    for i in range(p):
        try:
            _, y_pos = scm.simulate(intervene_on={i: +1.0}, n_samples=N_MC_TAU)
            _, y_neg = scm.simulate(intervene_on={i: -1.0}, n_samples=N_MC_TAU)
            tau[i] = (float(np.mean(y_pos)) - float(np.mean(y_neg))) / max(std_y_obs, 1e-9)
        except Exception:
            tau[i] = np.nan
    return tau


CELLS = ("ridge_int", "ridge_cond", "lgbm_path", "lgbm_int")


def _shap_ridge_int(ridge, Xs_train, ys_train):
    bg = shap.sample(Xs_train, N_SHAP_BG, random_state=42)
    expl = shap.KernelExplainer(ridge.predict, bg)
    # KernelExplainer.shap_values has no random_state kwarg; its nsamples
    # subset-sampling reads np.random global state. Pin it here so reruns
    # with the same chunk-id reproduce the same SHAP values bit-for-bit.
    np.random.seed(BASE_SEED + 17)
    sv = expl.shap_values(Xs_train, nsamples=128, silent=True)
    return np.abs(sv).mean(axis=0)


def _shap_ridge_cond(ridge, Xs_train):
    # shap.LinearExplainer with feature_perturbation="correlation_dependent"
    # is the closed-form conditional Linear SHAP (Aas, Jullum & Løland 2021)
    # — exact for linear models under multivariate-normal X.
    expl = shap.LinearExplainer(
        ridge, Xs_train, feature_perturbation="correlation_dependent")
    sv = expl.shap_values(Xs_train)
    return np.abs(sv).mean(axis=0)


def _shap_lgbm_path(lgbm_model, Xs_train):
    # Path-dependent TreeSHAP via LightGBM's pred_contrib (LightGBM default;
    # conditional-style approximation).
    contribs = lgbm_model.predict(Xs_train, pred_contrib=True)
    sv = contribs[:, :-1]  # last column is the bias / expected value.
    return np.abs(sv).mean(axis=0)


def _shap_lgbm_int(lgbm_model, Xs_train):
    bg = shap.sample(Xs_train, N_SHAP_BG, random_state=42)
    expl = shap.TreeExplainer(
        lgbm_model, data=bg, feature_perturbation="interventional")
    np.random.seed(BASE_SEED + 19)
    sv = expl.shap_values(Xs_train)
    return np.abs(sv).mean(axis=0)


def analyse_one(scm, family: str, num_features: int):
    X_t, y_t = scm()
    X_np = X_t.detach().numpy().astype(float)
    y_np = y_t.detach().numpy().astype(float)
    if np.std(y_np) < 1e-6 or not np.isfinite(X_np).all() or not np.isfinite(y_np).all():
        return None

    Xs = StandardScaler().fit_transform(X_np)
    ys = StandardScaler().fit_transform(y_np.reshape(-1, 1)).ravel()
    p = Xs.shape[1]
    std_y_obs = float(np.std(y_np))

    obs_corr = np.array([abs(np.corrcoef(Xs[:, i], ys)[0, 1]) for i in range(p)])
    r2_full = r2_score(ys, Ridge(alpha=0.1).fit(Xs, ys).predict(Xs))
    int_effect = np.zeros(p)
    for i in range(p):
        mask = [j for j in range(p) if j != i]
        if mask:
            r2_red = r2_score(ys, Ridge(alpha=0.1).fit(Xs[:, mask], ys).predict(Xs[:, mask]))
            int_effect[i] = max(0.0, r2_full - r2_red)
        else:
            int_effect[i] = r2_full

    Xs_tr = Xs[:N_SHAP_SAMPLES]
    ys_tr = ys[:N_SHAP_SAMPLES]
    ridge = Ridge(alpha=0.1).fit(Xs_tr, ys_tr)

    lgbm = lgb.LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        min_child_samples=20, verbosity=-1, random_state=0,
    ).fit(Xs_tr, ys_tr)

    mean_abs_shap_by_cell = {}
    for cell in CELLS:
        try:
            if cell == "ridge_int":
                v = _shap_ridge_int(ridge, Xs_tr, ys_tr)
            elif cell == "ridge_cond":
                v = _shap_ridge_cond(ridge, Xs_tr)
            elif cell == "lgbm_path":
                v = _shap_lgbm_path(lgbm, Xs_tr)
            elif cell == "lgbm_int":
                v = _shap_lgbm_int(lgbm, Xs_tr)
            else:
                v = None
        except Exception as e:
            v = None
            mean_abs_shap_by_cell[cell + "_err"] = repr(e)[:200]
        mean_abs_shap_by_cell[cell] = v

    # True causal tau via simulate(intervene_on=...).
    true_tau = true_tau_via_simulate(scm, p, std_y_obs)

    if not np.isfinite(true_tau).any():
        return None

    finite_tau = np.isfinite(true_tau)
    n_finite = int(finite_tau.sum())

    out = {
        "family": family,
        "p": p,
        "obs_corr": obs_corr,
        "int_effect": int_effect,
        "true_tau": true_tau,
    }

    # Per-cell metrics. Cell `ridge_int` is the existing "the SHAP" — its
    # arrays alias the legacy field names (mean_abs_shap, causal_misleading,
    # ...) so the original reducer / shap_failure_identifiable.tex pipeline
    # still works against the extended pickle.
    primary = "ridge_int"
    for cell in CELLS:
        v = mean_abs_shap_by_cell.get(cell)
        out[f"mean_abs_shap_{cell}"] = v
        if v is None:
            cm = np.zeros(p, dtype=bool)
            shap_top = np.zeros(p, dtype=bool)
            rho = float("nan")
        else:
            shap_thresh = np.quantile(v, 1.0 - SHAP_TOPK_FRAC)
            shap_top = v >= shap_thresh
            cm = shap_top & (np.abs(true_tau) < TAU_NEAR_ZERO) & finite_tau
            rho_v, _ = spearmanr(v, np.abs(true_tau))
            rho = float(rho_v)
        out[f"shap_top_{cell}"] = shap_top
        out[f"causal_misleading_{cell}"] = cm
        out[f"has_causal_misleading_{cell}"] = bool(cm.any())
        out[f"n_causal_misleading_{cell}"] = int(cm.sum())
        out[f"frac_causal_misleading_{cell}"] = (
            float(cm.sum() / n_finite) if n_finite else float("nan")
        )
        out[f"shap_truetau_spearman_{cell}"] = rho

    # Legacy aliases (cell ridge_int): keep the original schema so downstream
    # consumers of the existing pickle keep working.
    mean_abs_shap = mean_abs_shap_by_cell[primary]
    if mean_abs_shap is None:
        # Should not happen — this is the original cell.
        return None
    rho_shap_int, _ = spearmanr(mean_abs_shap, int_effect)
    rho_shap_truetau = out[f"shap_truetau_spearman_{primary}"]
    rho_corr_truetau, _ = spearmanr(obs_corr, np.abs(true_tau))
    out.update({
        "mean_abs_shap": mean_abs_shap,
        "shap_top": out[f"shap_top_{primary}"],
        "causal_misleading": out[f"causal_misleading_{primary}"],
        "has_causal_misleading": out[f"has_causal_misleading_{primary}"],
        "n_causal_misleading": out[f"n_causal_misleading_{primary}"],
        "frac_causal_misleading": out[f"frac_causal_misleading_{primary}"],
        "shap_int_spearman": float(rho_shap_int),
        "shap_truetau_spearman": float(rho_shap_truetau),
        "corr_truetau_spearman": float(rho_corr_truetau),
    })

    for name, model in PROBES.items():
        if model is None:
            continue
        with torch.no_grad():
            X_in = torch.from_numpy(Xs.astype(np.float32)).unsqueeze(0)
            y_in = torch.from_numpy(ys.astype(np.float32)).unsqueeze(0)
            tau_hat = model(X_in, y_in).squeeze(0).cpu().numpy().astype(float)
        out[f"probe_{name}_tau"] = tau_hat
        rho_p_int, _ = spearmanr(np.abs(tau_hat), int_effect)
        rho_p_true, _ = spearmanr(np.abs(tau_hat), np.abs(true_tau))
        out[f"probe_{name}_int_spearman"] = float(rho_p_int)
        out[f"probe_{name}_truetau_spearman"] = float(rho_p_true)
        # Of the causal-misleading features (cell ridge_int, the legacy
        # definition the existing reducer/probe row was tuned against),
        # what fraction does the probe correctly flag?
        cm_legacy = out["causal_misleading"]
        if cm_legacy.any():
            out[f"probe_{name}_correct_on_causal_misleading"] = float(
                (np.abs(tau_hat[cm_legacy]) < PROBE_THRESH).mean()
            )
        else:
            out[f"probe_{name}_correct_on_causal_misleading"] = float("nan")
        # Probe Pearson with true tau on this DGP.
        sd_p = float(np.std(tau_hat))
        sd_t = float(np.std(true_tau))
        if sd_p > 0 and sd_t > 0:
            out[f"probe_{name}_truetau_pearson"] = float(np.corrcoef(tau_hat, true_tau)[0, 1])
        else:
            out[f"probe_{name}_truetau_pearson"] = float("nan")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--out-dir", default="results/chunks_ident")
    ap.add_argument("--family", default="mix",
                    choices=["lingam", "anm", "tree", "mix"])
    args = ap.parse_args()

    chunk_seed = BASE_SEED * 1_000_003 + args.chunk_id
    rng = np.random.default_rng(chunk_seed)
    torch.manual_seed(chunk_seed)

    out_dir = os.path.join(os.path.dirname(__file__), "..", args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"chunk_{args.chunk_id:05d}.pkl")
    print(f"[chunk {args.chunk_id}] family={args.family} seed={chunk_seed} n={args.chunk_size}")

    families_pool = ["lingam", "anm", "tree"] if args.family == "mix" else [args.family]
    results, errors = [], 0
    for idx in range(args.chunk_size):
        family = str(rng.choice(families_pool))
        try:
            scm, meta = sample_scm(family, rng)
            r = analyse_one(scm, **meta)
            if r is None:
                errors += 1
                continue
            results.append(r)
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  err idx={idx} family={family}: {e}")
            continue
        if (idx + 1) % 50 == 0:
            recent = results[-min(50, len(results)):]
            print(f"  [{idx+1:4d}/{args.chunk_size}] cm-rate {np.mean([r['has_causal_misleading'] for r in recent])*100:.0f}% errs={errors}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump({"chunk_id": args.chunk_id, "seed": chunk_seed,
                     "results": results, "errors": errors}, f)
    print(f"[chunk {args.chunk_id}] done: {len(results)} ok, {errors} errors -> {out_path}")


if __name__ == "__main__":
    main()
