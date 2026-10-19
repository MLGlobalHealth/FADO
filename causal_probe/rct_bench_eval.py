"""Run the causal-method suite on RCT_Bench and compare to the RCT gold standard.

For each trial we know the randomized Treatment, so the unadjusted difference in
means on the primary outcome is an unbiased estimate of the true ATE. Every
method estimates that same ATE; we score each on the standardized scale
d = ATE / std(Y) (comparable across trials with wildly different raw units).

Methods (each returns a RAW-outcome-unit ATE; we standardize at the end):
  * rct_unadjusted   difference in means (the ground truth; sanity anchor)
  * ols_adjusted     OLS coef of T in Y ~ T + X (covariate-adjusted linear ATE)
  * fado_labeled     FADO told which column is T (FADOBaseline.estimate_ate)
  * fado_hidden      FADO with T hidden among X + proxy/leak/noise decoys
  * drlearner        econml DRLearner (doubly-robust), raw .ate()
  * lingam           DirectLiNGAM total effect of T on Y (raw units)
  * shap_rank        LightGBM TreeSHAP importance rank of T among features (foil)

CausalPFN / DoPFN are added in phase 2 (extra deps / GPU repo).

Runs on a single GPU workstation (.venv has econml/lingam/shap/lightgbm/torch). FADO uses the
canonical FINAL foundation-L seed0 checkpoint.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

from causal_probe.rct_bench_loader import load_trial, TrialData


# ----------------------------- FADO (probe) --------------------------------
_FADO = {}  # lazy cache: ckpt -> FADOBaseline


def _fado(ckpt: str, device: str, n_rows: int):
    key = (ckpt, device, n_rows)
    if key not in _FADO:
        from causal_probe.fado_baseline import FADOBaseline
        _FADO[key] = FADOBaseline(ckpt, device=device, n_rows=n_rows, seed=2025)
    return _FADO[key]


def fado_labeled(td: TrialData, ckpt: str, device="cpu", n_rows=512) -> float:
    """FADO reads tau off the known Treatment column -> raw-unit ATE."""
    if td.p == 0:
        # still works: X is just T; FADOBaseline prepends T so give empty cov
        X = np.zeros((td.n, 0))
    else:
        X = td.X
    return float(_fado(ckpt, device, n_rows).estimate_ate(X, td.T, td.Y))


_MODEL = {}


def _load_probe(ckpt: str, device: str):
    if ckpt not in _MODEL:
        from causal_probe.eval import _load_model
        _MODEL[ckpt] = _load_model(ckpt, device=device)
    return _MODEL[ckpt]


def fado_hidden(td: TrialData, ckpt: str, device="cpu", n_rows=512, k_noise=3,
                seed=2025) -> dict:
    """FADO with T hidden among {X, proxy P_T=T+noise, leak L_Y=Y+noise, noise}.
    Mirrors the Hillstrom construction: assemble, column-shuffle, standardize,
    read pred at the (shuffled) T index. Returns raw-unit tau at T plus the
    decoy taus and T's rank, so we can check T is recovered above the decoys."""
    import torch
    rng = np.random.default_rng(seed)
    T = td.T.astype(float); Y = td.Y.astype(float)
    sdT = T.std() or 1.0
    proxy = T + rng.normal(0, 0.3, size=td.n) * (sdT if sdT > 0 else 1.0)
    leak = Y + rng.normal(0, 0.3, size=td.n) * (Y.std() or 1.0)
    noise = rng.normal(0, 1, size=(td.n, k_noise))
    blocks = [T[:, None], proxy[:, None], leak[:, None]]
    labels = ["T", "P_T", "L_Y"]
    if td.p:
        blocks.append(td.X); labels += list(td.x_names)
    blocks.append(noise); labels += [f"noise{i}" for i in range(k_noise)]
    Xaug = np.column_stack(blocks)
    perm = rng.permutation(Xaug.shape[1])
    Xaug = Xaug[:, perm]; labels = [labels[i] for i in perm]
    T_idx = labels.index("T"); P_idx = labels.index("P_T"); L_idx = labels.index("L_Y")

    # standardize columns + y
    mu = Xaug.mean(0); sd = Xaug.std(0); sd[sd == 0] = 1.0
    Xs = (Xaug - mu) / sd
    muy = Y.mean(); sdy = Y.std() or 1.0
    ys = (Y - muy) / sdy
    # subsample rows to n_rows
    idx = np.arange(len(ys))
    if len(ys) > n_rows:
        idx = rng.choice(len(ys), n_rows, replace=False)
    model = _load_probe(ckpt, device)
    Xt = torch.tensor(Xs[idx][None], dtype=torch.float32, device=device)
    yt = torch.tensor(ys[idx][None], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(Xt, yt).squeeze(0).cpu().numpy()  # length p, standardized tau
    tau_std_T = float(pred[T_idx])
    raw_T = tau_std_T * sdy / (2.0 * sdT)  # back to raw outcome units
    order = np.argsort(-np.abs(pred))
    T_rank = int(np.where(order == T_idx)[0][0]) + 1
    return {"raw": raw_T, "tau_std_T": tau_std_T,
            "tau_std_P_T": float(pred[P_idx]), "tau_std_L_Y": float(pred[L_idx]),
            "T_rank": T_rank, "n_aug_cols": Xaug.shape[1]}


# ----------------------------- baselines -----------------------------------
def ols_adjusted(td: TrialData) -> float:
    from numpy.linalg import lstsq
    D = np.column_stack([np.ones(td.n), td.T.astype(float)] +
                        ([td.X] if td.p else []))
    beta, *_ = lstsq(D, td.Y.astype(float), rcond=None)
    return float(beta[1])


def drlearner(td: TrialData, seed=0) -> float:
    if td.p == 0:
        return float("nan")
    try:
        from econml.dr import DRLearner
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        n_est = 30 if td.n < 300 else 80
        est = DRLearner(
            model_regression=RandomForestRegressor(n_estimators=n_est, random_state=seed),
            model_propensity=RandomForestClassifier(n_estimators=n_est, random_state=seed),
            model_final=RandomForestRegressor(n_estimators=n_est, random_state=seed),
            cv=2, random_state=seed)
        # covariates go in X (final CATE model); ate() averages CATE -> ATE
        est.fit(td.Y.astype(float), td.T.astype(int), X=td.X, W=None)
        return float(est.ate(X=td.X))
    except Exception as e:
        return float("nan")


def lingam_tau_raw(td: TrialData) -> float:
    """DirectLiNGAM total effect of T on Y in raw units."""
    try:
        import lingam
        M = np.column_stack([td.T.astype(float),
                             (td.X if td.p else np.zeros((td.n, 0))),
                             td.Y.astype(float)])
        m = lingam.DirectLiNGAM(); m.fit(M)
        B = m.adjacency_matrix_
        A = np.linalg.inv(np.eye(B.shape[0]) - B)  # total effects
        return float(A[-1, 0])  # effect of col0 (T) on last col (Y)
    except Exception:
        return float("nan")


_CPFN_PATCHED = [False]


def _patch_causalpfn_small_n() -> None:
    """CausalPFN's causal_estimator.py:394 sets an internal GradientBoosting
    nuisance model's min_samples_leaf=int(n/100), which underflows to 0 for
    n<100 and trips sklearn's validator -> it silently fails on every trial with
    fewer than 100 subjects. Floor it at 1 so CausalPFN runs on the small trials
    too (fair full-coverage comparison with FADO)."""
    if _CPFN_PATCHED[0]:
        return
    import causalpfn.causal_estimator as ce
    orig = ce.GradientBoostingRegressor

    def _safe_gbr(*a, **k):
        # factory: return a genuine GradientBoostingRegressor (keeps sklearn
        # introspection/clone happy) with min_samples_leaf floored at 1
        if "min_samples_leaf" in k:
            k["min_samples_leaf"] = max(1, int(k["min_samples_leaf"]))
        return orig(*a, **k)

    ce.GradientBoostingRegressor = _safe_gbr
    _CPFN_PATCHED[0] = True


def causalpfn_ate(td: TrialData, device="cpu") -> float:
    """CausalPFN known-treatment ATE (raw outcome units)."""
    if td.p == 0:
        return float("nan")
    try:
        _patch_causalpfn_small_n()
        from causalpfn import ATEEstimator
        est = ATEEstimator(device=device)
        est.fit(td.X, td.T.astype(int), td.Y.astype(float))
        return float(np.asarray(est.estimate_ate()).ravel()[0])
    except Exception:
        return float("nan")


def shap_rank(td: TrialData) -> dict:
    """LightGBM TreeSHAP: rank of T among [T, X] by mean|SHAP|. Foil (not an ATE)."""
    if td.p == 0:
        return {"T_rank": 1, "n_feat": 1}
    try:
        import lightgbm as lgb
        feats = np.column_stack([td.T.astype(float), td.X])
        ybin = (td.Y > np.median(td.Y)).astype(int)
        clf = lgb.LGBMClassifier(n_estimators=200, verbose=-1)
        clf.fit(feats, ybin)
        contrib = clf.predict(feats, pred_contrib=True)[:, :-1]  # drop bias
        imp = np.abs(contrib).mean(0)
        order = np.argsort(-imp)
        return {"T_rank": int(np.where(order == 0)[0][0]) + 1, "n_feat": feats.shape[1]}
    except Exception:
        return {"T_rank": None, "n_feat": td.p + 1}


# ------------------------------- driver ------------------------------------
def run_trial(td: TrialData, ckpt: str, device: str, do_fado: bool) -> dict:
    sdy = float(np.std(td.Y)) or 1.0
    rct = td.rct_ate_raw()
    rec = {
        "trial": td.trial, "n": td.n, "p": td.p, "primary": td.primary,
        "primary_type": td.primary_type, "n_arms": td.n_arms_original,
        "std_y": sdy, "rct_ate_raw": rct, "rct_d": rct / sdy,
        "notes": td.notes,
        "raw": {"rct_unadjusted": rct, "ols_adjusted": ols_adjusted(td),
                "drlearner": drlearner(td), "lingam": lingam_tau_raw(td),
                "causalpfn": causalpfn_ate(td, device)},
        "shap_rank": shap_rank(td),
    }
    if do_fado:
        rec["raw"]["fado_labeled"] = fado_labeled(td, ckpt, device)
        rec["fado_hidden"] = fado_hidden(td, ckpt, device)
        rec["raw"]["fado_hidden"] = rec["fado_hidden"]["raw"]
    # standardized d for every raw estimate
    rec["d"] = {k: (v / sdy if v is not None and np.isfinite(v) else None)
                for k, v in rec["raw"].items()}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/RCT_Bench/cleaned_data")
    ap.add_argument("--ckpt", default="causal_probe/results/probe_intersample_L_seed0.ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="causal_probe/results/rct_bench_eval.json")
    ap.add_argument("--limit", type=int, default=0, help="only first N trials (smoke)")
    ap.add_argument("--trials", default="", help="comma list e.g. trial1,trial9 (smoke)")
    ap.add_argument("--no-fado", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "trial*.csv")),
                   key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))
    if args.trials:
        want = set(args.trials.split(","))
        paths = [p for p in paths if os.path.basename(p).replace(".csv", "") in want]
    if args.limit:
        paths = paths[:args.limit]

    out = []
    for p in paths:
        td = load_trial(p)
        if td is None:
            print(f"skip {os.path.basename(p)}"); continue
        rec = run_trial(td, args.ckpt, args.device, do_fado=not args.no_fado)
        out.append(rec)
        d = rec["d"]
        fh = f" fado_h={d.get('fado_hidden'):+.2f}(rk{rec.get('fado_hidden',{}).get('T_rank','-')})" if "fado_hidden" in d and d["fado_hidden"] is not None else ""
        fl = f" fado_l={d.get('fado_labeled'):+.2f}" if d.get("fado_labeled") is not None else ""
        print(f"{td.trial:>9} n={td.n:>4} p={td.p:>2} | rct_d={rec['rct_d']:+.2f} "
              f"ols={_f(d['ols_adjusted'])} drl={_f(d['drlearner'])} "
              f"lingam={_f(d['lingam'])}{fl}{fh}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}  ({len(out)} trials)")


def _f(x):
    return f"{x:+.2f}" if x is not None and np.isfinite(x) else "  nan"


if __name__ == "__main__":
    main()
