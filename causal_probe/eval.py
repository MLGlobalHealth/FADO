"""Evaluate a trained causal probe against held-out random SCMs + motifs.

Metrics:
  * Per-episode: Pearson, Spearman, MSE, MAE between tau_hat and true tau.
  * Sign accuracy on features with |tau| > 0.1.
  * AUROC for detecting nonzero-effect features (label: |tau| > 0.1).
  * Zero-true-effect MAE (especially for features that are associationally
    predictive but causally zero).
  * Motif-level table: per-feature prediction vs baselines vs truth.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.model import CausalProbe, ProbeConfig
from causal_probe.motifs import ALL_MOTIFS, motif_scm
from causal_probe.scm import LinearNonGaussianSCM


# Per-SCM seeds for eval are drawn from the upper int32 half so they are
# guaranteed disjoint from train.py's [0, 2**31) range; this prevents any
# train/eval data leakage even if the user accidentally invokes both with
# the same --seed value.
_EVAL_SCM_SEED_LO = 2**31
_EVAL_SCM_SEED_HI = 2**32


# Frozen canonical paper-headline mixture for foundation-model eval. Uniform
# over 10 SCM families (10 _PAPER_MIXTURE_PRIOR families; gp_cde dropped for
# numerical stability of the held-out draw). Pinned in eval.py, not in
# train.py, so the eval benchmark stays fixed across training-prior
# rotations.
_PAPER_MIXTURE_PRIOR = {
    "linear": 0.10, "nonlinear": 0.10, "mlp": 0.10,
    "hidden": 0.10, "mlp_hidden": 0.10, "mixed": 0.10,
    "treatment": 0.10, "polynomial": 0.10, "spline": 0.10,
    "discrete_mix": 0.10,
}

# Single-family list (excluding "mixture" + "gp_cde") used by callers like
# scripts/cross_family_heatmap*.py and run_baseline.py that want to iterate
# over the canonical eval families.
ALL_EVAL_SCM_TYPES = (
    "linear", "nonlinear", "mlp", "hidden", "mlp_hidden", "mixed",
    "treatment", "polynomial", "spline", "gp_cde", "discrete_mix",
)


def _make_eval_scm(scm_type: str, p: int, rng: np.random.Generator,
                   noise: str = "laplace", nonlinear_mc: int = 2048):
    """Construct an SCM of the given family for held-out eval."""
    seed = rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI)
    sub_rng = np.random.default_rng(seed)
    if scm_type == "mixture":
        families = list(_PAPER_MIXTURE_PRIOR.keys())
        weights = np.asarray(list(_PAPER_MIXTURE_PRIOR.values()), dtype=np.float64)
        weights = weights / weights.sum()
        family = str(sub_rng.choice(families, p=weights))
        return _make_eval_scm(family, p, sub_rng, noise=noise, nonlinear_mc=nonlinear_mc)
    if scm_type == "linear":
        return LinearNonGaussianSCM(p=p, rng=sub_rng, noise=noise)
    if scm_type == "nonlinear":
        from causal_probe.scm_nonlinear import NonlinearSCM
        return NonlinearSCM(p=p, rng=sub_rng, n_mc=nonlinear_mc)
    if scm_type == "mlp":
        from causal_probe.scm_mlp import MLPSCM
        return MLPSCM(p=p, rng=sub_rng, n_mc=nonlinear_mc)
    if scm_type == "hidden":
        from causal_probe.scm_hidden import LinearNonGaussianSCMHidden
        return LinearNonGaussianSCMHidden(p=p, rng=sub_rng, noise=noise)
    if scm_type == "mlp_hidden":
        from causal_probe.scm_mlp_hidden import MLPSCMHidden
        return MLPSCMHidden(p=p, rng=sub_rng, n_mc=nonlinear_mc)
    if scm_type == "mixed":
        from causal_probe.scm_mixed import LinearMixedSCM
        return LinearMixedSCM(p=p, rng=sub_rng, noise=noise)
    if scm_type == "treatment":
        from causal_probe.scm_treatment import DesignatedTreatmentSCM
        return DesignatedTreatmentSCM(p=max(p, 2), rng=sub_rng,
                                      n_mc=nonlinear_mc, noise=noise)
    if scm_type == "polynomial":
        from causal_probe.scm_polynomial import PolynomialSCM
        return PolynomialSCM(p=p, rng=sub_rng, n_mc=nonlinear_mc, noise=noise)
    if scm_type == "spline":
        from causal_probe.scm_spline import SplineAdditiveSCM
        return SplineAdditiveSCM(p=p, rng=sub_rng, n_mc=nonlinear_mc, noise=noise)
    if scm_type == "gp_cde":
        from causal_probe.scm_gp_cde import GPGridSCM
        return GPGridSCM(p=p, rng=sub_rng, n_mc=nonlinear_mc, noise=noise)
    if scm_type == "discrete_mix":
        from causal_probe.scm_discrete import DiscreteMixedSCM
        return DiscreteMixedSCM(p=p, rng=sub_rng, noise=noise)
    raise ValueError(f"unknown scm_type: {scm_type}")


def _spearman_pearson(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, float]:
    from scipy.stats import spearmanr, pearsonr
    if pred.size < 2 or np.std(pred) == 0 or np.std(tgt) == 0:
        return float("nan"), float("nan")
    sp, _ = spearmanr(pred, tgt)
    pe, _ = pearsonr(pred, tgt)
    return float(sp), float(pe)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Simple AUROC for binary labels."""
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _load_model(ckpt_path: str, device: str = "cpu"):
    """Load any checkpoint, handling both legacy `cfg`+CausalProbe ckpts
    and the new `arch`+`arch_kwargs` format that supports any registered
    architecture."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "arch" in ckpt and "arch_kwargs" in ckpt:
        from causal_probe.models import build_arch
        model = build_arch(ckpt["arch"], **ckpt["arch_kwargs"]).to(device)
    else:
        cfg = ProbeConfig(**ckpt["cfg"])
        model = CausalProbe(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def evaluate_random(
    model: CausalProbe,
    n_scms: int,
    p: int,
    n_rows: int,
    rng: np.random.Generator,
    device: str,
    noise: str = "laplace",
    scm_type: str = "linear",
    nonlinear_mc: int = 2048,
    raw_out: dict | None = None,
    p_pad: int | None = None,
) -> dict:
    """Run eval on n_scms random held-out SCMs of the given family.

    If ``raw_out`` is provided, the per-SCM arrays (preds, truths, marginal,
    multivariate; each shape (n_scms, p)) are stashed into it under keys
    ``random_pred / random_true / random_marginal / random_multivariate``
    so the caller can bootstrap-CI metrics over SCMs.

    If ``p_pad`` is set and > p, each X is padded with (p_pad - p) iid
    standard-normal noise columns and the model is run on the padded
    input. This matches the variable-p training distribution (which
    always pads to p_max = 15) and removes the OOD shape mismatch that
    otherwise penalizes mixed-p foundation models at eval. Metrics are
    reported over the FIRST p columns (the real features); the padded
    columns are evaluated separately and reported under "padded".
    """
    preds, truths, assocs, regs = [], [], [], []
    preds_pad = [] if (p_pad is not None and p_pad > p) else None
    for _ in range(n_scms):
        scm = _make_eval_scm(scm_type, p, rng, noise=noise, nonlinear_mc=nonlinear_mc)
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI)))
        X_real = samp.X.astype(np.float32)
        if p_pad is not None and p_pad > p:
            n_pad = p_pad - p
            pad_seed = int(rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI))
            pad_noise = np.random.default_rng(pad_seed).standard_normal((n_rows, n_pad)).astype(np.float32)
            X_model = np.concatenate([X_real, pad_noise], axis=1)
        else:
            X_model = X_real
        X_t = torch.from_numpy(X_model).unsqueeze(0).to(device)
        y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            full_pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        preds.append(full_pred[:p])
        truths.append(samp.tau)
        if preds_pad is not None:
            preds_pad.append(full_pred[p:])
        assocs.append(BASELINES["marginal"](samp.X, samp.y))
        regs.append(BASELINES["multivariate"](samp.X, samp.y))

    preds = np.stack(preds)        # (n_scms, p)
    truths = np.stack(truths)
    assocs = np.stack(assocs)
    regs = np.stack(regs)

    if raw_out is not None:
        raw_out["random_pred"] = preds.astype(np.float32)
        raw_out["random_true"] = truths.astype(np.float32)
        raw_out["random_marginal"] = assocs.astype(np.float32)
        raw_out["random_multivariate"] = regs.astype(np.float32)

    flat_p = preds.reshape(-1)
    flat_t = truths.reshape(-1)
    flat_a = assocs.reshape(-1)
    flat_r = regs.reshape(-1)

    sp_model, pe_model = _spearman_pearson(flat_p, flat_t)
    sp_assoc, pe_assoc = _spearman_pearson(flat_a, flat_t)
    sp_reg, pe_reg = _spearman_pearson(flat_r, flat_t)

    mse_model = float(np.mean((flat_p - flat_t) ** 2))
    mse_zero = float(np.mean(flat_t ** 2))
    mse_assoc = float(np.mean((flat_a - flat_t) ** 2))
    mse_reg = float(np.mean((flat_r - flat_t) ** 2))

    # R^2 = 1 - SS_res / SS_tot
    ss_tot = float(np.sum((flat_t - flat_t.mean()) ** 2))
    r2_model = 1.0 - float(np.sum((flat_p - flat_t) ** 2)) / max(ss_tot, 1e-12)
    r2_assoc = 1.0 - float(np.sum((flat_a - flat_t) ** 2)) / max(ss_tot, 1e-12)
    r2_reg = 1.0 - float(np.sum((flat_r - flat_t) ** 2)) / max(ss_tot, 1e-12)

    # Sign accuracy on signal features
    signal_mask = np.abs(flat_t) > 0.1
    if signal_mask.any():
        sign_acc_model = float(np.mean(np.sign(flat_p[signal_mask]) == np.sign(flat_t[signal_mask])))
        sign_acc_assoc = float(np.mean(np.sign(flat_a[signal_mask]) == np.sign(flat_t[signal_mask])))
    else:
        sign_acc_model = sign_acc_assoc = float("nan")

    # Nonzero detection AUROC
    labels_nonzero = (np.abs(flat_t) > 0.1).astype(int)
    auroc_model = _auroc(np.abs(flat_p), labels_nonzero)
    auroc_assoc = _auroc(np.abs(flat_a), labels_nonzero)
    auroc_reg = _auroc(np.abs(flat_r), labels_nonzero)

    # Zero-effect MAE — MAE over features where |tau| <= 0.1 (should be small)
    zero_mask = np.abs(flat_t) <= 0.1
    mae_zero_model = float(np.mean(np.abs(flat_p[zero_mask] - flat_t[zero_mask]))) if zero_mask.any() else float("nan")
    mae_zero_assoc = float(np.mean(np.abs(flat_a[zero_mask] - flat_t[zero_mask]))) if zero_mask.any() else float("nan")

    out = {
        "n_scms": n_scms, "p": p, "n_rows": n_rows, "noise": noise,
        "scm_type": scm_type,
        "pearson": {"model": pe_model, "marginal": pe_assoc, "multivariate": pe_reg},
        "spearman": {"model": sp_model, "marginal": sp_assoc, "multivariate": sp_reg},
        "mse": {"model": mse_model, "marginal": mse_assoc, "multivariate": mse_reg, "zero": mse_zero},
        "r2": {"model": r2_model, "marginal": r2_assoc, "multivariate": r2_reg},
        "sign_acc_signal": {"model": sign_acc_model, "marginal": sign_acc_assoc},
        "auroc_nonzero": {"model": auroc_model, "marginal": auroc_assoc, "multivariate": auroc_reg},
        "mae_zero_features": {"model": mae_zero_model, "marginal": mae_zero_assoc},
        "signal_frac": float(signal_mask.mean()),
    }
    if preds_pad is not None:
        pp = np.stack(preds_pad).reshape(-1)
        out["p_pad"] = p_pad
        # Padded columns are pure-noise: ground truth tau is exactly 0 by
        # construction (the SCM doesn't see them). Report how tight the
        # model's predictions are around zero — the foundation-model
        # "doesn't confuse correlation for causation" stress test.
        out["padded_features"] = {
            "n_padded": int(p_pad - p),
            "mae": float(np.mean(np.abs(pp))),
            "rmse": float(np.sqrt(np.mean(pp ** 2))),
            "max_abs": float(np.max(np.abs(pp))),
            "frac_within_0_05": float(np.mean(np.abs(pp) < 0.05)),
            "frac_within_0_1": float(np.mean(np.abs(pp) < 0.1)),
        }
    return out


def evaluate_motifs(
    model: CausalProbe,
    p: int,
    n_rows: int,
    rng: np.random.Generator,
    device: str,
    n_repeats: int = 30,
    raw_out: dict | None = None,
) -> dict:
    """For each motif, average predictions + baselines over n_repeats SCM instances.

    If ``raw_out`` is provided, per-instance arrays for each motif are stashed
    under keys ``motif_<name>_pred / true / marginal / multivariate`` (each
    shape (n_repeats, p)) so the caller can compute bootstrap CIs on the
    per-feature mean tau_hat.
    """
    out = {}
    for name, make in ALL_MOTIFS.items():
        # Skip motifs that can't construct at this p (e.g.
        # J_latent_clique_3x3 needs p>=10). Probe by constructing one
        # spec inside a try; if it raises, log and continue. Also skip
        # if n_repeats==0 so downstream array-shape arithmetic stays
        # 2D.
        if n_repeats <= 0:
            out[name] = {"n_repeats": 0, "rows": [], "skipped": "n_repeats=0"}
            continue
        try:
            _probe_spec = make(p=p, rng=np.random.default_rng(rng.integers(0, 2**31)))
            del _probe_spec
        except (ValueError, AssertionError) as e:
            print(f"  skipping motif {name}: {e}", flush=True)
            out[name] = {"n_repeats": 0, "rows": [], "skipped": str(e)}
            continue
        per_feat = defaultdict(list)
        # Per-instance raw arrays for bootstrap.
        raw_pred = np.empty((n_repeats, p), dtype=np.float32)
        raw_true = np.empty((n_repeats, p), dtype=np.float32)
        raw_assoc = np.empty((n_repeats, p), dtype=np.float32)
        raw_multi = np.empty((n_repeats, p), dtype=np.float32)
        for r in range(n_repeats):
            spec = make(p=p, rng=np.random.default_rng(rng.integers(0, 2**31)))
            scm = motif_scm(spec, rng=np.random.default_rng(rng.integers(0, 2**31)))
            samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
            X_t = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(device)
            y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(X_t, y_t).squeeze(0).cpu().numpy()
            assoc = BASELINES["marginal"](samp.X, samp.y)
            mreg = BASELINES["multivariate"](samp.X, samp.y)
            raw_pred[r] = pred
            raw_true[r] = samp.tau
            raw_assoc[r] = assoc
            raw_multi[r] = mreg
            for i in range(p):
                per_feat[i].append((float(samp.tau[i]), float(pred[i]),
                                    float(assoc[i]), float(mreg[i])))
        if raw_out is not None:
            raw_out[f"motif_{name}_pred"] = raw_pred
            raw_out[f"motif_{name}_true"] = raw_true
            raw_out[f"motif_{name}_marginal"] = raw_assoc
            raw_out[f"motif_{name}_multivariate"] = raw_multi
        rows = []
        for i in range(p):
            arr = np.asarray(per_feat[i])
            rows.append({
                "feature": i,
                "tau_true_mean": float(arr[:, 0].mean()),
                "tau_hat_mean": float(arr[:, 1].mean()),
                "assoc_mean": float(arr[:, 2].mean()),
                "multivariate_mean": float(arr[:, 3].mean()),
                "tau_hat_std": float(arr[:, 1].std()),
            })
        out[name] = {"n_repeats": n_repeats, "rows": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="causal_probe/results/probe.ckpt")
    ap.add_argument("--n-random-scms", type=int, default=500)
    ap.add_argument("--motif-repeats", type=int, default=30)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--p-pad", type=int, default=None,
                    help="If set and > --p, pad each random-SCM X with (p-pad - p) "
                         "iid standard-normal noise columns before feeding to the "
                         "model. Matches the variable-p training distribution "
                         "(which pads to p_max). Metrics on real columns; padded "
                         "columns reported separately under 'padded_features'.")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--scm-type",
                    choices=["linear", "nonlinear", "mlp", "hidden", "mlp_hidden", "mixed",
                             "treatment", "polynomial", "spline", "gp_cde", "discrete_mix",
                             "mixture"],
                    default="linear",
                    help="SCM family used for held-out random eval. "
                         "'mixture' draws each held-out SCM from the frozen "
                         "_PAPER_MIXTURE_PRIOR (10 families) — the paper's "
                         "foundation-model headline benchmark.")
    ap.add_argument("--noise", choices=["laplace", "gaussian", "heavy"], default="laplace",
                    help="Noise distribution for linear and hidden SCM families "
                         "(ignored by nonlinear / mlp / mlp_hidden / mixed).")
    ap.add_argument("--nonlinear-mc", type=int, default=2048,
                    help="MC sample count for nonlinear / mlp / mlp_hidden tau labels.")
    ap.add_argument("--out-json", default="causal_probe/results/eval.json")
    ap.add_argument("--out-npz", default=None,
                    help="If set, also save per-SCM (and per-motif-instance) "
                         "raw arrays for bootstrap-CI analysis.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = _load_model(args.ckpt, device=args.device)
    print(f"Loaded {args.ckpt}")

    raw: dict | None = {} if args.out_npz else None

    random_res = evaluate_random(model, args.n_random_scms, args.p, args.n_rows,
                                 rng=rng, device=args.device,
                                 noise=args.noise, scm_type=args.scm_type,
                                 nonlinear_mc=args.nonlinear_mc,
                                 raw_out=raw,
                                 p_pad=args.p_pad)
    motif_res = evaluate_motifs(model, args.p, args.n_rows, rng=rng,
                                device=args.device, n_repeats=args.motif_repeats,
                                raw_out=raw)

    # Pretty-print random results
    print()
    print(f"=== Random held-out SCMs (n={args.n_random_scms}, p={args.p}, n_rows={args.n_rows}) ===")
    for metric in ("pearson", "spearman", "r2", "auroc_nonzero", "sign_acc_signal",
                   "mae_zero_features", "mse"):
        sub = random_res.get(metric, {})
        if isinstance(sub, dict):
            row = "  ".join(f"{k}={v:+.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in sub.items())
            print(f"  {metric:<22s} {row}")
    print(f"  signal_frac (|tau|>0.1): {random_res['signal_frac']:.3f}")

    # Pretty-print motif table
    print()
    print(f"=== Motif behavior (avg over {args.motif_repeats} repeats) ===")
    for name, mres in motif_res.items():
        print(f"\n--- {name} ---")
        print(f"{'feat':<6}{'tau_true':>10}{'tau_hat':>10}{'assoc':>10}{'multi':>10}{'±hat_std':>10}")
        for row in mres["rows"]:
            print(
                f"{row['feature']:<6d}{row['tau_true_mean']:>+10.3f}"
                f"{row['tau_hat_mean']:>+10.3f}{row['assoc_mean']:>+10.3f}"
                f"{row['multivariate_mean']:>+10.3f}{row['tau_hat_std']:>10.3f}"
            )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"random": random_res, "motifs": motif_res}, f, indent=2)
    print(f"\nwrote {args.out_json}")

    if args.out_npz and raw is not None:
        Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_npz, **raw)
        print(f"wrote {args.out_npz} with {len(raw)} arrays")


if __name__ == "__main__":
    main()
