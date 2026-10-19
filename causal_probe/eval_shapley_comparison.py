"""Compare FADO to oracle-graph causal-Shapley baselines (do-Shapley, Causal
Shapley) plus the associational baselines, on held-out linear SCMs and the
causal motifs.

The two Shapley baselines are given the TRUE causal graph (oracle); FADO and
the associational baselines see only (X, y). Scoring is correlation / sign /
error against the true tau, identical to eval.py.

Usage:
    python -m causal_probe.eval_shapley_comparison \
        --ckpt causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt \
        --n-random-scms 300 --motif-repeats 40 --p 5 --n-rows 512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.baselines import baseline_marginal, baseline_multivariate
from causal_probe.eval import _load_model, _make_eval_scm, _EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI
from causal_probe.motifs import ALL_MOTIFS, motif_scm
from causal_probe.shapley_causal import both_shapley


METHODS = ["model", "do_shapley", "causal_shapley", "marginal", "multivariate"]


def _fado_pred(model, X, y, device):
    X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(X_t, y_t).squeeze(0).cpu().numpy()


def _all_preds(model, scm, samp, device, n_mc, rng):
    """Return dict method -> length-p prediction vector for one SCM sample.

    do-Shapley and Causal Shapley share one coalition pass (both_shapley) so the
    MC interventions on nonlinear SCMs are computed once, not twice.
    """
    phi_do, phi_cs = both_shapley(scm, n_mc=n_mc,
                                  rng=np.random.default_rng(rng.integers(0, 2**31)))
    return {
        "model": _fado_pred(model, samp.X, samp.y, device),
        "do_shapley": phi_do,
        "causal_shapley": phi_cs,
        "marginal": baseline_marginal(samp.X, samp.y),
        "multivariate": baseline_multivariate(samp.X, samp.y),
    }


def _metrics(pred: np.ndarray, tau: np.ndarray) -> dict:
    from scipy.stats import spearmanr, pearsonr
    pe = sp = float("nan")
    if pred.size >= 2 and np.std(pred) > 0 and np.std(tau) > 0:
        pe = float(pearsonr(pred, tau)[0])
        sp = float(spearmanr(pred, tau)[0])
    mae = float(np.mean(np.abs(pred - tau)))
    signal = np.abs(tau) > 0.1
    sign_acc = (float(np.mean(np.sign(pred[signal]) == np.sign(tau[signal])))
                if signal.any() else float("nan"))
    zero = np.abs(tau) <= 0.1
    mae_zero = float(np.mean(np.abs(pred[zero]))) if zero.any() else float("nan")
    return {"pearson": pe, "spearman": sp, "mae": mae,
            "sign_acc": sign_acc, "mae_zero": mae_zero}


def evaluate_random(model, n_scms, p, n_rows, rng, device, scm_type="linear",
                    noise="laplace", n_mc=3000, nonlinear_mc=4096):
    """Flatten all (SCM, feature) pairs of one family and score each method vs true tau."""
    stacks = {m: [] for m in METHODS}
    taus = []
    for _ in range(n_scms):
        scm = _make_eval_scm(scm_type, p, rng, noise=noise, nonlinear_mc=nonlinear_mc)
        samp = scm.sample(
            n=n_rows,
            rng=np.random.default_rng(rng.integers(_EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI)),
        )
        preds = _all_preds(model, scm, samp, device, n_mc, rng)
        for m in METHODS:
            stacks[m].append(preds[m])
        taus.append(samp.tau)
    tau = np.concatenate(taus)
    return {m: _metrics(np.concatenate(stacks[m]), tau) for m in METHODS}


def evaluate_motifs(model, p, n_rows, rng, device, n_repeats=40):
    """Per-motif: pooled correlation over (repeat, feature) + per-feature means."""
    out = {}
    for name, make in ALL_MOTIFS.items():
        try:
            make(p=p, rng=np.random.default_rng(rng.integers(0, 2**31)))
        except (ValueError, AssertionError) as e:
            out[name] = {"skipped": str(e)}
            continue
        stacks = {m: [] for m in METHODS}
        taus = []
        per_feat_pred = {m: np.zeros((n_repeats, p)) for m in METHODS}
        per_feat_tau = np.zeros((n_repeats, p))
        for r in range(n_repeats):
            spec = make(p=p, rng=np.random.default_rng(rng.integers(0, 2**31)))
            scm = motif_scm(spec, rng=np.random.default_rng(rng.integers(0, 2**31)))
            samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
            preds = _all_preds(model, scm, samp, device, n_mc=3000, rng=rng)  # motifs are linear -> exact
            for m in METHODS:
                stacks[m].append(preds[m])
                per_feat_pred[m][r] = preds[m]
            taus.append(samp.tau)
            per_feat_tau[r] = samp.tau
        tau = np.concatenate(taus)
        out[name] = {
            "metrics": {m: _metrics(np.concatenate(stacks[m]), tau) for m in METHODS},
            "tau_mean": per_feat_tau.mean(0).tolist(),
            "pred_mean": {m: per_feat_pred[m].mean(0).tolist() for m in METHODS},
        }
    return out


def _fmt_row(label, d, keys):
    cells = "".join(
        f"{(d[k] if not np.isnan(d[k]) else float('nan')):>+10.3f}"
        if isinstance(d[k], float) else f"{d[k]:>10}"
        for k in keys
    )
    return f"{label:<16}{cells}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt")
    ap.add_argument("--families", default="linear,nonlinear,mlp,hidden,mlp_hidden",
                    help="comma-separated SCM families for the random-SCM sweep")
    ap.add_argument("--n-random-scms", type=int, default=200,
                    help="SCMs per family (MC families are slower; consider fewer)")
    ap.add_argument("--n-random-scms-mc", type=int, default=80,
                    help="SCMs for MC (nonlinear) families, overrides --n-random-scms")
    ap.add_argument("--motif-repeats", type=int, default=40)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--noise", default="laplace")
    ap.add_argument("--n-mc", type=int, default=3000,
                    help="MC rows per interventional mean for nonlinear Shapley")
    ap.add_argument("--nonlinear-mc", type=int, default=4096,
                    help="MC rows for the SCMs' own tau labels")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="causal_probe/results/shapley_comparison.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = _load_model(args.ckpt, device=args.device)
    print(f"Loaded {args.ckpt}\n")

    _MC_FAMILIES = {"nonlinear", "mlp", "mlp_hidden", "gp_cde"}
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    metric_keys = ["pearson", "spearman", "sign_acc", "mae", "mae_zero"]
    per_family = {}
    for fam in families:
        n = args.n_random_scms_mc if fam in _MC_FAMILIES else args.n_random_scms
        print(f"[random sweep] family={fam}  n_scms={n} ...", flush=True)
        per_family[fam] = evaluate_random(
            model, n, args.p, args.n_rows, rng=rng, device=args.device,
            scm_type=fam, noise=args.noise, n_mc=args.n_mc,
            nonlinear_mc=args.nonlinear_mc)

    motifs = evaluate_motifs(model, args.p, args.n_rows, rng=rng,
                             device=args.device, n_repeats=args.motif_repeats)

    print(f"\n=== Random held-out SCMs by family (p={args.p}, n_rows={args.n_rows}) "
          f"— Pearson r with true tau ===")
    print(f"{'family':<14}" + "".join(f"{m[:12]:>14}" for m in METHODS))
    for fam in families:
        cells = "".join(f"{per_family[fam][m]['pearson']:>+14.3f}" for m in METHODS)
        print(f"{fam:<14}{cells}")
    print(f"\n=== Same sweep — Spearman r with true tau ===")
    print(f"{'family':<14}" + "".join(f"{m[:12]:>14}" for m in METHODS))
    for fam in families:
        cells = "".join(f"{per_family[fam][m]['spearman']:>+14.3f}" for m in METHODS)
        print(f"{fam:<14}{cells}")
    print(f"\n=== Detailed metrics, family=linear ===")
    print(f"{'method':<16}" + "".join(f"{k:>10}" for k in metric_keys))
    for m in METHODS:
        if "linear" in per_family:
            print(_fmt_row(m, per_family["linear"][m], metric_keys))

    print("\n=== Per-motif Pearson r with true tau (pooled over "
          f"{args.motif_repeats} repeats x {args.p} features) ===")
    motif_names = [n for n in motifs if "metrics" in motifs[n]]
    print(f"{'motif':<26}" + "".join(f"{m[:12]:>14}" for m in METHODS))
    for n in motif_names:
        cells = "".join(f"{motifs[n]['metrics'][m]['pearson']:>+14.3f}" for m in METHODS)
        print(f"{n:<26}{cells}")

    print("\n=== Per-motif sign accuracy on signal features ===")
    print(f"{'motif':<26}" + "".join(f"{m[:12]:>14}" for m in METHODS))
    for n in motif_names:
        cells = "".join(f"{motifs[n]['metrics'][m]['sign_acc']:>+14.3f}" for m in METHODS)
        print(f"{n:<26}{cells}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"random_by_family": per_family, "motifs": motifs,
                   "config": vars(args)}, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
