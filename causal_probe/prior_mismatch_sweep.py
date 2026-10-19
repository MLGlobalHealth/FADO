"""Prior-mismatch sweep: how FADO's accuracy degrades as the TEST distribution
moves away from the fixed training prior the foundation model was trained on.

Addresses the question ("vary how misspecified the prior is,
report degradation vs mismatch") along TWO complementary mismatch axes, run on
the single canonical foundation-L checkpoint (NOT retrained per condition).

The foundation model is trained on ``train._MIXED_PRIOR_DEFAULT`` — a fixed
mixture over 11 SCM families, with ``--noise heavy`` and heterogeneous
``p in [3, 15]`` padded to 15 (see ``train_foundation.slurm``). Both axes
below hold that model fixed and vary only the TEST distribution.

AXIS 1 — discrete family mismatch (``--axes family``)
    Evaluate the model on each held-out eval family, labelling each by its
    WEIGHT in the training prior ``_MIXED_PRIOR_DEFAULT``. This is the
    "SCM-class mismatch" view: families the prior up-weights (e.g. linear at
    0.30) vs families it barely covers (gp_cde / polynomial / spline /
    treatment / discrete_mix at 0.05 each). Reports mean Pearson r + a
    bootstrap CI over SCMs per family. This is NOT the cross_family matrix
    (that trains one specialist per family); here it is the ONE foundation
    model vs families at varying prior coverage.

    HONESTY NOTE on gp_cde: the project's paper-eval benchmark prior
    (``eval._PAPER_MIXTURE_PRIOR``) DROPS gp_cde (too numerically fragile for
    a headline), but the TRAINING prior ``_MIXED_PRIOR_DEFAULT`` DOES include
    gp_cde at weight 0.05. So gp_cde is the lowest-weight / most-fragile
    TRAINING family, not a family the model literally never saw. The JSON
    records each family's exact training weight (0.0 == genuinely never
    trained) so the figure/caption can describe the mismatch truthfully.

AXIS 2 — continuous out-of-support sweep (``--axes param``) — the load-bearing
degradation CURVE.
    Pick ONE interpretable continuous SCM parameter that the training prior
    draws from a BOUNDED range, and sweep the TEST value from inside that
    range out to well beyond it, plotting FADO r vs the parameter.

    Chosen parameter: linear-SCM EDGE-WEIGHT MAGNITUDE.
    In ``scm.LinearNonGaussianSCM`` every nonzero structural edge weight is
    drawn ``uniform(weight_lo, weight_hi)`` with the training defaults
    ``weight_lo=0.5, weight_hi=2.0`` (see ``scm.py``). So during training the
    model only ever saw per-edge coupling magnitudes in **[0.5, 2.0]**. We
    sweep a SINGLE fixed magnitude ``w`` (set weight_lo == weight_hi == w) so
    every edge has coupling exactly ``w``, and move ``w`` from below the
    in-prior floor up to ~3x the ceiling: the structural coupling becomes
    progressively stronger than anything in the training distribution. The
    scientific question: is degradation graceful (good — robust to mild
    misspecification) or a cliff (an honest limitation). The in-prior band
    [0.5, 2.0] is recorded in the JSON so the figure can shade it.

    Why edge-weight magnitude and not edge_prob / polynomial degree: those are
    FIXED scalar defaults in the constructors (edge_prob=0.35, degree sampled
    from a fixed {1,2,3} categorical), not drawn from a tunable continuous
    [a, b] the prior "samples from". Edge-weight magnitude is the cleanest
    genuinely range-backed continuous knob, and it stays on the LINEAR family
    where tau is exact (closed-form), so the degradation curve isn't confounded
    by Monte-Carlo label noise.

CLI::

    python -m causal_probe.prior_mismatch_sweep \
        --ckpt <foundation-L.ckpt> --device cuda \
        --axes all --n-scms 200 --seed 100 \
        --out causal_probe/results/prior_mismatch_sweep.json

    # dry-run: build SCMs for both axes, print classification + sweep points
    # + the prior range, exit 0 WITHOUT loading the ckpt or touching a GPU.
    python -m causal_probe.prior_mismatch_sweep --dry-run --axes all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# AXIS 1 config: eval families + their TRAINING-prior weight.
# ---------------------------------------------------------------------------
# The canonical foundation-L model was trained with --scm-type mixture, which
# dispatches to train._MIXED_PRIOR_DEFAULT. We re-import that dict so this file
# stays in lock-step with whatever prior the sprint last trained on, rather
# than hard-coding a copy that can silently drift.
def _training_prior() -> dict[str, float]:
    from causal_probe.train import _MIXED_PRIOR_DEFAULT
    w = dict(_MIXED_PRIOR_DEFAULT)
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


# All held-out eval families (eval.ALL_EVAL_SCM_TYPES, minus the "mixture"
# meta-family which isn't a single SCM class). Imported lazily in the helper.
def _eval_families() -> tuple[str, ...]:
    from causal_probe.eval import ALL_EVAL_SCM_TYPES
    return tuple(ALL_EVAL_SCM_TYPES)


# ---------------------------------------------------------------------------
# AXIS 2 config: linear-SCM edge-weight magnitude sweep.
# ---------------------------------------------------------------------------
# Training prior draws each edge weight uniform(WEIGHT_LO_TRAIN, WEIGHT_HI_TRAIN).
# These MUST match scm.LinearNonGaussianSCM's defaults; asserted at runtime.
WEIGHT_LO_TRAIN = 0.5
WEIGHT_HI_TRAIN = 2.0

# Sweep a single fixed magnitude w (weight_lo == weight_hi == w) across these
# points: two below the in-prior floor, several inside [0.5, 2.0], and a tail
# out to 3x the ceiling (6.0) where coupling is far stronger than training.
PARAM_SWEEP_POINTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _assert_prior_range_matches_code() -> None:
    """Fail loudly if scm.py's defaults drift from the documented [a, b]."""
    import inspect
    from causal_probe.scm import LinearNonGaussianSCM
    sig = inspect.signature(LinearNonGaussianSCM.__init__)
    lo = sig.parameters["weight_lo"].default
    hi = sig.parameters["weight_hi"].default
    if (lo, hi) != (WEIGHT_LO_TRAIN, WEIGHT_HI_TRAIN):
        raise AssertionError(
            f"LinearNonGaussianSCM edge-weight prior range drifted: code has "
            f"[{lo}, {hi}] but prior_mismatch_sweep documents "
            f"[{WEIGHT_LO_TRAIN}, {WEIGHT_HI_TRAIN}]. Update WEIGHT_*_TRAIN "
            f"and the module docstring, then re-run."
        )


# ---------------------------------------------------------------------------
# SCM seeds: reuse eval.py's upper-int32 band so there's zero train/eval leak.
# ---------------------------------------------------------------------------
def _scm_seed_bounds() -> tuple[int, int]:
    from causal_probe.eval import _EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI
    return _EVAL_SCM_SEED_LO, _EVAL_SCM_SEED_HI


def _make_linear_scm_fixed_weight(p: int, w: float, rng: np.random.Generator,
                                  noise: str):
    """A linear non-Gaussian SCM with EVERY edge weight fixed to magnitude w."""
    from causal_probe.scm import LinearNonGaussianSCM
    lo, hi = _scm_seed_bounds()
    sub_rng = np.random.default_rng(rng.integers(lo, hi))
    return LinearNonGaussianSCM(
        p=p, rng=sub_rng, noise=noise,
        weight_lo=float(w), weight_hi=float(w),
    )


# ---------------------------------------------------------------------------
# Core model-run helper: feed one SCM's sample through the model, with the
# p_pad padding the foundation model expects (trained padded to p_max=15).
# Returns (pred_on_real_cols (p,), true_tau (p,)).
# ---------------------------------------------------------------------------
def _predict_one(model, scm, p: int, n_rows: int, rng: np.random.Generator,
                 device: str, p_pad: int | None):
    import torch
    lo, hi = _scm_seed_bounds()
    samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(lo, hi)))
    X_real = samp.X.astype(np.float32)
    if p_pad is not None and p_pad > p:
        n_pad = p_pad - p
        pad_seed = int(rng.integers(lo, hi))
        pad_noise = np.random.default_rng(pad_seed).standard_normal(
            (n_rows, n_pad)).astype(np.float32)
        X_model = np.concatenate([X_real, pad_noise], axis=1)
    else:
        X_model = X_real
    X_t = torch.from_numpy(X_model).unsqueeze(0).to(device)
    y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        full_pred = model(X_t, y_t).squeeze(0).cpu().numpy()
    return full_pred[:p], samp.tau


def _pearson_flat(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred.reshape(-1)
    t = true.reshape(-1)
    if p.size < 2 or np.std(p) == 0 or np.std(t) == 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def _bootstrap_ci_over_scms(preds: np.ndarray, truths: np.ndarray,
                            n_boot: int, seed: int) -> tuple[float, float, float]:
    """Point Pearson + percentile 95% CI, resampling SCMs (rows) with replacement.

    preds / truths are (n_scms, p); each bootstrap draw resamples whole SCMs
    then flattens, so the CI reflects SCM-to-SCM variability.
    """
    point = _pearson_flat(preds, truths)
    n = preds.shape[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = _pearson_flat(preds[idx], truths[idx])
    lo = float(np.nanpercentile(vals, 2.5))
    hi = float(np.nanpercentile(vals, 97.5))
    return point, lo, hi


# ---------------------------------------------------------------------------
# AXIS 1 runner.
# ---------------------------------------------------------------------------
def run_axis_family(model, *, families, prior, n_scms, p, n_rows, noise,
                    nonlinear_mc, p_pad, seed, n_boot, device) -> dict:
    from causal_probe.eval import _make_eval_scm
    results = {}
    for fam in families:
        rng = np.random.default_rng(seed)  # same eval draws across families
        preds, truths = [], []
        for _ in range(n_scms):
            scm = _make_eval_scm(fam, p, rng, noise=noise, nonlinear_mc=nonlinear_mc)
            pred_i, tau_i = _predict_one(model, scm, p, n_rows, rng, device, p_pad)
            preds.append(pred_i)
            truths.append(tau_i)
        preds = np.stack(preds)
        truths = np.stack(truths)
        point, lo, hi = _bootstrap_ci_over_scms(preds, truths, n_boot, seed)
        train_w = float(prior.get(fam, 0.0))
        results[fam] = {
            "pearson": point, "ci_lo": lo, "ci_hi": hi,
            "train_prior_weight": train_w,
            "in_training_prior": train_w > 0.0,
            "n_scms": int(n_scms),
        }
        print(f"  {fam:<14s} r={point:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"train_w={train_w:.3f}  "
              f"({'IN' if train_w > 0 else 'OUT-OF'}-prior)", flush=True)
    return results


# ---------------------------------------------------------------------------
# AXIS 2 runner.
# ---------------------------------------------------------------------------
def run_axis_param(model, *, points, n_scms, p, n_rows, noise, p_pad,
                   seed, n_boot, device) -> dict:
    curve = []
    for w in points:
        rng = np.random.default_rng(seed)  # same eval draws across sweep points
        preds, truths = [], []
        for _ in range(n_scms):
            scm = _make_linear_scm_fixed_weight(p, w, rng, noise)
            pred_i, tau_i = _predict_one(model, scm, p, n_rows, rng, device, p_pad)
            preds.append(pred_i)
            truths.append(tau_i)
        preds = np.stack(preds)
        truths = np.stack(truths)
        point, lo, hi = _bootstrap_ci_over_scms(preds, truths, n_boot, seed)
        in_prior = WEIGHT_LO_TRAIN <= w <= WEIGHT_HI_TRAIN
        curve.append({
            "edge_weight": float(w), "pearson": point,
            "ci_lo": lo, "ci_hi": hi, "in_prior_range": in_prior,
            "n_scms": int(n_scms),
        })
        print(f"  w={w:<5.2f} r={point:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"({'in' if in_prior else 'OUT-of'} [{WEIGHT_LO_TRAIN},{WEIGHT_HI_TRAIN}])",
              flush=True)
    return {
        "param_name": "linear_scm_edge_weight_magnitude",
        "family": "linear",
        "prior_range": [WEIGHT_LO_TRAIN, WEIGHT_HI_TRAIN],
        "sweep_points": [float(w) for w in points],
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# Dry-run: no ckpt, no GPU — just report the two designs.
# ---------------------------------------------------------------------------
def _dry_run(axes: str, p: int, n_scms: int, noise: str) -> None:
    _assert_prior_range_matches_code()
    prior = _training_prior()
    families = _eval_families()

    print("=== prior_mismatch_sweep DRY RUN ===")
    print(f"axes={axes}  p={p}  n_scms={n_scms}  noise={noise}")
    print()

    if axes in ("family", "all"):
        print("--- AXIS 1: discrete family mismatch ---")
        print("Classification vs TRAINING prior (train._MIXED_PRIOR_DEFAULT, normalized):")
        in_fams = [f for f in families if prior.get(f, 0.0) > 0.0]
        out_fams = [f for f in families if prior.get(f, 0.0) == 0.0]
        for fam in families:
            w = prior.get(fam, 0.0)
            tag = "IN-prior" if w > 0 else "OUT-OF-prior (never trained)"
            print(f"    {fam:<14s} train_w={w:.3f}   {tag}")
        print(f"  IN-prior families  ({len(in_fams)}): {', '.join(in_fams)}")
        print(f"  OUT-OF-prior families ({len(out_fams)}): "
              f"{', '.join(out_fams) if out_fams else '(none — all eval families appear in the training prior)'}")
        # Lowest-weight in-prior family = the de-facto weakest-coverage case.
        if in_fams:
            lowest = min(in_fams, key=lambda f: prior[f])
            print(f"  lowest-weight (weakest coverage) in-prior family: "
                  f"{lowest} (train_w={prior[lowest]:.3f})")
        print()

    if axes in ("param", "all"):
        print("--- AXIS 2: continuous out-of-support sweep ---")
        print("param = linear-SCM edge-weight magnitude "
              "(scm.LinearNonGaussianSCM weight_lo==weight_hi==w)")
        print(f"  prior samples this from [a, b] = "
              f"[{WEIGHT_LO_TRAIN}, {WEIGHT_HI_TRAIN}] (uniform per edge)")
        print(f"  sweep points (w): {list(PARAM_SWEEP_POINTS)}")
        for w in PARAM_SWEEP_POINTS:
            region = "in-prior" if WEIGHT_LO_TRAIN <= w <= WEIGHT_HI_TRAIN else "OUT-of-prior"
            print(f"    w={w:<5.2f}  {region}")
        # Smoke-build ONE SCM per axis to prove construction works on CPU.
        rng = np.random.default_rng(0)
        scm = _make_linear_scm_fixed_weight(p, PARAM_SWEEP_POINTS[-1], rng, noise)
        samp = scm.sample(n=64, rng=np.random.default_rng(1))
        print(f"  smoke-built linear SCM at w={PARAM_SWEEP_POINTS[-1]}: "
              f"X={samp.X.shape} tau={samp.tau.shape} "
              f"|tau|_max={np.abs(samp.tau).max():.3f}")
        print()

    if axes in ("family", "all"):
        from causal_probe.eval import _make_eval_scm
        rng = np.random.default_rng(0)
        scm = _make_eval_scm(families[0], p, rng, noise=noise, nonlinear_mc=256)
        samp = scm.sample(n=64, rng=np.random.default_rng(2))
        print(f"  smoke-built eval SCM '{families[0]}': "
              f"X={samp.X.shape} tau={samp.tau.shape}")

    print("\nDRY RUN OK (no ckpt loaded, no GPU touched).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None,
                    help="Foundation-L checkpoint. Required unless --dry-run.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--n-scms", type=int, default=200)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--p", type=int, default=5,
                    help="Real feature count per eval SCM (metrics over these).")
    ap.add_argument("--p-pad", type=int, default=15,
                    help="Pad each X to this width with iid-noise columns before "
                         "feeding the model. Foundation-L trained padded to "
                         "p_max=15; default 15 matches that. Set <= --p to disable.")
    ap.add_argument("--noise", choices=["laplace", "gaussian", "heavy", "mixture"],
                    default="heavy",
                    help="Noise family. Default 'heavy' = the foundation "
                         "model's training/eval noise condition.")
    ap.add_argument("--nonlinear-mc", type=int, default=2048,
                    help="MC sample count for nonlinear-family tau labels (axis 1).")
    ap.add_argument("--n-boot", type=int, default=5000,
                    help="Bootstrap resamples for the per-point/per-family CI.")
    ap.add_argument("--axes", choices=["family", "param", "all"], default="all")
    ap.add_argument("--out", default="causal_probe/results/prior_mismatch_sweep.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the SCMs for both axes, print the family "
                         "classification + the param sweep points + the prior "
                         "[a,b] range, then exit 0 WITHOUT loading the ckpt "
                         "or touching a GPU.")
    args = ap.parse_args()

    if args.dry_run:
        _dry_run(args.axes, args.p, args.n_scms, args.noise)
        return

    if args.ckpt is None:
        ap.error("--ckpt is required unless --dry-run")

    _assert_prior_range_matches_code()
    from causal_probe.eval import _load_model
    model = _load_model(args.ckpt, device=args.device)
    print(f"Loaded {args.ckpt} on {args.device}")

    p_pad = args.p_pad if (args.p_pad is not None and args.p_pad > args.p) else None
    prior = _training_prior()
    families = _eval_families()

    out: dict = {
        "config": {
            "ckpt": args.ckpt, "device": args.device, "seed": args.seed,
            "n_scms": args.n_scms, "n_rows": args.n_rows, "p": args.p,
            "p_pad": p_pad, "noise": args.noise, "nonlinear_mc": args.nonlinear_mc,
            "n_boot": args.n_boot, "axes": args.axes,
        },
        "training_prior": prior,
    }

    if args.axes in ("family", "all"):
        print("\n=== AXIS 1: discrete family mismatch ===")
        out["axis_family"] = run_axis_family(
            model, families=families, prior=prior, n_scms=args.n_scms,
            p=args.p, n_rows=args.n_rows, noise=args.noise,
            nonlinear_mc=args.nonlinear_mc, p_pad=p_pad, seed=args.seed,
            n_boot=args.n_boot, device=args.device,
        )

    if args.axes in ("param", "all"):
        print("\n=== AXIS 2: continuous out-of-support sweep "
              "(linear edge-weight magnitude) ===")
        out["axis_param"] = run_axis_param(
            model, points=PARAM_SWEEP_POINTS, n_scms=args.n_scms, p=args.p,
            n_rows=args.n_rows, noise=args.noise, p_pad=p_pad, seed=args.seed,
            n_boot=args.n_boot, device=args.device,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
