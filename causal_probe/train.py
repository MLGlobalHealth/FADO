"""Online training loop for the causal probe.

Each step samples a batch of random linear non-Gaussian SCMs, draws n rows
from each, applies a random column permutation (labels permuted
accordingly), and updates the model to predict the standardized tau
contrast vector.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from causal_probe.model import count_params
from causal_probe.models import build_arch, list_archs
from causal_probe.scm import LinearNonGaussianSCM
from causal_probe.scm_utils import sample_noise


_FINETUNE_BINARY_PRIOR = {
    "binary_outcome": 0.40,
    "linear": 0.18, "nonlinear": 0.08, "mlp": 0.08, "hidden": 0.08,
    "mixed": 0.10, "treatment": 0.08,
}

# Mixture prior of the p=5 FADO model (paper Sec. 3, "FADO: p=5" in
# tab:checkpoints): five SCM families.
_FADO_P5_PRIOR = {
    "linear": 0.35, "nonlinear": 0.15, "mlp": 0.15, "hidden": 0.15, "mixed": 0.20,
}

# Mixture prior of the larger foundation model
# (probe_intersample_L_seed0.ckpt: intersample arch, d_model=384, 6+6 layers,
# p in [3, 15], 150k steps), used for tab:acic-fp, tab:foundation-per-family,
# the uncertainty / selective-prediction results, the RCT negative controls,
# Causal Chambers and tab:shapley-comparison. Eleven SCM families.
_FOUNDATION_L_PRIOR = {
    "linear": 0.30, "nonlinear": 0.13, "mlp": 0.13,
    "hidden": 0.13, "mlp_hidden": 0.05, "mixed": 0.16,
    # treatment = IHDP/Lalonde-shape RCT family (binary T column). Was the
    # suspect in the original v4 hangs; re-enabled now that the
    # grad-finiteness guard around clip_grad_norm_ skips non-finite-grad
    # batches.
    "treatment": 0.05,
    "polynomial": 0.05,  # univariate polynomial structural eqs, k in {1,2,3}
    "spline": 0.05,      # additive spline structural eqs, smoother than polynomial
    # gp_cde is the multivariate-nonlinear (joint function of parents) family
    # via exact Kronecker-grid GP sampling. Cholesky on random kernel
    # hyperparameters is the main numerical risk; the ±8σ clamp in
    # scm_utils plus the grad-finiteness guard are the defenses.
    "gp_cde": 0.05,
    # discrete-mix is scm_mixed (binary) extended to k in {2,3,4,5} levels.
    # Same continuous-latent structural eqs + quantile-binned presentation;
    # no new structural form, so the lowest-risk family.
    "discrete_mix": 0.05,
}

_PRIORS = {"fado_p5": _FADO_P5_PRIOR, "foundation_l": _FOUNDATION_L_PRIOR}

# Back-compat alias: helpers that inspect "the foundation's training prior".
_MIXED_PRIOR_DEFAULT = _FOUNDATION_L_PRIOR


def _inject_sensor_decoy(
    X: np.ndarray, tau: np.ndarray, n_rows: int, noise: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a NEAR-DETERMINISTIC SENSOR DECOY column to a sampled (X, tau).

    Motivation: the Causal Chambers eval exposed a prior-coverage gap.
    Physical sensor readouts (e.g. ``current`` ~= f(red, green, blue) + tiny
    noise) are near-deterministic linear functions of SEVERAL true causes of
    Y, yet the training prior only ever showed the model NOISY single-parent
    proxy children. Those sensor columns therefore drew false-positive
    tau_hat ~= 0.4. A controlled sweep (proxy_determinism_sweep.py) confirmed
    DETERMINISM (noise scale -> 0, several parents) is the trigger, not plain
    Y-correlation. This injection teaches the model to zero such columns: D is
    a pure leaf (no outgoing edges), so do(D) moves nothing and tau_D == 0
    exactly by construction.

    Given a sampled (X, tau) with C existing columns, draws
    ``D = w . X[:, parents] + ns * sd(signal) * eps`` for a random 1-3 column
    subset (biased toward larger |tau|), random signed weights, eps from the
    same noise family, and ``ns`` log-uniform over [3e-3, 0.5]. D is
    standardized like the other columns, inserted at a RANDOM position, and
    the tau target is extended with an exact 0 at D's position.
    """
    C = X.shape[1]
    n_parents = int(rng.integers(1, min(3, C) + 1))
    # Bias parent selection toward larger-|tau| columns, but only when enough
    # columns have nonzero |tau| to draw n_parents WITHOUT replacement (else
    # np.choice raises "Fewer non-zero entries in p than size"). Fall back to
    # uniform otherwise (all-noise/all-zero-tau episodes, or few active causes).
    abs_tau = np.abs(tau)
    if np.count_nonzero(abs_tau) >= n_parents:
        pick_p = abs_tau / abs_tau.sum()
    else:
        pick_p = np.full(C, 1.0 / C)
    parents = rng.choice(C, size=n_parents, replace=False, p=pick_p)
    w = rng.uniform(0.5, 2.0, size=n_parents) * rng.choice([-1.0, 1.0], size=n_parents)
    signal = X[:, parents] @ w
    sd_sig = max(float(signal.std()), 1e-12)
    ns = float(np.exp(rng.uniform(np.log(3e-3), np.log(0.5))))
    eps = sample_noise(n_rows, rng, noise=noise)
    D = signal + ns * sd_sig * eps
    D = (D - D.mean()) / max(float(D.std()), 1e-12)

    d_pos = int(rng.integers(0, C + 1))
    X_out = np.insert(X, d_pos, D.astype(X.dtype), axis=1)
    tau_out = np.insert(tau, d_pos, 0.0)
    return X_out, tau_out, d_pos


def _build_one_scm(
    seed: int, p_this: int, p_pad: int, n_rows: int,
    scm_type_this: str, noise: str, motif_name: Optional[str],
    nonlinear_mc: int, inject_sensor_decoy: bool = False,
    sensor_decoy_loss_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one SCM episode. Top-level / pickle-safe so Pool workers can call it.

    If ``inject_sensor_decoy`` is True, the SCM is drawn at ``p_this - 1``
    features and a near-deterministic sensor decoy column (tau == 0) is
    appended, so the episode still presents ``p_this`` feature columns. The
    caller (``_build_batch``) gates this on ``--sensor-decoy-prob`` and only
    sets the flag when ``p_this - 1`` stays at or above ``p_min``.
    """
    rng = np.random.default_rng(seed)
    if inject_sensor_decoy:
        p_this = p_this - 1  # one column is reserved for the appended decoy
    scm = None
    if motif_name is not None:
        from causal_probe.motifs import ALL_MOTIFS, motif_scm
        try:
            spec = ALL_MOTIFS[motif_name](p=p_this)
            scm = motif_scm(spec, rng=np.random.default_rng(rng.integers(0, 2**31)))
        except (ValueError, AssertionError):
            scm = None  # motif incompatible with p_this; fall through to family
    if scm is not None:
        pass
    elif scm_type_this == "nonlinear":
        from causal_probe.scm_nonlinear import NonlinearSCM
        scm = NonlinearSCM(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), n_mc=nonlinear_mc, noise=noise)
    elif scm_type_this == "hidden":
        from causal_probe.scm_hidden import LinearNonGaussianSCMHidden
        scm = LinearNonGaussianSCMHidden(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), noise=noise)
    elif scm_type_this == "mlp":
        from causal_probe.scm_mlp import MLPSCM
        scm = MLPSCM(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), n_mc=nonlinear_mc, noise=noise)
    elif scm_type_this == "mlp_hidden":
        from causal_probe.scm_mlp_hidden import MLPSCMHidden
        scm = MLPSCMHidden(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), n_mc=nonlinear_mc, noise=noise)
    elif scm_type_this == "mixed":
        from causal_probe.scm_mixed import LinearMixedSCM
        scm = LinearMixedSCM(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), noise=noise)
    elif scm_type_this == "treatment":
        from causal_probe.scm_treatment import DesignatedTreatmentSCM
        scm = DesignatedTreatmentSCM(
            p=max(p_this, 2),  # treatment SCM needs >= 2 features
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            n_mc=nonlinear_mc, noise=noise,
        )
    elif scm_type_this == "polynomial":
        from causal_probe.scm_polynomial import PolynomialSCM
        scm = PolynomialSCM(
            p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)),
            n_mc=nonlinear_mc, noise=noise,
        )
    elif scm_type_this == "spline":
        from causal_probe.scm_spline import SplineAdditiveSCM
        scm = SplineAdditiveSCM(
            p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)),
            n_mc=nonlinear_mc, noise=noise,
        )
    elif scm_type_this == "gp_cde":
        from causal_probe.scm_gp_cde import GPGridSCM
        scm = GPGridSCM(
            p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)),
            n_mc=nonlinear_mc, noise=noise,
        )
    elif scm_type_this == "discrete_mix":
        from causal_probe.scm_discrete import DiscreteMixedSCM
        scm = DiscreteMixedSCM(
            p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)),
            noise=noise,
        )
    elif scm_type_this == "binary_outcome":
        from causal_probe.scm_binary_outcome import BinaryOutcomeSCM
        scm = BinaryOutcomeSCM(
            p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)),
            n_mc=nonlinear_mc, noise=noise,
        )
    else:
        scm = LinearNonGaussianSCM(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), noise=noise)
    sample = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
    perm = rng.permutation(p_this)
    X = sample.X[:, perm]
    tau = sample.tau[perm]
    n_cols = p_this
    # Per-position loss weights: 1.0 everywhere except (optionally) the
    # injected sensor-decoy position, whose tau==0 label can be up-weighted so
    # convergence on the mixture objective cannot wash the lesson out (the
    # q=0.3, weight=1.0 run showed a transient fix at 30k steps that regressed
    # by 62k).
    lw = np.ones(n_cols, dtype=np.float64)
    if inject_sensor_decoy:
        # Append the near-deterministic sensor decoy (tau == 0) BEFORE padding,
        # so it survives the pad-noise concat + final permutation exactly like a
        # true feature. p_this was already decremented by 1 above, so the
        # episode ends up with the caller's intended p_this feature columns.
        X, tau, d_pos = _inject_sensor_decoy(X, tau, n_rows, noise, rng)
        n_cols = p_this + 1
        lw = np.insert(lw, d_pos, sensor_decoy_loss_weight)
    if p_pad > n_cols:
        pad_noise = rng.standard_normal((n_rows, p_pad - n_cols)).astype(np.float32)
        X = np.concatenate([X, pad_noise], axis=1)
        tau = np.concatenate([tau, np.zeros(p_pad - n_cols, dtype=np.float64)])
        lw = np.concatenate([lw, np.ones(p_pad - n_cols, dtype=np.float64)])
        final_perm = rng.permutation(p_pad)
        X = X[:, final_perm]
        tau = tau[final_perm]
        lw = lw[final_perm]
    return X, sample.y, tau, lw


def _build_batch(
    batch_size: int, p: int, n_rows: int, rng: np.random.Generator,
    noise: str = "laplace", motif_mix: float = 0.0, scm_type: str = "linear",
    nonlinear_mc: int = 4096, p_min: Optional[int] = None, p_max: Optional[int] = None,
    pad_to: Optional[int] = None, prior_weights: Optional[dict[str, float]] = None,
    pool=None, sensor_decoy_prob: float = 0.0,
    sensor_decoy_loss_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X (B, n, p_pad), y (B, n), tau (B, p_pad) arrays with column permutations.

    If p_min/p_max are set, each SCM samples a random p in [p_min, p_max]
    and is padded to pad_to (or p_max) with pure-noise features so the
    batch tensors are rectangular. Padded tau entries are 0.

    If ``sensor_decoy_prob > 0``, each sampled dataset independently has that
    probability of getting a near-deterministic sensor decoy column appended
    (see ``_inject_sensor_decoy``). The SCM is then drawn at one fewer feature
    so the presented column count is unchanged; injection is skipped when that
    would drop below ``p_min``. All decoy RNG draws are guarded behind the
    probability check, so ``sensor_decoy_prob == 0`` reproduces the pre-flag
    RNG stream bit-for-bit.

    Per-SCM episodes are independent, so when ``pool`` is a multiprocessing
    Pool instance the batch is generated in parallel via ``pool.starmap``.
    The parent rng deterministically pre-decides per-SCM seeds and configs
    so output is bit-identical between sequential and parallel paths.
    """
    from causal_probe.motifs import ALL_MOTIFS
    motif_keys = list(ALL_MOTIFS.keys())
    heterog = p_min is not None and p_max is not None
    P_pad = pad_to if pad_to is not None else (p_max if heterog else p)
    use_prior_mixture = scm_type in ("mixture", "mixture_binary")
    if use_prior_mixture:
        if prior_weights is None:
            prior_weights = _FINETUNE_BINARY_PRIOR if scm_type == "mixture_binary" else _MIXED_PRIOR_DEFAULT
        mixture_keys = list(prior_weights.keys())
        mixture_weights = np.asarray([prior_weights[k] for k in mixture_keys], dtype=np.float64)
        mixture_weights = mixture_weights / mixture_weights.sum()

    configs = []
    for _ in range(batch_size):
        p_this = int(rng.integers(p_min, p_max + 1)) if heterog else p
        if use_prior_mixture:
            scm_type_this = mixture_keys[int(rng.choice(len(mixture_keys), p=mixture_weights))]
        else:
            scm_type_this = scm_type
        use_motif = motif_mix > 0 and rng.random() < motif_mix
        motif_name = str(rng.choice(motif_keys)) if use_motif else None
        seed = int(rng.integers(0, 2**31))
        # Decoy decision LAST and fully guarded, so q == 0 leaves the RNG
        # stream (and thus reproducibility of existing seeds) untouched.
        inject_decoy = False
        if sensor_decoy_prob > 0.0 and rng.random() < sensor_decoy_prob:
            # Respect p_min: reserving a column for the decoy must not push the
            # true-feature count below the floor. p_this - 1 must stay >= the
            # smaller of p_min (heterog) and 1 (need >=1 real column to seed D).
            floor = p_min if p_min is not None else 1
            if p_this - 1 >= floor:
                inject_decoy = True
        configs.append((seed, p_this, P_pad, n_rows, scm_type_this, noise,
                        motif_name, nonlinear_mc, inject_decoy,
                        sensor_decoy_loss_weight))

    if pool is None:
        results = [_build_one_scm(*c) for c in configs]
    else:
        results = pool.starmap(_build_one_scm, configs)

    Xs, ys, taus, lws = zip(*results)
    return (
        np.stack(Xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.stack(taus).astype(np.float32),
        np.stack(lws).astype(np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--p-min", type=int, default=None,
                    help="If set along with --p-max, sample p per episode and pad to p-max")
    ap.add_argument("--p-max", type=int, default=None)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-rows-min", type=int, default=None,
                    help="If set with --n-rows-max, sample n_rows per batch "
                         "log-uniformly from [n-rows-min, n-rows-max].")
    ap.add_argument("--n-rows-max", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-row-layers", type=int, default=2)
    ap.add_argument("--n-col-layers", type=int, default=2)
    ap.add_argument("--motif-mix", type=float, default=0.0,
                    help="Probability of sampling a motif SCM instead of random.")
    ap.add_argument("--noise", choices=["laplace", "mixture", "gaussian", "heavy"], default="laplace")
    ap.add_argument("--scm-type",
                    choices=["linear", "nonlinear", "hidden", "mlp", "mlp_hidden", "mixed",
                             "treatment", "polynomial", "spline", "gp_cde", "discrete_mix",
                             "mixture", "mixture_binary", "binary_outcome"],
                    default="linear",
                    help="Single-family training, or 'mixture' to sample scm_type per "
                         "episode from the --prior weights. _build_one_scm already "
                         "dispatches all 11 families; this list just exposes them on the CLI.")
    ap.add_argument("--prior", choices=sorted(_PRIORS), default="fado_p5",
                    help="Family weights for --scm-type mixture: 'fado_p5' (5 families, "
                         "the p=5 FADO model) or 'foundation_l' (11 families, the "
                         "larger intersample foundation model).")
    ap.add_argument("--nonlinear-mc", type=int, default=4096,
                    help="MC size for nonlinear SCM label computation")
    ap.add_argument("--sensor-decoy-loss-weight", type=float, default=1.0,
                    help="Loss weight applied at the injected sensor-decoy "
                         "position (default 1.0 = plain MSE). The q=0.3, "
                         "weight=1.0 run learned to zero the decoys mid-training "
                         "but regressed by 62k steps; up-weighting keeps the "
                         "decoy-zero constraint binding at convergence.")
    ap.add_argument("--sensor-decoy-prob", type=float, default=0.0,
                    help="Probability per sampled dataset of appending a "
                         "NEAR-DETERMINISTIC SENSOR DECOY column (tau == 0). "
                         "Motivation: the Causal Chambers eval exposed a "
                         "prior-coverage gap -- physical sensor readouts (e.g. "
                         "current ~= f(red,green,blue) + tiny noise) are "
                         "near-deterministic linear functions of SEVERAL true "
                         "causes of Y, but the prior only ever showed NOISY "
                         "single-parent proxy children, so those sensors drew "
                         "false-positive tau_hat ~= 0.4. proxy_determinism_sweep.py "
                         "confirmed DETERMINISM (not plain Y-correlation) is the "
                         "trigger. With this probability the SCM is drawn at one "
                         "fewer feature and a decoy D = w.X[:,parents] + "
                         "ns*sd(signal)*eps (1-3 parents, ns log-uniform in "
                         "[3e-3, 0.5]) is inserted at a random position with an "
                         "exact-0 tau label, teaching the model to zero such "
                         "leaves. Default 0.0 = exactly the pre-flag behavior "
                         "(RNG stream unchanged). Injection is skipped when "
                         "reserving the column would drop below --p-min.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-out", type=str, default="causal_probe/results/probe.ckpt")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--arch", type=str, default="rowcol",
                    help="Architecture variant; one of "
                         f"{list_archs()}. rowcol = legacy row-then-col "
                         "transformer; attnpool = same trunk + role-codebook "
                         "attention-pooled head.")
    ap.add_argument("--n-roles", type=int, default=4,
                    help="attnpool only: size of the learned role codebook. "
                         "Ignored by other archs.")
    ap.add_argument("--n-pool-heads", type=int, default=0,
                    help="attnpool only: heads for the codebook cross-attn. "
                         "0 = use --n-heads. Ignored by other archs.")
    ap.add_argument("--no-row-attn", action="store_true",
                    help="Architecture ablation (handoff §9.5): drop row-level attention. rowcol only.")
    ap.add_argument("--no-col-attn", action="store_true",
                    help="Architecture ablation: drop column-level attention. rowcol only.")
    ap.add_argument("--no-type-emb", action="store_true",
                    help="Architecture ablation: drop the feature/target type embedding. rowcol only.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Periodic checkpoint cadence (0 = save only at the end).")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Resume from this checkpoint path; restores model, "
                         "optimizer, scheduler, and step counter.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="If >0, run a lightweight held-out eval every this "
                         "many steps. Appends one JSON line per eval to "
                         "${ckpt_out}.metrics.jsonl and saves "
                         "${ckpt_out}.best.ckpt whenever pearson improves. "
                         "Default 0 = no intermediate eval.")
    ap.add_argument("--eval-n-scms", type=int, default=100,
                    help="Number of held-out SCMs per intermediate eval. "
                         "Default 100 (vs 500 for the post-training eval) "
                         "keeps each intermediate eval to ~30s on L40S.")
    ap.add_argument("--eval-motifs", type=str, default="",
                    help="Comma-separated motif names to include in the "
                         "intermediate eval (e.g. "
                         "G_chained_mediators_k3,I_latent_clique_2x2,J_latent_clique_3x3). "
                         "Empty = skip motif eval (default). Each motif "
                         "adds ~5s per intermediate eval with --eval-motif-repeats.")
    ap.add_argument("--eval-motif-repeats", type=int, default=10,
                    help="Per-motif repeat count for intermediate eval. "
                         "10 is enough for a noisy mid-training estimate; "
                         "the final post-training eval uses 50.")
    ap.add_argument("--best-by", type=str, default="pearson_model",
                    help="Metric key from the intermediate eval that selects "
                         "the .best.ckpt. Default 'pearson_model' (random "
                         "held-out). Use e.g. 'motif_J_latent_clique_3x3' "
                         "to optimise for OOD generalisation instead.")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="If >0, generate per-batch SCM episodes in parallel "
                         "via a multiprocessing.Pool of this many workers. "
                         "Default 0 = sequential. Useful when SCM label compute "
                         "(e.g. mixture/nonlinear/MLP MC) is the wall-clock "
                         "bottleneck and the host has many CPU cores.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    arch_kwargs = dict(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_row_layers=args.n_row_layers,
        n_col_layers=args.n_col_layers,
        no_row_attn=args.no_row_attn,
        no_col_attn=args.no_col_attn,
        no_type_emb=args.no_type_emb,
    )
    if args.arch == "attnpool":
        arch_kwargs["n_roles"] = args.n_roles
        if args.n_pool_heads > 0:
            arch_kwargs["n_pool_heads"] = args.n_pool_heads
    model = build_arch(args.arch, **arch_kwargs).to(args.device)
    print(f"arch={args.arch}")
    print(f"model params: {count_params(model)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Linear warmup + cosine decay.
    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume_from is not None and Path(args.resume_from).exists():
        ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "opt_state" in ckpt:
            opt.load_state_dict(ckpt["opt_state"])
        if "sched_state" in ckpt:
            sched.load_state_dict(ckpt["sched_state"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume_from} at step {start_step}", flush=True)

    def _save(path: str, step: int) -> None:
        ck = {
            "state_dict": model.state_dict(),
            "opt_state": opt.state_dict(),
            "sched_state": sched.state_dict(),
            "step": step,
            "arch": args.arch,
            "arch_kwargs": arch_kwargs,
            "args": vars(args),
        }
        torch.save(ck, path)

    sample_n_rows = (
        args.n_rows_min is not None and args.n_rows_max is not None
        and args.n_rows_min < args.n_rows_max
    )
    if sample_n_rows:
        log_min = float(np.log(args.n_rows_min))
        log_max = float(np.log(args.n_rows_max))

    pool = None
    if args.num_workers > 0:
        from multiprocessing import Pool
        pool = Pool(args.num_workers)
        print(f"batch generation: parallel via Pool({args.num_workers})", flush=True)

    # Intermediate-eval state (only used if --eval-every > 0).
    metrics_path = str(args.ckpt_out) + ".metrics.jsonl"
    best_ckpt_path = str(args.ckpt_out) + ".best.ckpt"
    best_eval_pearson = float("-inf")

    def _intermediate_eval(step_done: int) -> dict:
        """Lightweight held-out eval; returns metrics dict, also appends jsonl."""
        from causal_probe.eval import evaluate_random, evaluate_motifs
        from causal_probe.motifs import ALL_MOTIFS
        eval_rng = np.random.default_rng(int(args.seed) * 1_000_003 + step_done)
        model.eval()
        try:
            res = evaluate_random(
                model, n_scms=int(args.eval_n_scms), p=int(args.p),
                n_rows=int(args.n_rows), rng=eval_rng, device=args.device,
                noise=args.noise, scm_type="linear",
                nonlinear_mc=int(args.nonlinear_mc),
            )
            motif_corrs: dict[str, float] = {}
            wanted = [m.strip() for m in args.eval_motifs.split(",") if m.strip()]
            wanted = [m for m in wanted if m in ALL_MOTIFS]
            if wanted:
                # Subset eval: build a thin ALL_MOTIFS-like view by patching the
                # global temporarily. Simpler: call evaluate_motifs with the full
                # set but small n_repeats, then keep only requested keys.
                motif_res = evaluate_motifs(
                    model, p=int(args.p), n_rows=int(args.n_rows),
                    rng=eval_rng, device=args.device,
                    n_repeats=int(args.eval_motif_repeats),
                )
                for name in wanted:
                    rows = motif_res.get(name, {}).get("rows", [])
                    if not rows:
                        motif_corrs[f"motif_{name}"] = float("nan")
                        continue
                    tt = np.array([r["tau_true_mean"] for r in rows])
                    th = np.array([r["tau_hat_mean"] for r in rows])
                    if tt.std() < 1e-6 or th.std() < 1e-6:
                        motif_corrs[f"motif_{name}"] = float("nan")
                    else:
                        motif_corrs[f"motif_{name}"] = float(np.corrcoef(tt, th)[0, 1])
        finally:
            model.train()
        out = {
            "step": step_done,
            "wall": time.time() - t0,
            "pearson_model": res["pearson"]["model"],
            "pearson_marginal": res["pearson"]["marginal"],
            "pearson_multivariate": res["pearson"]["multivariate"],
            "auroc_nonzero_model": res["auroc_nonzero"]["model"],
            "mse_model": res["mse"]["model"],
            "n_scms": res["n_scms"],
        }
        out.update(motif_corrs if wanted else {})
        return out

    t0 = time.time()
    recent_losses = []
    for step in range(start_step, args.max_steps):
        if sample_n_rows:
            n_rows_step = int(np.round(np.exp(rng.uniform(log_min, log_max))))
            n_rows_step = max(args.n_rows_min, min(args.n_rows_max, n_rows_step))
        else:
            n_rows_step = args.n_rows
        X_np, y_np, tau_np, lw_np = _build_batch(
            args.batch_size, args.p, n_rows_step, rng,
            noise=args.noise, motif_mix=args.motif_mix,
            scm_type=args.scm_type, nonlinear_mc=args.nonlinear_mc,
            p_min=args.p_min, p_max=args.p_max,
            prior_weights=_PRIORS[args.prior] if args.scm_type == "mixture" else None,
            pool=pool, sensor_decoy_prob=args.sensor_decoy_prob,
            sensor_decoy_loss_weight=args.sensor_decoy_loss_weight,
        )
        X = torch.from_numpy(X_np).to(args.device)
        y = torch.from_numpy(y_np).to(args.device)
        tau = torch.from_numpy(tau_np).to(args.device)
        lw = torch.from_numpy(lw_np).to(args.device)

        pred = model(X, y)
        # Guard: clip tau labels and skip NaN/Inf batches entirely so one
        # rogue SCM can't blow up the head (seen with polynomial SCM at
        # long topological chains).
        tau = torch.clamp(tau, min=-5.0, max=5.0)
        # Weighted MSE: identical to mse_loss when all weights are 1.0
        # (the default). Only injected sensor-decoy positions ever get a
        # weight != 1.0, via --sensor-decoy-loss-weight.
        loss = ((pred - tau) ** 2 * lw).sum() / lw.sum()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            sched.step()
            print(f"step {step + 1}: SKIP (non-finite loss)", flush=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Pre-clip grad-finite guard. Without this, a single NaN grad
        # gets propagated to ALL params by clip_grad_norm_ (clip_coef
        # = max_norm / (NaN + eps) = NaN, then grads *= NaN in-place),
        # opt.step writes NaN into every weight, and forward → NaN on
        # every subsequent batch. The non-finite-loss guard above then
        # silently `continue`s on every step; the log block below is
        # skipped along with it, so the job appears to hang for hours
        # but is actually churning through NaN-SKIPs.
        grads_finite = True
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all().item():
                grads_finite = False
                break
        if not grads_finite:
            opt.zero_grad(set_to_none=True)
            sched.step()
            print(f"step {step + 1}: SKIP (non-finite grad)", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        recent_losses.append(float(loss.detach().cpu().item()))
        recent_losses = recent_losses[-args.log_every:]

        if (step + 1) % args.log_every == 0 or step == 0:
            with torch.no_grad():
                pred_np = pred.detach().cpu().numpy()
                tau_cpu = tau.detach().cpu().numpy()
                # Batch-wise Pearson between predicted and true tau.
                flat_p = pred_np.reshape(-1)
                flat_t = tau_cpu.reshape(-1)
                if flat_p.std() > 0 and flat_t.std() > 0:
                    rho = float(np.corrcoef(flat_p, flat_t)[0, 1])
                else:
                    rho = float("nan")
            wall = time.time() - t0
            print(
                f"step {step + 1:5d}/{args.max_steps}  "
                f"loss={np.mean(recent_losses):.4f}  "
                f"pearson_tau={rho:+.3f}  "
                f"lr={sched.get_last_lr()[0]:.1e}  "
                f"wall={wall:.1f}s",
                flush=True,
            )

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            _save(args.ckpt_out, step + 1)
            print(f"checkpoint at step {step + 1} -> {args.ckpt_out}", flush=True)

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            metrics = _intermediate_eval(step + 1)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(
                f"eval step {step + 1}: pearson_model={metrics['pearson_model']:+.4f} "
                f"auroc={metrics['auroc_nonzero_model']:.3f} (n={metrics['n_scms']})",
                flush=True,
            )
            if metrics["pearson_model"] > best_eval_pearson:
                best_eval_pearson = metrics["pearson_model"]
                _save(best_ckpt_path, step + 1)
                print(f"new best pearson_model={best_eval_pearson:+.4f} -> {best_ckpt_path}", flush=True)

    _save(args.ckpt_out, args.max_steps)
    print(f"saved {args.ckpt_out}")
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
