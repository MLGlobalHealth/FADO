"""Online training loop for the causal probe.

Each step samples a batch of random linear non-Gaussian SCMs, draws n rows
from each, applies a random column permutation (labels permuted
accordingly), and updates the model to predict the standardized tau
contrast vector.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from causal_probe.model import CausalProbe, ProbeConfig, count_params
from causal_probe.scm import LinearNonGaussianSCM


_MIXED_PRIOR_DEFAULT = {
    "linear": 0.35, "nonlinear": 0.15, "mlp": 0.15,
    "hidden": 0.15, "mixed": 0.20,
}


def _build_one_scm(
    seed: int, p_this: int, p_pad: int, n_rows: int,
    scm_type_this: str, noise: str, motif_name: Optional[str],
    nonlinear_mc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one SCM episode. Top-level / pickle-safe so Pool workers can call it."""
    rng = np.random.default_rng(seed)
    if motif_name is not None:
        from causal_probe.motifs import ALL_MOTIFS, motif_scm
        spec = ALL_MOTIFS[motif_name](p=p_this)
        scm = motif_scm(spec, rng=np.random.default_rng(rng.integers(0, 2**31)))
    elif scm_type_this == "nonlinear":
        from causal_probe.scm_nonlinear import NonlinearSCM
        scm = NonlinearSCM(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), n_mc=nonlinear_mc)
    elif scm_type_this == "hidden":
        from causal_probe.scm_hidden import LinearNonGaussianSCMHidden
        scm = LinearNonGaussianSCMHidden(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), noise=noise)
    elif scm_type_this == "mlp":
        from causal_probe.scm_mlp import MLPSCM
        scm = MLPSCM(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), n_mc=nonlinear_mc)
    elif scm_type_this == "mlp_hidden":
        from causal_probe.scm_mlp_hidden import MLPSCMHidden
        scm = MLPSCMHidden(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), n_mc=nonlinear_mc)
    elif scm_type_this == "mixed":
        from causal_probe.scm_mixed import LinearMixedSCM
        scm = LinearMixedSCM(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), noise=noise)
    else:
        scm = LinearNonGaussianSCM(p=p_this, rng=np.random.default_rng(rng.integers(0, 2**31)), noise=noise)
    sample = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
    perm = rng.permutation(p_this)
    X = sample.X[:, perm]
    tau = sample.tau[perm]
    if p_pad > p_this:
        pad_noise = rng.standard_normal((n_rows, p_pad - p_this)).astype(np.float32)
        X = np.concatenate([X, pad_noise], axis=1)
        tau = np.concatenate([tau, np.zeros(p_pad - p_this, dtype=np.float64)])
        final_perm = rng.permutation(p_pad)
        X = X[:, final_perm]
        tau = tau[final_perm]
    return X, sample.y, tau


def _build_batch(
    batch_size: int, p: int, n_rows: int, rng: np.random.Generator,
    noise: str = "laplace", motif_mix: float = 0.0, scm_type: str = "linear",
    nonlinear_mc: int = 4096, p_min: Optional[int] = None, p_max: Optional[int] = None,
    pad_to: Optional[int] = None, prior_weights: Optional[dict[str, float]] = None,
    pool=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (B, n, p_pad), y (B, n), tau (B, p_pad) arrays with column permutations.

    If p_min/p_max are set, each SCM samples a random p in [p_min, p_max]
    and is padded to pad_to (or p_max) with pure-noise features so the
    batch tensors are rectangular. Padded tau entries are 0.

    Per-SCM episodes are independent, so when ``pool`` is a multiprocessing
    Pool instance the batch is generated in parallel via ``pool.starmap``.
    The parent rng deterministically pre-decides per-SCM seeds and configs
    so output is bit-identical between sequential and parallel paths.
    """
    from causal_probe.motifs import ALL_MOTIFS
    motif_keys = list(ALL_MOTIFS.keys())
    heterog = p_min is not None and p_max is not None
    P_pad = pad_to if pad_to is not None else (p_max if heterog else p)
    use_prior_mixture = scm_type == "mixture"
    if use_prior_mixture:
        if prior_weights is None:
            prior_weights = _MIXED_PRIOR_DEFAULT
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
        configs.append((seed, p_this, P_pad, n_rows, scm_type_this, noise, motif_name, nonlinear_mc))

    if pool is None:
        results = [_build_one_scm(*c) for c in configs]
    else:
        results = pool.starmap(_build_one_scm, configs)

    Xs, ys, taus = zip(*results)
    return (
        np.stack(Xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.stack(taus).astype(np.float32),
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
    ap.add_argument("--scm-type", choices=["linear", "nonlinear", "hidden", "mlp", "mlp_hidden", "mixed", "mixture"], default="linear",
                    help="'mixture' samples scm_type per episode from a default prior mix")
    ap.add_argument("--nonlinear-mc", type=int, default=4096,
                    help="MC size for nonlinear SCM label computation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-out", type=str, default="causal_probe/results/probe.ckpt")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-row-attn", action="store_true",
                    help="Architecture ablation (handoff §9.5): drop row-level attention.")
    ap.add_argument("--no-col-attn", action="store_true",
                    help="Architecture ablation: drop column-level attention.")
    ap.add_argument("--no-type-emb", action="store_true",
                    help="Architecture ablation: drop the feature/target type embedding.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Periodic checkpoint cadence (0 = save only at the end).")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Resume from this checkpoint path; restores model, "
                         "optimizer, scheduler, and step counter.")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="If >0, generate per-batch SCM episodes in parallel "
                         "via a multiprocessing.Pool of this many workers. "
                         "Default 0 = sequential. Useful when SCM label compute "
                         "(e.g. mixture/nonlinear/MLP MC) is the wall-clock "
                         "bottleneck and the host has many CPU cores.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = ProbeConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_row_layers=args.n_row_layers,
        n_col_layers=args.n_col_layers,
        no_row_attn=args.no_row_attn,
        no_col_attn=args.no_col_attn,
        no_type_emb=args.no_type_emb,
    )
    model = CausalProbe(cfg).to(args.device)
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
            "cfg": cfg.__dict__,
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

    t0 = time.time()
    recent_losses = []
    for step in range(start_step, args.max_steps):
        if sample_n_rows:
            n_rows_step = int(np.round(np.exp(rng.uniform(log_min, log_max))))
            n_rows_step = max(args.n_rows_min, min(args.n_rows_max, n_rows_step))
        else:
            n_rows_step = args.n_rows
        X_np, y_np, tau_np = _build_batch(
            args.batch_size, args.p, n_rows_step, rng,
            noise=args.noise, motif_mix=args.motif_mix,
            scm_type=args.scm_type, nonlinear_mc=args.nonlinear_mc,
            p_min=args.p_min, p_max=args.p_max,
            pool=pool,
        )
        X = torch.from_numpy(X_np).to(args.device)
        y = torch.from_numpy(y_np).to(args.device)
        tau = torch.from_numpy(tau_np).to(args.device)

        pred = model(X, y)
        # Guard: clip tau labels and skip NaN/Inf batches entirely so one
        # rogue SCM can't blow up the head (seen with polynomial SCM at
        # long topological chains).
        tau = torch.clamp(tau, min=-5.0, max=5.0)
        loss = torch.nn.functional.mse_loss(pred, tau)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
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

    _save(args.ckpt_out, args.max_steps)
    print(f"saved {args.ckpt_out}")
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
