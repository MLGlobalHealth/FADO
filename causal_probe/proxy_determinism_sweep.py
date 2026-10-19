"""Controlled sweep: WHY does FADO false-positive on causal-chamber sensor
decoys but not on the paper's synthetic proxy decoys?

Hypothesis
----------
Chamber decoys like ``current`` (see causal_chambers_eval.py) are
NEAR-DETERMINISTIC functions of SEVERAL true causes of Y (the LED drive
settings), whereas the training prior's proxy decoys (semi_synth_real.py)
are NOISY children of a SINGLE cause (P = X_j + 0.3 * eps).  Two candidate
triggers for the false positive, which this grid is designed to separate:

  (a) determinism: the FP switches on as the decoy's own noise -> 0
      and/or as its parent count grows (D leaves the prior's proxy family
      and starts looking like an "effective cause" — a near-exact linear
      summary of the causal drivers);
  (b) plain correlation: the FP simply tracks |corr(D, Y)|, and chamber
      decoys just happen to be more Y-correlated than prior proxies.

Design
------
Base SCM: ``LinearNonGaussianSCM`` (identifiable linear non-Gaussian
regime, Laplace noise) with p true features + Y, Y forced LAST in the
topological order so it has ancestors.  Instances are rejection-sampled
until Y has >= 3 true causes (|tau| >= 0.1) and >= 2 non-causes
(|tau| <= 0.01).

ONE decoy column D = g(parents) + noise_scale * sd(g(parents)) * eps,
g linear with random signed weights in [0.5, 2], eps unit Laplace, is
inserted at a random column position.  D never enters Y's structural
equation, so tau_true_D = 0 exactly.  Grid axes:

  * noise_scale in {1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003, 0}
    (fraction of the parent-driven sd; 0 = perfectly deterministic)
  * n_parents in {1, 2, 3}
  * parent overlap:
      - "causes_y": parents are the strongest true causes of Y
        (top-|tau|), so corr(D, Y) is high;
      - "mixed": non-cause-leaning mix — floor(n/2) strongest causes +
        ceil(n/2) weakest non-causes (n=1 -> a pure non-cause parent),
        so corr(D, Y) is low AT THE SAME determinism level.

How the grid separates determinism from correlation
---------------------------------------------------
Within a (parents, overlap) column, shrinking noise_scale from 0.25 to 0
moves corr(D, Y) by only ~3% (corr scales as 1/sqrt(1 + ns^2)) while
determinism goes from "noisy child" to "exact function".  A FP that jumps
in that range is a determinism effect, not a correlation effect.
Conversely the mandated high-corr-but-noisy cell — overlap=causes_y,
n_parents=1 (single strongest cause), noise_scale=0.5 — has corr(D, Y)
~ 0.89 * corr(parent, Y) (high) but is unambiguously noisy; if the FP is
correlation-driven it must already fire there.  The "mixed" block holds
determinism fixed while cutting corr(D, Y), giving the reverse contrast
(deterministic but Y-decorrelated, maximally so at n_parents=1).

Per cell (48 cells x --n-scms seeded instances, base SCM and base data
SHARED across cells within a seed so only the decoy varies) we record:
FADO mean |tau_hat_D| (the FP), mean |tau_hat| on true causes
(separation), the marginal (2*corr) and multivariate-ridge baselines at
D, and mean/min/max |corr(D, Y)|.

Cost: 48 cells x 40 instances ~ 2k forward passes at (n=512, p+1 cols)
— minutes on a 4090.

Usage
-----
  uv run python -m causal_probe.proxy_determinism_sweep --dry-run
  uv run python -m causal_probe.proxy_determinism_sweep \
      --ckpt ckpts/probe_intersample_L_seed0.ckpt --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from causal_probe.baselines import BASELINES
from causal_probe.bootstrap_uncertainty import _bootstrap_preds
from causal_probe.scm import LinearNonGaussianSCM
from causal_probe.scm_utils import sample_laplace_unit

NOISE_SCALES = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003, 0.0)
N_PARENTS = (1, 2, 3)
OVERLAPS = ("causes_y", "mixed")

CAUSE_TAU_MIN = 0.1     # |tau| >= this -> "true cause" of Y
NONCAUSE_TAU_MAX = 0.01  # |tau| <= this -> "non-cause"
MAX_REJECTION_TRIES = 500


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------


def build_base_instance(seed: int, p: int, n_rows: int) -> dict:
    """Rejection-sample one base SCM (Y last in topo order, >=3 causes,
    >=2 non-causes) and draw its observational data + the per-instance
    decoy ingredients that are SHARED across grid cells (eps draw, decoy
    column position, per-config weights)."""
    rng = np.random.default_rng(seed)
    for _ in range(MAX_REJECTION_TRIES):
        scm_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        scm = LinearNonGaussianSCM(p=p, rng=scm_rng, noise="laplace", y_position=p)
        abs_tau = np.abs(scm.tau)
        causes = [i for i in range(p) if abs_tau[i] >= CAUSE_TAU_MIN]
        noncauses = [i for i in range(p) if abs_tau[i] <= NONCAUSE_TAU_MAX]
        if len(causes) >= 3 and len(noncauses) >= 2:
            break
    else:
        raise RuntimeError(
            f"seed {seed}: no SCM with >=3 causes and >=2 non-causes "
            f"in {MAX_REJECTION_TRIES} tries"
        )
    samp = scm.sample(n=n_rows, rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))))
    causes_sorted = sorted(causes, key=lambda i: -abs_tau[i])       # strongest first
    noncauses_sorted = sorted(noncauses, key=lambda i: abs_tau[i])  # weakest first

    # Shared across all cells of this instance: decoy noise draw (so the
    # noise_scale axis is a pure scale sweep), decoy column position, and
    # per-(overlap, n_parents) linear weights.
    eps_D = sample_laplace_unit(n_rows, rng)
    d_pos = int(rng.integers(0, p + 1))
    weights = {}
    for overlap in OVERLAPS:
        for n_par in N_PARENTS:
            w = rng.uniform(0.5, 2.0, size=n_par) * rng.choice([-1.0, 1.0], size=n_par)
            weights[(overlap, n_par)] = w
    return {
        "scm": scm,
        "X": samp.X,            # (n, p) population-standardized
        "y": samp.y,            # (n,)
        "tau_true": scm.tau,    # (p,)
        "causes": causes_sorted,
        "noncauses": noncauses_sorted,
        "eps_D": eps_D,
        "d_pos": d_pos,
        "weights": weights,
    }


def select_parents(overlap: str, n_parents: int, causes: list[int], noncauses: list[int]) -> list[int]:
    if overlap == "causes_y":
        return causes[:n_parents]
    if overlap == "mixed":
        n_c = n_parents // 2            # n=1 -> 0 causes, 1 non-cause
        n_nc = n_parents - n_c
        return noncauses[:n_nc] + causes[:n_c]
    raise ValueError(f"unknown overlap {overlap!r}")


def make_decoy_variant(inst: dict, overlap: str, n_parents: int, noise_scale: float) -> dict:
    """Return the (X_full, y, indices) for one grid cell on one instance."""
    X, y = inst["X"], inst["y"]
    parents = select_parents(overlap, n_parents, inst["causes"], inst["noncauses"])
    w = inst["weights"][(overlap, n_parents)]
    signal = X[:, parents] @ w
    sd_sig = float(signal.std())
    D = signal + noise_scale * sd_sig * inst["eps_D"]
    D = (D - D.mean()) / max(float(D.std()), 1e-12)

    d_pos = inst["d_pos"]
    X_full = np.insert(X, d_pos, D, axis=1)
    # Column index bookkeeping after the insert.
    shift = lambda j: j + 1 if j >= d_pos else j  # noqa: E731
    cause_cols = [shift(j) for j in inst["causes"]]
    # Sample-standardize at the model boundary (matches semi_synth_real.py).
    X_std = (X_full - X_full.mean(axis=0)) / X_full.std(axis=0).clip(min=1e-9)
    y_std = (y - y.mean()) / max(float(y.std()), 1e-12)
    corr_dy = float(np.corrcoef(D, y_std)[0, 1])
    return {
        "X_std": X_std,
        "y_std": y_std,
        "d_col": d_pos,
        "cause_cols": cause_cols,
        "parents": parents,
        "corr_dy": corr_dy,
    }


def iter_cells():
    for overlap in OVERLAPS:
        for n_par in N_PARENTS:
            for ns in NOISE_SCALES:
                yield overlap, n_par, ns


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(args) -> dict:
    import torch
    from causal_probe.eval import _load_model

    model = _load_model(args.ckpt, device=args.device)
    root = np.random.default_rng(args.seed)
    inst_seeds = [int(root.integers(0, 2**31)) for _ in range(args.n_scms)]

    cells = list(iter_cells())
    acc = {c: {"fp": [], "sep": [], "marg": [], "multi": [], "corr": [],
               "sigma_D": [], "sigma_true": []} for c in cells}
    boot_rng = np.random.default_rng(args.seed + 12345)

    for s, seed in enumerate(inst_seeds):
        inst = build_base_instance(seed, p=args.p, n_rows=args.n_rows)
        variants = [(c, make_decoy_variant(inst, *c)) for c in cells]
        # Batch the model forwards in chunks (same shapes across variants).
        X_batch = np.stack([v["X_std"] for _, v in variants]).astype(np.float32)
        y_batch = np.stack([v["y_std"] for _, v in variants]).astype(np.float32)
        preds = []
        with torch.no_grad():
            for lo in range(0, len(variants), args.batch_size):
                X_t = torch.from_numpy(X_batch[lo:lo + args.batch_size]).to(args.device)
                y_t = torch.from_numpy(y_batch[lo:lo + args.batch_size]).to(args.device)
                preds.append(model(X_t, y_t).cpu().numpy())
        preds = np.concatenate(preds, axis=0)  # (n_cells, p+1)

        for (cell, v), tau_hat in zip(variants, preds):
            a = acc[cell]
            a["fp"].append(float(abs(tau_hat[v["d_col"]])))
            a["sep"].append(float(np.mean(np.abs(tau_hat[v["cause_cols"]]))))
            a["marg"].append(float(abs(BASELINES["marginal"](v["X_std"], v["y_std"])[v["d_col"]])))
            a["multi"].append(float(abs(BASELINES["multivariate"](v["X_std"], v["y_std"])[v["d_col"]])))
            a["corr"].append(float(abs(v["corr_dy"])))
            if args.n_boot > 0:
                # Per-feature bootstrap sigma: B row-resamples of the 512-row
                # context through the SAME model; sd over the B predictions
                # (reuses bootstrap_uncertainty._bootstrap_preds).
                boot = _bootstrap_preds(
                    model, v["X_std"].astype(np.float32), v["y_std"].astype(np.float32),
                    args.n_boot, boot_rng, args.device, chunk=args.boot_chunk,
                )  # (n_boot, p+1)
                b_std = boot.std(axis=0, ddof=1)
                a["sigma_D"].append(float(b_std[v["d_col"]]))
                a["sigma_true"].append(float(np.mean(b_std[v["cause_cols"]])))
        print(f"[{s + 1}/{args.n_scms}] instance seed={seed} done", flush=True)

    results = []
    for cell in cells:
        overlap, n_par, ns = cell
        a = acc[cell]
        cell_out = {
            "overlap": overlap,
            "n_parents": n_par,
            "noise_scale": ns,
            "fp_model_mean": float(np.mean(a["fp"])),
            "fp_model_sd": float(np.std(a["fp"])),
            "fp_model_per_scm": a["fp"],
            "sep_true_cause_mean": float(np.mean(a["sep"])),
            "marginal_D_mean_abs": float(np.mean(a["marg"])),
            "multivariate_D_mean_abs": float(np.mean(a["multi"])),
            "corr_DY_mean_abs": float(np.mean(a["corr"])),
            "corr_DY_min_abs": float(np.min(a["corr"])),
            "corr_DY_max_abs": float(np.max(a["corr"])),
        }
        if a["sigma_D"]:
            # Bootstrap sigma at the decoy (test: does uncertainty rise as the
            # decoy approaches non-identifiability?) and, as a control, averaged
            # over the identified TRUE-CAUSE positions (should stay flat).
            cell_out["sigma_D_mean"] = float(np.mean(a["sigma_D"]))
            cell_out["sigma_D_sd"] = float(np.std(a["sigma_D"]))
            cell_out["sigma_D_per_scm"] = a["sigma_D"]
            cell_out["sigma_true_cause_mean"] = float(np.mean(a["sigma_true"]))
            cell_out["sigma_true_cause_sd"] = float(np.std(a["sigma_true"]))
        results.append(cell_out)
    return {
        "config": {
            "ckpt": args.ckpt, "seed": args.seed, "n_scms": args.n_scms,
            "n_rows": args.n_rows, "p": args.p, "n_boot": args.n_boot,
            "noise_scales": list(NOISE_SCALES), "n_parents": list(N_PARENTS),
            "overlaps": list(OVERLAPS),
            "cause_tau_min": CAUSE_TAU_MIN, "noncause_tau_max": NONCAUSE_TAU_MAX,
        },
        "cells": results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_tables(out: dict) -> None:
    cells = {(c["overlap"], c["n_parents"], c["noise_scale"]): c for c in out["cells"]}
    has_sigma = any("sigma_D_mean" in c for c in out["cells"])
    metrics = [
        ("fp_model_mean", "FADO FP mean |tau_hat_D|"),
        ("corr_DY_mean_abs", "mean |corr(D, Y)|"),
        ("marginal_D_mean_abs", "marginal baseline |2*corr| at D"),
        ("multivariate_D_mean_abs", "multivariate ridge |tau_hat| at D"),
        ("sep_true_cause_mean", "FADO mean |tau_hat| on true causes"),
    ]
    if has_sigma:
        metrics += [
            ("sigma_D_mean", "FADO bootstrap sigma at decoy D"),
            ("sigma_true_cause_mean", "FADO bootstrap sigma on true causes"),
        ]
    for metric, label in metrics:
        for overlap in OVERLAPS:
            print(f"\n=== {label} (overlap={overlap}) ===")
            header = "noise_scale " + "".join(f"  np={n:<6d}" for n in N_PARENTS)
            print(header)
            for ns in NOISE_SCALES:
                row = f"{ns:<12.3f}"
                for n_par in N_PARENTS:
                    row += f"  {cells[(overlap, n_par, ns)][metric]:<9.4f}"
                print(row)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(args) -> None:
    print("DRY RUN: one instance per cell, no checkpoint loaded.")
    corr_all = []
    for i, cell in enumerate(iter_cells()):
        overlap, n_par, ns = cell
        inst = build_base_instance(args.seed + i, p=args.p, n_rows=args.n_rows)
        v = make_decoy_variant(inst, overlap, n_par, ns)
        corr_all.append(abs(v["corr_dy"]))
        print(
            f"cell overlap={overlap:<8s} np={n_par} ns={ns:<6.3f} "
            f"X_std={v['X_std'].shape} y_std={v['y_std'].shape} "
            f"d_col={v['d_col']} parents={v['parents']} "
            f"|corr(D,Y)|={abs(v['corr_dy']):.3f}"
        )
    print(
        f"\n{len(corr_all)} cells; |corr(D,Y)| range "
        f"[{min(corr_all):.3f}, {max(corr_all):.3f}]"
    )
    print("dry run OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None, help="FADO checkpoint (required unless --dry-run)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-scms", type=int, default=40, help="SCM instances per cell")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--p", type=int, default=7, help="true features (6-8); decoy makes p+1")
    ap.add_argument("--batch-size", type=int, default=16, help="model forward batch size")
    ap.add_argument("--n-boot", type=int, default=100,
                    help="row-resamples for per-feature bootstrap sigma at D "
                         "(and on true causes); 0 disables the sigma columns")
    ap.add_argument("--boot-chunk", type=int, default=25,
                    help="bootstrap resamples pushed through the model at once")
    ap.add_argument("--out", default="causal_probe/results/proxy_determinism_sweep.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="one instance per cell, print shapes + corr(D,Y), no ckpt")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
        sys.exit(0)
    if not args.ckpt:
        ap.error("--ckpt is required unless --dry-run")

    out = run_sweep(args)
    print_tables(out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
