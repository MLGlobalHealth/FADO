"""IHDP semi-synthetic decoy benchmark: real IHDP covariates + OUR synthetic outcomes.

This is the "safe" IHDP experiment: it mirrors the ACIC-2016 false-positive
design in causal_probe/semisynth_acic_fp.py, swapping only the covariate
source. It is deliberately NOT the standard Hill-2011 IHDP ATE benchmark —
that comparison lives in causal_probe/ihdp_eval.py and is out of scope here.

Covariates: the 25 IHDP covariates (fredjo npz, train+test concatenated,
n=747), which are identical across the 100 replicates, so we use replicate 0.
Everything downstream — outcome families (linear / mixed_nonlinear /
interaction), decoy construction (proxy / leak / noise / confounder-child),
regimes (no confounding / hidden confounder), the false-positive metric
(mean |tau_hat| at decoy positions), and n_scms=40 per family — is imported
from semi_synth_real.make_benchmark via semisynth_acic_fp.run_regime, so the
resulting table is directly comparable to the paper's ACIC table
(tables/acic_fp). The ACIC replication check is disabled (expected={}) since
there is no prior IHDP run to replicate.

Run:
  uv run python -m causal_probe.ihdp_decoy_eval --dry-run
  uv run python -m causal_probe.ihdp_decoy_eval \
      --ckpt causal_probe/results/probe_intersample_L_seed0.ckpt --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_ihdp_covariates(cache_dir: str) -> tuple[np.ndarray, bool]:
    """Return (X, identical) where X is the (n, p) IHDP covariate matrix.

    Uses replicate 0 of the fredjo npz (train+test concatenated). ``identical``
    reports whether replicate 0 and replicate 1 covariates match up to a row
    permutation (each replicate reshuffles rows for its train/test split, but
    the 747 subjects are the same), as a sanity check on the "covariates are
    shared across replicates" assumption. Zero-variance columns are dropped
    (mirrors semisynth_acic_fp.load_acic2016; none expected for IHDP).
    """
    from causal_probe.ihdp_eval import load_ihdp

    x, _t, _yf, _mu0, _mu1 = load_ihdp(Path(cache_dir))

    def _row_sorted(a: np.ndarray) -> np.ndarray:
        return np.array(sorted(map(tuple, np.round(a, 6))))

    identical = bool(np.allclose(_row_sorted(x[:, :, 0]), _row_sorted(x[:, :, 1])))
    X = x[:, :, 0].astype(np.float64)
    keep = X.std(axis=0) > 0
    return X[:, keep], identical


def _dry_run(X: np.ndarray, args: argparse.Namespace) -> None:
    """Build one SCM instance per family x regime, print shapes + decoy
    positions, exit before any model load."""
    from causal_probe.semi_synth_real import make_benchmark

    for hidden in (False, True):
        label = "hidden_confounder" if hidden else "no_confounding"
        print(f"\n[dry-run] regime: {label}")
        colrng = np.random.default_rng(args.seed)
        for s, family in enumerate(f.strip() for f in args.families.split(",")):
            # same RNG structure as run_regime: per-family colrng from
            # args.seed, per-SCM benchmark rng 1000*seed + s (here s indexes
            # families since we build one instance each).
            cols = colrng.choice(X.shape[1], args.p_real_cap, replace=False)
            b = make_benchmark(
                X[:, cols], np.random.default_rng(1000 * args.seed + s),
                n_causal=args.n_causal, outcome_family=family,
                hidden_confound=hidden,
            )
            kinds = b["kinds"]
            names = b["feature_names"]
            decoys = [(j, kinds[j], names[j]) for j in range(len(kinds))
                      if kinds[j] != "real"]
            print(f"  {family:<16} X_full={b['X_full'].shape}  y={b['y'].shape}  "
                  f"true_tau={b['true_tau'].shape}")
            print("    decoy positions: "
                  + ", ".join(f"{j}={k}[{nm}]" for j, k, nm in decoys))
    print("\n[dry-run] OK — exiting before model load")


def main() -> None:
    from causal_probe.semisynth_acic_fp import add_comparator_args, run_regime

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Probe checkpoint (required unless --dry-run or "
                         "--comparators-only).")
    ap.add_argument("--ihdp-dir", default="data/ihdp",
                    help="Cache dir for the fredjo IHDP .npz files.")
    ap.add_argument("--source-label", default="ihdp")
    ap.add_argument("--families", default="linear,mixed_nonlinear,interaction")
    ap.add_argument("--n-scms", type=int, default=40)
    ap.add_argument("--p-real-cap", type=int, default=10,
                    help="Random covariate-subset size per SCM (matches ACIC).")
    ap.add_argument("--n-causal", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="causal_probe/results/ihdp_decoy_fp.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch IHDP X, build one SCM per family x regime, "
                         "print shapes + decoy positions, exit (no model).")
    add_comparator_args(ap)
    args = ap.parse_args()

    if args.ckpt is None and not (args.dry_run or args.comparators_only):
        ap.error("--ckpt is required unless --dry-run or --comparators-only")

    X, identical = load_ihdp_covariates(args.ihdp_dir)
    if args.p_real_cap > X.shape[1]:
        print(f"p_real_cap {args.p_real_cap} > p={X.shape[1]}; clipping.")
        args.p_real_cap = X.shape[1]
    print(f"IHDP covariates: {X.shape[0]} subjects x {X.shape[1]} usable cols "
          f"(replicate 0; identical across replicates up to row order: {identical}); "
          f"each SCM uses a random {args.p_real_cap}-col subset")

    if args.dry_run:
        _dry_run(X, args)
        return

    if args.comparators_only:
        model = torch = None
    else:
        import torch

        from causal_probe.eval import _load_model

        model = _load_model(args.ckpt, device=args.device)

    results = {}
    for hidden in (False, True):
        label = "hidden_confounder" if hidden else "no_confounding"
        print(f"\n=== regime: {label} ===")
        results[label] = run_regime(model, X, args, hidden, torch, expected={})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
