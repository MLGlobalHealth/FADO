"""FADO accuracy vs number of features p.

Evaluates a single checkpoint across a grid of p on representative families
(identifiable / nonlinear / latent-confounded), reporting Pearson & Spearman
with true tau. The canonical model is trained on heterogeneous p in [3,15], so
p<=15 is in-distribution and p>15 is extrapolation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from causal_probe.eval import _load_model, evaluate_random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--families", default="linear,nonlinear,hidden")
    ap.add_argument("--p-grid", default="3,5,8,10,13,15,20")
    ap.add_argument("--n-scms", type=int, default=150)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--nonlinear-mc", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="causal_probe/results/shapley_comparison/p_sweep_seed0_final.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = _load_model(args.ckpt, device=args.device)
    fams = [f.strip() for f in args.families.split(",")]
    pgrid = [int(x) for x in args.p_grid.split(",")]
    print(f"Loaded {args.ckpt}\n")

    out = {}
    for fam in fams:
        out[fam] = {}
        for p in pgrid:
            r = evaluate_random(model, args.n_scms, p, args.n_rows, rng=rng,
                                device=args.device, scm_type=fam,
                                nonlinear_mc=args.nonlinear_mc)
            out[fam][p] = {"pearson": r["pearson"]["model"],
                           "spearman": r["spearman"]["model"],
                           "auroc": r["auroc_nonzero"]["model"]}
            print(f"  {fam:<10} p={p:<3} pearson={r['pearson']['model']:+.3f} "
                  f"spearman={r['spearman']['model']:+.3f}", flush=True)

    print(f"\n=== FADO Pearson r vs p  (train range p3-15; p=20 is extrapolation) ===")
    hdr = "family     " + "".join(f"{('p='+str(p)):>9}" for p in pgrid)
    print(hdr)
    for fam in fams:
        print(f"{fam:<11}" + "".join(f"{out[fam][p]['pearson']:>+9.3f}" for p in pgrid))
    print(f"\n=== Spearman r vs p ===")
    print(hdr)
    for fam in fams:
        print(f"{fam:<11}" + "".join(f"{out[fam][p]['spearman']:>+9.3f}" for p in pgrid))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"sweep": out, "config": vars(args)}, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
