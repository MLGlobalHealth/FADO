"""Cross-family generalization for the intersample (canonical-arch) models.

Addresses the concern ("test families = training families, unfair advantage") using
the SAME architecture as the other experiments: single-family intersample
specialists tested on HELD-OUT families, plus the mixture foundation model.
Off-diagonal average = transfer to families the specialist never trained on.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from causal_probe.eval import _load_model, evaluate_random

R = str(Path(__file__).resolve().parent / "results")
# (row label, checkpoint, own training family or None for the mixture foundation)
ROWS = [
    ("nonlinear-spec",  f"{R}/probe_nonlinear_p5_20k_v2_intersample.ckpt", "nonlinear"),
    ("mlp-spec",        f"{R}/probe_mlp_p5_20k_intersample.ckpt",          "mlp"),
    ("hidden-spec",     f"{R}/probe_hidden_p5_20k_intersample.ckpt",       "hidden"),
    ("mlp_hidden-spec", f"{R}/probe_mlp_hidden_p5_25k_intersample.ckpt",   "mlp_hidden"),
    ("mixed-spec",      f"{R}/probe_mixed_p5_20k_intersample.ckpt",        "mixed"),
    ("FOUNDATION (L)",  f"{R}/probe_foundation_intersample_L_p3_15_intersample_L_seed0.ckpt", None),
]
FAMILIES = ["linear", "nonlinear", "mlp", "hidden", "mlp_hidden", "mixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scms", type=int, default=100)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", default="heavy")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=f"{R}/shapley_comparison/cross_family_intersample.json")
    args = ap.parse_args()

    mat = {}
    for label, ckpt, own in ROWS:
        rng = np.random.default_rng(args.seed)   # same eval SCMs per row
        model = _load_model(ckpt, device=args.device)
        mat[label] = {"own": own, "cells": {}}
        # Foundation was trained padded to p_max=15; the p=5 specialists were
        # trained at fixed p=5. Evaluate each in its NATIVE width.
        p_pad = 15 if own is None else None
        for fam in FAMILIES:
            r = evaluate_random(model, args.n_scms, args.p, args.n_rows, rng=rng,
                                device=args.device, scm_type=fam, noise=args.noise,
                                p_pad=p_pad)
            mat[label]["cells"][fam] = r["pearson"]["model"]
        print(f"{label:<18} " + " ".join(f"{fam[:4]}={mat[label]['cells'][fam]:+.2f}" for fam in FAMILIES), flush=True)

    print("\n=== Cross-family Pearson matrix (rows=train, cols=eval; noise=%s) ===" % args.noise)
    print(f"{'':<18}" + "".join(f"{f[:9]:>10}" for f in FAMILIES))
    for label, _, _ in ROWS:
        print(f"{label:<18}" + "".join(f"{mat[label]['cells'][f]:>+10.3f}" for f in FAMILIES))

    print("\n=== Held-out off-diagonal average (transfer to non-training families) ===")
    for label, _, own in ROWS:
        cells = mat[label]["cells"]
        if own is None:
            avg = np.mean([cells[f] for f in FAMILIES])
            print(f"  {label:<18} per-family avg = {avg:.3f}  (mixture: in-distribution)")
        else:
            off = [cells[f] for f in FAMILIES if f != own]
            print(f"  {label:<18} off-diag avg = {np.mean(off):.3f}  (held out; own family '{own}' excluded)")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"matrix": {k: v["cells"] for k, v in mat.items()},
                   "own": {k: v["own"] for k, v in mat.items()},
                   "config": vars(args)}, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
