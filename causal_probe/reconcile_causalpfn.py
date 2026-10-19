"""Reconcile our CausalPFN setup with the published numbers.

Imports the upstream benchmarks/ harness directly (cloned to
external/CausalPFN_upstream/) and runs CausalPFN's ATEEstimator on IHDP
and RealCauseLalondeCPS using their exact metric definition:

    rel_err = abs(hat_ate - true_ate) / abs(true_ate)   per realization
    report = mean ± SEM across realizations

Published targets (Balazadeh et al. 2025, Table 1):
    IHDP        ATE rel-err = 0.20 ± 0.04
    Lalonde-CPS ATE rel-err = 0.13 ± 0.01

If we don't reproduce these inside their own harness, our recreation is
broken. If we do, we can proceed to add FADO as a baseline in the same
loop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "external" / "CausalPFN_upstream"
sys.path.insert(0, str(UPSTREAM))

from benchmarks import IHDPDataset, RealCauseLalondeCPSDataset  # noqa: E402
from benchmarks.base import ATE_Dataset  # noqa: E402
from causalpfn import ATEEstimator  # noqa: E402


def run(dataset, name: str, n_tables: int, device: str, cache_dir: str | None) -> dict:
    rel_errs = []
    for r in range(min(n_tables, len(dataset))):
        _, ate_dset = dataset[r]
        ate_dset: ATE_Dataset
        t0 = time.time()
        kwargs = {"device": device}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        est = ATEEstimator(**kwargs)
        est.fit(ate_dset.X, ate_dset.t, ate_dset.y)
        hat = float(est.estimate_ate())
        rel = abs(hat - ate_dset.true_ate) / max(abs(ate_dset.true_ate), 1e-12)
        rel_errs.append(rel)
        print(f"  {name} rep {r+1}/{n_tables}: true={ate_dset.true_ate:+.4f} "
              f"hat={hat:+.4f} rel_err={rel:.4f}  ({time.time() - t0:.1f}s)")

    arr = np.asarray(rel_errs)
    out = {
        "dataset": name,
        "n_tables": int(arr.size),
        "rel_err_mean": float(arr.mean()),
        "rel_err_sem": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "rel_errs": arr.tolist(),
    }
    print(f"\n  {name}: rel_err = {out['rel_err_mean']:.3f} ± {out['rel_err_sem']:.3f}  (n={out['n_tables']})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ihdp", "lalonde_cps"])
    ap.add_argument("--n-tables", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import os
    cache_dir = args.cache_dir or os.environ.get("CAUSALPFN_CACHE_DIR")

    results = {}
    if "ihdp" in args.datasets:
        print("=== IHDP (target: 0.20 ± 0.04) ===")
        results["ihdp"] = run(IHDPDataset(n_tables=args.n_tables), "IHDP",
                              args.n_tables, args.device, cache_dir)
    if "lalonde_cps" in args.datasets:
        print("\n=== RealCause Lalonde-CPS (target: 0.13 ± 0.01) ===")
        results["lalonde_cps"] = run(RealCauseLalondeCPSDataset(n_tables=args.n_tables),
                                     "LalondeCPS", args.n_tables, args.device, cache_dir)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:<12s}  rel_err = {v['rel_err_mean']:.3f} ± {v['rel_err_sem']:.3f}  (n={v['n_tables']})")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
