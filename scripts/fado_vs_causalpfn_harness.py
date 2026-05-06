"""FADO vs CausalPFN inside CausalPFN's own benchmark harness.

Mirrors the eval loop of upstream notebooks/causal_effect.ipynb but as a
plain script. Per realization:

    rel_err_method = abs(hat - true_ate) / abs(true_ate)

aggregated as mean ± SEM across realizations. This is the same metric
that produced CausalPFN's published Table 1 numbers; we already
reproduced those (IHDP 0.20, LalondeCPS 0.13) via
scripts/reconcile_causalpfn.py, so any FADO row on this scale is
directly comparable to CausalPFN's published result.

FADO note: trained for column-symmetric per-feature contrasts on
synthetic SCM priors. Designated-treatment ATE on real-data benchmarks
is OOD for FADO. A fair-comparison row, not a competitive one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "external" / "CausalPFN_upstream"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(REPO))

from benchmarks import (  # noqa: E402
    IHDPDataset, RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset,
)
from benchmarks.base import ATE_Dataset  # noqa: E402
from causalpfn import ATEEstimator  # noqa: E402

from causal_probe.fado_baseline import FADOBaseline  # noqa: E402


def make_datasets(names: list[str], n_tables: int) -> dict:
    catalog = {
        "ihdp": lambda: IHDPDataset(n_tables=n_tables),
        "lalonde_cps": lambda: RealCauseLalondeCPSDataset(n_tables=n_tables),
        "lalonde_psid": lambda: RealCauseLalondePSIDDataset(n_tables=n_tables),
    }
    return {n: catalog[n]() for n in names if n in catalog}


def make_methods(args, ckpt_dir: Path) -> dict:
    methods: dict = {}
    if "causalpfn" in args.methods:
        def _causalpfn_estimate(X, t, y):
            kwargs = {"device": args.device}
            if args.causalpfn_cache:
                kwargs["cache_dir"] = args.causalpfn_cache
            est = ATEEstimator(**kwargs)
            est.fit(X, t, y)
            return float(est.estimate_ate())
        methods["causalpfn"] = _causalpfn_estimate

    for ckpt_name in args.fado_ckpts:
        ckpt_path = ckpt_dir / f"{ckpt_name}.ckpt"
        if not ckpt_path.exists():
            print(f"  [warn] FADO ckpt missing: {ckpt_path}")
            continue
        fado = FADOBaseline(
            ckpt=str(ckpt_path), device=args.device,
            n_rows=args.fado_n_rows, seed=args.seed,
        )
        methods[f"fado_{ckpt_name}"] = lambda X, t, y, _f=fado: _f.estimate_ate(X, t, y)
    return methods


def evaluate(dataset, name: str, methods: dict, n_tables: int) -> dict:
    per_method: dict = {m: {"rel_errs": [], "hats": [], "trues": [], "times": []} for m in methods}
    for r in range(min(n_tables, len(dataset))):
        _, ate_dset = dataset[r]
        ate_dset: ATE_Dataset
        true_ate = ate_dset.true_ate
        for m, fn in methods.items():
            t0 = time.time()
            try:
                hat = float(fn(ate_dset.X, ate_dset.t, ate_dset.y))
            except Exception as e:
                print(f"  {name} rep {r+1}/{n_tables} [{m}] failed: {e}")
                continue
            dt = time.time() - t0
            rel = abs(hat - true_ate) / max(abs(true_ate), 1e-12)
            per_method[m]["rel_errs"].append(rel)
            per_method[m]["hats"].append(hat)
            per_method[m]["trues"].append(float(true_ate))
            per_method[m]["times"].append(dt)
            print(f"  {name} rep {r+1}/{n_tables} [{m}]: "
                  f"true={true_ate:+.3f} hat={hat:+.3f} rel={rel:.3f} ({dt:.1f}s)")

    summary: dict = {"dataset": name, "methods": {}}
    for m, d in per_method.items():
        arr = np.asarray(d["rel_errs"])
        if arr.size == 0:
            summary["methods"][m] = {"n": 0}
            continue
        summary["methods"][m] = {
            "n": int(arr.size),
            "rel_err_mean": float(arr.mean()),
            "rel_err_sem": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
            "hat_mean": float(np.mean(d["hats"])),
            "true_mean": float(np.mean(d["trues"])),
            "time_mean": float(np.mean(d["times"])),
        }

    print(f"\n=== {name} summary ===")
    for m, s in summary["methods"].items():
        if s.get("n", 0) > 0:
            print(f"  {m:<32s} rel_err = {s['rel_err_mean']:.3f} ± {s['rel_err_sem']:.3f}  "
                  f"(n={s['n']}, hat={s['hat_mean']:+.3f} vs true={s['true_mean']:+.3f}, "
                  f"{s['time_mean']:.1f}s/rep)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ihdp", "lalonde_cps", "lalonde_psid"])
    ap.add_argument("--methods", nargs="+", default=["causalpfn", "fado"],
                    help="If 'fado' present, runs all --fado-ckpts.")
    ap.add_argument("--fado-ckpts", nargs="+",
                    default=["probe_p13_20k", "probe_mixed_p13_25k", "probe_p20_30k"])
    ap.add_argument("--fado-n-rows", type=int, default=512)
    ap.add_argument("--n-tables", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt-dir", default="causal_probe/results")
    ap.add_argument("--causalpfn-cache", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    import os
    args.causalpfn_cache = args.causalpfn_cache or os.environ.get("CAUSALPFN_CACHE_DIR")

    datasets = make_datasets(args.datasets, args.n_tables)
    methods = make_methods(args, Path(args.ckpt_dir))

    print(f"datasets: {list(datasets.keys())}")
    print(f"methods:  {list(methods.keys())}")
    print()

    results = {}
    for dname, ds in datasets.items():
        print(f"\n=== {dname} (n_tables={args.n_tables}) ===")
        results[dname] = evaluate(ds, dname, methods, args.n_tables)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
