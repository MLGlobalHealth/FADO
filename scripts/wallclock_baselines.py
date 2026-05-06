"""Per-call wallclock for FADO + the headline baselines on a fixed test set.

Pre-samples N linear-non-Gaussian SCMs at p=5, n=512 and times each method's
per-SCM cost end-to-end (fit + explain for per-dataset methods; pure inference
for amortized methods). Saves median/IQR + raw timings to a JSON cache so
re-runs only fill missing cells.

Methods covered:
  marginal, ridge      — trivial reference
  shap                 — LightGBM-regressor + TreeSHAP, signed
  doubleml             — LinearDML per-feature
  causal_forest        — CausalForestDML per-feature
  causalpfn            — CausalPFN ATEEstimator per-feature
  probe                — FADO trained checkpoint (forward pass only)

Usage:
  uv run python -m scripts.wallclock_baselines --device cpu
  uv run python -m scripts.wallclock_baselines --device cuda --methods probe,causalpfn

Per-method timeouts: each method gets `--max-seconds-per-method` total budget
(default 600s on CPU). Methods that exceed it after their first SCM are
truncated, and the partial median is recorded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "causal_probe" / "results"
CACHE = RESULTS / "wallclock_baselines.json"

DEFAULT_METHODS = ["marginal", "ridge", "shap", "doubleml",
                   "causal_forest", "causalpfn", "probe"]
PROBE_CKPT = REPO / "causal_probe" / "results" / "probe_main_p5_15k.ckpt"


def _sample_scms(p: int, n_scms: int, seed: int) -> list:
    """N linear-non-Gaussian (Laplace) SCMs at p, fixed seed."""
    from causal_probe.scm import LinearNonGaussianSCM
    rng = np.random.default_rng(seed)
    scms = []
    for _ in range(n_scms):
        s = int(rng.integers(0, 2**31))
        scms.append(LinearNonGaussianSCM(p=p, rng=np.random.default_rng(s),
                                         noise="laplace"))
    return scms


def time_method(
    method: str, scms: list, n_rows: int, device: str, seed: int,
    max_seconds: float,
) -> dict:
    """Return {timings_s: [...], median_s, p25_s, p75_s, n_completed, n_skipped}."""
    from causal_probe.run_baseline import METHODS
    fn = METHODS[method]
    kwargs: dict = {}
    if method == "causalpfn":
        kwargs["device"] = device
    elif method == "probe":
        kwargs["ckpt_path"] = str(PROBE_CKPT)
        kwargs["device"] = device

    timings = []
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    skipped = 0
    for k, scm in enumerate(scms):
        samp = scm.sample(
            n=n_rows,
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )
        X = samp.X.astype(np.float64)
        y = samp.y.astype(np.float64)
        # Trigger jit / lazy-load on first call so it doesn't pollute timings
        if k == 0:
            try:
                _ = fn(X, y, **kwargs)
            except Exception as e:
                print(f"  [{method}] warmup failed: {e}")
                return {"error": str(e), "n_completed": 0,
                        "timings_s": [], "median_s": float("nan")}
        t0 = time.perf_counter()
        try:
            _ = fn(X, y, **kwargs)
        except Exception as e:
            print(f"  [{method}] SCM {k}: {e}")
            skipped += 1
            continue
        dt = time.perf_counter() - t0
        timings.append(dt)
        elapsed = time.perf_counter() - started
        if elapsed > max_seconds:
            print(f"  [{method}] hit {max_seconds:.0f}s budget after "
                  f"{len(timings)} timings; truncating")
            break

    if not timings:
        return {"error": "no successful timings", "n_completed": 0,
                "n_skipped": skipped, "timings_s": [], "median_s": float("nan")}
    arr = np.array(timings)
    return {
        "timings_s": [float(t) for t in arr],
        "median_s": float(np.median(arr)),
        "p25_s": float(np.percentile(arr, 25)),
        "p75_s": float(np.percentile(arr, 75)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "n_completed": int(len(arr)),
        "n_skipped": int(skipped),
    }


def cell_key(method: str, device: str, p: int, n_rows: int) -> str:
    return f"{method}::{device}::p{p}::n{n_rows}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help="Comma-separated method names")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-scms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--max-seconds-per-method", type=float, default=600.0)
    ap.add_argument("--force", action="store_true",
                    help="Re-time even if cached")
    args = ap.parse_args()

    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())

    methods = args.methods.split(",")
    scms = _sample_scms(args.p, args.n_scms, args.seed)
    print(f"Sampled {len(scms)} SCMs (linear, p={args.p}, n_rows={args.n_rows})")

    for m in methods:
        key = cell_key(m, args.device, args.p, args.n_rows)
        if key in cache and not args.force:
            r = cache[key]
            print(f"[cache] {m:>15s} ({args.device}): "
                  f"median={r.get('median_s', float('nan')):.4f}s "
                  f"({r.get('n_completed', 0)} runs)")
            continue
        print(f"[run]   {m:>15s} ({args.device})...")
        res = time_method(m, scms, args.n_rows, args.device, args.seed,
                          args.max_seconds_per_method)
        cache[key] = res
        CACHE.write_text(json.dumps(cache, indent=2))
        if "error" in res:
            print(f"  -> ERROR: {res['error']}")
        else:
            print(f"  -> median={res['median_s']:.4f}s "
                  f"(p25={res['p25_s']:.4f} p75={res['p75_s']:.4f}) "
                  f"n={res['n_completed']}")

    print(f"\nwrote {CACHE}")


if __name__ == "__main__":
    main()
