"""IHDP apples-to-apples: probe vs CausalPFN over the standard 100 replicates.

IHDP is the headline benchmark in CausalPFN's paper. Each of 100 replicates
has the same 25 IHDP covariates with synthetic potential outcomes attached
(Hill 2011; NPCI hosting via fredjo). True per-replicate ATE = mean(mu1 - mu0).

We score *|ε_ATE|* on the probe's reporting scale, τ = (ATE/std_y)·2·std_T,
so probe and CausalPFN are read off the same column. PEHE is not reported —
the probe outputs a population-level per-feature contrast, not a per-unit
CATE, so PEHE is out of scope for this comparison.

Data layout per .npz: keys ``x``, ``t``, ``yf``, ``ycf``, ``mu0``, ``mu1``,
each shaped (n, 100). We use train+test concatenated so true ATE is on the
full population for that replicate, and so is std_y / std_T.

Probe input convention: T is included as one column of X (column 0 by
construction here); the probe is column-symmetric so position is
irrelevant. CausalPFN gets T explicitly and adjusts for the remaining
columns.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from causal_probe.eval import _load_model


URLS_TRAIN = [
    "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz",
    "http://www.fredjo.com/files/ihdp_npci_1-100.train.npz",
]
URLS_TEST = [
    "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz",
    "http://www.fredjo.com/files/ihdp_npci_1-100.test.npz",
]


def _fetch_npz(urls: list[str], cache_path: Path) -> np.lib.npyio.NpzFile:
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=False)
    last = None
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                blob = r.read()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(blob)
            return np.load(cache_path, allow_pickle=False)
        except Exception as e:
            last = e
    raise RuntimeError(f"IHDP fetch failed: {last}")


def load_ihdp(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns concatenated train+test arrays per replicate.

    Shapes: x (n_total, 25, 100), t (n_total, 100), yf (n_total, 100),
    mu0 (n_total, 100), mu1 (n_total, 100).
    """
    tr = _fetch_npz(URLS_TRAIN, cache_dir / "ihdp_npci_1-100.train.npz")
    te = _fetch_npz(URLS_TEST, cache_dir / "ihdp_npci_1-100.test.npz")
    x = np.concatenate([tr["x"], te["x"]], axis=0)
    t = np.concatenate([tr["t"], te["t"]], axis=0)
    yf = np.concatenate([tr["yf"], te["yf"]], axis=0)
    mu0 = np.concatenate([tr["mu0"], te["mu0"]], axis=0)
    mu1 = np.concatenate([tr["mu1"], te["mu1"]], axis=0)
    return x, t, yf, mu0, mu1


def _scale(ate_raw: float, std_y: float, std_T: float) -> float:
    """Probe reporting scale: τ = (ATE/std_y)·2·std_T."""
    return (ate_raw / max(std_y, 1e-9)) * 2.0 * std_T


def _probe_tau_T(model, X: np.ndarray, y: np.ndarray, t_idx: int, *, device: str) -> float:
    """Standardize columns and y, run probe, return τ̂[t_idx]."""
    Xs = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-9)
    ys = (y - y.mean()) / y.std().clip(min=1e-9)
    X_t = torch.from_numpy(Xs.astype(np.float32)).unsqueeze(0).to(device)
    y_t = torch.from_numpy(ys.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(X_t, y_t).squeeze(0).cpu().numpy()
    return float(pred[t_idx])


def _causalpfn_ate(X_cov: np.ndarray, T: np.ndarray, y: np.ndarray,
                   *, device: str, cache_dir: str | None) -> float | None:
    from causalpfn import ATEEstimator
    kwargs = {"device": device}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    try:
        est = ATEEstimator(**kwargs)
        est.fit(X_cov.astype(np.float64), T.astype(np.int64), y.astype(np.float64))
        return float(est.estimate_ate())
    except Exception as e:
        print(f"    CausalPFN fit failed: {e}")
        return None


def evaluate(
    *, methods: list[str], probe_ckpts: dict[str, str],
    n_replicates: int, cache_dir: Path, causalpfn_cache: str | None,
    device: str, n_rows: int | None,
) -> dict:
    x, t, yf, mu0, mu1 = load_ihdp(cache_dir)
    n_total, p_x, R = x.shape
    n_replicates = min(n_replicates, R)
    print(f"IHDP loaded: n_total={n_total}, p={p_x}, R={R} (using R={n_replicates})")

    # Per-method per-replicate error + standardized predictions.
    results = {m: {"err": [], "tau_hat": [], "tau_true": []} for m in methods}

    # Probe model cache.
    probe_models: dict[str, object] = {}
    for tag, ckpt in probe_ckpts.items():
        if tag in methods:
            print(f"  loading probe '{tag}': {ckpt}")
            probe_models[tag] = _load_model(ckpt, device=device)

    for r in range(n_replicates):
        X_cov = x[:, :, r].astype(np.float64)              # (n, 25)
        T = t[:, r].astype(np.int64)                        # (n,)
        Y = yf[:, r].astype(np.float64)                     # (n,)
        true_ATE = float((mu1[:, r] - mu0[:, r]).mean())
        std_y = float(np.std(Y).clip(min=1e-9))
        std_T = float(np.std(T).clip(min=1e-9))
        tau_true = _scale(true_ATE, std_y, std_T)

        # Probe input: prepend T as column 0 of a (n, 26) matrix.
        X_with_T = np.concatenate([T.reshape(-1, 1).astype(np.float64), X_cov], axis=1)
        if n_rows is not None and X_with_T.shape[0] > n_rows:
            rng = np.random.default_rng(2025 + r)
            sel = rng.choice(X_with_T.shape[0], size=n_rows, replace=False)
            X_probe = X_with_T[sel]; Y_probe = Y[sel]
            T_cf = T[sel]; std_y_sub = float(np.std(Y_probe).clip(min=1e-9))
            std_T_sub = float(np.std(T_cf).clip(min=1e-9))
        else:
            X_probe = X_with_T; Y_probe = Y
            std_y_sub = std_y; std_T_sub = std_T

        for m in methods:
            if m.startswith("probe_"):
                tag = m[len("probe_"):]
                model = probe_models.get(m) or probe_models.get(tag)
                if model is None:
                    continue
                tau_hat = _probe_tau_T(model, X_probe, Y_probe, t_idx=0, device=device)
                # Probe is trained on standardized inputs/outputs, so tau_hat
                # is already on the standardized scale. Compare directly to
                # tau_true (also on the standardized scale).
                err = abs(tau_hat - tau_true)
            elif m == "causalpfn":
                ate_raw = _causalpfn_ate(X_cov, T, Y, device=device, cache_dir=causalpfn_cache)
                if ate_raw is None:
                    continue
                tau_hat = _scale(ate_raw, std_y, std_T)
                err = abs(tau_hat - tau_true)
            else:
                raise ValueError(f"unknown method: {m}")
            results[m]["err"].append(err)
            results[m]["tau_hat"].append(tau_hat)
            results[m]["tau_true"].append(tau_true)

        if (r + 1) % 10 == 0:
            line = f"[rep {r+1}/{n_replicates}] τ_true={tau_true:+.3f}"
            for m in methods:
                if results[m]["err"]:
                    line += f"  {m} err={results[m]['err'][-1]:.3f}"
            print(line)

    # Aggregate.
    summary: dict = {
        "n_replicates": n_replicates,
        "tau_true_mean": float(np.mean([t for m in methods for t in results[m]["tau_true"][:n_replicates]] or [0])),
        "methods": {},
    }
    for m in methods:
        errs = np.asarray(results[m]["err"])
        if errs.size == 0:
            summary["methods"][m] = {"n": 0}
            continue
        summary["methods"][m] = {
            "n": int(errs.size),
            "err_mean": float(errs.mean()),
            "err_ci": [float(np.percentile(errs, 2.5)), float(np.percentile(errs, 97.5))],
            "err_std": float(errs.std(ddof=1)) if errs.size > 1 else 0.0,
            "tau_hat_mean": float(np.mean(results[m]["tau_hat"])),
        }

    print("\n=== IHDP-100 |ε_ATE| (probe scale) ===")
    for m, s in summary["methods"].items():
        if s.get("n", 0) > 0:
            print(f"  {m:<28s} n={s['n']:3d}  mean|err|={s['err_mean']:.3f}  "
                  f"95% CI=[{s['err_ci'][0]:.3f}, {s['err_ci'][1]:.3f}]  "
                  f"τ̂_mean={s['tau_hat_mean']:+.3f}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["probe_mixed_p13_25k", "probe_p20_30k", "causalpfn"])
    ap.add_argument("--ckpt-dir", default="causal_probe/results",
                    help="Directory holding probe_*.ckpt files.")
    ap.add_argument("--ihdp-dir", default="data/ihdp",
                    help="Cache for downloaded IHDP .npz.")
    ap.add_argument("--n-replicates", type=int, default=100)
    ap.add_argument("--n-rows", type=int, default=None,
                    help="If set, subsample probe inputs to this many rows "
                         "(CausalPFN always uses the full replicate).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--causalpfn-cache", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import os
    causalpfn_cache = args.causalpfn_cache or os.environ.get("CAUSALPFN_CACHE_DIR")

    probe_ckpts = {
        "probe_mixed_p13_25k": str(Path(args.ckpt_dir) / "probe_mixed_p13_25k.ckpt"),
        "probe_p20_30k":       str(Path(args.ckpt_dir) / "probe_p20_30k.ckpt"),
    }

    res = evaluate(
        methods=args.methods, probe_ckpts=probe_ckpts,
        n_replicates=args.n_replicates, cache_dir=Path(args.ihdp_dir),
        causalpfn_cache=causalpfn_cache, device=args.device,
        n_rows=args.n_rows,
    )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
