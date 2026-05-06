"""n-scaling eval: how does held-out Pearson change as the context size
n varies at test time? A well-calibrated PFN-style probe should improve
monotonically with n.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.eval import _load_model, _spearman_pearson, _auroc
from causal_probe.scm import LinearNonGaussianSCM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-grid", nargs="+", type=int,
                    default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--n-scms", type=int, default=150)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", default="laplace")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    out_json = args.out_json or (
        str(Path(args.ckpt).with_suffix("").as_posix()) + "_nscale.json"
    )
    model = _load_model(args.ckpt, device=args.device)
    print(f"ckpt = {args.ckpt}, n_grid = {args.n_grid}")

    rng = np.random.default_rng(args.seed)
    # Generate a FIXED pool of SCMs so each n uses the same generating processes.
    scms = [
        LinearNonGaussianSCM(
            p=args.p, rng=np.random.default_rng(rng.integers(0, 2**31)),
            noise=args.noise,
        )
        for _ in range(args.n_scms)
    ]
    results = {}
    for n in args.n_grid:
        preds, truths = [], []
        for scm in scms:
            samp = scm.sample(n=int(n), rng=np.random.default_rng(rng.integers(0, 2**31)))
            X_t = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(args.device)
            y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(args.device)
            with torch.no_grad():
                pred = model(X_t, y_t).squeeze(0).cpu().numpy()
            preds.append(pred)
            truths.append(samp.tau)
        preds = np.stack(preds).reshape(-1)
        truths = np.stack(truths).reshape(-1)
        sp, pe = _spearman_pearson(preds, truths)
        r2 = 1.0 - float(np.sum((preds - truths) ** 2)) / max(float(np.sum((truths - truths.mean()) ** 2)), 1e-12)
        labels_nz = (np.abs(truths) > 0.1).astype(int)
        au = _auroc(np.abs(preds), labels_nz)
        zero_mask = np.abs(truths) <= 0.1
        mae_zero = float(np.mean(np.abs(preds[zero_mask] - truths[zero_mask]))) if zero_mask.any() else float("nan")
        results[int(n)] = {
            "pearson": pe, "spearman": sp, "r2": r2,
            "auroc_nonzero": au, "mae_zero": mae_zero,
        }
        print(f"n={n:5d}  Pearson={pe:+.4f}  R2={r2:+.4f}  AUROC={au:+.4f}  MAE0={mae_zero:.4f}")

    with open(out_json, "w") as f:
        json.dump({"ckpt": args.ckpt, "n_scms": args.n_scms, "results": results}, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
