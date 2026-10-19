"""Type-flip ablation for IntersampleProbe (arch_intersample).

Parallel to causal_probe/ablation_type_flip.py (which is rowcol-specific).
Same 3 configs (normal / all_zero / mispointed); the only difference is
the custom forward path, which has to mirror IntersampleProbe.forward
instead of CausalProbe.forward.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from causal_probe.eval import _load_model
from causal_probe.scm import LinearNonGaussianSCM


def custom_forward(model, X: torch.Tensor, y: torch.Tensor,
                   type_ids: torch.Tensor) -> torch.Tensor:
    """Reimplements IntersampleProbe.forward with caller-supplied type_ids."""
    if X.dim() != 3 or y.dim() != 2:
        raise ValueError(f"X must be (B,n,p), y must be (B,n); got {X.shape}, {y.shape}")
    B, n, p = X.shape
    if type_ids.shape != (p + 1,):
        raise ValueError(f"type_ids must be ({p+1},); got {type_ids.shape}")
    Z = torch.cat([X, y.unsqueeze(-1)], dim=-1)             # (B, n, p+1)
    h = model.value_proj(Z.unsqueeze(-1))                    # (B, n, p+1, d)
    te = model.type_emb(type_ids.to(X.device))
    h = h + te.view(1, 1, p + 1, -1)
    for row_blk, inter_blk in zip(model.row_blocks, model.intersample_blocks):
        h_r = h.reshape(B * n, p + 1, -1)
        h_r = row_blk(h_r).reshape(B, n, p + 1, -1)
        h = inter_blk(h_r)
    col_emb = h.mean(dim=1)                                  # (B, p+1, d)
    feat_emb = col_emb[:, :p, :]
    return model.head(feat_emb).squeeze(-1)


def _safe_spearman(a, b):
    return float(spearmanr(a, b)[0])


def _mae_zero(pred, truth, eps: float = 0.1):
    mask = np.abs(truth) <= eps
    return float(np.mean(np.abs(pred[mask] - truth[mask]))) if mask.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-scms", type=int, default=500)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--noise", default="laplace", choices=["laplace", "gaussian", "heavy"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    model = _load_model(args.ckpt, device=args.device)

    rng = np.random.default_rng(args.seed)
    p = args.p

    type_ids_normal = torch.zeros(p + 1, dtype=torch.long); type_ids_normal[p] = 1
    type_ids_zero = torch.zeros(p + 1, dtype=torch.long)
    type_ids_mispointed = torch.zeros(p + 1, dtype=torch.long); type_ids_mispointed[0] = 1

    configs = [
        ("normal", type_ids_normal),
        ("all_zero", type_ids_zero),
        ("mispointed", type_ids_mispointed),
    ]

    preds = {label: [] for label, _ in configs}
    truths = []

    print(f"Type-flip (intersample) ablation: ckpt={args.ckpt}, n_scms={args.n_scms}, p={p}, noise={args.noise}")
    for _ in range(args.n_scms):
        scm_seed = rng.integers(2**31, 2**32)
        sample_seed = rng.integers(2**31, 2**32)
        scm = LinearNonGaussianSCM(p=p, rng=np.random.default_rng(scm_seed), noise=args.noise)
        samp = scm.sample(n=args.n_rows, rng=np.random.default_rng(sample_seed))
        X = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(args.device)
        y = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(args.device)
        with torch.no_grad():
            for label, tid in configs:
                pred = custom_forward(model, X, y, tid).squeeze(0).cpu().numpy()
                preds[label].append(pred)
        truths.append(samp.tau)

    truths_flat = np.stack(truths).reshape(-1)
    normal_flat = np.stack(preds["normal"]).reshape(-1)
    results = {}
    for label in ["normal", "all_zero", "mispointed"]:
        flat = np.stack(preds[label]).reshape(-1)
        results[label] = {
            "pearson_truth": float(pearsonr(flat, truths_flat)[0]),
            "spearman_truth": _safe_spearman(flat, truths_flat),
            "mae_zero_features": _mae_zero(flat, truths_flat),
            "mean_abs_diff_vs_normal": (0.0 if label == "normal"
                                        else float(np.mean(np.abs(flat - normal_flat)))),
            "pearson_vs_normal": (1.0 if label == "normal"
                                  else float(pearsonr(flat, normal_flat)[0])),
        }

    print()
    print(f"{'config':<14}{'Pearson(τ̂,τ)':>16}  {'Δ vs normal':>13}  "
          f"{'Pearson(τ̂,τ̂_normal)':>23}  {'MAE-zero':>10}")
    for k in ["normal", "all_zero", "mispointed"]:
        r = results[k]
        print(f"{k:<14}{r['pearson_truth']:>16.4f}  "
              f"{r['mean_abs_diff_vs_normal']:>13.4f}  "
              f"{r['pearson_vs_normal']:>23.4f}  "
              f"{r['mae_zero_features']:>10.4f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "n_scms": args.n_scms, "p": p,
                "n_rows": args.n_rows, "seed": args.seed, "noise": args.noise,
                "results": results,
            }, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
