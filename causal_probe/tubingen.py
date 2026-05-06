"""Tübingen cause-effect pairs benchmark.

Each pair is a real two-variable dataset with known causal direction
(column 1 causes column 2, typically). We run the causal probe twice:

  Task 1:  X = A,  y = B  →  |tau_hat_{A→B}|
  Task 2:  X = B,  y = A  →  |tau_hat_{B→A}|

Classify direction by whichever |tau_hat| is larger. Report accuracy
on the (pairs × 1) benchmark.

The model requires p >= 2 features; we pad each 1-feature input with
k_noise noise columns and shuffle column order.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from causal_probe.eval import _load_model


TUBINGEN_BASE = "https://webdav.tuebingen.mpg.de/cause-effect"


def _fetch_meta() -> list[dict]:
    with urllib.request.urlopen(f"{TUBINGEN_BASE}/pairmeta.txt", timeout=15) as r:
        raw = r.read().decode()
    pairs = []
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        pid, cs, ce, es, ee, weight = parts[:6]
        pairs.append({
            "id": pid,
            "cause_cols": (int(cs), int(ce)),
            "effect_cols": (int(es), int(ee)),
            "weight": float(weight),
        })
    return pairs


def _fetch_pair(pair_id: str) -> np.ndarray:
    with urllib.request.urlopen(f"{TUBINGEN_BASE}/pair{pair_id}.txt", timeout=30) as r:
        raw = r.read().decode()
    rows = []
    for line in raw.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return np.asarray(rows, dtype=np.float64)


def _predict_tau_for_feature(
    model, X_feat: np.ndarray, y_feat: np.ndarray, rng: np.random.Generator,
    *, n_rows: int, n_repeats: int, k_noise: int, device: str,
) -> float:
    """Present (X_feat, y_feat) to the model, padded with noise columns and
    shuffled. Return the model's |tau_hat| at the position of X_feat.
    """
    n = X_feat.shape[0]
    # Standardize
    x = (X_feat - X_feat.mean()) / (X_feat.std() + 1e-9)
    y = (y_feat - y_feat.mean()) / (y_feat.std() + 1e-9)

    preds = []
    for _ in range(n_repeats):
        if n <= n_rows:
            idx = np.arange(n)
        else:
            idx = rng.choice(n, size=n_rows, replace=False)
        xi = x[idx]; yi = y[idx]
        # Noise padding
        noise = rng.standard_normal((len(idx), k_noise)).astype(np.float64)
        X_aug = np.concatenate([xi.reshape(-1, 1), noise], axis=1)
        # Column permutation
        order = rng.permutation(X_aug.shape[1])
        feat_pos = int(np.where(order == 0)[0][0])
        X_aug = X_aug[:, order]
        X_t = torch.from_numpy(X_aug.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(yi.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        preds.append(abs(float(pred[feat_pos])))
    return float(np.mean(preds))


def evaluate(
    ckpt: str, *, n_rows: int, n_repeats: int, k_noise: int, device: str,
    seed: int, max_pairs: int | None,
) -> dict:
    pairs = _fetch_meta()
    if max_pairs:
        pairs = pairs[:max_pairs]
    print(f"Tübingen: evaluating {len(pairs)} pairs")
    model = _load_model(ckpt, device=device)
    rng = np.random.default_rng(seed)

    correct = 0
    total = 0
    total_weight = 0.0
    weighted_correct = 0.0
    per_pair = []
    for pair in pairs:
        try:
            data = _fetch_pair(pair["id"])
        except Exception as e:
            print(f"  skip {pair['id']}: {e}")
            continue
        # Restrict to 2-variable pairs (most of them). Skip multivariate.
        if data.shape[1] != 2:
            continue
        cs, ce = pair["cause_cols"]
        es, ee = pair["effect_cols"]
        # 1-indexed column numbers; skip if not single-column cause/effect
        if cs != ce or es != ee:
            continue
        cause_col = cs - 1
        effect_col = es - 1
        A = data[:, cause_col]  # true cause
        B = data[:, effect_col]  # true effect

        tau_AtoB = _predict_tau_for_feature(
            model, A, B, rng,
            n_rows=n_rows, n_repeats=n_repeats, k_noise=k_noise, device=device,
        )
        tau_BtoA = _predict_tau_for_feature(
            model, B, A, rng,
            n_rows=n_rows, n_repeats=n_repeats, k_noise=k_noise, device=device,
        )
        pred_correct = tau_AtoB > tau_BtoA
        correct += int(pred_correct)
        total += 1
        w = pair["weight"]
        total_weight += w
        if pred_correct:
            weighted_correct += w
        per_pair.append({
            "id": pair["id"],
            "tau_AtoB": tau_AtoB,
            "tau_BtoA": tau_BtoA,
            "correct": bool(pred_correct),
            "weight": w,
        })
        print(f"  pair {pair['id']}: |tau_{{A→B}}|={tau_AtoB:.3f} "
              f"|tau_{{B→A}}|={tau_BtoA:.3f}  {'✓' if pred_correct else '✗'}")

    acc = correct / total if total else 0.0
    w_acc = weighted_correct / total_weight if total_weight else 0.0
    print()
    print(f"Tübingen direction accuracy:   {acc:.3f} ({correct}/{total})")
    print(f"Weighted accuracy:             {w_acc:.3f}")
    return {
        "ckpt": ckpt, "n_pairs_evaluated": total,
        "direction_accuracy": acc, "weighted_accuracy": w_acc,
        "per_pair": per_pair,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--k-noise", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    res = evaluate(
        args.ckpt, n_rows=args.n_rows, n_repeats=args.n_repeats,
        k_noise=args.k_noise, device=args.device,
        seed=args.seed, max_pairs=args.max_pairs,
    )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
