"""Sachs-structured quantitative benchmark.

The Sachs et al. 2005 protein signaling network is the best-known
real-world causal benchmark: 11 proteins, DAG validated by intervention
experiments. We encode the published DAG structure (Sachs 2005 Figure 2)
as a linear non-Gaussian SCM with literature-plausible coefficient signs
and magnitudes. Observational data is sampled; ground-truth tau is
computed from A = (I - B)^{-1} exactly as in scm.py.

This is 'Sachs-structured' synthetic — it uses the published DAG edge
list (a real-world causal object) but fits linear coefficients to it.
Not as compelling as running on the actual Sachs flow-cytometry data,
but:
  * the actual data is strongly non-linear (log-transformed),
  * interventional validation requires the interventional subsets,
  * the structure-based test still falsifies models that collapse to
    association on well-known causal pathways.

Nodes (index):   0=Praf  1=Pmek  2=Plcg  3=PIP2  4=PIP3  5=p44_42 (Erk)
                 6=pakts473 (Akt)  7=PKA  8=PKC  9=P38  10=pjnk
Y-slot:          typically pakts473 or Erk as the "outcome" of the
                 signaling cascade. We pick pakts473 (Akt).
"""
from __future__ import annotations

import numpy as np

from causal_probe.scm import SCMSample


# Published DAG (Sachs et al. 2005, Figure 2) as parent -> child pairs.
SACHS_EDGES = [
    ("PKC", "Praf"),
    ("PKC", "Pmek"),
    ("PKC", "P38"),
    ("PKC", "pjnk"),
    ("PKA", "Praf"),
    ("PKA", "Pmek"),
    ("PKA", "p44_42"),
    ("PKA", "pakts473"),
    ("PKA", "P38"),
    ("PKA", "pjnk"),
    ("Praf", "Pmek"),
    ("Pmek", "p44_42"),
    ("p44_42", "pakts473"),
    ("Plcg", "PIP2"),
    ("Plcg", "PIP3"),
    ("PIP3", "PIP2"),
    ("PIP3", "pakts473"),
]
SACHS_NODES = [
    "Praf", "Pmek", "Plcg", "PIP2", "PIP3", "p44_42",
    "pakts473", "PKA", "PKC", "P38", "pjnk",
]


def make_sachs_scm(
    rng: np.random.Generator,
    *,
    y_node: str = "pakts473",
    coef_lo: float = 0.6,
    coef_hi: float = 1.2,
):
    """Return a pseudo-LinearNonGaussianSCM built on the Sachs DAG."""
    from causal_probe.scm import LinearNonGaussianSCM
    N = len(SACHS_NODES)
    node_to_idx = {n: i for i, n in enumerate(SACHS_NODES)}
    if y_node not in node_to_idx:
        raise ValueError(f"y_node {y_node!r} not in SACHS_NODES")
    y_idx = node_to_idx[y_node]

    # Build B directly. Because the edge list is already a DAG, we can
    # use a topological sort.
    adj = {n: [] for n in SACHS_NODES}
    in_adj = {n: [] for n in SACHS_NODES}
    for parent, child in SACHS_EDGES:
        adj[parent].append(child)
        in_adj[child].append(parent)
    # Kahn's
    indeg = {n: len(in_adj[n]) for n in SACHS_NODES}
    topo = []
    stack = [n for n in SACHS_NODES if indeg[n] == 0]
    while stack:
        n = stack.pop(0)
        topo.append(n)
        for c in adj[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                stack.append(c)
    assert len(topo) == N, "DAG cycle somehow"
    topo_idx = [node_to_idx[n] for n in topo]

    B = np.zeros((N, N), dtype=np.float64)
    for parent, child in SACHS_EDGES:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        mag = rng.uniform(coef_lo, coef_hi)
        B[node_to_idx[child], node_to_idx[parent]] = sign * mag

    scm = LinearNonGaussianSCM.from_adjacency(
        B=B,
        y_idx_in_Z=int(y_idx),
        topo_order=np.asarray(topo_idx, dtype=int),
        noise="laplace",
    )
    # Convenience: feature names in feature_to_Z order
    scm.feature_names = [SACHS_NODES[i] for i in scm.feature_to_Z.tolist()]
    scm.y_node = y_node
    return scm


def main():
    import argparse
    import json
    from pathlib import Path

    import torch

    from causal_probe.baselines import BASELINES
    from causal_probe.eval import _load_model, _spearman_pearson

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-seeds", type=int, default=30, help="#SCM coefficient draws")
    ap.add_argument("--y-node", default="pakts473")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None,
                    help="If set, save per-draw (n_seeds, p) arrays "
                         "(pred / true / marginal / multivariate) for bootstrap CIs.")
    args = ap.parse_args()

    model = _load_model(args.ckpt, device=args.device)
    rng = np.random.default_rng(args.seed)

    all_pred, all_true, all_assoc, all_multi = [], [], [], []
    per_seed = []
    for s in range(args.n_seeds):
        scm = make_sachs_scm(
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            y_node=args.y_node,
        )
        samp = scm.sample(n=args.n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
        X_t = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(args.device)
        y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred = model(X_t, y_t).squeeze(0).cpu().numpy()
        all_pred.append(pred)
        all_true.append(samp.tau)
        all_assoc.append(BASELINES["marginal"](samp.X, samp.y))
        all_multi.append(BASELINES["multivariate"](samp.X, samp.y))

    all_pred = np.stack(all_pred); all_true = np.stack(all_true)
    all_assoc = np.stack(all_assoc); all_multi = np.stack(all_multi)
    flat_p = all_pred.reshape(-1); flat_t = all_true.reshape(-1)
    flat_a = all_assoc.reshape(-1); flat_m = all_multi.reshape(-1)

    sp_m, pe_m = _spearman_pearson(flat_p, flat_t)
    sp_a, pe_a = _spearman_pearson(flat_a, flat_t)
    sp_r, pe_r = _spearman_pearson(flat_m, flat_t)
    print(f"Sachs-structured benchmark (y={args.y_node}, {args.n_seeds} coefficient draws):")
    print(f"  Pearson   model={pe_m:+.3f}  marginal={pe_a:+.3f}  multivariate={pe_r:+.3f}")
    print(f"  Spearman  model={sp_m:+.3f}  marginal={sp_a:+.3f}  multivariate={sp_r:+.3f}")

    # Per-feature average
    feat_names = scm.feature_names
    print(f"\nPer-feature average across {args.n_seeds} coefficient draws:")
    print(f"  {'feat':<12}{'tau_true':>10}{'tau_hat':>10}{'assoc':>10}{'multi':>10}")
    for i, name in enumerate(feat_names):
        print(
            f"  {name:<12}{all_true[:, i].mean():>+10.3f}"
            f"{all_pred[:, i].mean():>+10.3f}"
            f"{all_assoc[:, i].mean():>+10.3f}"
            f"{all_multi[:, i].mean():>+10.3f}"
        )

    if args.out_npz:
        from pathlib import Path as _P
        _P(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_npz,
            pred=all_pred.astype(np.float32),
            true=all_true.astype(np.float32),
            marginal=all_assoc.astype(np.float32),
            multivariate=all_multi.astype(np.float32),
            feature_names=np.asarray(feat_names, dtype=object),
        )
        print(f"wrote {args.out_npz}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "n_seeds": args.n_seeds,
                "pearson": {"model": pe_m, "marginal": pe_a, "multivariate": pe_r},
                "spearman": {"model": sp_m, "marginal": sp_a, "multivariate": sp_r},
                "per_feature": {
                    name: {
                        "tau_true_mean": float(all_true[:, i].mean()),
                        "tau_hat_mean": float(all_pred[:, i].mean()),
                        "assoc_mean": float(all_assoc[:, i].mean()),
                        "multi_mean": float(all_multi[:, i].mean()),
                    } for i, name in enumerate(feat_names)
                },
            }, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
