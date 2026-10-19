"""FADO + CAT showcase: reproduce the "Causal Hans" explanation graph-free.

CAT's flagship running example (Zečević et al., CLeaR 2026, Sec. 3): a 4-variable
SCM over Age (A), Nutrition (N), Health (H), Mobility (M):

    A -> N   (alpha > 0)
    A -> H   (beta  < 0,  |beta| large: age dominates health)
    N -> H   (gamma > 0)
    H -> M   (delta > 0)

For the patient "Hans" (old, good nutrition, poor health, poor mobility) CAT
articulates the why-question "Why is Hans's Mobility below average?" as

    "Mobility is bad because of bad Health, which is mostly due to high Age,
     although Food Habits are good."

We reproduce that articulation by driving CAT with four different providers of
the per-edge causal effect:

    ground_truth   direct structural coefficients (exact)
    fado           FADO's per-feature effect estimate, GRAPH-FREE (queried with
                   each node as the outcome and its non-descendants as inputs;
                   needs a topological order, not the DAG)
    do_shapley     oracle-graph do-Shapley (Jung et al., ICML 2022)          [ref 3]
    causal_shapley oracle-graph Causal Shapley (Heskes et al., NeurIPS 2020)  [ref 1]

Success = FADO's articulation (rendered sentence AND rule fingerprint) matches
the ground-truth / oracle articulation, i.e. FADO recovers the same
excitation/inhibition pattern and the same dominant ("mostly") cause WITHOUT a
given graph. FADO must run in its identifiable regime (non-Gaussian noise), else
Prop. 3 kicks in and the signs are not recovered.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from causal_probe.scm import LinearNonGaussianSCM
from causal_probe.cat_articulation import articulate, render, fingerprint

# Variable <-> Z-index
A, N, H, M = 0, 1, 2, 3
NAMES = {A: "Age", N: "Food Habits", H: "Health", M: "Mobility"}
TOPO = np.array([A, N, H, M])                 # topological order
PARENTS = {M: [H], H: [A, N], A: [], N: []}   # given structure (CAT assumes this)
# Coefficients (paper's sign pattern; |beta| > gamma so Age dominates Health)
COEF = {(N, A): 0.8, (H, A): -1.5, (H, N): 0.6, (M, H): 1.0}


def build_hans_scm(noise: str = "laplace") -> LinearNonGaussianSCM:
    B = np.zeros((4, 4))
    for (child, parent), w in COEF.items():
        B[child, parent] = w
    return LinearNonGaussianSCM.from_adjacency(
        B, y_idx_in_Z=M, topo_order=TOPO, noise=noise)


def _nondesc_inputs(child: int) -> list[int]:
    """Non-descendant candidate causes of `child` (everything topologically
    before it). Uses only a topological order, not the DAG."""
    order = TOPO.tolist()
    return order[: order.index(child)]


# --------------------------------------------------------------------------
# Effect providers: each returns effect(child, parent) -> signed float
# --------------------------------------------------------------------------

def provider_ground_truth(noise: str = "laplace"):
    """Ground-truth per-edge effect on the SAME total-effect (tau) scale the
    oracle providers use, so the columns are comparable. For each child, the
    exact standardized +-1 total effect of each non-descendant on it is the
    sub-SCM's tau (built with the child as outcome). On these direct edges the
    total effect equals what do-Shapley / Causal Shapley return, so ground-truth
    and oracle now agree. (The raw structural coefficients are 1.0 / -1.5 / 0.6;
    they differ from the total effect only by the 2*std standardization, which
    is why the earlier table's 'oracle' looked like it overshot ground truth.)"""
    cache = {child: _true_edge(child, noise) for child in [H, M]}
    return lambda c, p: cache.get(c, {}).get(p, 0.0)


def _true_edge(child: int, noise: str):
    """Exact standardized total effect (tau scale) of each non-descendant on
    `child`, read off the sub-SCM's ground-truth tau labels."""
    cols = _nondesc_inputs(child)
    if not cols:
        return {}
    idx = cols + [child]
    remap = {old: new for new, old in enumerate(idx)}
    Bsub = np.zeros((len(idx), len(idx)))
    for (c, p), w in COEF.items():
        if c in remap and p in remap:
            Bsub[remap[c], remap[p]] = w
    topo_sub = np.array([remap[o] for o in TOPO.tolist() if o in remap])
    sub = LinearNonGaussianSCM.from_adjacency(
        Bsub, y_idx_in_Z=remap[child], topo_order=topo_sub, noise=noise)
    order = [c for c in TOPO.tolist() if remap.get(c, -1) in sub.feature_to_Z.tolist()]
    return {col: float(v) for col, v in zip(order, sub.tau)}


def provider_fado(model, D: np.ndarray, device: str = "cpu"):
    """FADO's graph-free per-edge effect. For each child, query FADO with the
    child as the outcome and its non-descendants as inputs; cache the vector."""
    import torch
    cache: dict[int, dict[int, float]] = {}
    for child in [H, M]:
        cols = _nondesc_inputs(child)
        X = D[:, cols].astype(np.float32)
        y = D[:, child].astype(np.float32)
        Xt = torch.from_numpy(X).unsqueeze(0).to(device)
        yt = torch.from_numpy(y).unsqueeze(0).to(device)
        with torch.no_grad():
            tau = model(Xt, yt).squeeze(0).cpu().numpy()
        cache[child] = {col: float(tau[i]) for i, col in enumerate(cols)}
    return lambda c, p: cache.get(c, {}).get(p, 0.0)


def _oracle_edge(method: str, child: int, noise: str):
    """Oracle per-edge effects: build the sub-SCM with `child` as outcome and
    its non-descendants as features, then run do/causal Shapley on it."""
    from causal_probe.shapley_causal import ORACLE_GRAPH_BASELINES as SHAP
    cols = _nondesc_inputs(child)
    if not cols:
        return {}
    idx = cols + [child]                       # sub-DAG node set (keep order)
    remap = {old: new for new, old in enumerate(idx)}
    Bsub = np.zeros((len(idx), len(idx)))
    for (c, p), w in COEF.items():
        if c in remap and p in remap:
            Bsub[remap[c], remap[p]] = w
    topo_sub = np.array([remap[o] for o in TOPO.tolist() if o in remap])
    sub = LinearNonGaussianSCM.from_adjacency(
        Bsub, y_idx_in_Z=remap[child], topo_order=topo_sub, noise=noise)
    vec = SHAP[method](sub)                     # length = #features, feature order
    feat_cols = [i for i in idx if i != child]  # original col ids, feature order
    # sub.feature_to_Z orders features by topo rank; map back to original cols
    order = [c for c in TOPO.tolist() if remap.get(c, -1) in sub.feature_to_Z.tolist()]
    return {col: float(v) for col, v in zip(order, vec)}


def provider_oracle(method: str, noise: str):
    cache = {child: _oracle_edge(method, child, noise) for child in [H, M]}
    return lambda c, p: cache.get(c, {}).get(p, 0.0)


def run(providers: dict, hans_obs: dict, mu: dict) -> dict:
    out = {}
    for name, eff in providers.items():
        art = articulate(M, lambda n: PARENTS[n], eff, hans_obs, mu)
        out[name] = {
            "sentence": render(art, names=NAMES),
            "fingerprint": fingerprint(art),
            "effects": {f"{NAMES[c]}<-{NAMES[p]}": round(eff(c, p), 3)
                        for c in (M, H) for p in PARENTS[c]},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="FADO checkpoint (omit to skip FADO)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--noise", default="laplace")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out-json", default="causal_probe/results/cat_hans.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scm = build_hans_scm(noise=args.noise)

    # Hans: old (A high), good nutrition (N high) -> the SCM drives H, M low.
    hans_std = {A: 1.5, N: 1.0}
    hans_std[H] = COEF[(H, A)] * hans_std[A] + COEF[(H, N)] * hans_std[N]  # < 0
    hans_std[M] = COEF[(M, H)] * hans_std[H]                              # < 0
    mu = {A: 0.0, N: 0.0, H: 0.0, M: 0.0}  # standardized population means

    providers = {"ground_truth": provider_ground_truth(args.noise)}

    if args.ckpt:
        from causal_probe.eval import _load_model
        model = _load_model(args.ckpt, device=args.device)
        samp = scm.sample(n=args.n_rows, rng=rng)
        # full standardized data matrix in [A, N, H, M] order
        D = np.column_stack([samp.X[:, 0], samp.X[:, 1], samp.X[:, 2], samp.y])
        providers["fado"] = provider_fado(model, D, device=args.device)

    providers["do_shapley"] = provider_oracle("do_shapley", args.noise)
    providers["causal_shapley"] = provider_oracle("causal_shapley", args.noise)

    results = run(providers, hans_std, mu)

    ref = results["ground_truth"]["fingerprint"]
    print(f"\nCAT's published articulation for 'Why is Hans's Mobility below average?':")
    print('  "Mobility is bad because of bad Health, which is mostly due to high '
          'Age, although Food Habits are good."\n')
    for name, r in results.items():
        match = "  [matches ground-truth rules]" if r["fingerprint"] == ref else \
                "  [DIFFERS from ground-truth]"
        tag = "" if name == "ground_truth" else match
        print(f"[{name}]{tag}")
        print(f"   {r['sentence']}")
        print(f"   effects: {r['effects']}\n")

    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
