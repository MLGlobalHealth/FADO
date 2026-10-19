"""FADO vs a deep structural causal model (NOTEARS-MLP).

NOTEARS-MLP (Zheng et al. 2020, "Learning Sparse Nonparametric DAGs") learns a
nonlinear SCM from observational data: an MLP structural equation per node under
a continuous acyclicity constraint. We read off the same per-feature
interventional effect FADO targets by simulating do(X_i=+1) vs do(X_i=-1)
through the *learned* SCM (fix X_i, propagate to its descendants via the learned
MLPs, average Y), and compare Pearson r with the true tau on the SAME synthetic
SCMs FADO is evaluated on. One fit yields all p features (no per-column loop).
"""
from __future__ import annotations
import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ----------------------------- NOTEARS-MLP (deep SCM) ----------------------
def _dag_from_W(W, thr=0.3):
    W = np.asarray(W)
    A = (np.abs(W) > thr).astype(int)
    np.fill_diagonal(A, 0)
    return A  # A[i,j]=1 => edge i->j


def _topo(A):
    d = A.shape[0]
    indeg = A.sum(0).astype(int)
    order, stack = [], [i for i in range(d) if indeg[i] == 0]
    A2 = A.copy()
    while stack:
        n = stack.pop()
        order.append(n)
        for j in range(d):
            if A2[n, j]:
                A2[n, j] = 0
                indeg[j] -= 1
                if indeg[j] == 0:
                    stack.append(j)
    if len(order) < d:                       # cycle fallback
        order += [i for i in range(d) if i not in order]
    return order


def _descendants(A, i):
    d = A.shape[0]
    seen, stack = set(), [i]
    while stack:
        n = stack.pop()
        for j in range(d):
            if A[n, j] and j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def notears_mlp_tau(X, y, hidden=10, lambda1=0.01, lambda2=0.01, max_iter=50):
    import torch
    from notears.nonlinear import NotearsMLP, notears_nonlinear
    Z = np.column_stack([X, y]).astype(np.float64)
    Z = (Z - Z.mean(0)) / Z.std(0).clip(min=1e-9)
    n, d = Z.shape
    yi = d - 1
    _prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)           # NOTEARS package assumes double
    try:
        model = NotearsMLP(dims=[d, hidden, 1], bias=True)
        notears_nonlinear(model, Z, lambda1=lambda1, lambda2=lambda2,
                          max_iter=max_iter, w_threshold=0.0)
    finally:
        torch.set_default_dtype(_prev)
    A = _dag_from_W(model.fc1_to_adj(), thr=0.3)
    order = _topo(A)

    dtype = next(model.parameters()).dtype     # NOTEARS trains in float64

    def do(i, v):
        z = torch.tensor(Z.copy(), dtype=dtype)
        z[:, i] = v
        desc = _descendants(A, i)
        with torch.no_grad():
            for j in order:
                if j in desc:
                    z[:, j] = model(z)[:, j]
        return float(z[:, yi].mean())

    tau = np.full(X.shape[1], np.nan)
    for i in range(X.shape[1]):
        try:
            tau[i] = do(i, +1.0) - do(i, -1.0)
        except Exception:
            pass
    return tau


# ----------------------------- FADO ----------------------------------------
def fado_tau(model, X, y, device="cpu", p_pad=15, seed=0):
    """FADO per-feature tau; padded to the foundation's p=15 training width
    (matching the paper's foundation evals). Returns the p real-feature taus."""
    import torch
    p = X.shape[1]
    Xs = (X - X.mean(0)) / X.std(0).clip(min=1e-9)
    ys = (y - y.mean()) / (y.std() or 1)
    if p_pad and p_pad > p:                       # iid-noise padding, as in eval.py
        rng = np.random.default_rng(seed)
        Xs = np.column_stack([Xs, rng.standard_normal((Xs.shape[0], p_pad - p))])
    with torch.no_grad():
        pred = model(torch.tensor(Xs[None], dtype=torch.float32, device=device),
                     torch.tensor(ys[None], dtype=torch.float32, device=device)
                     ).squeeze(0).cpu().numpy()
    return pred[:p]                               # drop padded columns


def _scms(family, p, n, seed0):
    out = []
    for s in range(n):
        rng = np.random.default_rng(seed0 + s)
        if family == "linear":
            from causal_probe.scm import LinearNonGaussianSCM
            out.append(LinearNonGaussianSCM(p=p, rng=rng, noise="laplace"))
        elif family == "nonlinear":
            from causal_probe.scm_nonlinear import NonlinearSCM
            out.append(NonlinearSCM(p=p, rng=rng, n_mc=2048))
        elif family == "mlp":
            from causal_probe.scm_mlp import MLPSCM
            out.append(MLPSCM(p=p, rng=rng, n_mc=2048))
        elif family == "hidden":
            from causal_probe.scm_hidden import LinearNonGaussianSCMHidden
            out.append(LinearNonGaussianSCMHidden(p=p, rng=rng, noise="laplace"))
    return out


def _pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="causal_probe/results/probe_intersample_L_seed0.ckpt")
    ap.add_argument("--families", default="linear,nonlinear,mlp,hidden")
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-scms", type=int, default=40)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out-json", default="causal_probe/results/dscm_compare.json")
    args = ap.parse_args()
    from causal_probe.eval import _load_model
    model = _load_model(args.ckpt, device=args.device)

    out = []
    for fam in args.families.split(","):
        fam = fam.strip()
        scms = _scms(fam, args.p, args.n_scms, 1000 * args.seed)
        fado_all, dscm_all, true_all = [], [], []
        rng = np.random.default_rng(args.seed)
        for k, scm in enumerate(scms):
            samp = scm.sample(n=args.n_rows, rng=np.random.default_rng(rng.integers(0, 2**31)))
            X, y, tau = samp.X.astype(np.float64), samp.y.astype(np.float64), samp.tau
            fado_all.append(fado_tau(model, X, y, args.device))
            dscm_all.append(notears_mlp_tau(X, y))
            true_all.append(tau)
        F = np.concatenate(fado_all); D = np.concatenate(dscm_all); T = np.concatenate(true_all)
        rec = {"family": fam, "n_scms": args.n_scms,
               "fado_pearson": _pearson(F, T),
               "notears_mlp_pearson": _pearson(D, T),
               "notears_valid_frac": float(np.isfinite(D).mean())}
        out.append(rec)
        print(f"{fam:>10} | FADO r={rec['fado_pearson']:.2f}  "
              f"NOTEARS-MLP (deep SCM) r={rec['notears_mlp_pearson']:.2f}  "
              f"(valid {100*rec['notears_valid_frac']:.0f}%)")
    print(f"\nMEAN | FADO r={np.nanmean([r['fado_pearson'] for r in out]):.2f}  "
          f"NOTEARS-MLP r={np.nanmean([r['notears_mlp_pearson'] for r in out]):.2f}")
    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
