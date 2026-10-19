"""Quantitative FADO->CAT agreement sweep.

Depth-1 CAT articulation of the outcome Y ("why is Y below/above average") from
each provider's per-feature effect on Y, across the synthetic SCM families with
known graphs. We ask: does FADO's GRAPH-FREE effect estimate drive the SAME CAT
articulation as the oracle-graph estimators (do-Shapley [ref 3], Causal Shapley
[ref 1]) and the ground-truth SCM?

For a depth-1 articulation of Y, CAT's per-feature decision is:
  * excitation vs inhibition of feature j = sign( effect_j ) combined with the
    instance's observed deviations. For a FIXED instance the two providers see
    the same deviations, so their excitation/inhibition labels agree for
    feature j iff sign(effect_j^A) == sign(effect_j^B). Hence the rule-label
    agreement is exactly the effect-SIGN agreement (instance-independent).
  * the dominant ("mostly") cause = argmax_j |effect_j|.

So we report, FADO vs each reference:
  sign_agreement    : fraction of true causal features where FADO and the
                      reference agree on excitation-vs-inhibition (= sign)
  dominant_agreement: fraction of SCMs where FADO and the reference pick the
                      same dominant cause among the true causal features
  articulation_match: fraction of SCMs where the WHOLE depth-1 articulation
                      is identical (all causal-feature signs + dominant agree)
with 95% bootstrap CIs over SCMs. "True causal features" = |ground-truth tau| >
--tau-thresh (decoys with tau~0 have undefined sign and are excluded).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from causal_probe.eval import _make_eval_scm, _load_model
from causal_probe.baselines import BASELINES
from causal_probe.shapley_causal import do_shapley, causal_shapley

FAMILIES = ["linear", "nonlinear", "mlp", "hidden", "mlp_hidden"]
REFS = ["ground_truth", "do_shapley", "causal_shapley"]


def _sign(a):
    return np.sign(a)


def _boot_ci(per_scm_vals, weights=None, n_boot=2000, seed=13):
    """Bootstrap CI of a (weighted) mean over SCMs."""
    v = np.asarray(per_scm_vals, dtype=float)
    ok = np.isfinite(v)
    v = v[ok]
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)[ok]
    if v.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    point = float(np.average(v, weights=w))
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, v.size, v.size)
        draws.append(float(np.average(v[idx], weights=w[idx])))
    return [round(point, 3), round(float(np.percentile(draws, 2.5)), 3),
            round(float(np.percentile(draws, 97.5)), 3)]


def run_family(model, scm_type, *, n_scms, p, n_rows, tau_thresh, device, rng):
    prov = {r: [] for r in REFS}
    prov["fado"] = []
    causal_mask = []
    for _ in range(n_scms):
        scm = _make_eval_scm(scm_type, p, rng, noise="laplace")
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(1 << 62)))
        Xt = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(device)
        yt = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            prov["fado"].append(model(Xt, yt).squeeze(0).cpu().numpy())
        prov["ground_truth"].append(np.asarray(samp.tau, float))
        try:
            prov["do_shapley"].append(np.asarray(do_shapley(scm), float))
            prov["causal_shapley"].append(np.asarray(causal_shapley(scm), float))
        except Exception:
            prov["do_shapley"].append(np.full(p, np.nan))
            prov["causal_shapley"].append(np.full(p, np.nan))
        causal_mask.append(np.abs(np.asarray(samp.tau, float)) > tau_thresh)

    fado = np.stack(prov["fado"])
    masks = np.stack(causal_mask)
    out = {"family": scm_type, "n_scms": n_scms, "p": p,
           "mean_causal_feats_per_scm": round(float(masks.sum(1).mean()), 2),
           "vs": {}}
    for ref in REFS:
        R = np.stack(prov[ref])
        sign_ok_per_scm, dom_ok_per_scm, art_ok_per_scm, wts = [], [], [], []
        for s in range(n_scms):
            m = masks[s]
            if m.sum() == 0 or not np.all(np.isfinite(R[s])):
                sign_ok_per_scm.append(np.nan); dom_ok_per_scm.append(np.nan)
                art_ok_per_scm.append(np.nan); wts.append(0.0)
                continue
            sa = (_sign(fado[s][m]) == _sign(R[s][m]))
            sign_ok_per_scm.append(float(sa.mean()))
            # dominant among causal features
            cf = np.where(m)[0]
            dom_f = cf[np.argmax(np.abs(fado[s][m]))]
            dom_r = cf[np.argmax(np.abs(R[s][m]))]
            dom_ok = float(dom_f == dom_r)
            dom_ok_per_scm.append(dom_ok)
            art_ok_per_scm.append(float(sa.all() and dom_ok == 1.0))
            wts.append(float(m.sum()))
        out["vs"][ref] = {
            "sign_agreement": _boot_ci(sign_ok_per_scm, weights=wts),
            "dominant_agreement": _boot_ci(dom_ok_per_scm),
            "articulation_match": _boot_ci(art_ok_per_scm),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-scms", type=int, default=40)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--tau-thresh", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out-json", default="causal_probe/results/cat_agreement_sweep.json")
    args = ap.parse_args()

    model = _load_model(args.ckpt, device=args.device)
    rng = np.random.default_rng(args.seed)
    results = []
    for fam in FAMILIES:
        rec = run_family(model, fam, n_scms=args.n_scms, p=args.p,
                         n_rows=args.n_rows, tau_thresh=args.tau_thresh,
                         device=args.device, rng=rng)
        results.append(rec)
        print(f"\n=== {fam} (n_scms={args.n_scms}, p={args.p}, "
              f"~{rec['mean_causal_feats_per_scm']} causal feats/SCM) ===")
        print(f"{'FADO vs':<16}{'sign(excit/inhib)':>26}{'dominant cause':>22}"
              f"{'full articulation':>22}")
        for ref in REFS:
            v = rec["vs"][ref]
            def f(x): return f"{x[0]:.2f} [{x[1]:.2f},{x[2]:.2f}]"
            print(f"{ref:<16}{f(v['sign_agreement']):>26}"
                  f"{f(v['dominant_agreement']):>22}{f(v['articulation_match']):>22}")

    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
