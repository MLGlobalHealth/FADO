"""is FADO's per-feature target tied to the +-1 z-score contrast?

Eq.1 defines the per-feature effect via a fixed symmetric contrast
(do(X_i = +sigma) vs do(X_i = -sigma), i.e. a 2-SD swing). One
asks whether this is a modeling *limitation* — whether soft / continuous /
asymmetric interventions would change the story.

This sweep answers it empirically on the nonlinear families (linear is
analytic: tau_i = A[y,i] * (width) * std_i / std_y, so every symmetric or
asymmetric contrast gives a profile that is an exact rescaling of the
default -> per-feature Pearson r == 1 by construction; nothing to test).

For each nonlinear SCM we recompute the ground-truth per-feature profile
under a range of intervention contrasts by reusing the SCM's own
`_simulate_intervention` with different anchor values:

  sym_c      : do(mu + c*sigma) vs do(mu - c*sigma)   (width sweep; c=1 is default)
  asym_0_+1  : do(mu + sigma)   vs do(mu)             (one-sided, positive)
  asym_-1_0  : do(mu)           vs do(mu - sigma)     (one-sided, negative)
  soft_pm1   : do(X ~ N(mu+sigma, (0.5 sigma)^2)) vs N(mu-sigma, .) (stochastic/soft)

Two things are reported per family:

  target_stability[c] = mean over SCMs of Pearson(tau(contrast), tau(default sym_1.0))
      -> is the *causal profile itself* (what FADO is trained to predict)
         stable under the contrast choice?
  fado_recovery[c]    = pooled Pearson(tau_hat, tau(contrast)) over all (SCM, feature)
      -> does FADO's single learned head recover each contrast's profile
         about equally well?

If both are ~flat across contrasts, the +-1 choice in Eq.1 is a
representative summary, not a restrictive assumption.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from causal_probe.eval import _make_eval_scm, _load_model


CONTRASTS = ["sym_0.5", "sym_1.0", "sym_1.5", "sym_2.0",
             "asym_0_+1", "asym_-1_0", "soft_pm1"]
DEFAULT = "sym_1.0"
FAMILIES = ["nonlinear", "mlp", "mlp_hidden"]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _anchors(kind: str, mu: float, sd: float, n_mc: int, rng: np.random.Generator):
    """Return (val_plus, val_minus) intervention values (scalar or (n_mc,) array)."""
    if kind.startswith("sym_"):
        c = float(kind.split("_")[1])
        return mu + c * sd, mu - c * sd
    if kind == "asym_0_+1":
        return mu + sd, mu
    if kind == "asym_-1_0":
        return mu, mu - sd
    if kind == "soft_pm1":
        vp = mu + sd + 0.5 * sd * rng.standard_normal(n_mc)
        vm = mu - sd + 0.5 * sd * rng.standard_normal(n_mc)
        return vp, vm
    raise ValueError(f"unknown contrast {kind!r}")


def contrast_tau(scm, kind: str, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """Per-feature effect vector under `kind`. Profile is used scale-free."""
    yidx = int(scm.y_idx_in_Z)
    std_y = float(scm._std_Z[yidx])
    feats = scm.feature_to_Z.tolist()
    tau = np.empty(len(feats), dtype=np.float64)
    for i, zi in enumerate(feats):
        zi = int(zi)
        mu = float(scm._mean_Z[zi])
        sd = float(scm._std_Z[zi])
        r_anchor = np.random.default_rng(rng.integers(1 << 62))
        vp, vm = _anchors(kind, mu, sd, n_mc, r_anchor)
        rp = np.random.default_rng(rng.integers(1 << 62))
        rm = np.random.default_rng(rng.integers(1 << 62))
        Zp = scm._simulate_intervention(n_mc, rp, zi, vp)
        Zm = scm._simulate_intervention(n_mc, rm, zi, vm)
        tau[i] = (float(Zp[:, yidx].mean()) - float(Zm[:, yidx].mean())) / std_y
    return tau


def run_family(model, scm_type, *, n_scms, p, n_rows, n_mc, device, rng):
    # per-SCM Pearson(tau_c, tau_default); and pooled (tau_hat, tau_c) across SCMs
    stab = {c: [] for c in CONTRASTS}
    selfcheck = []                       # Pearson(tau(sym_1.0), scm.tau built-in)
    pooled_hat = []                      # list of tau_hat per SCM
    pooled_tau = {c: [] for c in CONTRASTS}
    for _ in range(n_scms):
        scm = _make_eval_scm(scm_type, p, rng, noise="laplace", nonlinear_mc=n_mc)
        samp = scm.sample(n=n_rows, rng=np.random.default_rng(rng.integers(1 << 62)))
        X_t = torch.from_numpy(samp.X.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(samp.y.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            tau_hat = model(X_t, y_t).squeeze(0).cpu().numpy()
        taus = {c: contrast_tau(scm, c, n_mc, rng) for c in CONTRASTS}
        selfcheck.append(_pearson(taus[DEFAULT], samp.tau))
        for c in CONTRASTS:
            stab[c].append(_pearson(taus[c], taus[DEFAULT]))
            pooled_tau[c].append(taus[c])
        pooled_hat.append(tau_hat)
    hat_flat = np.concatenate(pooled_hat)
    out = {
        "family": scm_type,
        "n_scms": n_scms, "p": p, "n_mc": n_mc,
        "selfcheck_default_vs_builtin": round(float(np.nanmean(selfcheck)), 3),
        "target_stability": {c: round(float(np.nanmean(stab[c])), 3) for c in CONTRASTS},
        "fado_recovery": {
            c: round(_pearson(hat_flat, np.concatenate(pooled_tau[c])), 3)
            for c in CONTRASTS
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-scms", type=int, default=40)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-mc", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out-json", default="causal_probe/results/q3_contrast_sweep.json")
    args = ap.parse_args()

    model = _load_model(args.ckpt, device=args.device)
    rng = np.random.default_rng(args.seed)
    results = []
    for fam in FAMILIES:
        rec = run_family(model, fam, n_scms=args.n_scms, p=args.p,
                         n_rows=args.n_rows, n_mc=args.n_mc,
                         device=args.device, rng=rng)
        results.append(rec)
        print(f"\n=== {fam} (n_scms={args.n_scms}, p={args.p}) "
              f"[self-check tau(sym_1.0) vs built-in = {rec['selfcheck_default_vs_builtin']}] ===")
        print(f"{'contrast':<12}{'profile_vs_default':>20}{'FADO_recovery':>16}")
        for c in CONTRASTS:
            flag = "  <- default" if c == DEFAULT else ""
            print(f"{c:<12}{rec['target_stability'][c]:>20.3f}"
                  f"{rec['fado_recovery'][c]:>16.3f}{flag}")

    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
