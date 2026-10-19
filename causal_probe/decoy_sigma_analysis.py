"""Does FADO's row-bootstrap sigma flag its own decoy false positives?

Hypothesis. The single-checkpoint reliability signal
(causal_probe/bootstrap_uncertainty.py: resample the context rows B times,
sigma_j = sd over bootstrap replicates of tau_hat_j) is calibrated on held-out
synthetic SCMs — abstaining on the top-sigma quintile cuts error sharply. It
has never been applied to the constructions where FADO is KNOWN to emit false
positives: confounder-children and proxies on the ACIC/IHDP semi-synthetic
decoy benchmark, and DAG-null sensor decoys in the causal-chambers light
tunnel. If the model's uncertainty is elevated on exactly those features,
selective prediction turns a known failure mode into an abstention; if sigma
is flat (confidently wrong), the "don't trust this" signal does not transfer
from noise-driven error to structurally-driven error, which is itself a
finding.

Decision rule. At 20% abstention (drop the top-sigma 20% of features, pooled
within a part), report per decoy class the fraction of false-positive mass
(sum of |tau_hat| over the class; truth is 0 there by construction) removed,
against the fraction of true-causal features lost. Sigma "works" for a decoy
class if it removes a clearly larger share of that class's FP mass than the
20% a random abstention would remove, at a sub-20% loss of true-causal
features; Spearman(sigma, |tau_hat - tau_true|) pooled per part is the
accompanying calibration number.

Constructions.
  * Part "acic": the semi-synthetic decoy benchmark of
    causal_probe/semisynth_acic_fp.py / semi_synth_real.make_benchmark —
    real covariates (ACIC-2016 x.csv, or the IHDP covariate matrix via
    --covariates ihdp when the ACIC csv is not on disk), synthetic outcome
    with known tau, decoy columns proxy / leak / noise, and (in the
    hidden-confounder regime) observed children of an unobserved confounder.
    --n-scms instances per outcome family x both regimes, same RNG structure
    as semisynth_acic_fp.run_regime (colrng from --seed, per-instance
    benchmark rng 1000*seed+s, row-subsample rng default_rng(s)).
  * Part "chambers": the light-tunnel configs of
    causal_probe/causal_chambers_eval.py (Y=vis_3 and Y=vis_1), with the
    hardware-verified ground truth built by build_ground_truth (CPU-only).
    Never-intervened sensor readouts are DAG-null decoys (tau_true = 0 by the
    published chamber DAG); intervened features split into true-causal
    (|tau_true| > SIGN_THRESHOLD) and intervened-null. Each of --n-scms
    context draws (512 rows, permuted columns) is one construction instance.

Per construction: FADO point tau_hat on the context, plus sigma = sd over
--n-boot row-resamples of the same context (causal_probe.bootstrap_uncertainty
._bootstrap_preds, batched through the model in chunks).

Outputs (JSON + printed tables): per feature-class mean |tau_hat| / mean
sigma; pooled Spearman/Pearson(sigma, |tau_hat - tau_true|); abstention
curves at 0/10/20/30/50% (remaining FP mass per decoy class, true-causal
features lost); and the 20%-abstention decision table.

Usage:
  uv run python -m causal_probe.decoy_sigma_analysis --dry-run \
      --covariates ihdp --ihdp-dir data/ihdp
  python -m causal_probe.decoy_sigma_analysis \
      --ckpt <ckpt> --device cuda --parts all --out causal_probe/results/decoy_sigma.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.bootstrap_uncertainty import _bootstrap_preds

ABSTAIN_LEVELS = [0, 10, 20, 30, 50]
DECISION_LEVEL = 20
# Feature classes. Decoy classes have tau_true = 0 by construction.
DECOY_CLASSES = ["proxy", "leak", "noise", "conf_child", "real_null",
                 "chamber_dag_null", "chamber_interv_null"]
CAUSAL_CLASS = "true_causal"


# ---------------------------------------------------------------------------
# Construction builders. Each yields dicts with:
#   X_ctx (n_rows, p) standardized context, y_ctx (n_rows,),
#   classes list[str] (per feature), tau_true (p,), meta dict
# ---------------------------------------------------------------------------


def _semisynth_classes(kinds: list[str], names: list[str]) -> list[str]:
    out = []
    for k, nm in zip(kinds, names):
        if k == "real":
            out.append(CAUSAL_CLASS if "*" in nm else "real_null")
        elif k == "conf":
            out.append("conf_child")
        else:  # proxy / leak / noise
            out.append(k)
    return out


def build_semisynth_constructions(X: np.ndarray, args) -> list[dict]:
    """ACIC/IHDP semi-synthetic decoy instances.

    RNG structure copied from semisynth_acic_fp.run_regime (which we cannot
    call directly: it runs model inference inline with no bootstrap hook).
    Per family: colrng = default_rng(seed); instance s uses benchmark rng
    default_rng(1000*seed + s), row-subsample rng default_rng(s), and
    standardizes X_full/y over ALL rows before subsampling — identical to the
    FP-benchmark protocol, so tau_hat here sits on the same benchmark as the
    reference FP columns (up to n_scms).
    """
    from causal_probe.semi_synth_real import make_benchmark

    out = []
    for hidden in (False, True):
        regime = "hidden_confounder" if hidden else "no_confounding"
        for family in (f.strip() for f in args.families.split(",")):
            colrng = np.random.default_rng(args.seed)
            for s in range(args.n_scms):
                cols = colrng.choice(X.shape[1], args.p_real_cap, replace=False)
                b = make_benchmark(X[:, cols], np.random.default_rng(1000 * args.seed + s),
                                   n_causal=args.n_causal, outcome_family=family,
                                   hidden_confound=hidden)
                X_full, y = b["X_full"], b["y"]
                n = X_full.shape[0]
                rng = np.random.default_rng(s)
                idx = (rng.choice(n, args.n_rows, replace=False)
                       if n > args.n_rows else np.arange(n))
                Xs = (X_full - X_full.mean(0)) / X_full.std(0).clip(min=1e-9)
                ys = (y - y.mean()) / (y.std() or 1)
                out.append({
                    "X_ctx": Xs[idx], "y_ctx": ys[idx],
                    "classes": _semisynth_classes(b["kinds"], b["feature_names"]),
                    "tau_true": np.asarray(b["true_tau"], dtype=np.float64),
                    "meta": {"part": "acic", "regime": regime, "family": family,
                             "scm": s},
                })
    return out


def build_chamber_constructions(args) -> list[dict]:
    """Light-tunnel configs; one construction per (config, context draw).

    Ground truth + data loading via causal_chambers_eval (CPU-only). Columns
    of each context are randomly permuted (guard against position artifacts)
    and un-permuted in the stored construction via the recorded perm.
    """
    from causal_probe.causal_chambers_eval import (
        LT_CONFIGS, SIGN_THRESHOLD, _experiment_df, _load_dataset,
        _standardize, build_ground_truth, REFERENCE_EXPERIMENT)

    dataset = _load_dataset(args.data_dir)
    out = []
    for label, outcome, features in LT_CONFIGS:
        truth = build_ground_truth(dataset, outcome, features)
        tau_true = np.array([r["tau_true"] for r in truth["rows"]])
        classes = []
        for r in truth["rows"]:
            if not r["intervened"]:
                classes.append("chamber_dag_null")
            elif abs(r["tau_true"]) > SIGN_THRESHOLD:
                classes.append(CAUSAL_CLASS)
            else:
                classes.append("chamber_interv_null")
        ref = _experiment_df(dataset, REFERENCE_EXPERIMENT)
        Xs, ys = _standardize(ref[features].values.astype(np.float64),
                              ref[outcome].values.astype(np.float64))
        n_total, p = Xs.shape
        rng = np.random.default_rng(args.seed)
        n_draws = args.chamber_draws if args.chamber_draws else args.n_scms
        for d in range(n_draws):
            idx = (rng.choice(n_total, size=args.n_rows, replace=False)
                   if n_total > args.n_rows else np.arange(n_total))
            perm = rng.permutation(p)
            out.append({
                "X_ctx": Xs[idx][:, perm], "y_ctx": ys[idx],
                "perm": perm,  # model outputs are un-permuted before storage
                "classes": classes, "tau_true": tau_true,
                "meta": {"part": "chambers", "config": label,
                         "outcome": outcome, "features": features, "draw": d},
            })
    return out


# ---------------------------------------------------------------------------
# Model pass: point tau_hat + bootstrap sigma per construction
# ---------------------------------------------------------------------------


def _point_pred(model, X: np.ndarray, y: np.ndarray, device: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32)[None]).to(device),
                     torch.from_numpy(y.astype(np.float32)[None]).to(device)
                     ).squeeze(0).cpu().numpy().astype(np.float64)


def run_constructions(model, constructions: list[dict], args) -> list[dict]:
    """Attach tau_hat (point), boot_mean, sigma to each construction record."""
    rng = np.random.default_rng(args.seed + 1)
    records = []
    for i, c in enumerate(constructions):
        X, y = c["X_ctx"], c["y_ctx"]
        tau_hat = _point_pred(model, X, y, args.device)
        boot = _bootstrap_preds(model, X, y, args.n_boot,
                                np.random.default_rng(rng.integers(0, 2**31)),
                                args.device)
        b_mean, sigma = boot.mean(axis=0), boot.std(axis=0, ddof=1)
        if "perm" in c:  # chambers contexts are column-permuted; undo
            inv = np.empty_like(c["perm"]); inv[c["perm"]] = np.arange(len(c["perm"]))
            tau_hat, b_mean, sigma = tau_hat[inv], b_mean[inv], sigma[inv]
        records.append({"meta": c["meta"], "classes": c["classes"],
                        "tau_true": c["tau_true"].tolist(),
                        "tau_hat": tau_hat.tolist(),
                        "boot_mean": b_mean.tolist(),
                        "sigma": sigma.tolist()})
        if (i + 1) % 10 == 0 or i + 1 == len(constructions):
            print(f"  [{i + 1}/{len(constructions)}] {c['meta']}", flush=True)
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _flatten(records: list[dict]):
    cls, th, tt, sg = [], [], [], []
    for r in records:
        cls += r["classes"]
        th += r["tau_hat"]
        tt += r["tau_true"]
        sg += r["sigma"]
    return (np.array(cls), np.array(th, dtype=np.float64),
            np.array(tt, dtype=np.float64), np.array(sg, dtype=np.float64))


def analyze(records: list[dict], label: str) -> dict:
    from scipy.stats import pearsonr, spearmanr

    cls, tau_hat, tau_true, sigma = _flatten(records)
    err = np.abs(tau_hat - tau_true)
    classes_present = [c for c in [CAUSAL_CLASS] + DECOY_CLASSES if (cls == c).any()]

    per_class = {}
    for c in classes_present:
        m = cls == c
        per_class[c] = {"n": int(m.sum()),
                        "mean_abs_tau_hat": float(np.abs(tau_hat[m]).mean()),
                        "mean_sigma": float(sigma[m].mean()),
                        "mean_abs_err": float(err[m].mean())}

    res = {"label": label, "n_features": int(cls.size), "per_class": per_class,
           "spearman_sigma_err": float(spearmanr(sigma, err)[0]),
           "pearson_sigma_err": float(pearsonr(sigma, err)[0])}

    # Abstention: drop the top-sigma k features (pooled within this part).
    order = np.argsort(-sigma)  # descending sigma
    curve = {}
    for lvl in ABSTAIN_LEVELS:
        k = int(round(lvl / 100 * cls.size))
        dropped = np.zeros(cls.size, dtype=bool)
        dropped[order[:k]] = True
        row = {"n_dropped": k}
        for c in classes_present:
            m = cls == c
            mass = np.abs(tau_hat[m]).sum()
            kept_mass = np.abs(tau_hat[m & ~dropped]).sum()
            row[c] = {
                "frac_features_dropped": float(dropped[m].mean()),
                "frac_mass_removed": float(1 - kept_mass / mass) if mass > 0 else 0.0,
                "remaining_mean_abs_tau_hat":
                    (float(np.abs(tau_hat[m & ~dropped]).mean())
                     if (m & ~dropped).any() else float("nan")),
            }
        curve[str(lvl)] = row
    res["abstention_curve"] = curve

    d = curve[str(DECISION_LEVEL)]
    res["decision_at_20"] = {
        "true_causal_features_lost":
            d[CAUSAL_CLASS]["frac_features_dropped"] if CAUSAL_CLASS in d else float("nan"),
        "fp_mass_removed": {c: d[c]["frac_mass_removed"]
                            for c in classes_present if c != CAUSAL_CLASS},
    }
    return res


def _print_analysis(res: dict) -> None:
    print(f"\n=== {res['label']} ({res['n_features']} feature observations) ===")
    print(f"  Spearman(sigma, |err|) = {res['spearman_sigma_err']:+.3f}   "
          f"Pearson = {res['pearson_sigma_err']:+.3f}")
    print(f"  {'class':<20}{'n':>6}{'mean|tau_hat|':>15}{'mean sigma':>12}{'mean|err|':>11}")
    for c, v in res["per_class"].items():
        print(f"  {c:<20}{v['n']:>6}{v['mean_abs_tau_hat']:>15.4f}"
              f"{v['mean_sigma']:>12.4f}{v['mean_abs_err']:>11.4f}")
    print(f"\n  abstention (drop top-sigma X%): frac of class FP mass removed"
          f" [frac of class features dropped]")
    classes = [c for c in res["per_class"] if c != CAUSAL_CLASS]
    hdr = "  {:>4}".format("X%") + f"{'causal lost':>13}" + "".join(f"{c[:14]:>16}" for c in classes)
    print(hdr)
    for lvl in ABSTAIN_LEVELS:
        row = res["abstention_curve"][str(lvl)]
        causal = (row[CAUSAL_CLASS]["frac_features_dropped"]
                  if CAUSAL_CLASS in row else float("nan"))
        cells = "".join(
            f"  {row[c]['frac_mass_removed']:>5.2f}[{row[c]['frac_features_dropped']:>4.2f}]  "
            for c in classes)
        print(f"  {lvl:>4}{causal:>13.2f}{cells}")
    d20 = res["decision_at_20"]
    print(f"  decision@20%: true-causal features lost = "
          f"{d20['true_causal_features_lost']:.2f}; FP mass removed: "
          + "  ".join(f"{c}={v:.2f}" for c, v in d20["fp_mass_removed"].items()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_covariates(args) -> np.ndarray:
    if args.covariates == "acic":
        from causal_probe.semisynth_acic_fp import load_acic2016
        X = load_acic2016(args.acic_path)
        print(f"ACIC-2016 covariates: {X.shape[0]} x {X.shape[1]} usable cols")
    else:
        from causal_probe.ihdp_decoy_eval import load_ihdp_covariates
        X, identical = load_ihdp_covariates(args.ihdp_dir)
        print(f"IHDP covariates: {X.shape[0]} x {X.shape[1]} usable cols "
              f"(identical across replicates: {identical})")
    if args.p_real_cap > X.shape[1]:
        print(f"p_real_cap {args.p_real_cap} > p={X.shape[1]}; clipping.")
        args.p_real_cap = X.shape[1]
    return X


def _dry_run_report(name: str, constructions: list[dict]) -> None:
    counts: dict[str, int] = {}
    for c in constructions:
        for k in c["classes"]:
            counts[k] = counts.get(k, 0) + 1
    print(f"[dry-run] part={name}: {len(constructions)} constructions, "
          f"{sum(counts.values())} feature observations")
    for k in [CAUSAL_CLASS] + DECOY_CLASSES:
        if k in counts:
            print(f"    {k:<20}{counts[k]:>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None,
                    help="FADO checkpoint (required unless --dry-run)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-scms", type=int, default=10,
                    help="SCM instances per family x regime (part acic) and "
                         "context draws per config (part chambers, unless "
                         "--chamber-draws)")
    ap.add_argument("--n-boot", type=int, default=100,
                    help="row-bootstrap resamples per construction")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--out", default="causal_probe/results/decoy_sigma.json")
    ap.add_argument("--parts", default="all", choices=["acic", "chambers", "all"])
    ap.add_argument("--dry-run", action="store_true",
                    help="build constructions, print per-class counts, exit 0 "
                         "(no checkpoint / GPU)")
    # part acic (semisynth) options
    ap.add_argument("--covariates", default="acic", choices=["acic", "ihdp"],
                    help="covariate source for the semisynth part")
    ap.add_argument("--acic-path", default="data/acic2016/x.csv")
    ap.add_argument("--ihdp-dir", default="data/ihdp")
    ap.add_argument("--families", default="linear,mixed_nonlinear,interaction")
    ap.add_argument("--p-real-cap", type=int, default=10)
    ap.add_argument("--n-causal", type=int, default=3)
    # part chambers options
    ap.add_argument("--data-dir", default="/tmp/causalchamber_data")
    ap.add_argument("--chamber-draws", type=int, default=None,
                    help="context draws per chamber config (default: --n-scms)")
    args = ap.parse_args()

    if args.ckpt is None and not args.dry_run:
        ap.error("--ckpt is required unless --dry-run")

    parts = ["acic", "chambers"] if args.parts == "all" else [args.parts]
    constructions: dict[str, list[dict]] = {}
    if "acic" in parts:
        X = _load_covariates(args)
        constructions["acic"] = build_semisynth_constructions(X, args)
    if "chambers" in parts:
        constructions["chambers"] = build_chamber_constructions(args)

    if args.dry_run:
        for name, cons in constructions.items():
            _dry_run_report(name, cons)
        print("[dry-run] OK — exiting before model load")
        return

    from causal_probe.eval import _load_model
    model = _load_model(args.ckpt, device=args.device)
    print(f"Loaded {args.ckpt}; B={args.n_boot} row-resamples per construction")

    all_records: dict[str, list[dict]] = {}
    analyses: dict[str, dict] = {}
    for name, cons in constructions.items():
        print(f"\n--- part {name}: {len(cons)} constructions ---")
        recs = run_constructions(model, cons, args)
        all_records[name] = recs
        analyses[name] = analyze(recs, name)
        _print_analysis(analyses[name])
    if len(all_records) > 1:
        pooled = [r for recs in all_records.values() for r in recs]
        analyses["pooled"] = analyze(pooled, "pooled (all parts)")
        _print_analysis(analyses["pooled"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()},
                   "analyses": analyses, "records": all_records}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
