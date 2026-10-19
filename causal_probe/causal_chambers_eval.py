"""Causal Chambers (light tunnel) benchmark: FADO vs real physical do-effects.

First test of the probe on a real physical system with hardware-verified
interventional ground truth: the light tunnel of Gamella, Peters & Buhlmann,
"Causal chambers as a real-world physical testbed for AI methodology",
Nature Machine Intelligence 2025 (doi:10.1038/s42256-024-00964-x).

Data (downloaded via the ``causalchamber`` PyPI package):
  dataset ``lt_interventions_standard_v1``
  * ``uniform_reference`` (N=10,000): every actuator is exogenously SET by
    the chamber, uniformly at random and independently of everything else
    (red/green/blue ~ U{0..85}, pol_1/pol_2 ~ U{-15..25} deg, auxiliary
    LEDs l_ij ~ U{0..170}).  This is FADO's "observational" input, and --
    because the actuators are randomized hardware settings, not passively
    observed variables -- it is simultaneously a randomized experiment:
    E[Y | X_i = x] under this protocol equals E[Y | do(X_i = x)] with the
    remaining actuators at their reference distributions.
  * single-target intervention experiments ``uniform_<f>_mid`` /
    ``uniform_<f>_strong`` (N=1,000 each): the target actuator is moved to
    a disjoint, higher range (e.g. red ~ U{86..170} / U{171..255}) while
    all other actuators keep their reference distributions.

Estimand.  FADO reports tau_i ~= ( E[Y | do(X_i = mu_i + sd_i)]
- E[Y | do(X_i = mu_i - sd_i)] ) / sd_Y with mu/sd from the observational
data.  Ground truth here, per intervened actuator f:

  1. mu_f, sd_f, mu_Y, sd_Y from ``uniform_reference``.
  2. Pool rows of {reference, <f>_mid, <f>_strong}.  In every one of these
     experiments f is exogenously set and all *other* actuators follow the
     same reference distributions, so the pooled conditional mean
     m(x) = E[Y | f = x] is an interventional dose-response curve.
  3. tau_true[f] = ( m(mu_f + sd_f) - m(mu_f - sd_f) ) / sd_Y, with m(x0)
     estimated by a local mean over pooled rows with |f - x0| <= 0.25*sd_f
     (hundreds-to-thousands of rows per window; window counts are stored
     in the output JSON as n_lo/n_hi).
     NOTE the evaluation points mu_f +/- sd_f lie inside the reference
     support, so the local means draw (almost) exclusively on reference
     rows -- the ground truth rests on the reference protocol itself being
     a randomized experiment.  The mid/strong single-target experiments
     contribute (a) the linear-extrapolation robustness number below,
     (b) hardware verification of nulls (e.g. pol interventions move vis_3
     by -1.8 sd but vis_1 by ~0.0 sd), reported in the "shift" table.
  4. Robustness: tau_true_linear[f] = OLS fit of Y on f using ONLY the
     mid/strong intervention rows, evaluated at mu_f +/- sd_f
     (an extrapolation; it disagrees with tau_true where the physical
     dose-response is nonlinear, e.g. Malus' cos^2 law for the polarizers).

Features that are never intervened on (current, angle_1/2, ir_*/vis_*) are
sensor readouts.  The chamber's known ground-truth DAG (shipped with the
package: ``causalchamber.ground_truth.graph(chamber="lt",
configuration="standard")``, mirroring Fig. 3 / App. II of the paper) gives
them no outgoing edges -- they cannot cause anything -- so tau_true = 0.
This is asserted programmatically at runtime.  They are deliberately
included as decoys: e.g. vis_1 (same light measured upstream of the
polarizers) has Pearson corr ~0.90 with Y=vis_3 yet zero causal effect.

Two configurations (both light tunnel, p = 10 features + Y):
  * Y = vis_3 (visible-light intensity after both polarizers).
    features = red, green, blue, pol_1, pol_2, l_31, l_32 (intervened;
    truth measured) + current, angle_1, vis_1 (DAG-null sensor decoys).
    Nonzero truth: red/green/blue ~ +0.5/+0.9/+1.7 sd.  pol_1/pol_2 truth
    ~ 0 at +/-1 sd (cos^2 is flat near alignment) despite huge mid/strong
    shifts; l_31/l_32 (auxiliary IR/UV LEDs at sensor 3) ~ 0.
  * Y = vis_1 (visible-light intensity upstream of both polarizers).
    features = red, green, blue, pol_1, pol_2, l_11, l_12 (intervened)
    + current, angle_1, ir_2 (DAG-null sensor decoys).  Here pol_1/pol_2
    are hardware-verified nulls (polarizers sit downstream of sensor 1),
    while red/green/blue remain strong -- the falsification twin of the
    vis_3 run.

Metrics per configuration: Pearson r(tau_hat, tau_true) over the 10
features, sign agreement on features with |tau_true| > 0.1, MAE; mean +/-
sd over --n-seeds row subsamples (n_rows context rows, feature columns
permuted per seed).  Marginal/multivariate baselines from
causal_probe.baselines are computed on the full standardized reference.

Comparator suite (--methods full).  Every graph-free method from
``causal_probe.run_baseline.METHODS`` (marginal, ridge, permutation, shap,
lingam, pc, ges, fci, notears, doubleml, causal_forest, causalpfn) plus the
Do-PFN per-column baseline runs ONCE, deterministically, on the same
standardized (X, y) the FADO pass draws its contexts from -- but on ALL
n=10,000 observational rows.  PROTOCOL ASYMMETRY, deliberate and recorded
in the output JSON: the comparators are not context-limited, whereas FADO
sees --n-rows-row subsamples averaged over --n-seeds draws.  SHAP is
TreeSHAP over a LightGBM regressor fit to the full (X, y) (mean |SHAP| per
feature, signed by the marginal correlation, /sd_y; see
``run_baseline.shap_tau``); permutation importance is signed permutation
importance of a LightGBM fit on a held-out split (``permutation_tau``).
Slow comparators (notears, doubleml, causal_forest, causalpfn, dopfn) are
dropped by --skip-slow; dopfn additionally needs --dopfn-repo (and
realistically a GPU, as does causalpfn).  Methods that error on this data
(e.g. discovery methods on near-deterministic sensor mechanisms) are
caught and recorded as ``error`` in the JSON.  --comparators-only produces
the whole comparator table with no checkpoint / GPU.

Graph-oracle skylines.  Two comparators receive the chamber's PUBLISHED
ground-truth DAG (``causalchamber.ground_truth.graph("lt", "standard")``)
-- information no data-driven method has:

  * backdoor_linear / backdoor_rf: for each feature X_i, adjust for its
    graph parents Pa(X_i) -- always a valid backdoor adjustment set when
    every parent is measured, which holds here (every DAG node is a
    recorded column of the reference experiment; constant columns are
    dropped, as a variable that never varies cannot open a backdoor path
    in this data).  tau_i = (E[Y|do(X_i=mu+sd)] - E[Y|do(X_i=mu-sd)])/sd_Y
    is estimated by regression adjustment on the observational reference
    data: OLS on standardized data (backdoor_linear: 2*beta_i) or a
    random-forest plug-in (backdoor_rf: mean over rows of
    f(x_i=+1, Z) - f(x_i=-1, Z)).  Actuators are DAG roots (empty
    adjustment set, so backdoor reduces to the marginal regression, which
    is the correct do-estimate under the randomized reference protocol);
    sensor decoys have their causes adjusted away, leaving only sensor
    noise -- the classical-causal-inference answer to the decoy trap.
    NOTE the oracle privilege is twofold: the true graph AND access to all
    measured chamber columns for adjustment (e.g. l_11/l_12 when
    explaining lt_vis3), not just the 10 features the other methods see.

  * do-Shapley (Jung et al., ICML 2022) and Causal Shapley (Heskes et al.,
    NeurIPS 2020) are deliberately NOT run.  The repo's implementation
    (``causal_probe/shapley_causal.py``, used on synthetic SCMs by
    ``eval_shapley_comparison.py``) evaluates the coalition value function
    v(S) = E[Y | do(X_S = x_S)] via ``scm.simulate_do`` (or an exact
    forward pass through the structural matrix B) -- MECHANISM access: the
    ability to simulate the system under arbitrary JOINT interventions.
    No data-driven method has that for a real physical system, and data +
    graph alone do not supply it without fitting a full generative model
    of every mechanism (each coalition is a separate multi-node
    identification/estimation problem).  The honest graph-oracle skyline
    on real data is therefore the backdoor estimator above; that exact
    do-/Causal Shapley requires mechanistic access no method operating on
    observational data has is itself a point the paper makes.

Usage:
  uv run python -m causal_probe.causal_chambers_eval --dry-run
  uv run python -m causal_probe.causal_chambers_eval --comparators-only --skip-slow
  python -m causal_probe.causal_chambers_eval --ckpt <ckpt> --device cuda --methods full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_probe.baselines import BASELINES

DATASET_NAME = "lt_interventions_standard_v1"
REFERENCE_EXPERIMENT = "uniform_reference"
LOCAL_BANDWIDTH = 0.25  # local-mean window, in units of sd_f
SIGN_THRESHOLD = 0.1    # |tau_true| above this counts as signal for sign agreement

# Graph-free comparators, dispatched through causal_probe.run_baseline.METHODS
# (dopfn goes through causal_probe.dopfn_per_column_baseline and needs
# --dopfn-repo). All run once on the FULL standardized observational data --
# see "Comparator suite" in the module docstring for the protocol asymmetry
# vs FADO's n_rows-limited contexts.
GRAPH_FREE_COMPARATORS = [
    "marginal", "ridge", "permutation", "shap", "lingam",
    "pc", "ges", "fci", "notears", "doubleml", "causal_forest",
    "causalpfn", "dopfn",
]
SLOW_COMPARATORS = {"notears", "doubleml", "causal_forest", "causalpfn", "dopfn"}
# Skylines that receive the chamber's published true DAG (see docstring).
GRAPH_ORACLE_COMPARATORS = ["backdoor_linear", "backdoor_rf"]
ALL_COMPARATORS = GRAPH_FREE_COMPARATORS + GRAPH_ORACLE_COMPARATORS

COMPARATOR_PROTOCOL_NOTE = (
    "Comparators run once, deterministically, on ALL observational rows of "
    "uniform_reference (standardized), while FADO is evaluated on n_rows-row "
    "context subsamples averaged over n_seeds draws -- an asymmetry that "
    "favors the comparators. backdoor_* additionally receive the published "
    "true DAG and may adjust for measured chamber columns outside the "
    "10-feature set. do-/Causal Shapley are absent by design: their value "
    "function E[Y|do(X_S=x_S)] requires mechanism access (simulate_do), "
    "which no data-driven method has on a real physical system; "
    "backdoor_linear/backdoor_rf are the honest graph-oracle skylines here."
)

# (config label, outcome, feature list). Intervened features must have
# uniform_<f>_mid (and optionally _strong) experiments in the dataset;
# the rest must be DAG-null sensors (checked at runtime).
LT_CONFIGS = [
    ("lt_vis3", "vis_3",
     ["red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32",
      "current", "angle_1", "vis_1"]),
    ("lt_vis1", "vis_1",
     ["red", "green", "blue", "pol_1", "pol_2", "l_11", "l_12",
      "current", "angle_1", "ir_2"]),
]


def _load_dataset(data_dir: str):
    from causalchamber.datasets import Dataset
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return Dataset(name=DATASET_NAME, root=data_dir, download=True)


def _experiment_df(dataset, name: str):
    return dataset.get_experiment(name).as_pandas_dataframe()


def _intervention_experiments(dataset, feature: str) -> list[str]:
    """Single-target experiments for `feature` (mid, strong if present)."""
    avail = set(dataset.available_experiments())
    return [nm for nm in (f"uniform_{feature}_mid", f"uniform_{feature}_strong")
            if nm in avail]


def _check_dag(features: list[str], intervened: list[str], outcome: str) -> dict[str, bool]:
    """Check the design against the chamber's known causal graph.

    Never-intervened decoys MUST have no outgoing edges (their tau_true=0
    rests on the DAG).  Intervened features may or may not be parents of the
    outcome -- non-parents (e.g. pol_1 for Y=vis_1) are deliberate
    hardware-verified nulls -- so parenthood is recorded, not asserted.
    """
    from causalchamber.ground_truth import graph
    A = graph(chamber="lt", configuration="standard")  # adjacency DataFrame
    for f in features:
        if f in intervened:
            continue
        children = [c for c in A.columns if A.loc[f, c] == 1]
        assert not children, f"decoy {f} has outgoing edges {children}; tau_true=0 unjustified"
    return {f: bool(A.loc[f, outcome] == 1) for f in features}


def _local_mean(x: np.ndarray, y: np.ndarray, x0: float, h: float) -> tuple[float, int]:
    mask = np.abs(x - x0) <= h
    n = int(mask.sum())
    return (float(y[mask].mean()) if n else float("nan")), n


def build_ground_truth(dataset, outcome: str, features: list[str]) -> dict:
    """Ground-truth tau per feature (see module docstring, steps 1-4)."""
    ref = _experiment_df(dataset, REFERENCE_EXPERIMENT)
    mu_y, sd_y = float(ref[outcome].mean()), float(ref[outcome].std())

    rows = []
    for f in features:
        mu_f, sd_f = float(ref[f].mean()), float(ref[f].std())
        exps = _intervention_experiments(dataset, f)
        x_lo, x_hi = mu_f - sd_f, mu_f + sd_f
        h = LOCAL_BANDWIDTH * sd_f
        row = {"feature": f, "mu": mu_f, "sd": sd_f,
               "intervened": bool(exps), "experiments": exps}
        if exps:
            pooled = [ref] + [_experiment_df(dataset, e) for e in exps]
            x = np.concatenate([d[f].values for d in pooled]).astype(np.float64)
            y = np.concatenate([d[outcome].values for d in pooled]).astype(np.float64)
            m_lo, n_lo = _local_mean(x, y, x_lo, h)
            m_hi, n_hi = _local_mean(x, y, x_hi, h)
            row.update(tau_true=(m_hi - m_lo) / sd_y, n_lo=n_lo, n_hi=n_hi)
            # Linear extrapolation from the intervention experiments alone.
            xi = np.concatenate([_experiment_df(dataset, e)[f].values for e in exps])
            yi = np.concatenate([_experiment_df(dataset, e)[outcome].values for e in exps])
            slope, intercept = np.polyfit(xi.astype(np.float64), yi.astype(np.float64), 1)
            row["tau_true_linear"] = float(slope * (x_hi - x_lo) / sd_y)
            # Raw shift of each intervention experiment vs reference (sd_y units):
            # hardware verification of effect direction / nullity out of support.
            row["shift_sd"] = {
                e: float((_experiment_df(dataset, e)[outcome].mean() - mu_y) / sd_y)
                for e in exps
            }
        else:
            # Never-intervened sensor readout; DAG-null (asserted in _check_dag).
            row.update(tau_true=0.0, n_lo=0, n_hi=0,
                       tau_true_linear=0.0, shift_sd={})
        rows.append(row)

    intervened = [r["feature"] for r in rows if r["intervened"]]
    dag_parent = _check_dag(features, intervened, outcome)
    for r in rows:
        r["dag_parent_of_y"] = dag_parent[r["feature"]]
    return {"outcome": outcome, "sd_y": sd_y, "mu_y": mu_y, "rows": rows}


def _standardize(X: np.ndarray, y: np.ndarray):
    mu_X = X.mean(axis=0); sd_X = X.std(axis=0).clip(min=1e-9)
    mu_y = float(y.mean()); sd_y = float(y.std().clip(min=1e-9))
    return (X - mu_X) / sd_X, (y - mu_y) / sd_y


def _print_truth_table(label: str, truth: dict) -> None:
    print(f"\n=== {label}: Y={truth['outcome']} (sd_Y={truth['sd_y']:.2f}) ===")
    print(f"{'feature':<10}{'type':<12}{'mu':>9}{'sd':>8}{'tau_true':>10}"
          f"{'tau_lin':>9}{'n_lo':>6}{'n_hi':>6}  experiments")
    for r in truth["rows"]:
        kind = ("intervened" if r["dag_parent_of_y"] else "interv-null") \
            if r["intervened"] else "DAG-null"
        print(f"{r['feature']:<10}{kind:<12}{r['mu']:>9.2f}{r['sd']:>8.2f}"
              f"{r['tau_true']:>+10.3f}{r['tau_true_linear']:>+9.3f}"
              f"{r['n_lo']:>6d}{r['n_hi']:>6d}  {','.join(r['experiments'])}")
    shifts = {r["feature"]: r["shift_sd"] for r in truth["rows"] if r["shift_sd"]}
    print("  out-of-support shifts vs reference (sd_Y units):")
    for f, s in shifts.items():
        print(f"    {f:<8}" + "  ".join(f"{e.split('_')[-1]}={v:+.2f}" for e, v in s.items()))


def _metrics(tau_hat: np.ndarray, tau_true: np.ndarray) -> dict:
    """Pearson r / sign agreement / MAE on the finite entries of tau_hat.

    Some comparators (doubleml, causalpfn, dopfn) return NaN for features
    they cannot fit; metrics are computed on the finite subset and the
    finite/total counts are recorded. FADO never returns NaN, so this is a
    no-op for the model path.
    """
    tau_hat = np.asarray(tau_hat, dtype=np.float64)
    tau_true = np.asarray(tau_true, dtype=np.float64)
    finite = np.isfinite(tau_hat)
    out = {"n_features_finite": int(finite.sum()),
           "n_features_total": int(tau_hat.size)}
    if finite.sum() < 2:
        out.update(pearson_r=float("nan"), mae=float("nan"),
                   n_signal=0, sign_agreement=float("nan"))
        return out
    th, tt = tau_hat[finite], tau_true[finite]
    signal = np.abs(tt) > SIGN_THRESHOLD
    out.update(
        pearson_r=float(np.corrcoef(th, tt)[0, 1]),
        mae=float(np.mean(np.abs(th - tt))),
        n_signal=int(signal.sum()),
        sign_agreement=(float(np.mean(np.sign(th[signal]) == np.sign(tt[signal])))
                        if signal.any() else float("nan")),
    )
    return out


# ---------------------------------------------------------------------------
# Graph-oracle skyline: backdoor adjustment on the published true DAG
# ---------------------------------------------------------------------------

def _lt_true_graph():
    from causalchamber.ground_truth import graph
    return graph(chamber="lt", configuration="standard")


def _has_directed_path(A, src: str, dst: str) -> bool:
    """True if the DAG (adjacency DataFrame, A.loc[u, v]==1 for u->v) has a
    directed path src -> ... -> dst."""
    seen, stack = {src}, [src]
    while stack:
        u = stack.pop()
        for v in A.columns:
            if A.loc[u, v] == 1 and v not in seen:
                if v == dst:
                    return True
                seen.add(v)
                stack.append(v)
    return False


def backdoor_tau(ref, features: list[str], outcome: str, *,
                 kind: str = "linear", seed: int = 0):
    """Backdoor-adjustment skyline given the chamber's published true DAG.

    For each feature f: adjustment set Z = Pa(f) from the true graph,
    restricted to non-constant recorded columns (all parents ARE recorded in
    the reference experiment; constants cannot open a backdoor path here).
    Estimate tau = (E[Y|do(f=mu+sd)] - E[Y|do(f=mu-sd)]) / sd_Y by regression
    adjustment on the observational reference data:
      kind="linear": OLS of standardized Y on [1, f_std, Z_std]; tau = 2*beta_f.
      kind="rf":     RandomForest fit of Y on [f_std, Z_std]; tau = plug-in
                     mean over rows of rf(+1, Z) - rf(-1, Z).
    Returns (tau array, per-feature metadata list).
    """
    A = _lt_true_graph()
    y_raw = ref[outcome].values.astype(np.float64)
    ys = (y_raw - y_raw.mean()) / max(float(y_raw.std()), 1e-9)
    taus = np.full(len(features), np.nan, dtype=np.float64)
    meta = []
    for k, f in enumerate(features):
        parents = [u for u in A.index if A.loc[u, f] == 1]
        assert outcome not in parents, f"outcome {outcome} is a parent of {f}?"
        zcols = [u for u in parents
                 if u in ref.columns and float(ref[u].std()) > 1e-9]
        info = {"feature": f, "adjustment_set": zcols,
                "parents_dropped_constant_or_unrecorded":
                    [u for u in parents if u not in zcols],
                "has_directed_path_to_y": _has_directed_path(A, f, outcome)}
        x_raw = ref[f].values.astype(np.float64)
        sd_f = float(x_raw.std())
        if sd_f < 1e-9:
            info["error"] = "constant treatment column"
            meta.append(info)
            continue
        xs = (x_raw - x_raw.mean()) / sd_f
        Z = np.empty((len(xs), 0), dtype=np.float64)
        if zcols:
            Zr = ref[zcols].values.astype(np.float64)
            Z = (Zr - Zr.mean(axis=0)) / Zr.std(axis=0).clip(min=1e-9)
        if kind == "linear":
            D = np.column_stack([np.ones_like(xs), xs, Z])
            beta, *_ = np.linalg.lstsq(D, ys, rcond=None)
            taus[k] = 2.0 * float(beta[1])
        elif kind == "rf":
            from sklearn.ensemble import RandomForestRegressor
            D = np.column_stack([xs, Z])
            rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                       n_jobs=-1, random_state=seed)
            rf.fit(D, ys)
            D_hi = D.copy(); D_hi[:, 0] = +1.0
            D_lo = D.copy(); D_lo[:, 0] = -1.0
            taus[k] = float(np.mean(rf.predict(D_hi) - rf.predict(D_lo)))
        else:
            raise ValueError(f"unknown backdoor kind {kind!r}")
        meta.append(info)
    return taus, meta


# ---------------------------------------------------------------------------
# Comparator runner
# ---------------------------------------------------------------------------

def _dopfn_comparator(Xs: np.ndarray, ys: np.ndarray,
                      dopfn_repo: str, device: str) -> np.ndarray:
    """Do-PFN per-column on (Xs, ys). Needs a checked-out Do-PFN repo; the
    DoPFNRegressor opens artifacts via relative paths, so chdir for the call."""
    import os
    from causal_probe.dopfn_per_column_baseline import _load_dopfn, dopfn_tau
    repo = str(Path(dopfn_repo).resolve())
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        model = _load_dopfn(repo, device=device)
        return dopfn_tau(model, Xs, ys)
    finally:
        os.chdir(cwd)


def run_comparators(
    methods: list[str],
    Xs: np.ndarray,
    ys: np.ndarray,
    ref,
    features: list[str],
    outcome: str,
    tau_true: np.ndarray,
    *,
    seed: int,
    device: str,
    dopfn_repo: str | None,
) -> dict:
    """Run each comparator once on the full data; catch per-method errors."""
    import time
    out = {}
    for m in methods:
        t0 = time.time()
        entry: dict = {}
        print(f"  [comparator] {m:<16}", end="", flush=True)
        try:
            if m in ("backdoor_linear", "backdoor_rf"):
                tau, adj = backdoor_tau(
                    ref, features, outcome,
                    kind=m.removeprefix("backdoor_"), seed=seed)
                entry["adjustment"] = adj
            elif m == "dopfn":
                if dopfn_repo is None:
                    raise RuntimeError(
                        "requires --dopfn-repo (checked-out Do-PFN repo with artifacts/)")
                tau = _dopfn_comparator(Xs, ys, dopfn_repo, device)
            else:
                from causal_probe.run_baseline import METHODS as DISPATCH
                kw = {}
                if m == "permutation":
                    kw["seed"] = seed
                elif m == "causalpfn":
                    kw["device"] = device
                tau = DISPATCH[m](Xs, ys, **kw)
            tau = np.asarray(tau, dtype=np.float64)
            entry["tau"] = tau.tolist()
            entry["metrics"] = _metrics(tau, tau_true)
            entry["runtime_s"] = round(time.time() - t0, 2)
            mt = entry["metrics"]
            print(f" r={mt['pearson_r']:+.3f}  mae={mt['mae']:.3f}"
                  f"  ({entry['runtime_s']:.1f}s)")
        except Exception as e:  # noqa: BLE001 -- record and continue
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["runtime_s"] = round(time.time() - t0, 2)
            print(f" ERROR ({entry['runtime_s']:.1f}s): {entry['error'][:110]}")
        out[m] = entry
    return out


def _print_comparator_table(features: list[str], tau_true: np.ndarray,
                            comp: dict) -> None:
    print(f"\n  {'method':<16}{'r':>8}{'sign':>7}{'MAE':>8}{'time':>9}")
    for m, e in comp.items():
        if "error" in e:
            print(f"  {m:<16}   error: {e['error'][:80]}")
            continue
        mt = e["metrics"]
        print(f"  {m:<16}{mt['pearson_r']:>+8.3f}{mt['sign_agreement']:>7.2f}"
              f"{mt['mae']:>8.3f}{e['runtime_s']:>8.1f}s")
    ok = [m for m, e in comp.items() if "error" not in e]
    if not ok:
        return
    print(f"\n  {'feature':<10}{'tau_true':>9}"
          + "".join(f"{m[:9]:>10}" for m in ok))
    for i, f in enumerate(features):
        cells = "".join(
            f"{comp[m]['tau'][i]:>+10.3f}" if np.isfinite(comp[m]["tau"][i])
            else f"{'nan':>10}" for m in ok)
        print(f"  {f:<10}{tau_true[i]:>+9.3f}{cells}")


def evaluate_config(
    label: str,
    outcome: str,
    features: list[str],
    dataset,
    ckpt: str | None,
    device: str,
    seed: int,
    n_rows: int,
    n_seeds: int,
    dry_run: bool,
    comparators: list[str] | None = None,
    run_model: bool = True,
    dopfn_repo: str | None = None,
) -> dict:
    truth = build_ground_truth(dataset, outcome, features)
    tau_true = np.array([r["tau_true"] for r in truth["rows"]])
    _print_truth_table(label, truth)

    ref = _experiment_df(dataset, REFERENCE_EXPERIMENT)
    X_raw = ref[features].values.astype(np.float64)
    y_raw = ref[outcome].values.astype(np.float64)
    Xs, ys = _standardize(X_raw, y_raw)
    print(f"  observational input: X {Xs.shape}, y {ys.shape} "
          f"(context n_rows={n_rows}, n_seeds={n_seeds})")

    result = {
        "label": label, "outcome": outcome, "features": features,
        "dataset": DATASET_NAME, "reference_experiment": REFERENCE_EXPERIMENT,
        "local_bandwidth_sd": LOCAL_BANDWIDTH, "sign_threshold": SIGN_THRESHOLD,
        "n_obs": int(Xs.shape[0]), "p": int(Xs.shape[1]),
        "truth": truth["rows"], "tau_true": tau_true.tolist(),
    }

    # Baselines on the full standardized reference.
    assoc = BASELINES["marginal"](Xs, ys)
    multi = BASELINES["multivariate"](Xs, ys)
    result["assoc"] = assoc.tolist()
    result["multi"] = multi.tolist()
    result["metrics_assoc"] = _metrics(assoc, tau_true)
    result["metrics_multi"] = _metrics(multi, tau_true)

    if dry_run:
        if comparators:
            print(f"  [dry-run] comparators that would run "
                  f"(full {Xs.shape[0]}-row data, single pass): "
                  f"{', '.join(comparators)}")
        return result

    if comparators:
        print(f"\n  comparator suite on the full {Xs.shape[0]}-row "
              f"standardized observational data (single run each):")
        result["comparators"] = run_comparators(
            comparators, Xs, ys, ref, features, outcome, tau_true,
            seed=seed, device=device, dopfn_repo=dopfn_repo)
        _print_comparator_table(features, tau_true, result["comparators"])

    if not run_model:
        return result

    import torch
    from causal_probe.eval import _load_model
    model = _load_model(ckpt, device=device)

    rng = np.random.default_rng(seed)
    n_total, p = Xs.shape
    preds, per_seed_metrics = [], []
    for _ in range(n_seeds):
        idx = (rng.choice(n_total, size=n_rows, replace=False)
               if n_total > n_rows else np.arange(n_total))
        perm = rng.permutation(p)  # guard against column-position artifacts
        X_ctx = Xs[idx][:, perm]; y_ctx = ys[idx]
        X_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_perm = model(X_t, y_t).squeeze(0).cpu().numpy()
        pred = np.empty(p); pred[perm] = pred_perm  # un-permute
        preds.append(pred)
        per_seed_metrics.append(_metrics(pred, tau_true))
    preds = np.stack(preds)
    tau_hat = preds.mean(axis=0)

    result["tau_hat"] = tau_hat.tolist()
    result["tau_hat_std"] = preds.std(axis=0).tolist()
    result["tau_hat_per_seed"] = preds.tolist()
    result["metrics"] = _metrics(tau_hat, tau_true)
    for key in ("pearson_r", "mae", "sign_agreement"):
        vals = np.array([m[key] for m in per_seed_metrics])
        result["metrics"][f"{key}_seed_mean"] = float(np.nanmean(vals))
        result["metrics"][f"{key}_seed_sd"] = float(np.nanstd(vals))

    print(f"\n  {'feature':<10}{'tau_true':>10}{'tau_hat':>10}{'±sd':>8}"
          f"{'assoc':>9}{'multi':>9}")
    for i, f in enumerate(features):
        print(f"  {f:<10}{tau_true[i]:>+10.3f}{tau_hat[i]:>+10.3f}"
              f"{preds.std(axis=0)[i]:>8.3f}{assoc[i]:>+9.3f}{multi[i]:>+9.3f}")
    m = result["metrics"]
    print(f"  pearson_r = {m['pearson_r']:+.4f}  (seeds {m['pearson_r_seed_mean']:+.4f}"
          f" ± {m['pearson_r_seed_sd']:.4f})")
    print(f"  sign_agreement (|tau|>{SIGN_THRESHOLD}, n={m['n_signal']}) = "
          f"{m['sign_agreement']:.3f}")
    print(f"  MAE = {m['mae']:.4f}   [assoc r={result['metrics_assoc']['pearson_r']:+.3f}"
          f" mae={result['metrics_assoc']['mae']:.3f};"
          f" multi r={result['metrics_multi']['pearson_r']:+.3f}"
          f" mae={result['metrics_multi']['mae']:.3f}]")
    return result


def _resolve_comparators(spec: str, skip_slow: bool) -> list[str]:
    if spec == "default":
        return []
    if spec == "full":
        methods = list(ALL_COMPARATORS)
    else:
        methods = [s.strip() for s in spec.split(",") if s.strip()]
        unknown = [m for m in methods if m not in ALL_COMPARATORS]
        if unknown:
            raise SystemExit(
                f"unknown comparator(s) {unknown}; choose from {ALL_COMPARATORS}")
    if skip_slow:
        methods = [m for m in methods if m not in SLOW_COMPARATORS]
    return methods


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None,
                    help="probe checkpoint (required unless --dry-run/--comparators-only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="row-subsample/column-permutation repeats")
    ap.add_argument("--data-dir", default="/tmp/causalchamber_data")
    ap.add_argument("--out", default=None,
                    help="output JSON (default causal_probe/results/"
                         "causal_chambers.json, or ..._comparators.json "
                         "under --comparators-only)")
    ap.add_argument("--chamber", default="lt", choices=["lt", "wt", "all"])
    ap.add_argument("--methods", default="default",
                    help="'default' (marginal+multivariate only, as before), "
                         "'full' (whole comparator suite incl. graph-oracle "
                         "backdoor skylines), or a comma-separated subset of: "
                         + ",".join(ALL_COMPARATORS))
    ap.add_argument("--skip-slow", action="store_true",
                    help=f"drop slow comparators: {sorted(SLOW_COMPARATORS)}")
    ap.add_argument("--comparators-only", action="store_true",
                    help="run ground truth + comparator suite with no FADO "
                         "model (no --ckpt/GPU needed); implies --methods full "
                         "unless --methods is given")
    ap.add_argument("--dopfn-repo", default=None,
                    help="checked-out Do-PFN repo (with artifacts/) for the "
                         "dopfn comparator")
    ap.add_argument("--dry-run", action="store_true",
                    help="download data, print ground-truth tables and the "
                         "comparator list, skip all methods and the model")
    args = ap.parse_args()

    if args.comparators_only and args.methods == "default":
        args.methods = "full"
    comparators = _resolve_comparators(args.methods, args.skip_slow)
    if not args.dry_run and not args.comparators_only and args.ckpt is None:
        ap.error("--ckpt is required unless --dry-run or --comparators-only")
    if args.out is None:
        args.out = ("causal_probe/results/causal_chambers_comparators.json"
                    if args.comparators_only
                    else "causal_probe/results/causal_chambers.json")
    if args.chamber in ("wt", "all"):
        print("[warn] wind tunnel skipped: no published wt analogue of "
              f"{DATASET_NAME} (uniform reference + single-target interventions); "
              "see module docstring / open questions.")
        if args.chamber == "wt":
            return

    dataset = _load_dataset(args.data_dir)
    results = {"dataset": DATASET_NAME, "seed": args.seed, "n_rows": args.n_rows,
               "n_seeds": args.n_seeds, "ckpt": args.ckpt,
               "comparator_methods": comparators,
               "comparator_protocol": COMPARATOR_PROTOCOL_NOTE if comparators else None,
               "configs": []}
    for label, outcome, features in LT_CONFIGS:
        results["configs"].append(evaluate_config(
            label, outcome, features, dataset,
            ckpt=args.ckpt, device=args.device, seed=args.seed,
            n_rows=args.n_rows, n_seeds=args.n_seeds, dry_run=args.dry_run,
            comparators=comparators, run_model=not args.comparators_only,
            dopfn_repo=args.dopfn_repo,
        ))

    if args.dry_run:
        print(f"\n[dry-run] comparators that would run under "
              f"--methods {args.methods}"
              f"{' --skip-slow' if args.skip_slow else ''}: "
              f"{', '.join(comparators) if comparators else '(none)'}")
        print("[dry-run] ground truth built; exiting before model load.")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
