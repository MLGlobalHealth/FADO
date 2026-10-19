"""Graph-free comparator suite shared by the real-RCT eval harnesses.

Used by ``rct_negative_controls.py``, ``star_rct.py`` and ``twins_eval.py``:
run every graph-free baseline method on the SAME standardized (X, y)
construction the probe sees, and read off each method's score at the
treatment column T. Explainability-paper framing: the interesting columns
are predictive attribution (SHAP, permutation importance) vs FADO on
known-null / known-effect treatment columns.

Graph-oracle methods (parent-adjustment with the true graph, etc.) are
deliberately absent: no ground-truth causal graph exists for these real
datasets, so only graph-free estimators are comparable.

Method taxonomy:

* **Vector methods** (``marginal``, ``shap``, ``permutation``, ``lingam``):
  one fit yields all p scores; reused by import from
  ``causal_probe.run_baseline.METHODS``. Cheap, CPU-friendly.
* **Per-column methods** (``doubleml``, ``causal_forest``, ``drlearner``,
  ``causalpfn``, ``dopfn``): each fit designates one treatment column, so
  this module evaluates them at the T column only (the column of interest
  in all three harnesses). All are gated behind the harnesses'
  ``--skip-slow``. CausalPFN / Do-PFN imports are heavy (model-checkpoint
  downloads) and may be absent locally; they are imported lazily inside
  the method functions so ``--dry-run`` and cheap-method runs work
  without them.

Scaling convention: every method reports tau on the probe's reporting
scale (Gelman 2008 "divide by 2 SDs"): response of std-Y to a 2-std swing
in raw X. For the binarized per-column estimators this is
``2 * ate * std(x) / (gap * std(y))`` with ``gap`` the above-/below-median
mean gap — the convention of ``hillstrom_drlearner.drlearner_tau`` and
``run_baseline.causalpfn_tau``. (Deviation from
``dopfn_per_column_baseline.dopfn_tau``, which uses the plain
``ate / std(y)``; the two agree for a balanced binary treatment.)

Any method erroring on a dataset is recorded as an ``error`` string in
the result dict; the suite continues with the remaining methods.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from causal_probe.baselines import BASELINES

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

CHEAP_METHODS = ("marginal", "shap", "permutation", "lingam")
SLOW_METHODS = ("doubleml", "causal_forest", "drlearner", "causalpfn", "dopfn")
ALL_METHODS = CHEAP_METHODS + SLOW_METHODS


def default_methods(
    skip_slow: bool = False, dopfn_repo: str | None = None,
) -> list[str]:
    """Default method list: cheap always; slow unless skip_slow; dopfn only
    when a checked-out Do-PFN repo path is supplied."""
    ms = list(CHEAP_METHODS)
    if not skip_slow:
        ms += [m for m in SLOW_METHODS if m != "dopfn" or dopfn_repo]
    return ms


def top_value_frequency(x: np.ndarray) -> float:
    """Fraction of rows carrying the modal value of x (near-constant screen:
    the realdata bench excludes controls with top_freq > 0.9)."""
    x = np.asarray(x)
    _, counts = np.unique(x, return_counts=True)
    return float(counts.max() / x.size)


# ---------------------------------------------------------------------------
# Per-column (designated-treatment) methods: score ONE column t_idx.
# Bodies mirror the corresponding full-vector functions in run_baseline.py /
# hillstrom_drlearner.py / dopfn_per_column_baseline.py.
# ---------------------------------------------------------------------------

def _doubleml_col(X: np.ndarray, y: np.ndarray, i: int, **_kw) -> float:
    """LinearDML at column i (mirrors run_baseline.doubleml_tau)."""
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor

    t = X[:, i].astype(np.float64)
    if np.std(t) < 1e-9:
        return float("nan")
    X_rest = np.delete(X, i, axis=1).astype(np.float64)
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=5,
                                      min_samples_leaf=10, random_state=0),
        model_t=RandomForestRegressor(n_estimators=50, max_depth=5,
                                      min_samples_leaf=10, random_state=0),
        discrete_treatment=False, cv=2, random_state=0,
    )
    est.fit(y.astype(np.float64), t, W=X_rest)
    ate = float(est.ate(X=None))
    std_y = float(np.std(y).clip(min=1e-9))
    return 2.0 * ate * float(np.std(t)) / std_y


def _causal_forest_col(X: np.ndarray, y: np.ndarray, i: int, **_kw) -> float:
    """CausalForestDML at column i (mirrors run_baseline.causal_forest_tau)."""
    from econml.dml import CausalForestDML

    t = X[:, i].astype(np.float64)
    if np.std(t) < 1e-9:
        return float("nan")
    X_rest = np.delete(X, i, axis=1).astype(np.float64)
    est = CausalForestDML(
        n_estimators=100, min_samples_leaf=10, max_depth=8,
        discrete_treatment=False, cv=2, random_state=0,
    )
    est.fit(y.astype(np.float64), t, X=X_rest)
    ate = float(est.ate(X=X_rest))
    std_y = float(np.std(y).clip(min=1e-9))
    return 2.0 * ate * float(np.std(t)) / std_y


def _drlearner_col(
    X: np.ndarray, y: np.ndarray, i: int, *,
    drlearner_light: bool = True, seed: int = 0, **_kw,
) -> float:
    """Kennedy-2020 DR-Learner at column i, reusing the Hillstrom convention
    (median binarization + gap rescaling) from hillstrom_drlearner."""
    from causal_probe.hillstrom_drlearner import drlearner_tau
    return float(drlearner_tau(X, y, i, light=drlearner_light, random_state=seed))


def _causalpfn_col(
    X: np.ndarray, y: np.ndarray, i: int, *,
    device: str = "cpu", cache_dir: str | None = None, **_kw,
) -> float:
    """CausalPFN ATEEstimator at column i (single-column version of
    run_baseline.causalpfn_tau: median binarization + gap rescaling)."""
    from causalpfn import ATEEstimator

    if cache_dir is None:
        cache_dir = os.environ.get("CAUSALPFN_CACHE_DIR")

    x_i = X[:, i].astype(np.float64)
    med = float(np.median(x_i))
    t = (x_i > med).astype(np.int64)
    if t.sum() < 5 or (1 - t).sum() < 5:
        return float("nan")
    gap = float(x_i[t == 1].mean() - x_i[t == 0].mean())
    if gap < 1e-9:
        return float("nan")
    std_X_i = float(np.std(x_i).clip(min=1e-9))
    std_y = float(np.std(y).clip(min=1e-9))
    X_rest = np.delete(X, i, axis=1).astype(np.float64)
    kwargs = {"device": device}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    est = ATEEstimator(**kwargs)
    est.fit(X_rest, t, y.astype(np.float64))
    ate = float(est.estimate_ate())
    return 2.0 * ate * std_X_i / (gap * std_y)


_DOPFN_MODEL = None  # cached across calls (checkpoint load is expensive)


def _dopfn_col(
    X: np.ndarray, y: np.ndarray, i: int, *,
    dopfn_repo: str | None = None, device: str = "cpu", **_kw,
) -> float:
    """Do-PFN at column i (single-column version of
    dopfn_per_column_baseline.dopfn_tau, rescaled to the shared gap
    convention). Requires a checked-out Do-PFN repo; the DoPFNRegressor
    opens artifact files via relative paths, so we chdir for the call."""
    global _DOPFN_MODEL
    import contextlib
    import io

    import torch

    if dopfn_repo is None:
        raise RuntimeError("dopfn requires --dopfn-repo (checked-out Do-PFN repo)")
    repo = str(Path(dopfn_repo).resolve())

    x_i = X[:, i].astype(np.float64)
    med = float(np.median(x_i))
    t = (x_i > med).astype(np.float32)
    if t.sum() < 5 or (1 - t).sum() < 5:
        return float("nan")
    gap = float(x_i[t > 0.5].mean() - x_i[t < 0.5].mean())
    if gap < 1e-9:
        return float("nan")
    std_X_i = float(np.std(x_i).clip(min=1e-9))
    std_y = float(np.std(y).clip(min=1e-9))
    X_rest = np.delete(X, i, axis=1).astype(np.float32)
    D = np.concatenate([t[:, None], X_rest], axis=1).astype(np.float32)

    cwd = os.getcwd()
    os.chdir(repo)
    try:
        if _DOPFN_MODEL is None:
            from causal_probe.dopfn_per_column_baseline import _load_dopfn
            _DOPFN_MODEL = _load_dopfn(repo, device=device)
        with contextlib.redirect_stderr(io.StringIO()):
            _DOPFN_MODEL.fit(D, y.astype(np.float32))
            cate = _DOPFN_MODEL.predict_cate(torch.tensor(D).clone())
    finally:
        os.chdir(cwd)
    ate = float(np.asarray(cate).mean())
    return 2.0 * ate * std_X_i / (gap * std_y)


_PER_COLUMN = {
    "doubleml": _doubleml_col,
    "causal_forest": _causal_forest_col,
    "drlearner": _drlearner_col,
    "causalpfn": _causalpfn_col,
    "dopfn": _dopfn_col,
}


def _vector_tau(method: str, X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Full-vector methods, reused by import from run_baseline.METHODS."""
    if method == "marginal":
        return np.asarray(BASELINES["marginal"](X, y), dtype=np.float64)
    from causal_probe.run_baseline import METHODS
    kw = {"seed": seed} if method == "permutation" else {}
    return np.asarray(METHODS[method](X, y, **kw), dtype=np.float64)


# ---------------------------------------------------------------------------
# Suite driver
# ---------------------------------------------------------------------------

def run_comparator_suite(
    X: np.ndarray,
    y: np.ndarray,
    t_idx: int,
    *,
    methods: list[str] | None = None,
    skip_slow: bool = False,
    seed: int = 0,
    device: str = "cpu",
    dopfn_repo: str | None = None,
    drlearner_light: bool = True,
    extra_cols: dict[str, int] | None = None,
    return_full: bool = False,
    verbose: bool = False,
) -> dict:
    """Run every requested graph-free method on (X, y); extract the T column.

    Parameters
    ----------
    X, y : the SAME standardized construction the probe sees.
    t_idx : column index of the (randomized) treatment T.
    methods : explicit list; default = ``default_methods(skip_slow, dopfn_repo)``.
    extra_cols : {name: col_idx} of additional columns of interest (e.g. a
        permuted-T placebo). Reported for vector methods only (free there;
        per-column methods would need a second full fit).
    return_full : include the full per-feature tau vector for vector methods.

    Returns
    -------
    {method: {"tau_hat_T": float, ...} | {"error": str}} — per-method dict
    with ``tau_hat_T`` (score at t_idx), optional ``tau_hat_<name>`` for
    extra_cols, optional ``tau`` (full vector), or an ``error`` string if
    the method raised. Non-finite scores are stored as None (JSON null).
    """
    if methods is None:
        methods = default_methods(skip_slow=skip_slow, dopfn_repo=dopfn_repo)
    elif skip_slow:
        methods = [m for m in methods if m not in SLOW_METHODS]
    extra_cols = extra_cols or {}

    def _f(v: float) -> float | None:
        return float(v) if np.isfinite(v) else None

    out: dict[str, dict] = {}
    for m in methods:
        if verbose:
            print(f"    comparator: {m} ...", flush=True)
        try:
            if m in _PER_COLUMN:
                val = _PER_COLUMN[m](
                    X, y, t_idx,
                    device=device, dopfn_repo=dopfn_repo,
                    drlearner_light=drlearner_light, seed=seed,
                )
                out[m] = {"tau_hat_T": _f(val)}
            else:
                tau = _vector_tau(m, X, y, seed)
                rec = {"tau_hat_T": _f(tau[t_idx])}
                for nm, ci in extra_cols.items():
                    rec[f"tau_hat_{nm}"] = _f(tau[ci])
                if return_full:
                    rec["tau"] = [_f(v) for v in tau]
                out[m] = rec
        except Exception as e:  # record + continue: one bad method must not kill the run
            out[m] = {"error": f"{type(e).__name__}: {e}"}
    return out


def summarize_tau_T(records: list[dict], methods: list[str]) -> dict:
    """Aggregate |tau_hat_T| per method across a list of per-construction
    comparator dicts (as produced by run_comparator_suite). Returns
    {method: {n, mean_abs_tau_hat_T, max_abs_tau_hat_T, n_errors}}."""
    summary = {}
    for m in methods:
        vals, n_err = [], 0
        for rec in records:
            r = rec.get(m)
            if r is None:
                continue
            if "error" in r:
                n_err += 1
            elif r.get("tau_hat_T") is not None and np.isfinite(r["tau_hat_T"]):
                vals.append(abs(r["tau_hat_T"]))
        summary[m] = {
            "n": len(vals),
            "mean_abs_tau_hat_T": float(np.mean(vals)) if vals else None,
            "max_abs_tau_hat_T": float(np.max(vals)) if vals else None,
            "n_errors": n_err,
        }
    return summary
