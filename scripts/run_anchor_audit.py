"""Run the symmetric-vs-curve anchor audit across the paper's real-data
benchmarks. Loads each dataset locally where possible and writes per-
dataset audit JSON plus a cross-dataset scatter.

Datasets:
  - hillstrom    via sklift.datasets.fetch_hillstrom (cached locally)
  - criteo       via the local .sklift_cache/criteo10.csv.gz
  - lalonde      via http://users.nber.org/~rdehejia/data/...
  - sachs        synthetic from the published DAG (no network)
  - tubingen     skipped here — the data webdav is not on the sandbox
                 allow-list; pre-cache the pairs with the repo's
                 tubingen.py and rerun with --include-tubingen.

The audit is deliberately observational (Nadaraya-Watson on E[Y|X_i=t])
so we can sanity-check whether the symmetric (-1, +1) contrast that the
existing FADO head targets has a non-trivial blind spot on these
datasets.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

from causal_probe.anchor_audit import (
    audit_feature, audit_table, flag_potentially_missed, T_GRID,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_hillstrom() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Hillstrom RCT. Returns (X, y, names) with continuous + categorical
    covariates one-hot expanded; treatment 'segment' encoded as +1/-1/0
    for Mens/Womens/None and exposed as a column too.
    """
    cache = os.environ.get("SKLIFT_HOME", os.path.expanduser("~/.cache/sklift"))
    os.makedirs(cache, exist_ok=True)
    from sklift.datasets import fetch_hillstrom
    bunch = fetch_hillstrom(target_col="visit", return_X_y_t=False, data_home=cache)
    df = bunch.data.copy()
    seg = bunch.treatment.values
    y = bunch.target.values.astype(np.float64)
    # Encode segment as a single signed-treatment column
    T = np.where(seg == "Mens E-Mail", 1.0, np.where(seg == "Womens E-Mail", -1.0, 0.0))
    df = df.copy()
    df["T_signed"] = T
    # One-hot encode any non-numeric columns.
    df = df.astype({c: "float64" for c in df.columns if df[c].dtype.kind in "fi"})
    df_oh = _onehot_str(df)
    X = df_oh.values.astype(np.float64)
    names = list(df_oh.columns)
    return X, y, names


def _onehot_str(df):
    """One-hot any non-numeric column; cast numeric columns to float."""
    import pandas as pd
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if s.dtype.kind in "biufc":
            out[c] = s.astype(float)
        else:
            d = pd.get_dummies(s.astype(str), prefix=c, drop_first=False).astype(float)
            out = pd.concat([out, d], axis=1)
    return out


def _load_criteo() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cached Criteo Uplift 10% subsample. Treatment T_true is 0/1; outcome
    'visit' is 0/1. We pull a manageable subsample to keep the audit fast.
    """
    import pandas as pd
    cache = os.environ.get("SKLIFT_HOME", os.path.expanduser("~/.cache/sklift"))
    path = os.path.join(cache, "criteo10.csv.gz")
    print(f"  loading {path}")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f)
    rng = np.random.default_rng(2025)
    n_use = min(20000, len(df))
    df = df.iloc[rng.choice(len(df), size=n_use, replace=False)].reset_index(drop=True)
    # Outcome and treatment columns may be 'visit' / 'treatment' depending on csv.
    y_col = "visit" if "visit" in df.columns else df.columns[-1]
    t_col = "treatment" if "treatment" in df.columns else None
    y = df[y_col].astype(np.float64).values
    cov_cols = [c for c in df.columns if c not in {y_col, "conversion", "exposure"}]
    if t_col is not None and t_col in cov_cols:
        # keep treatment as a column to audit alongside covariates
        pass
    X = df[cov_cols].astype(np.float64).values
    return X, y, cov_cols


def _fetch_text(urls: list[str]) -> str:
    last = None
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=15) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            last = e
    raise RuntimeError(f"fetch failed: {last}")


def _load_lalonde() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """LaLonde NSW + CPS3 control, outcome re78."""
    URLS_T = ["http://users.nber.org/~rdehejia/data/nswre74_treated.txt",
              "https://users.nber.org/~rdehejia/data/nswre74_treated.txt"]
    URLS_C = ["http://users.nber.org/~rdehejia/data/cps3_controls.txt",
              "https://users.nber.org/~rdehejia/data/cps3_controls.txt"]
    cols = ["T", "age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75", "re78"]

    def parse(raw: str) -> np.ndarray:
        rows = []
        for line in raw.splitlines():
            parts = line.strip().split()
            if len(parts) == len(cols):
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
        return np.asarray(rows, dtype=np.float64)

    treated = parse(_fetch_text(URLS_T))
    ctrl = parse(_fetch_text(URLS_C))
    data = np.vstack([treated, ctrl])
    df_cols = {c: i for i, c in enumerate(cols)}
    y = data[:, df_cols["re78"]]
    feat_names = ["T", "age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
    X = data[:, [df_cols[c] for c in feat_names]]
    return X, y, feat_names


def _load_sachs() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a sample from the Sachs-structured SCM (no network)."""
    from causal_probe.sachs_benchmark import make_sachs_scm
    rng = np.random.default_rng(2025)
    scm = make_sachs_scm(rng=rng, y_node="pakts473")
    sample = scm.sample(n=8000, rng=rng)
    X = sample.X.astype(np.float64)
    y = sample.y.astype(np.float64)
    # Use feature_to_Z to build readable names.
    from causal_probe.sachs_benchmark import SACHS_NODES
    z_to_name = {i: SACHS_NODES[i] for i in range(len(SACHS_NODES))}
    feat_names = [z_to_name.get(int(zi), f"z{int(zi)}") for zi in sample.feature_to_Z]
    return X, y, feat_names


def _load_tubingen_local(cache_dir: str) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """Try to load Tübingen pairs from a local cache directory, skipping
    multivariate or malformed pairs. Returns list of (id, X, Y, weight).
    Cache layout (matches the tuebingen.py download URL paths):
        cache_dir/pairmeta.txt
        cache_dir/pair0001.txt
        ...
    """
    out = []
    meta_path = Path(cache_dir) / "pairmeta.txt"
    if not meta_path.exists():
        return out
    with open(meta_path) as f:
        meta = f.read()
    for line in meta.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        pid, cs, ce, es, ee, w = parts[:6]
        cs, ce, es, ee = int(cs), int(ce), int(es), int(ee)
        if cs != ce or es != ee:
            continue   # multivariate, skip
        pair_path = Path(cache_dir) / f"pair{pid}.txt"
        if not pair_path.exists():
            continue
        try:
            data = np.loadtxt(pair_path)
        except Exception:
            continue
        if data.ndim != 2 or data.shape[1] != 2:
            continue
        X = data[:, cs - 1]
        Y = data[:, es - 1]
        out.append((pid, X.astype(np.float64), Y.astype(np.float64), float(w)))
    return out


# ---------------------------------------------------------------------------
# Per-dataset audit drivers
# ---------------------------------------------------------------------------


def _audit_multivariate(name: str, X: np.ndarray, y: np.ndarray, feat_names: list[str]) -> dict:
    """Audit each column of X against y; return summary dict."""
    rows = []
    for j, fn in enumerate(feat_names):
        Xj = X[:, j]
        # Skip degenerate columns (constant or near-constant).
        if Xj.std() < 1e-9:
            continue
        a = audit_feature(fn, Xj, y)
        rows.append({**a.__dict__, "feature_name": fn, "feature_idx": j})
    flagged = flag_potentially_missed(rows)
    return {
        "dataset": name,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "rows": rows,
        "flagged": flagged,
        "T_GRID": T_GRID.tolist(),
    }


def _audit_tubingen(pairs: list[tuple[str, np.ndarray, np.ndarray, float]]) -> dict:
    rows = []
    for pid, X, Y, w in pairs:
        a = audit_feature(f"pair{pid}", X, Y)
        d = a.__dict__.copy()
        d["pair_id"] = pid
        d["weight"] = w
        rows.append(d)
    flagged = flag_potentially_missed(rows)
    return {
        "dataset": "tubingen",
        "n_pairs": len(pairs),
        "rows": rows,
        "flagged": flagged,
        "T_GRID": T_GRID.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="causal_probe/results/anchor_audit")
    ap.add_argument(
        "--datasets", nargs="*",
        default=["hillstrom", "criteo", "lalonde", "sachs"],
        choices=["hillstrom", "criteo", "lalonde", "sachs", "tubingen"],
    )
    ap.add_argument("--tubingen-cache", default="data/tubingen_pairs",
                    help="Pre-downloaded Tübingen pair files (pairmeta.txt + pairNNNN.txt).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOADERS = {
        "hillstrom": _load_hillstrom,
        "criteo":    _load_criteo,
        "lalonde":   _load_lalonde,
        "sachs":     _load_sachs,
    }

    summary = {}
    for d in args.datasets:
        print(f"\n=== {d} ===", flush=True)
        if d == "tubingen":
            pairs = _load_tubingen_local(args.tubingen_cache)
            if not pairs:
                print(f"  no Tübingen pairs in {args.tubingen_cache}; skipping")
                continue
            res = _audit_tubingen(pairs)
        else:
            X, y, names = LOADERS[d]()
            print(f"  loaded {d}: X={X.shape}  y={y.shape}", flush=True)
            res = _audit_multivariate(d, X, y, names)
        out_path = out_dir / f"{d}.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        summary[d] = {
            "n_features": (
                len(res["rows"]) if "rows" in res else 0
            ),
            "n_flagged": len(res.get("flagged", [])),
            "out": str(out_path),
        }
        # Console preview of flagged features.
        flagged = res.get("flagged", [])
        if flagged:
            print(f"  flagged {len(flagged)} feature(s) (small symmetric contrast, large range):")
            for r in flagged[:10]:
                print(
                    f"    {r['name']!r:30s}"
                    f"  contrast_sym={r['contrast_sym']:+.3f}"
                    f"  range={r['range_curve']:.3f}"
                    f"  asymmetry={r['asymmetry']:.3f}"
                )
        else:
            print("  no features flagged (symmetric contrast looks fine)")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
