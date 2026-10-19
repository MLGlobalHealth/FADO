"""Inventory the RCT_Bench corpus: per-trial schema, arms, sizes, missingness.

Prints a summary table and writes rct_bench_inventory.json for the harness to
consume. Uses the naming contract: Treatment / YP_* / YS_* / X_* plus excluded
design fields (Crossover_Sequence, Crossover_Period, Assessment_Window,
Participant_ID).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DESIGN_FIELDS = {
    "Crossover_Sequence", "Crossover_Period", "Assessment_Window", "Participant_ID",
}


def col_kind(c: str) -> str:
    if c == "Treatment":
        return "T"
    if c in DESIGN_FIELDS:
        return "design"
    if c.startswith("YP_"):
        return "YP"
    if c.startswith("YS_"):
        return "YS"
    if c.startswith("X_"):
        return "X"
    return "other"


def outcome_type(s: pd.Series) -> str:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) == 0:
        return "empty"
    u = np.unique(v)
    if set(u.tolist()) <= {0.0, 1.0}:
        return "binary"
    return "continuous"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/RCT_Bench/cleaned_data")
    ap.add_argument("--out-json", default="causal_probe/results/rct_bench_inventory.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "trial*.csv")),
                   key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))
    rows = []
    for p in paths:
        tid = os.path.basename(p).replace(".csv", "")
        df = pd.read_csv(p, low_memory=False)
        kinds: dict[str, list[str]] = {}
        for c in df.columns:
            kinds.setdefault(col_kind(c), []).append(c)
        yp = kinds.get("YP", [])
        tcol = df["Treatment"] if "Treatment" in df.columns else pd.Series(dtype=object)
        arms = tcol.dropna().astype(str).value_counts()
        prim = yp[0] if yp else None
        rec = {
            "trial": tid,
            "n": int(len(df)),
            "n_X": len(kinds.get("X", [])),
            "n_YP": len(yp),
            "n_YS": len(kinds.get("YS", [])),
            "other_cols": kinds.get("other", []),
            "n_arms": int(arms.shape[0]),
            "arms": {k: int(v) for k, v in arms.items()},
            "primary": prim,
            "primary_type": outcome_type(df[prim]) if prim else None,
            "primary_missing_frac": (float(pd.to_numeric(df[prim], errors="coerce").isna().mean())
                                     if prim else None),
        }
        rows.append(rec)

    # Summary
    print(f"{'trial':>9} {'n':>6} {'nX':>4} {'nYP':>4} {'nYS':>4} {'arms':>4} "
          f"{'ptype':>10} {'miss%':>6}  primary")
    for r in rows:
        print(f"{r['trial']:>9} {r['n']:>6} {r['n_X']:>4} {r['n_YP']:>4} {r['n_YS']:>4} "
              f"{r['n_arms']:>4} {str(r['primary_type']):>10} "
              f"{100*(r['primary_missing_frac'] or 0):>6.1f}  {r['primary']}")

    n = len(rows)
    print("\n=== corpus summary ===")
    print(f"trials: {n}")
    print(f"n rows: min {min(r['n'] for r in rows)}, median {int(np.median([r['n'] for r in rows]))}, max {max(r['n'] for r in rows)}")
    print(f"n_X:    min {min(r['n_X'] for r in rows)}, median {int(np.median([r['n_X'] for r in rows]))}, max {max(r['n_X'] for r in rows)}")
    from collections import Counter
    print("arms distribution:", dict(Counter(r['n_arms'] for r in rows)))
    print("primary_type:", dict(Counter(r['primary_type'] for r in rows)))
    print("trials with >2 arms:", [r['trial'] for r in rows if r['n_arms'] > 2])
    print("trials with 'other' cols:", [(r['trial'], r['other_cols']) for r in rows if r['other_cols']][:20])
    print("trials with high primary missingness (>20%):",
          [(r['trial'], round(100*r['primary_missing_frac'],1)) for r in rows
           if r['primary_missing_frac'] and r['primary_missing_frac'] > 0.2])

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
