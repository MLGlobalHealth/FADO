"""DoPFN known-treatment ATE on RCT_Bench, run as a separate pass.

DoPFNRegressor opens its artifacts via relative paths, so the process must
chdir into the Do-PFN repo for the model's lifetime. We therefore run DoPFN on
its own and merge the column into the main results afterwards. Reuses the
project's _load_dopfn; the per-trial call mirrors dopfn_tau (treatment forced to
column 0, fit, predict_cate, ATE = mean CATE) in RAW outcome units.
"""
from __future__ import annotations
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2e", default=".",
                    help="repo root (for causal_probe import)")
    ap.add_argument("--dopfn-repo", required=True)
    ap.add_argument("--data-dir", required=True, help="absolute path to cleaned_data")
    ap.add_argument("--out-json", required=True, help="absolute path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-cap", type=int, default=1000,
                    help="subsample rows to this cap (PFN context limit)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--per-column", action="store_true",
                    help="fair mode: sweep every column as candidate treatment "
                         "(FADO's task), instead of the designated-treatment ATE")
    args = ap.parse_args()

    sys.path.insert(0, args.l2e)
    from causal_probe.rct_bench_loader import load_trial
    from causal_probe.dopfn_per_column_baseline import _load_dopfn, dopfn_tau
    import torch

    paths = sorted(glob.glob(os.path.join(args.data_dir, "trial*.csv")),
                   key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))

    os.chdir(args.dopfn_repo)               # DoPFN artifacts are relative
    model = _load_dopfn(args.dopfn_repo, device=args.device)
    rng = np.random.default_rng(args.seed)

    out = []
    for p in paths:
        td = load_trial(p)
        if td is None:
            continue
        rec = {"trial": None, "dopfn_raw": None}
        if td is not None:
            rec["trial"] = td.trial
            rec["std_y"] = float(np.std(td.Y))
            rec["rct_ate_raw"] = td.rct_ate_raw()
            if td.p > 0:
                T = td.T.astype(np.float32)
                X = td.X.astype(np.float32)
                y = td.Y.astype(np.float32)
                idx = np.arange(td.n)
                if td.n > args.n_cap:
                    idx = rng.choice(td.n, args.n_cap, replace=False)
                if args.per_column:
                    # Fair playing field: DoPFN faces FADO's task -- try EVERY
                    # column as the candidate treatment (T placed at index 0),
                    # no privileged designated-treatment. dopfn_tau returns a
                    # length-(1+p) vector of standardized d = ate_i/std_y.
                    feats = np.concatenate([T[idx, None], X[idx]], axis=1).astype(np.float32)
                    try:
                        tau = dopfn_tau(model, feats.astype(np.float64), y[idx].astype(np.float64))
                        rec["dopfn_percol_d"] = float(tau[0])          # treatment = col 0
                        rec["tau_vector"] = [None if not np.isfinite(v) else float(v) for v in tau]
                        finite = np.where(np.isfinite(tau))[0]
                        if 0 in finite:
                            order = finite[np.argsort(-np.abs(tau[finite]))]
                            rec["T_rank"] = int(np.where(order == 0)[0][0]) + 1
                        rec["n_passes"] = int(feats.shape[1])          # p+1 DoPFN calls/trial
                    except Exception as e:
                        rec["dopfn_err"] = str(e)[:120]
                else:
                    D = np.concatenate([T[idx, None], X[idx]], axis=1).astype(np.float32)
                    try:
                        with contextlib.redirect_stderr(io.StringIO()):
                            model.fit(D, y[idx])
                            cate = model.predict_cate(torch.tensor(D).clone())
                        rec["dopfn_raw"] = float(np.asarray(cate).mean())
                    except Exception as e:
                        rec["dopfn_err"] = str(e)[:120]
        out.append(rec)
        if args.per_column:
            dd = rec.get("dopfn_percol_d")
            print(f"{rec['trial']:>9} p={td.p:>2} | rct_d={rec['rct_ate_raw']/rec['std_y']:+.2f} "
                  f"dopfn_d={('%+.2f' % dd) if dd is not None else '  nan'} "
                  f"T_rank={rec.get('T_rank','-')}/{td.p+1}")
        else:
            d = rec["dopfn_raw"]
            dd = (d / rec["std_y"]) if d is not None else None
            print(f"{rec['trial']:>9} p={td.p:>2} | rct_d={rec['rct_ate_raw']/rec['std_y']:+.2f} "
                  f"dopfn_d={('%.2f' % dd) if dd is not None else ' nan'}")

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}  ({len(out)} trials)")


if __name__ == "__main__":
    main()
