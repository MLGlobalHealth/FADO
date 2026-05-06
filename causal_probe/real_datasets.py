"""Qualitative evaluation of the causal probe on real continuous-feature
datasets where no ground-truth tau exists.

We standardize X and y with population stats computed ON THE SAMPLE (not
a train/test split — the model is treating the whole dataset as one
observational context D), run the trained model, and compare the
predicted tau_hat to the marginal-association baseline.

Domain-knowledge sanity checks we can look for:
  * California Housing: MedInc (median income of a block) should be the
    strongest causal feature on median house value. AveBedrms, AveOccup
    should be weaker causal (more associational/proxy).
  * Wine Quality (red): alcohol content is the most consistent predictor
    of quality in the literature; volatile acidity negatively correlates
    but the causal story is contested.
  * Diabetes (sklearn built-in, 10 features): BMI and blood pressure
    are known to be causally related to disease progression; sex is
    associational (confounded by lifestyle/risk factors).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.eval import _load_model


DATASETS: dict[str, callable] = {}


def register(name: str):
    def wrap(fn):
        DATASETS[name] = fn
        return fn
    return wrap


@register("california_housing")
def _california_housing():
    from sklearn.datasets import fetch_california_housing
    ds = fetch_california_housing(as_frame=True)
    X = ds.data.values.astype(np.float64)
    y = ds.target.values.astype(np.float64)
    return X, y, list(ds.feature_names), "MedHouseVal (hundreds of thousands USD)"


@register("diabetes")
def _diabetes():
    from sklearn.datasets import load_diabetes
    ds = load_diabetes()
    X = ds.data.astype(np.float64)
    y = ds.target.astype(np.float64)
    return X, y, list(ds.feature_names), "disease progression 1yr (continuous)"


@register("wine_red")
def _wine_red():
    import urllib.request
    import io
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    with urllib.request.urlopen(url) as r:
        raw = r.read().decode()
    lines = [l for l in raw.splitlines() if l.strip()]
    header = [h.strip('"') for h in lines[0].split(";")]
    rows = [[float(v) for v in l.split(";")] for l in lines[1:]]
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, :-1], arr[:, -1], header[:-1], "quality (0-10 score)"


@register("boston")
def _boston():
    # sklearn removed this; load from the fetch_openml copy.
    from sklearn.datasets import fetch_openml
    ds = fetch_openml("boston", version=1, as_frame=True, parser="liac-arff")
    X = ds.data.select_dtypes(include=[np.number]).astype(np.float64).values
    y = np.asarray(ds.target, dtype=np.float64)
    names = [c for c in ds.data.columns if ds.data[c].dtype.kind in "fi"]
    return X, y, names, "MEDV (median home value $1000s)"


def _uci_csv(url: str, sep: str = ",", skip: int = 0, target_col: int = -1,
             names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import urllib.request
    with urllib.request.urlopen(url) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    lines = [l for l in raw.splitlines() if l.strip()]
    rows = []
    for l in lines[skip:]:
        parts = [p for p in l.split(sep) if p.strip() != ""]
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=np.float64)
    if target_col < 0:
        target_col = arr.shape[1] + target_col
    y = arr[:, target_col]
    X_cols = [i for i in range(arr.shape[1]) if i != target_col]
    X = arr[:, X_cols]
    if names is None:
        names = [f"X{i}" for i in range(X.shape[1])]
    else:
        names = [names[i] for i in X_cols]
    return X, y, names


@register("energy_efficiency")
def _energy():
    # UCI Energy Efficiency: 8 features, target = heating load.
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    try:
        df = pd.read_excel(url)
    except Exception:
        # Skip if xlrd/openpyxl absent on cluster
        raise
    df = df.dropna()
    X = df.iloc[:, :8].values.astype(np.float64)
    y = df.iloc[:, 8].values.astype(np.float64)  # Y1 = heating load
    names = [f"X{i+1}" for i in range(8)]
    return X, y, names, "Y1 heating load"


@register("abalone")
def _abalone():
    # UCI abalone. 7 continuous features (drop sex), y = rings (proxy for age).
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    cols = ["Sex", "Length", "Diameter", "Height", "Whole weight",
            "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    df = pd.read_csv(url, header=None, names=cols)
    df = df.drop(columns=["Sex"])  # categorical; drop for continuous probe
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.float64)
    return X, y, list(df.columns[:-1]), "Rings (age proxy)"


@register("auto_mpg")
def _auto_mpg():
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ["mpg", "cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year", "origin", "car_name"]
    df = pd.read_csv(url, sep=r"\s+", header=None, names=cols, na_values="?")
    df = df.drop(columns=["car_name", "origin"]).dropna()
    y = df["mpg"].values.astype(np.float64)
    X = df.drop(columns=["mpg"]).values.astype(np.float64)
    names = [c for c in df.columns if c != "mpg"]
    return X, y, names, "mpg (fuel efficiency)"


@register("friedman1")
def _friedman1():
    """Y = 10 sin(pi X1 X2) + 20 (X3 - 0.5)^2 + 10 X4 + 5 X5 + eps.
    X6..X10 are irrelevant (true tau ≈ 0)."""
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=1024, n_features=10, noise=1.0, random_state=42)
    names = [f"X{i+1}" for i in range(10)]
    return X, y, names, "friedman1 target (X1-X5 are causal; X6-X10 are noise)"


@register("friedman2")
def _friedman2():
    from sklearn.datasets import make_friedman2
    X, y = make_friedman2(n_samples=1024, noise=1.0, random_state=42)
    names = ["X1", "X2", "X3", "X4"]
    return X, y, names, "friedman2 target (all 4 features are causal)"


@register("friedman3")
def _friedman3():
    from sklearn.datasets import make_friedman3
    X, y = make_friedman3(n_samples=1024, noise=0.1, random_state=42)
    names = ["X1", "X2", "X3", "X4"]
    return X, y, names, "friedman3 target (all 4 features are causal)"


def _sachs_DISABLED():
    # All known Sachs data mirrors 404 as of this run. Kept the loader stub
    # for the future; for now rely on the Sachs-structured synthetic
    # benchmark in sachs_benchmark.py instead.
    raise RuntimeError("Sachs real data fetch disabled")


# @register("sachs_observational")
def _sachs_real():
    """Real Sachs 2005 flow-cytometry observational data (853 rows × 11 proteins).

    Target: pakts473 (Akt) — a key signaling endpoint.
    The other 10 proteins are the input features. Expert consensus DAG
    is encoded in sachs_benchmark.SACHS_EDGES; we can therefore compare
    the model's tau_hat on this REAL data to the total-effect prediction
    from the literature DAG.
    """
    import urllib.request, io
    import pandas as pd
    # Try multiple mirrors — bnlearn's repository changes URLs periodically.
    urls = [
        "https://www.bnlearn.com/book-crc/code/sachs.data.txt",
        "https://raw.githubusercontent.com/jmtomczak/causal-discovery/master/datasets/sachs.data.txt",
        "https://www.bnlearn.com/book-useR/code/ch4-sachs.data.txt",
    ]
    raw = None
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=10) as r:
                raw = r.read().decode()
                break
        except Exception:
            continue
    if raw is None:
        raise RuntimeError("Could not fetch Sachs data from any mirror")
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+")
    # Rename to match our Sachs node labels.
    rename = {"praf": "Praf", "pmek": "Pmek", "plcg": "Plcg",
              "PIP2": "PIP2", "PIP3": "PIP3", "p44.42": "p44_42",
              "pakts473": "pakts473", "PKA": "PKA", "PKC": "PKC",
              "P38": "P38", "pjnk": "pjnk"}
    df = df.rename(columns=rename)
    y = df["pakts473"].values.astype(np.float64)
    X_df = df.drop(columns=["pakts473"])
    return X_df.values.astype(np.float64), y, list(X_df.columns), "pakts473 (Akt activation)"


def evaluate_dataset(
    model, X: np.ndarray, y: np.ndarray, names: list[str], y_name: str,
    n_rows: int, device: str, rng: np.random.Generator, n_repeats: int = 10,
) -> dict:
    # Standardize columns
    mu_X = X.mean(axis=0)
    sd_X = X.std(axis=0, ddof=0).clip(min=1e-9)
    mu_y = float(y.mean()); sd_y = float(y.std(ddof=0).clip(min=1e-9))
    Xs = (X - mu_X) / sd_X
    ys = (y - mu_y) / sd_y

    p = Xs.shape[1]
    n_total = Xs.shape[0]

    # The model expects a fixed-p context. Run it n_repeats times on
    # resampled contexts of size n_rows (or the full dataset if smaller),
    # average the predictions.
    preds = []
    for _ in range(n_repeats):
        if n_total <= n_rows:
            X_ctx, y_ctx = Xs, ys
        else:
            idx = rng.choice(n_total, size=n_rows, replace=False)
            X_ctx = Xs[idx]; y_ctx = ys[idx]
        X_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(device)
        y_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            p_hat = model(X_t, y_t).squeeze(0).cpu().numpy()
        preds.append(p_hat)
    preds = np.stack(preds)                  # (n_repeats, p)
    tau_hat = preds.mean(axis=0)
    tau_hat_std = preds.std(axis=0)

    # Marginal baseline on the full sample (not the 512-row subsets).
    assoc = BASELINES["marginal"](Xs, ys)
    multi = BASELINES["multivariate"](Xs, ys)

    # Rank and normalize for clean display
    return {
        "y_name": y_name,
        "n_rows_full": n_total,
        "n_rows_context": min(n_rows, n_total),
        "n_repeats": n_repeats,
        "features": [
            {
                "name": nm,
                "tau_hat": float(tau_hat[i]),
                "tau_hat_std": float(tau_hat_std[i]),
                "assoc": float(assoc[i]),
                "multivariate": float(multi[i]),
            }
            for i, nm in enumerate(names)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    model = _load_model(args.ckpt, device=args.device)

    out = {}
    for name in args.datasets:
        print(f"\n=== dataset: {name} ===")
        try:
            X, y, names, y_name = DATASETS[name]()
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        if X.shape[1] > model.cfg.d_model:  # no meaningful cap but skip absurd cases
            pass
        # The trained model was trained at a specific p (heterogeneous or
        # fixed). If p doesn't match and model isn't heterogeneous, warn.
        print(f"  n={X.shape[0]}, p={X.shape[1]}, y={y_name}")
        res = evaluate_dataset(
            model, X, y, names, y_name,
            n_rows=args.n_rows, device=args.device,
            rng=np.random.default_rng(args.seed),
            n_repeats=args.n_repeats,
        )
        out[name] = res
        # Print side-by-side ranking
        feats = sorted(res["features"], key=lambda r: -abs(r["tau_hat"]))
        hdr = f"  {'rank':<5}{'feature':<22}{'tau_hat':>10}{'±std':>8}{'assoc':>10}{'multi':>10}"
        print(hdr)
        for i, r in enumerate(feats):
            print(f"  {i+1:<5}{r['name'][:20]:<22}{r['tau_hat']:>+10.3f}"
                  f"{r['tau_hat_std']:>8.3f}{r['assoc']:>+10.3f}{r['multivariate']:>+10.3f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"ckpt": args.ckpt, "results": out}, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
