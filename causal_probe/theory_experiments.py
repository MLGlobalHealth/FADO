"""Experiments illustrating the theory section.

This script produces two theorem-facing diagnostics:

1. Sample-size scaling under an identifiable prior (linear non-Gaussian) and
   a direction-ambiguous prior (linear Gaussian).
2. A two-SCM Gaussian toy where two causal models induce exactly the same
   observational distribution but imply different intervention effects.

The first diagnostic uses trained CausalProbe checkpoints. The second is
analytic/simulation-based: since the likelihoods are identical, the posterior
over the two SCMs remains equal to the prior for every n.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from causal_probe.baselines import BASELINES
from causal_probe.eval import _auroc, _load_model, _spearman_pearson
from causal_probe.scm import LinearNonGaussianSCM


def _json_float(x: float) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def _metrics(preds: np.ndarray, truths: np.ndarray) -> dict[str, float | None]:
    flat_p = preds.reshape(-1)
    flat_t = truths.reshape(-1)
    sp, pe = _spearman_pearson(flat_p, flat_t)
    ss_tot = float(np.sum((flat_t - flat_t.mean()) ** 2))
    r2 = 1.0 - float(np.sum((flat_p - flat_t) ** 2)) / max(ss_tot, 1e-12)
    labels_nonzero = (np.abs(flat_t) > 0.1).astype(int)
    zero_mask = np.abs(flat_t) <= 0.1
    return {
        "pearson": _json_float(pe),
        "spearman": _json_float(sp),
        "r2": _json_float(r2),
        "mse": _json_float(np.mean((flat_p - flat_t) ** 2)),
        "mae": _json_float(np.mean(np.abs(flat_p - flat_t))),
        "auroc_nonzero": _json_float(_auroc(np.abs(flat_p), labels_nonzero)),
        "mae_zero": _json_float(np.mean(np.abs(flat_p[zero_mask] - flat_t[zero_mask])))
        if zero_mask.any()
        else None,
    }


def run_nscale(
    *,
    name: str,
    ckpt: str,
    noise: str,
    p: int,
    n_grid: list[int],
    n_scms: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Evaluate one checkpoint on a fixed pool of SCMs across context sizes."""
    model = _load_model(ckpt, device=device)
    rng = np.random.default_rng(seed)
    max_n = max(n_grid)

    episodes = []
    for _ in range(n_scms):
        scm = LinearNonGaussianSCM(
            p=p,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            noise=noise,
        )
        # Draw one max-length context and use prefixes for smaller n. This
        # reduces Monte Carlo noise in the n-scaling curve.
        sample = scm.sample(
            n=max_n,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )
        episodes.append(sample)

    results: dict[str, Any] = {}
    for n in n_grid:
        preds, truths, marginals = [], [], []
        for sample in episodes:
            X_n = sample.X[:n]
            y_n = sample.y[:n]
            X_t = torch.from_numpy(X_n.astype(np.float32)).unsqueeze(0).to(device)
            y_t = torch.from_numpy(y_n.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(X_t, y_t).squeeze(0).cpu().numpy()
            preds.append(pred)
            truths.append(sample.tau)
            marginals.append(BASELINES["marginal"](X_n, y_n))
        pred_arr = np.stack(preds)
        truth_arr = np.stack(truths)
        marginal_arr = np.stack(marginals)
        results[str(n)] = {
            "model": _metrics(pred_arr, truth_arr),
            "marginal": _metrics(marginal_arr, truth_arr),
        }
        m = results[str(n)]["model"]
        print(
            f"{name:18s} n={n:5d} "
            f"MAE={m['mae']:.4f} Pearson={m['pearson']:.4f} R2={m['r2']:.4f}"
        )

    return {
        "name": name,
        "ckpt": ckpt,
        "noise": noise,
        "p": p,
        "n_scms": n_scms,
        "seed": seed,
        "results": results,
    }


def run_two_scm_toy(
    *,
    rho: float,
    n_grid: list[int],
    n_reps: int,
    seed: int,
) -> dict[str, Any]:
    """Two Gaussian SCMs with identical P(X,Y) and different tau.

    M1: X = e_x, Y = rho X + sqrt(1-rho^2) e_y, so tau_X = 2 rho.
    M2: Y = e_y, X = rho Y + sqrt(1-rho^2) e_x, so tau_X = 0.

    Both induce N(0, [[1, rho], [rho, 1]]). With equal prior mass, the
    posterior remains (1/2, 1/2) for every observational sample, and the
    squared-loss Bayes action is rho.
    """
    rng = np.random.default_rng(seed)
    cov = np.asarray([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    true_tau = {
        "M_x_to_y": 2.0 * rho,
        "M_y_to_x": 0.0,
    }
    bayes_tau = rho
    lower_bound_mse = rho**2

    rows: dict[str, Any] = {}
    for n in n_grid:
        marginal_estimates = {"M_x_to_y": [], "M_y_to_x": []}
        for model_name in marginal_estimates:
            draws = rng.multivariate_normal(
                mean=np.zeros(2),
                cov=cov,
                size=(n_reps, n),
            )
            x = draws[:, :, 0]
            y = draws[:, :, 1]
            x_center = x - x.mean(axis=1, keepdims=True)
            y_center = y - y.mean(axis=1, keepdims=True)
            cov_xy = np.mean(x_center * y_center, axis=1)
            var_x = np.mean(x_center**2, axis=1)
            marginal_tau = 2.0 * cov_xy / np.maximum(var_x, 1e-12)
            marginal_estimates[model_name] = marginal_tau

        risks = {}
        for estimator_name, est_by_model in {
            "posterior_mean": {
                "M_x_to_y": np.full(n_reps, bayes_tau),
                "M_y_to_x": np.full(n_reps, bayes_tau),
            },
            "marginal_association": marginal_estimates,
            "zero": {
                "M_x_to_y": np.zeros(n_reps),
                "M_y_to_x": np.zeros(n_reps),
            },
        }.items():
            mse_by_model = {
                model_name: float(np.mean((est - true_tau[model_name]) ** 2))
                for model_name, est in est_by_model.items()
            }
            risks[estimator_name] = {
                "mse_M_x_to_y": mse_by_model["M_x_to_y"],
                "mse_M_y_to_x": mse_by_model["M_y_to_x"],
                "max_mse": max(mse_by_model.values()),
                "mean_estimate_M_x_to_y": float(np.mean(est_by_model["M_x_to_y"])),
                "mean_estimate_M_y_to_x": float(np.mean(est_by_model["M_y_to_x"])),
            }

        rows[str(n)] = {
            "posterior_prob_M_x_to_y": 0.5,
            "posterior_prob_M_y_to_x": 0.5,
            "bayes_tau": bayes_tau,
            "true_tau": true_tau,
            "risks": risks,
        }
        print(
            f"toy n={n:5d} "
            f"Bayes max-MSE={risks['posterior_mean']['max_mse']:.4f} "
            f"marginal max-MSE={risks['marginal_association']['max_mse']:.4f}"
        )

    return {
        "rho": rho,
        "n_reps": n_reps,
        "seed": seed,
        "same_observational_covariance": cov.tolist(),
        "true_tau": true_tau,
        "bayes_tau": bayes_tau,
        "no_free_lunch_lower_bound_max_mse": lower_bound_mse,
        "results": rows,
    }


def plot_nscale(nscale: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    labels = {
        "identifiable_laplace": "linear non-Gaussian",
        "gaussian_ambiguous": "linear Gaussian",
    }
    colors = {
        "identifiable_laplace": "C0",
        "gaussian_ambiguous": "C3",
    }

    for key, payload in nscale["regimes"].items():
        ns = np.asarray([int(n) for n in payload["results"].keys()])
        order = np.argsort(ns)
        ns = ns[order]
        mae = np.asarray(
            [payload["results"][str(n)]["model"]["mae"] for n in ns],
            dtype=float,
        )
        pearson = np.asarray(
            [payload["results"][str(n)]["model"]["pearson"] for n in ns],
            dtype=float,
        )
        axes[0].plot(
            ns,
            mae,
            marker="o",
            color=colors.get(key),
            label=labels.get(key, key),
        )
        axes[1].plot(
            ns,
            pearson,
            marker="o",
            color=colors.get(key),
            label=labels.get(key, key),
        )

    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    axes[0].set_xlabel("context rows $n$")
    axes[1].set_xlabel("context rows $n$")
    axes[0].set_ylabel("MAE$(\\hat\\tau,\\tau)$")
    axes[1].set_ylabel("Pearson$(\\hat\\tau,\\tau)$")
    axes[0].set_title("Error vs. observational sample size")
    axes[1].set_title("Correlation vs. observational sample size")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_toy(toy: dict[str, Any], out_path: Path) -> None:
    ns = np.asarray([int(n) for n in toy["results"].keys()])
    ns = np.sort(ns)
    true_x_to_y = toy["true_tau"]["M_x_to_y"]
    true_y_to_x = toy["true_tau"]["M_y_to_x"]
    bayes_tau = toy["bayes_tau"]
    lower = toy["no_free_lunch_lower_bound_max_mse"]

    marginal_m2 = np.asarray(
        [
            toy["results"][str(n)]["risks"]["marginal_association"][
                "mean_estimate_M_y_to_x"
            ]
            for n in ns
        ]
    )
    risks = {
        "posterior mean": np.asarray(
            [
                toy["results"][str(n)]["risks"]["posterior_mean"]["max_mse"]
                for n in ns
            ]
        ),
        "marginal association": np.asarray(
            [
                toy["results"][str(n)]["risks"]["marginal_association"]["max_mse"]
                for n in ns
            ]
        ),
        "zero": np.asarray(
            [toy["results"][str(n)]["risks"]["zero"]["max_mse"] for n in ns]
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    axes[0].plot(ns, marginal_m2, marker="o", color="C1", label="marginal estimate")
    axes[0].axhline(true_y_to_x, color="black", linestyle="-", linewidth=1.0, label="true $\\tau$ under $Y\\to X$")
    axes[0].axhline(bayes_tau, color="C0", linestyle="--", linewidth=1.2, label="posterior mean")
    axes[0].axhline(true_x_to_y, color="C2", linestyle=":", linewidth=1.2, label="true $\\tau$ under $X\\to Y$")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("context rows $n$")
    axes[0].set_ylabel("estimated $\\tau_X$")
    axes[0].set_title("Same $P(X,Y)$, different effects")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    for label, vals in risks.items():
        axes[1].plot(ns, vals, marker="o", label=label)
    axes[1].axhline(lower, color="black", linestyle="--", linewidth=1.0, label="lower bound")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("context rows $n$")
    axes[1].set_ylabel("worst-case MSE")
    axes[1].set_title("No observational estimator escapes")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_toy_table(toy: dict[str, Any], out_path: Path) -> None:
    last_n = str(max(int(n) for n in toy["results"]))
    risks = toy["results"][last_n]["risks"]
    lines = [
        "% Auto-generated by causal_probe/theory_experiments.py",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Estimator & MSE if $X\\to Y$ & MSE if $Y\\to X$ & Worst-case MSE \\\\",
        "\\midrule",
    ]
    labels = {
        "posterior_mean": "Posterior mean",
        "marginal_association": "Marginal association",
        "zero": "Zero effect",
    }
    for key in ["posterior_mean", "marginal_association", "zero"]:
        row = risks[key]
        lines.append(
            f"{labels[key]} & "
            f"{row['mse_M_x_to_y']:.3f} & "
            f"{row['mse_M_y_to_x']:.3f} & "
            f"{row['max_mse']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            f"No-free-lunch lower bound & & & {toy['no_free_lunch_lower_bound_max_mse']:.3f} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laplace-ckpt", default="causal_probe/results/probe_main_p5_50k.ckpt")
    ap.add_argument("--gaussian-ckpt", default="causal_probe/results/probe_gauss_control_p5.ckpt")
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-grid", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--n-scms", type=int, default=100)
    ap.add_argument("--toy-reps", type=int, default=500)
    ap.add_argument("--toy-rho", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=20260429)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="causal_probe/results/theory_experiments.json")
    ap.add_argument("--fig-dir", default="paper/figures")
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    nscale = {
        "n_grid": args.n_grid,
        "regimes": {
            "identifiable_laplace": run_nscale(
                name="identifiable_laplace",
                ckpt=args.laplace_ckpt,
                noise="laplace",
                p=args.p,
                n_grid=args.n_grid,
                n_scms=args.n_scms,
                seed=args.seed,
                device=args.device,
            ),
            "gaussian_ambiguous": run_nscale(
                name="gaussian_ambiguous",
                ckpt=args.gaussian_ckpt,
                noise="gaussian",
                p=args.p,
                n_grid=args.n_grid,
                n_scms=args.n_scms,
                seed=args.seed + 1,
                device=args.device,
            ),
        },
    }
    toy = run_two_scm_toy(
        rho=args.toy_rho,
        n_grid=args.n_grid,
        n_reps=args.toy_reps,
        seed=args.seed + 2,
    )
    payload = {"nscale": nscale, "two_scm_toy": toy}
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_json}")

    plot_nscale(nscale, fig_dir / "f_theory_nscale.pdf")
    plot_toy(toy, fig_dir / "f_theory_toy_equivalence.pdf")
    write_toy_table(toy, fig_dir / "f_theory_toy_table.tex")


if __name__ == "__main__":
    main()
