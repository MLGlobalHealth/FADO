# FADO: Learning Causal Explanations with Prior-Fitted Networks

Code accompanying the AISTATS submission *FADO: Learning Causal Explanations
with Prior-Fitted Networks*. **Author and affiliation information has been
removed for double-blind review.**

This repository contains:

- Training and evaluation code for FADO (`causal_probe/`), including the SCM
  prior families and the model architectures (`causal_probe/models/`).
- Baselines and evaluation harnesses (DoubleML, CausalPFN, Do-PFN, causal
  forests, DR-Learner, LiNGAM, NOTEARS, GES, do-Shapley / Causal Shapley, etc.).
- Semi-synthetic and real-data benchmarks (ACIC-2016, IHDP, LaLonde, Criteo
  Uplift, Hillstrom, STAR, RCT_Bench, Causal Chambers, Sachs).
- The §1 SHAP-vs-causal motivating experiments through TabICL (`src/`).
- Trained checkpoints and cached evaluation artifacts
  (`causal_probe/results/`) sufficient to regenerate every paper table and
  figure without retraining.

## Install

We use [`uv`](https://docs.astral.sh/uv/) for environment management. From a
fresh clone:

```bash
uv venv
uv pip install -e .                   # core deps + tabicl[shap]
uv pip install -e '.[baselines]'      # optional: baseline libraries
```

Python ≥ 3.11 is required. The `tabicl` dependency is pinned to upstream
[`soda-inria/tabicl`](https://github.com/soda-inria/tabicl) at the SHA in
`pyproject.toml`.

## Models

Two trained FADO models are used in the paper. Both checkpoints are included.

| Checkpoint (`causal_probe/results/`) | Architecture | Training prior | Used for |
|---|---|---|---|
| `probe_main_p5_100k_mixture_heavy.ckpt` (5 MB) | row/column transformer, d=96, 3+3 layers, 4 heads; p=5 | 5 SCM families (`--prior fado_p5`), 100k steps | §3 "FADO: p=5"; tab:held-out-scms, tab:all-baselines, tab:offlabel-pfn, tab:seed-robust, tab:motifs, tab:ablations, tab:type-flip-ablation, tab:lalonde, tab:criteo, tab:shap-contrast, tab:sachs-ckpts, tab:semisynth-realx |
| `probe_intersample_L_seed0.ckpt` (57 MB, weights only) | intersample transformer, d=384, 6+6 layers, 8 heads; p ∈ [3, 15] | 11 SCM families (`--prior foundation_l`), 150k steps | tab:acic-fp, tab:foundation-per-family, tab:shapley-comparison, tab:uncertainty-coverage, tab:uncertainty-quality, fig:selective-prediction, tab:rct-negcontrols, tab:chambers-perfeature, the 124-trial RCT_Bench result |

Single-family variants listed in tab:checkpoints are also included
(`probe_*_{15,20,25,30}k.ckpt`, Gaussian controls, hetero-p).

The training prior weights for both models are in `causal_probe/train.py`
(`_FADO_P5_PRIOR`, `_FOUNDATION_L_PRIOR`); each checkpoint also stores its full
training arguments under the `args` key. Result JSONs from foundation-model
runs record the checkpoint under its original training filename,
`probe_foundation_intersample_L_p3_15_intersample_L_seed0.ckpt`, which holds
the same weights as `probe_intersample_L_seed0.ckpt`. The ensemble row of
tab:uncertainty-quality also used foundation-model seeds 1–4 (listed under
`ensemble_ckpts` in
`causal_probe/results/realdata_2026-09-29/uncertainty_estimator_compare.json`);
those checkpoints are not shipped, so that one evaluation cannot be rerun, but
the cached JSON is sufficient to rebuild the table. The p=5 checkpoint was
trained with an earlier revision of the SCM generators (before a ±8σ clamp in
the shared standardisation helper was added); retraining with this code
follows the same recipe but is not bit-for-bit identical.

## Reproduce paper tables and figures

From a fresh clone, using only the cached artifacts:

```bash
mkdir -p paper/tables paper/figures
uv run python scripts/build_tables.py             # main + appendix tables
uv run python scripts/make_result_tables.py       # uncertainty, negative controls, IHDP decoys, Chambers
uv run python scripts/paired_delta_baselines.py   # tab:all-baselines-paired-ci
uv run python scripts/make_selective_prediction_fig.py
PYTHONPATH=. uv run python scripts/cross_family_heatmap.py   # fig: cross-family heatmap (cached cells)
PYTHONPATH=. uv run python -m causal_probe.figure_pred_vs_causal  # fig:pred-vs-causal (Fig. 1)
```

Outputs land in `paper/tables/` and `paper/figures/`.
`tab:all-baselines` is a hand-formatted merge of the baseline JSONs read by
`scripts/paired_delta_baselines.py`; `tab:wallclock` comes from
`scripts/wallclock_baselines.py` + `scripts/wallclock_figure.py`; the
anchor-audit appendix from `scripts/run_anchor_audit.py` +
`scripts/plot_anchor_audit.py`; the τ̂-vs-τ scatter from `causal_probe/plots.py`.

## Reproduce from scratch (training and evaluation)

`reproduce.sh` lists every training and evaluation command with the exact
settings used, grouped by paper section; commands are commented out by default.
Training takes about 2 h (p=5 model) and 10 h (foundation model) on a single
L40S / RTX 4090.

```bash
bash reproduce.sh         # documents every step; commands are commented by default
```

Some benchmarks need external data that is not redistributed here: the
ACIC-2016 covariates (`data/acic2016/x.csv`, from the public `aciccomp2016`
package), Criteo Uplift (downloaded via `scikit-uplift`), RCT_Bench, and the
Causal Chambers light-tunnel data (downloaded on first use).

## Repository layout

```
causal_probe/        Core training, eval, baselines (no tabicl import)
  ├── train.py       Trainer (--prior fado_p5 | foundation_l)
  ├── eval.py        Held-out SCM evaluator
  ├── model.py       Row/column transformer (p=5 model)
  ├── models/        Architecture registry (intersample = foundation model)
  ├── scm*.py        Synthetic SCM generators by family
  ├── run_baseline.py Cross-regime baseline harness
  ├── semisynth_acic_fp.py, ihdp_decoy_eval.py      Semi-synthetic decoys
  ├── *_rct.py, rct_negative_controls.py, rct_bench_*.py   Real RCTs
  ├── bootstrap_*.py, *uncertainty*.py, selective_prediction_curve.py   Uncertainty
  ├── causal_chambers_eval.py                       Causal Chambers
  ├── shapley_causal.py, eval_shapley_comparison.py do-Shapley / Causal Shapley
  └── results/       Checkpoints and cached eval JSONs / NPZs
scripts/             Paper-artifact glue (table builders, figure makers)
src/                 §1 SHAP-vs-causal motivating experiments via TabICL
data/                IHDP and Tübingen benchmark inputs
pyproject.toml       Pinned upstream tabicl + baseline deps
```

## License

This code is released under the MIT License. See `LICENSE`.
