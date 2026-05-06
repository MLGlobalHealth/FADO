# FADO: Learning to Explain Causal Effects with Prior-Fitted Networks for Explanations

Code accompanying the NeurIPS submission *FADO: Learning to Explain Causal
Effects with Prior-Fitted Networks for Explanations*. **Author and affiliation information has
been removed for double-blind review.**

This repository contains:

- The training code for the FADO foundation probe (`causal_probe/`).
- Baselines and evaluation harnesses (DoubleML, CausalPFN, DoPFN, causal
  forests, LiNGAM, NOTEARS, GES, `causal-learn`, etc.).
- Real-data benchmarks (IHDP, Tübingen, Lalonde, Hillstrom, Criteo Uplift,
  Sachs).
- The §1 SHAP-vs-causal motivating experiments through TabICL (`src/`).
- Cached evaluation artifacts (`causal_probe/results/`) sufficient to
  regenerate every paper table and figure without retraining.

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

## Reproduce paper tables and figures

The headline foundation probe is checked into the repo at
`causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt` (≈ 5 MB), along
with all JSONs/NPZs needed by the table-build pipeline. From a fresh clone:

```bash
uv run python scripts/build_tables.py        # regenerates 18 .tex tables
uv run python scripts/cross_family_heatmap.py # regenerates the heatmap PDF
uv run python scripts/make_figures.py --which all
```

Outputs land in `paper/figures/` (created on demand) and printed to stdout.

## Reproduce from scratch (training)

To retrain the headline probe from scratch instead of using the cached
checkpoint, see `reproduce.sh` for the full command list. Headline training
takes roughly 5 hours on a single L40S/H100 GPU.

```bash
bash reproduce.sh         # documents every step; commands are commented by default
```

## Repository layout

```
causal_probe/        Core training, eval, baselines (no tabicl import)
  ├── train.py       Headline probe trainer
  ├── eval.py        Headline probe evaluator
  ├── run_baseline.py Cross-regime baseline harness (8 regimes × 6 baselines)
  ├── model.py       FADO architecture
  ├── scm*.py        Synthetic SCM generators by family
  ├── tubingen.py    Tübingen pairs benchmark
  ├── *_rct.py       Real-data RCT benchmarks (Hillstrom, Criteo, Lalonde)
  ├── sachs_benchmark.py   Sachs protein dataset
  └── results/       Cached eval JSONs / NPZs / checkpoints
scripts/             Paper-artifact glue (table builder, figure makers)
src/                 §1 SHAP-vs-causal motivating experiments via TabICL
data/                IHDP and Tübingen benchmark inputs
pyproject.toml       Pinned upstream tabicl + baseline deps
```

## License

This code is released under the MIT License. See `LICENSE`.
