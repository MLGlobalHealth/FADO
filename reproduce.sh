#!/usr/bin/env bash
# Reproduction script for "FADO: Learning to Explain Causal Effects with Prior-Fitted Networks for Explanations".
#
# All commands assume:
#   - the repo root is the current working directory
#   - .venv has been built via `uv venv && uv pip install -e .[baselines]`
#   - a single CUDA-capable GPU (≥ 16 GB) is available for training steps
#
# Each block is commented out by default. Uncomment the steps you wish to
# run. The cached JSONs/NPZs/checkpoints under `causal_probe/results/` are
# already sufficient to regenerate every paper table and figure without
# retraining; the training commands below are documented for full
# from-scratch reproduction.
#
# Wall-clock estimates were measured on a single NVIDIA L40S GPU.

set -euo pipefail

# -----------------------------------------------------------------------------
# Step 0: regenerate paper tables and figures from cached artifacts
# Wall clock: ~30 seconds
# -----------------------------------------------------------------------------
# uv run python scripts/build_tables.py
# uv run python scripts/cross_family_heatmap.py
# uv run python scripts/make_figures.py --which all

# -----------------------------------------------------------------------------
# Step 1: train the headline foundation probe (100k-step mixture-heavy)
# Output: causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt
# Wall clock: ~5h on 1× L40S
# -----------------------------------------------------------------------------
# uv run python -m causal_probe.train \
#     --regime mixture-heavy \
#     --p 5 --n 500 --steps 100000 \
#     --out causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt \
#     --seed 2025

# -----------------------------------------------------------------------------
# Step 2: headline evaluation (Pearson on linear-non-Gaussian heavy)
# Output: causal_probe/results/eval_main_p5_50k.json
# Wall clock: ~5 minutes on 1× L40S
# -----------------------------------------------------------------------------
# uv run python -m causal_probe.eval \
#     --ckpt causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt \
#     --regime linear-non-gaussian-heavy \
#     --p 5 --n 500 --seed 2025 \
#     --out causal_probe/results/eval_main_p5_50k.json

# -----------------------------------------------------------------------------
# Step 3: cross-regime baselines (DoubleML / CausalPFN / DoPFN / Causal Forest /
#         LiNGAM / NOTEARS / GES) over 8 regimes, used in tab:causal-baselines
#         and tab:held-out-scms
# Output: causal_probe/results/{baseline}_{regime}_{noise}.json
# Wall clock: ~6h CPU per regime × 8 regimes; embarrassingly parallel
# -----------------------------------------------------------------------------
# for regime in linear_p5_gaussian linear_p5_laplace linear_p8_laplace \
#               linear_p13_laplace mixed_p5_laplace mlp_p5_laplace \
#               nonlinear_p5_laplace hidden_p5_laplace; do
#   uv run python -m causal_probe.run_baseline \
#       --regime "${regime}" --seed 2025 \
#       --out-dir causal_probe/results/
# done

# -----------------------------------------------------------------------------
# Step 4: bootstrap CIs for paired-bootstrap CIs in tab:causal-baselines
# Output: causal_probe/results/bootstrap/{probe,lingam}_{regime}.npz
# Wall clock: ~30 minutes per regime
# -----------------------------------------------------------------------------
# uv run python -m causal_probe.bootstrap_cis \
#     --ckpt causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt \
#     --out-dir causal_probe/results/bootstrap/

# -----------------------------------------------------------------------------
# Step 5: real-data benchmarks (IHDP, Tübingen, Lalonde, Hillstrom, Criteo, Sachs)
# Wall clock: ~30 minutes total on CPU
# -----------------------------------------------------------------------------
# uv run python -m causal_probe.tubingen \
#     --ckpt causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt \
#     --out causal_probe/results/tubingen.json
# uv run python -m causal_probe.hillstrom_rct
# uv run python -m causal_probe.criteo_rct        # respects $SKLIFT_HOME
# uv run python -m causal_probe.lalonde_rct
# uv run python -m causal_probe.sachs_benchmark

# -----------------------------------------------------------------------------
# Step 6: §1 SHAP-vs-causal motivation through TabICL
# Output: results/shap_failure_results.pkl, results/shap_failure_table.tex
# Wall clock: ~4h on CPU (1000 DGPs)
# -----------------------------------------------------------------------------
# uv run python src/shap_failure_identifiable.py
# uv run python src/shap_failure_identifiable_reduce.py \
#     --chunks-dir results/chunks_ident \
#     --out-tex results/shap_failure_identifiable.tex

# -----------------------------------------------------------------------------
# Step 7: rebuild paper tables + figures from regenerated artifacts
# -----------------------------------------------------------------------------
# uv run python scripts/build_tables.py
# uv run python scripts/cross_family_heatmap.py
# uv run python scripts/make_figures.py --which all

echo "reproduce.sh: edit this script and uncomment steps to run them."
