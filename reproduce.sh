#!/usr/bin/env bash
# Reproduction script for "FADO: Learning Causal Explanations with
# Prior-Fitted Networks".
#
# All commands assume:
#   - the repo root is the current working directory
#   - .venv has been built via `uv venv && uv pip install -e '.[baselines]'`
#   - a single CUDA-capable GPU (>= 24 GB for the foundation model) for training
#
# Each block is commented out by default. Uncomment the steps you wish to run.
# The cached JSONs/NPZs/checkpoints under `causal_probe/results/` are already
# sufficient to regenerate every paper table and figure without retraining
# (Step 0); the remaining steps document full from-scratch reproduction.
#
# Two trained models are used in the paper (see README.md, "Models"):
#   FADO p=5     causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt
#   foundation   causal_probe/results/probe_intersample_L_seed0.ckpt
# Result JSONs written by the foundation-model runs record the checkpoint under
# its original training filename,
# probe_foundation_intersample_L_p3_15_intersample_L_seed0.ckpt; it is the same
# weights as probe_intersample_L_seed0.ckpt.

set -euo pipefail

# -----------------------------------------------------------------------------
# Step 0: regenerate paper tables and figures from cached artifacts (~1 minute)
# Output: paper/tables/*.tex, paper/figures/*.pdf
# -----------------------------------------------------------------------------
# mkdir -p paper/tables paper/figures
# uv run python scripts/build_tables.py
# uv run python scripts/make_result_tables.py
# uv run python scripts/paired_delta_baselines.py
# uv run python scripts/make_selective_prediction_fig.py
# uv run python scripts/cross_family_heatmap.py
# uv run python scripts/make_figures.py --which all

# -----------------------------------------------------------------------------
# Step 1a: train FADO p=5 (Sec. 3; "FADO: p=5" in tab:checkpoints)
# Output: causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt
# Wall clock: ~2 h on 1x L40S
# -----------------------------------------------------------------------------
# uv run python -m causal_probe.train \
#     --p 5 --n-rows 512 --batch-size 16 --max-steps 100000 \
#     --lr 3e-4 --warmup 200 \
#     --d-model 96 --n-heads 4 --n-row-layers 3 --n-col-layers 3 \
#     --scm-type mixture --prior fado_p5 --noise heavy --seed 0 \
#     --device cuda --ckpt-out causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt

# -----------------------------------------------------------------------------
# Step 1b: train the foundation model (intersample arch, p in [3, 15])
# Used for tab:acic-fp, tab:foundation-per-family, tab:shapley-comparison,
# tab:uncertainty-*, fig:selective-prediction, tab:rct-negcontrols,
# tab:chambers-perfeature and the 124-RCT result.
# Output: causal_probe/results/probe_intersample_L_seed0.ckpt
# Wall clock: ~10 h on 1x RTX 4090 / L40S
# -----------------------------------------------------------------------------
# uv run python -m causal_probe.train --arch intersample \
#     --p 15 --p-min 3 --p-max 15 --n-rows 512 --batch-size 8 --max-steps 150000 \
#     --lr 1e-4 --warmup 5000 \
#     --d-model 384 --n-heads 8 --n-row-layers 6 --n-col-layers 6 \
#     --scm-type mixture --prior foundation_l --noise heavy --seed 0 \
#     --num-workers 4 --save-every 1000 --eval-every 10000 \
#     --device cuda --ckpt-out causal_probe/results/probe_intersample_L_seed0.ckpt
# Seeds 1-4 (--seed 1..4) give the ensemble used in tab:uncertainty-quality.

# -----------------------------------------------------------------------------
# Step 2: held-out random-SCM evaluation (tab:held-out-scms)
# Output: causal_probe/results/eval_*.json
# Wall clock: ~5 minutes per checkpoint on 1x GPU
# -----------------------------------------------------------------------------
# uv run python -m causal_probe.eval \
#     --ckpt causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt \
#     --p 5 --scm-type linear --noise heavy --n-random-scms 500 \
#     --out-json causal_probe/results/eval_main_p5_50k.json
#
# Foundation model across the 10 eval families (tab:foundation-per-family):
# for fam in linear nonlinear mlp hidden mlp_hidden mixed treatment \
#            polynomial spline discrete_mix; do
#   uv run python -m causal_probe.eval \
#       --ckpt causal_probe/results/probe_intersample_L_seed0.ckpt \
#       --p 5 --p-pad 15 --scm-type "${fam}" --noise heavy --n-random-scms 500 \
#       --out-json "causal_probe/results/intersample_cf/eval_foundation_intersample_L_seed0_p5_pad15_${fam}.json"
# done

# -----------------------------------------------------------------------------
# Step 3: baselines on held-out SCMs (tab:all-baselines, tab:offlabel-pfn)
# Output: causal_probe/results/{method}_{setting}.json
# Wall clock: hours per setting on CPU; embarrassingly parallel
# -----------------------------------------------------------------------------
# bash scripts/run_baselines_grid.sh
# uv run python -m causal_probe.causalpfn_per_column_baseline --help
# uv run python -m causal_probe.dopfn_per_column_baseline --help

# -----------------------------------------------------------------------------
# Step 4: semi-synthetic ACIC-2016 decoys (tab:acic-fp) and IHDP decoys
# Needs data/acic2016/x.csv (ACIC 2016 covariates, from the public
# `aciccomp2016` package) and data/ihdp/.
# -----------------------------------------------------------------------------
# uv run python -m causal_probe.semisynth_acic_fp \
#     --ckpt causal_probe/results/probe_intersample_L_seed0.ckpt \
#     --acic-path data/acic2016/x.csv --out-json causal_probe/results/acic_fp_results.json
# uv run python -m causal_probe.ihdp_decoy_eval \
#     --ckpt causal_probe/results/probe_intersample_L_seed0.ckpt \
#     --ihdp-dir data/ihdp --out causal_probe/results/realdata_2026-09-29/ihdp_decoy_full.json

# -----------------------------------------------------------------------------
# Step 5: real RCTs with decoys (Sec. 4.4; tab:lalonde, tab:criteo, tab:shap-contrast)
# Wall clock: ~30 minutes total on CPU
# -----------------------------------------------------------------------------
# CK=causal_probe/results/probe_main_p5_100k_mixture_heavy.ckpt
# uv run python -m causal_probe.hillstrom_rct --ckpt "$CK"
# uv run python -m causal_probe.criteo_rct    --ckpt "$CK"   # respects $SKLIFT_HOME
# uv run python -m causal_probe.lalonde_rct   --ckpt "$CK"
# uv run python -m causal_probe.sachs_benchmark --ckpt "$CK"
#
# Negative controls (tab:rct-negcontrols) and the 124 RCT_Bench trials:
# CK=causal_probe/results/probe_intersample_L_seed0.ckpt
# uv run python -m causal_probe.rct_negative_controls --ckpt "$CK" --seed 2025 --n-repeats 5 \
#     --out causal_probe/results/realdata_2026-09-29/rct_negative_controls_full.json
# uv run python -m causal_probe.star_rct --ckpt "$CK" --negative-controls --seed 2025 --n-repeats 5 \
#     --out causal_probe/results/realdata_2026-09-29/star_negcontrols_full.json
# uv run python -m causal_probe.rct_bench_eval --ckpt "$CK" --data-dir /path/to/RCT_Bench/cleaned_data

# -----------------------------------------------------------------------------
# Step 6: uncertainty (Sec. 4.5; tab:uncertainty-coverage, tab:uncertainty-quality,
#         fig:selective-prediction)
# -----------------------------------------------------------------------------
# CK=causal_probe/results/probe_intersample_L_seed0.ckpt
# uv run python -m causal_probe.bootstrap_coverage --ckpt "$CK" --seed 100 --n-scms 200 --n-boot 200 \
#     --out causal_probe/results/realdata_2026-09-29/bootstrap_coverage.json
# uv run python -m causal_probe.uncertainty_estimator_compare --ckpt "$CK" --ckpt-dir causal_probe/results/ \
#     --out causal_probe/results/realdata_2026-09-29/uncertainty_estimator_compare.json
# uv run python -m causal_probe.causalpfn_uncertainty_compare --ckpt "$CK"
# uv run python -m causal_probe.selective_prediction_curve --ckpt "$CK" --seed 2025 \
#     --acic-path data/acic2016/x.csv \
#     --out causal_probe/results/realdata_2026-09-29/selective_prediction_curve.json

# -----------------------------------------------------------------------------
# Step 7: Causal Chambers (tab:chambers-perfeature) and causal-Shapley
#         comparison (tab:shapley-comparison)
# -----------------------------------------------------------------------------
# CK=causal_probe/results/probe_intersample_L_seed0.ckpt
# uv run python -m causal_probe.causal_chambers_eval --ckpt "$CK" --seed 2025 --n-seeds 5 \
#     --out causal_probe/results/realdata_2026-09-29/causal_chambers_full_L.json
# uv run python -m causal_probe.eval_shapley_comparison --ckpt "$CK" --p 5 --seed 100 \
#     --n-random-scms 200 --out-json causal_probe/results/shapley_comparison/shapley_comparison_intersample_L_seed0_FINAL.json

# -----------------------------------------------------------------------------
# Step 8: Sec. 1 SHAP-vs-causal motivation through TabICL
# Output: results/shap_failure_results.pkl, results/shap_failure_table.tex
# Wall clock: ~4h on CPU (1000 DGPs)
# -----------------------------------------------------------------------------
# uv run python src/shap_failure_identifiable.py
# uv run python src/shap_failure_identifiable_reduce.py \
#     --chunks-dir results/chunks_ident \
#     --out-tex results/shap_failure_identifiable.tex

echo "reproduce.sh: edit this script and uncomment steps to run them."
