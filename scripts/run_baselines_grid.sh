#!/bin/bash
# Run 9 baseline methods × 8 SCM settings = 72 cells, dump per-SCM .npz for paired-Δ CIs
# vs `probe_*.npz`. Restartable (skips cells whose .npz already exists).
#
# Designed for a single workstation with several cores rather than slurm. On 16
# cores the slow tail (causal_forest LinNG p=13) is ~50 min wall.
#
# Usage:
#   bash scripts/run_baselines_grid.sh                # run from repo root
#   PARALLEL=4 bash scripts/run_baselines_grid.sh    # adjust concurrency
#
# Each cell produces causal_probe/results/bootstrap/<method>_<tag>.{json,npz}.

set -uo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

OUT="$REPO/causal_probe/results/bootstrap"
LOG="$REPO/logs/baselines_grid"
mkdir -p "$OUT" "$LOG"

PY="$REPO/.venv/bin/python"
SEED=2025
NSCMS=100
NROWS=512
PARALLEL=${PARALLEL:-6}

# Settings: scm_type p noise tag
SETTINGS=(
  "linear     13 laplace  linear_p13"
  "linear     8  laplace  linear_p8"
  "mlp        5  laplace  mlp_p5"
  "nonlinear  5  laplace  poly_p5"
  "hidden     5  laplace  hidden_p5"
  "mixed      5  laplace  mixed_p5"
  "linear     5  laplace  linear_p5"
  "linear     5  gaussian gauss_p5"
)

METHODS=(pc ges fci notears causal_forest doubleml permutation marginal ridge)

JOBS_FILE=$(mktemp)
for m in "${METHODS[@]}"; do
  for s in "${SETTINGS[@]}"; do
    read -r scm_type p noise tag <<< "$s"
    npz="$OUT/${m}_${tag}.npz"
    [ -f "$npz" ] && continue
    echo "$m $scm_type $p $noise $tag" >> "$JOBS_FILE"
  done
done
N_JOBS=$(wc -l < "$JOBS_FILE")
echo "queued $N_JOBS cells (parallel=$PARALLEL)"

run_one() {
  local m=$1 scm_type=$2 p=$3 noise=$4 tag=$5
  local npz="$OUT/${m}_${tag}.npz"
  local json="$OUT/${m}_${tag}.json"
  local logf="$LOG/${m}_${tag}.log"
  local t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] start $m $tag" | tee -a "$LOG/_index.log"
  if OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
     "$PY" -m causal_probe.run_baseline \
       --method "$m" --scm-type "$scm_type" --p "$p" --noise "$noise" \
       --n-scms "$NSCMS" --n-rows "$NROWS" --seed "$SEED" \
       --out-json "$json" --out-npz "$npz" \
       > "$logf" 2>&1
  then
    local dt=$(( $(date +%s) - t0 ))
    echo "[$(date +%H:%M:%S)] done  $m $tag (${dt}s)" | tee -a "$LOG/_index.log"
  else
    local dt=$(( $(date +%s) - t0 ))
    echo "[$(date +%H:%M:%S)] FAIL  $m $tag (${dt}s) — see $logf" | tee -a "$LOG/_index.log"
  fi
}
export -f run_one
export OUT LOG PY SEED NSCMS NROWS REPO

xargs -a "$JOBS_FILE" -L1 -P "$PARALLEL" bash -c 'run_one "$@"' _

rm -f "$JOBS_FILE"
echo "all done."
ls "$OUT"/{notears,pc,ges,fci,doubleml,causal_forest,marginal,ridge,permutation}_*.npz 2>/dev/null | wc -l
