#!/bin/bash
# Reproduce the p-sweep (accuracy vs number of features):
# does the p=8 -> p=13 drop persist on the current single foundation model?
# One model (canonical foundation-L seed0 FINAL, trained on p3-15 mixture),
# evaluated on held-out linear non-Gaussian SCMs at each p, padded to 15.
# Result (Pearson r of tau_hat vs true tau, model column):
#   p=5: 0.99   p=8: 0.97   p=13: 0.87   p=15: 0.81
# vs the earlier single-p specialist models in tab:held-out-scms: 0.92 (p=8), 0.82 (p=13).
set -euo pipefail
CK=causal_probe/results/probe_intersample_L_seed0.ckpt   # = foundation-L seed0 FINAL
for p in 5 8 13 15; do
  python -m causal_probe.eval --ckpt "$CK" --p "$p" --p-pad 15 \
    --scm-type linear --noise laplace --n-random-scms 300 --motif-repeats 1 \
    --device cpu --out-json causal_probe/results/p_sweep/eval_linear_p${p}.json
done
