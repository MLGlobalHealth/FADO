"""Paired test: is FADO's null-variable error lower when the confounder is
observed than unobserved? (Pure shrinkage predicts no difference.) Trial is the
independent unit; bootstrap CI over trials + Wilcoxon signed-rank."""
import json, sys
import numpy as np
from scipy import stats

d = json.load(open(sys.argv[1]))
obs = {r['trial']: r for r in d if r['regime'] == 'observed'}
uno = {r['trial']: r for r in d if r['regime'] == 'unobserved'}
trials = list(obs)
n_seeds = len(obs[trials[0]].get('fado_seedvals', [1]))
print(f"{len(trials)} trials x {n_seeds} seeds\n")
print(f"{'method':>10}{'obs':>8}{'unobs':>8}{'Δ(u-o)':>9}{'95% CI (boot over trials)':>28}{'Wilcoxon p':>12}{'paired-t p':>11}")
rng = np.random.default_rng(0)
for m in ['fado', 'causalpfn', 'ols']:
    o = np.array([obs[t][m + '_abs'] for t in trials])
    u = np.array([uno[t][m + '_abs'] for t in trials])
    diff = u - o                       # >0 => worse when confounder hidden => genuine adjustment
    # bootstrap CI over trials
    boot = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        wp = stats.wilcoxon(u, o, alternative='greater').pvalue
    except Exception:
        wp = float('nan')
    tp = stats.ttest_rel(u, o, alternative='greater').pvalue
    print(f"{m:>10}{o.mean():>8.3f}{u.mean():>8.3f}{diff.mean():>+9.3f}"
          f"{f'[{lo:+.3f}, {hi:+.3f}]':>28}{wp:>12.4f}{tp:>11.4f}")
print("\nΔ>0 and CI excluding 0 => the adjustment response is real (not shrinkage).")
