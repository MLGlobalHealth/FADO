import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("causal_probe/results/realdata_2026-09-29/selective_prediction_curve.json"))
OI = {"blue":"#0072B2","orange":"#D55E00","grey":"#999999"}
benches = [("chambers","Causal Chambers"),("semisynth","Semi-synthetic (ACIC)")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
for ax,(key,title) in zip(axes, benches):
    c = d["curves"][key]
    x = np.array(c["abstain_fractions"])*100
    sig = np.array(c["sigma_abstention_error"])
    rnd = np.array(c["random_abstention_error_mean"]); rsd = np.array(c["random_abstention_error_sd"])
    ax.plot(x, sig, "-o", color=OI["blue"], lw=2, ms=4, label=r"abstain by $\sigma$")
    ax.plot(x, rnd, "--s", color=OI["grey"], lw=1.5, ms=3, label="abstain at random")
    ax.fill_between(x, rnd-rsd, rnd+rsd, color=OI["grey"], alpha=0.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("% features abstained", fontsize=9)
    sp = c.get("spearman_sigma_err")
    ax.text(0.04, 0.06, r"$\rho_{\sigma,\mathrm{err}}=%.2f$"%sp, transform=ax.transAxes, fontsize=8)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel(r"mean $|\hat\tau-\tau|$ on retained", fontsize=9)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("paper/figures/selective_prediction.pdf", bbox_inches="tight")
fig.savefig("paper/figures/selective_prediction.png", dpi=150, bbox_inches="tight")
print("wrote paper/figures/selective_prediction.{pdf,png}")
