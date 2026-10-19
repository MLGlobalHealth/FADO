"""Perturb-seq spike: FADO per-feature tau_hat vs measured single-gene CRISPRi effects.

EXPLORATORY / POST-SUBMISSION MATERIAL. Heavy caveats; see CAVEATS below and in
the output JSON. Sign-and-rank comparison only — NO magnitude claims.

Dataset
-------
Replogle et al. 2022, K562 "essential-scale" CRISPRi Perturb-seq, via the
scPerturb harmonization (Zenodo record 13350497):

    https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1

Size: 1.55 GB (documented from the Zenodo API on 2026-09-29). This was the
smallest viable option among the candidates researched:
  * ReplogleWeissman2022_K562_essential.h5ad  1547 MB  (CRISPRi, chosen)
  * ReplogleWeissman2022_rpe1.h5ad            1237 MB  (CRISPRi; RPE1, fewer
    control cells and less standard for this comparison)
  * NormanWeissman2019_filtered.h5ad           699 MB  (CRISPRa — activation,
    do(X=high); sign convention below is written for knockDOWN, so rejected)
  * ReplogleWeissman2022_K562_gwps.h5ad       8805 MB  (genome-wide; too big)
  * CausalBench (github.com/causalbench/causalbench) wraps the same Replogle
    data with its own preprocessed downloads of comparable size, but adds a
    package dependency and its own filtering choices; plain scanpy-convention
    processing of the scPerturb h5ad is more transparent, so CausalBench was
    not used.

Download step (not automated beyond a curl hint; ~1.5 GB):

    mkdir -p /tmp/perturbseq_data && curl -L -o \
      /tmp/perturbseq_data/ReplogleWeissman2022_K562_essential.h5ad \
      'https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1'

FADO ("probe") convention
-------------------------
The probe takes a standardized observational table (features X_1..X_p plus
target Y) and returns tau_hat_i per feature on the standardized scale,
tau_i ~= E[Y | do(X_i = +1 sd)] - E[Y | do(X_i = -1 sd)] in units of sd(Y).

Sign convention for the ground truth (READ CAREFULLY)
-----------------------------------------------------
CRISPRi knocks DOWN gene X_i: the intervention is do(X_i ~= low). We measure

    Delta_i = ( mean log-expr of Y in X_i-knockdown cells
                - mean log-expr of Y in non-targeting-control cells )
              / sd_control(Y).

If X_i POSITIVELY regulates Y, knocking X_i down DECREASES Y, so Delta_i < 0.
The causal effect of increasing X_i on Y therefore has sign(-Delta_i). FADO's
tau_hat is the response of Y to an INCREASE in X_i (low -> high contrast), so
we compare tau_hat against -Delta_i:

    correct sign      <=>  sign(tau_hat_i) == sign(-Delta_i)
    rank metric            Spearman rho( tau_hat, -Delta )

Because knockdown is an asymmetric, extreme intervention (X_i driven far below
its observational range, not a +/-1 sd contrast), magnitudes are NOT
comparable — we report sign agreement (on |Delta| >= threshold) and Spearman
rank correlation only.

Gene selection protocol (all choices recorded in the output JSON)
-----------------------------------------------------------------
Sufficient statistics are computed in one streaming pass: per-perturbation
per-gene sums and sums of squares of log-normalized expression
(counts-per-10k, log1p), plus control-cell stats and the full control-cell
matrix (cached to an .npz in --data-dir; the 1.5 GB h5ad is only read once).

  1. Valid perturbations: single-gene CRISPRi perturbations whose scPerturb
     `perturbation` label matches a measured gene, with >= --min-cells cells,
     and with VALIDATED on-target knockdown: Delta of the target gene's own
     expression <= --min-knockdown (default -0.5 control sds). This filters
     perturbations where CRISPRi silencing failed.
  2. Expressed genes: nonzero in >= --min-ctrl-nonzero-frac of control cells
     (default 0.25) and control mean log-expression >= --min-ctrl-mean.
     Mitochondrial genes (--exclude-gene-prefixes, default "MT-") are excluded
     from both Y and features: without this the Y criterion selects MT-ND2
     (moved by 372 perturbations), i.e. the mitochondrial-fraction stress/QC
     axis rather than gene regulation.
  3. Y = the expressed gene in the top --y-expr-quantile of control mean
     expression that is significantly moved (Welch |t| >= --t-signal) by the
     LARGEST number of valid perturbations. Ties break to higher expression.
  4. Signal features: valid perturbations (target gene expressed in controls,
     target != Y), ranked by |Delta_i| on Y among those with Welch
     |t| >= --t-signal; top --n-signal are taken (sign composition reported).
  5. Null decoys: --n-null valid perturbations with |t| < --t-null on Y and
     the smallest |Delta_i|, requiring >= --null-min-cells knockdown cells
     (well-powered measured nulls: a real intervention that measurably did
     ~nothing to Y).
  Total columns = n_signal + n_null features + Y  <= 15 (FADO envelope).

Observational input to FADO = non-targeting-control cells ONLY (scPerturb
label 'control'), log-normalized expression of the selected genes,
standardized on the control sample. Subsampled to --n-rows rows,
--n-repeats independent subsamples (default 5 seeds); tau_hat mean +/- sd.

CAVEATS (also embedded in the JSON)
------------------------------------------------------------------------
  * Intervention mismatch: CRISPRi ~80-95% knockdown is an extreme,
    asymmetric do(), not FADO's +/-1 sd contrast. Sign/rank only.
  * CRISPRi acts on transcription of X_i; observational "X_i" is mRNA
    abundance. Protein-level buffering means mRNA correlation structure and
    intervention response can decouple.
  * Single-cell log-normalized counts are zero-inflated, discrete, and depth-
    confounded — far from the continuous SCM families FADO was trained on.
    Library-size normalization removes only part of the technical covariance.
  * Gene-gene "effects" here are mostly indirect (through the whole
    regulatory network + cell state), and Delta is a total effect over ~7-9
    generations of growth post-infection, including selection/fitness
    composition effects on the measured population — not a clean
    instantaneous do() on a fixed population.
  * Selection uses the SAME interventional data that defines the ground
    truth (choosing Y as a gene hit by many perturbations biases toward
    trans-regulated programs); metrics are conditional on this selection.
  * p = 12 features is a vanishingly small slice of the network; every
    unmodeled co-regulated gene is an unobserved confounder of the observed
    columns. FADO assumes causal sufficiency over the table.
  * Trans effects in K562 are mostly small; with ~10 signal features the
    Spearman is over few points and noisy.
  * Non-targeting controls still carry dCas9-KRAB expression burden.

CLI
---
    uv run python -m causal_probe.perturbseq_eval --dry-run
    uv run python -m causal_probe.perturbseq_eval \
        --ckpt /path/to/probe.ckpt --data-dir /tmp/perturbseq_data

--dry-run runs the FULL selection + eval harness on a small synthetic
fixture (linear SCM with simulated knockdowns; clearly marked in the output)
so the pipeline is testable without the 1.5 GB download; if --ckpt is
omitted in dry-run mode a stub "model" (the marginal baseline) stands in for
FADO so shapes and JSON plumbing are exercised end to end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from causal_probe.baselines import BASELINES


def _load_any_model(ckpt_path: str, device: str = "cpu"):
    """Load either checkpoint format found in this repo:

    * probe-style: {'cfg': ProbeConfig kwargs, 'state_dict'} -> CausalProbe
      (causal_probe.eval._load_model);
    * foundation-style: {'arch': name, 'arch_kwargs', 'state_dict'} as written
      by causal_probe.train on the foundation sprint (the canonical
      probe_foundation_intersample_L_* checkpoints; 'intersample' is the
      historical name of the axial arch, remapped inside arch_axial).
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cfg" in ck:
        from causal_probe.eval import _load_model
        return _load_model(ckpt_path, device=device)
    arch = ck["arch"]
    if arch in ("intersample", "axial"):
        from causal_probe.models import arch_axial
        model = arch_axial.build(**ck["arch_kwargs"])
    else:
        from causal_probe.models import build_arch
        model = build_arch(arch, **ck["arch_kwargs"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model

H5AD_NAME = "ReplogleWeissman2022_K562_essential.h5ad"
H5AD_URL = (
    "https://zenodo.org/records/13350497/files/"
    "ReplogleWeissman2022_K562_essential.h5ad?download=1"
)
H5AD_SIZE_MB = 1547
STATS_CACHE = "replogle_k562_essential_stats.npz"

CONTROL_LABELS = {"control", "non-targeting", "nt", "ntc"}
OBS_PERT_CANDIDATES = ["perturbation", "gene", "perturbation_name", "target_gene"]

CAVEATS = [
    "CRISPRi knockdown is an extreme asymmetric do(X_i ~= low), not FADO's "
    "+/-1 sd contrast; comparison is sign and rank ONLY, no magnitude claims.",
    "CRISPRi intervenes on transcription; observational X_i is mRNA abundance; "
    "protein buffering can decouple the two.",
    "Single-cell log-normalized counts are zero-inflated, discrete, and "
    "sequencing-depth-confounded; far outside FADO's training families.",
    "Delta_i is a total network effect measured days after infection, "
    "including fitness/composition effects, not an instantaneous do().",
    "Gene selection uses the same interventional data as the ground truth; "
    "metrics are conditional on that selection.",
    "Only ~12 of ~8.5k measured genes enter the table; unmodeled co-regulated "
    "genes act as unobserved confounders (causal sufficiency violated).",
    "Few signal features => Spearman over few points; treat as qualitative.",
    "Non-targeting control cells still express dCas9-KRAB.",
]


# --------------------------------------------------------------------------
# Sufficient statistics (real data)
# --------------------------------------------------------------------------

def _compute_stats(h5ad_path: Path, cache_path: Path, max_ctrl_cells: int = 50000,
                   chunk: int = 4096) -> dict:
    """One streaming pass over the h5ad -> per-(perturbation, gene) sufficient
    stats of log-normalized expression + control-cell matrix. Cached to npz."""
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        return {k: z[k] for k in z.files}

    import anndata as ad
    import scipy.sparse as sp

    t0 = time.time()
    print(f"  loading {h5ad_path} (one-time; cached to {cache_path.name}) ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  loaded {adata.shape[0]} cells x {adata.shape[1]} genes "
          f"in {time.time()-t0:.0f}s")

    obs_col = next((c for c in OBS_PERT_CANDIDATES if c in adata.obs.columns), None)
    if obs_col is None:
        raise RuntimeError(
            f"no perturbation column among {OBS_PERT_CANDIDATES}; "
            f"obs columns: {list(adata.obs.columns)}")
    pert_series = adata.obs[obs_col].astype("category")
    codes = pert_series.cat.codes.to_numpy()
    pert_names = np.asarray(pert_series.cat.categories, dtype=str)
    gene_names = np.asarray(adata.var_names, dtype=str)
    n_cells, G = adata.shape
    P = len(pert_names)
    ctrl_code_set = {i for i, nm in enumerate(pert_names)
                     if nm.strip().lower() in CONTROL_LABELS}
    if not ctrl_code_set:
        raise RuntimeError(f"no control label found in {pert_names[:20]} ...")
    print(f"  perturbation column '{obs_col}': {P} labels, "
          f"controls = {sorted(pert_names[i] for i in ctrl_code_set)}")

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()

    S1 = np.zeros((P, G))
    S2 = np.zeros((P, G))
    n_per = np.zeros(P, dtype=np.int64)
    ctrl_nnz = np.zeros(G, dtype=np.int64)
    ctrl_rows: list[np.ndarray] = []
    is_ctrl = np.isin(codes, list(ctrl_code_set))

    for a in range(0, n_cells, chunk):
        b = min(a + chunk, n_cells)
        dense = np.asarray(X[a:b].todense(), dtype=np.float64)
        depth = dense.sum(axis=1, keepdims=True)
        depth[depth == 0] = 1.0
        logn = np.log1p(dense * (1e4 / depth))
        cc = codes[a:b]
        oh = sp.csr_matrix(
            (np.ones(b - a), (cc, np.arange(b - a))), shape=(P, b - a))
        S1 += oh @ logn
        S2 += oh @ (logn ** 2)
        np.add.at(n_per, cc, 1)
        m = is_ctrl[a:b]
        if m.any():
            ctrl_nnz += (dense[m] > 0).sum(axis=0).astype(np.int64)
            if sum(r.shape[0] for r in ctrl_rows) < max_ctrl_cells:
                ctrl_rows.append(logn[m].astype(np.float32))
        if (a // chunk) % 20 == 0:
            print(f"    rows {a}/{n_cells} ({time.time()-t0:.0f}s)")

    ctrl_matrix = np.concatenate(ctrl_rows, axis=0)[:max_ctrl_cells]
    ctrl_codes = np.asarray(sorted(ctrl_code_set), dtype=np.int64)
    ctrl_n = int(n_per[ctrl_codes].sum())
    ctrl_S1 = S1[ctrl_codes].sum(axis=0)
    ctrl_S2 = S2[ctrl_codes].sum(axis=0)

    # Map each perturbation label to a measured-gene index (-1 if no match).
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    pert_target = np.array([gene_idx.get(nm, -1) for nm in pert_names],
                           dtype=np.int64)
    pert_target[ctrl_codes] = -1

    stats = dict(
        gene_names=gene_names, pert_names=pert_names,
        pert_target=pert_target, n_per=n_per,
        S1=S1.astype(np.float32), S2=S2.astype(np.float32),
        ctrl_n=np.int64(ctrl_n), ctrl_S1=ctrl_S1, ctrl_S2=ctrl_S2,
        ctrl_nnz=ctrl_nnz, ctrl_matrix=ctrl_matrix,
        synthetic=np.bool_(False),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **stats)
    print(f"  stats cached ({time.time()-t0:.0f}s total, "
          f"{ctrl_matrix.shape[0]} control cells kept)")
    return stats


# --------------------------------------------------------------------------
# Synthetic fixture (--dry-run without the 1.5 GB download)
# --------------------------------------------------------------------------

def _synthetic_stats(seed: int = 0) -> dict:
    """Small linear-SCM world in the same sufficient-stats format.

    CLEARLY MARKED SYNTHETIC. 40 'genes' (gene00..gene38 + YGENE), the first
    30 individually 'perturbed' (knockdown = -2 sd shift on the target,
    propagated linearly to YGENE). 8 genes truly regulate YGENE; the rest are
    measured nulls. Exercises the full selection + eval + JSON path.
    """
    rng = np.random.default_rng(seed)
    G, n_pert_genes, n_ctrl, n_kd = 40, 30, 3000, 400
    y_gene = G - 1
    gene_names = np.array([f"gene{i:02d}" for i in range(G - 1)] + ["YGENE"],
                          dtype=str)
    beta = np.zeros(G)
    causal = rng.choice(n_pert_genes, size=8, replace=False)
    beta[causal] = rng.uniform(0.3, 0.9, size=8) * rng.choice([-1, 1], size=8)

    mu = rng.uniform(0.5, 2.0, size=G)
    mu[y_gene] = 3.5  # highly expressed so Y-selection can find it
    L = 0.15 * rng.standard_normal((G - 1, G - 1)) + np.eye(G - 1)

    def draw(n, kd_gene=None):
        base = rng.standard_normal((n, G - 1)) @ L.T
        if kd_gene is not None:
            base[:, kd_gene] -= 2.0  # knockdown: -2 sd on the target
        x = np.empty((n, G))
        x[:, : G - 1] = base + mu[: G - 1]
        x[:, y_gene] = mu[y_gene] + base @ beta[: G - 1] \
            + 0.5 * rng.standard_normal(n)
        return np.clip(x, 0.0, None)  # log-norm expression is nonnegative

    ctrl = draw(n_ctrl)
    pert_names = np.concatenate([gene_names[:n_pert_genes], ["control"]])
    P = len(pert_names)
    S1 = np.zeros((P, G), dtype=np.float32)
    S2 = np.zeros((P, G), dtype=np.float32)
    n_per = np.zeros(P, dtype=np.int64)
    for p in range(n_pert_genes):
        cells = draw(n_kd, kd_gene=p)
        S1[p], S2[p] = cells.sum(0), (cells ** 2).sum(0)
        n_per[p] = n_kd
    S1[-1], S2[-1], n_per[-1] = ctrl.sum(0), (ctrl ** 2).sum(0), n_ctrl

    pert_target = np.concatenate(
        [np.arange(n_pert_genes), [-1]]).astype(np.int64)
    return dict(
        gene_names=gene_names, pert_names=pert_names,
        pert_target=pert_target, n_per=n_per, S1=S1, S2=S2,
        ctrl_n=np.int64(n_ctrl), ctrl_S1=ctrl.sum(0), ctrl_S2=(ctrl ** 2).sum(0),
        ctrl_nnz=np.full(G, n_ctrl, dtype=np.int64), ctrl_matrix=ctrl.astype(np.float32),
        synthetic=np.bool_(True),
        _true_beta=beta,  # for the dry-run printout only
    )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def select_genes(stats: dict, *, n_signal: int, n_null: int, min_cells: int,
                 null_min_cells: int, min_knockdown: float,
                 min_ctrl_nonzero_frac: float, min_ctrl_mean: float,
                 y_expr_quantile: float, t_signal: float, t_null: float,
                 exclude_gene_prefixes: tuple[str, ...] = ("MT-",),
                 sign_balance: bool = False) -> dict:
    gene_names = stats["gene_names"]
    pert_target = np.asarray(stats["pert_target"])
    n_per = np.asarray(stats["n_per"], dtype=np.float64)
    S1 = np.asarray(stats["S1"], dtype=np.float64)
    S2 = np.asarray(stats["S2"], dtype=np.float64)
    ctrl_n = float(stats["ctrl_n"])
    mu_c = np.asarray(stats["ctrl_S1"]) / ctrl_n
    var_c = np.clip(np.asarray(stats["ctrl_S2"]) / ctrl_n - mu_c ** 2, 1e-12, None)
    sd_c = np.sqrt(var_c)
    nnz_frac = np.asarray(stats["ctrl_nnz"], dtype=np.float64) / ctrl_n

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = S1 / n_per[:, None]
        var_p = np.clip(S2 / n_per[:, None] - mean_p ** 2, 1e-12, None)
        delta = (mean_p - mu_c[None, :]) / sd_c[None, :]          # (P, G)
        t = (mean_p - mu_c[None, :]) / np.sqrt(
            var_p / n_per[:, None] + var_c[None, :] / ctrl_n)      # Welch t

    expressed = (nnz_frac >= min_ctrl_nonzero_frac) & (mu_c >= min_ctrl_mean)
    if exclude_gene_prefixes:
        excluded = np.array(
            [any(str(g).upper().startswith(pfx.upper())
                 for pfx in exclude_gene_prefixes) for g in gene_names])
        expressed &= ~excluded  # drops MT- genes from both Y and features

    # Valid perturbations: matched target gene, expressed target, enough
    # cells, validated on-target knockdown.
    valid = np.zeros(len(pert_target), dtype=bool)
    on_target_delta = np.full(len(pert_target), np.nan)
    for p, tg in enumerate(pert_target):
        if tg < 0 or n_per[p] < min_cells or not expressed[tg]:
            continue
        on_target_delta[p] = delta[p, tg]
        valid[p] = delta[p, tg] <= min_knockdown
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < n_signal + n_null + 1:
        raise RuntimeError(
            f"only {len(valid_idx)} valid perturbations after filters")

    # Y: expressed, top quantile by control mean, hit by most perturbations.
    expr_idx = np.where(expressed)[0]
    thr = np.quantile(mu_c[expr_idx], y_expr_quantile)
    y_cands = expr_idx[mu_c[expr_idx] >= thr]
    hits = (np.abs(t[valid_idx][:, y_cands]) >= t_signal).sum(axis=0)
    order = np.lexsort((mu_c[y_cands], hits))          # last key primary
    y_idx = int(y_cands[order[-1]])
    n_hits_y = int(hits[order[-1]])

    # Candidate features: valid perts, target expressed, target != Y.
    feats = np.array([p for p in valid_idx if pert_target[p] != y_idx])
    dY, tY = delta[feats, y_idx], t[feats, y_idx]

    sig_mask = np.abs(tY) >= t_signal
    if sign_balance:
        # Strongest |Delta| within each sign, half-and-half where available.
        # Without this the top-|Delta| list is typically all one sign (every
        # strong hit on a translation-factor Y is "knockdown lowers Y"),
        # which makes sign agreement and Spearman nearly degenerate.
        neg = feats[sig_mask & (dY < 0)][np.argsort(dY[sig_mask & (dY < 0)])]
        pos = feats[sig_mask & (dY > 0)][np.argsort(-dY[sig_mask & (dY > 0)])]
        half = n_signal // 2
        take_pos = pos[:min(half, len(pos))]
        take_neg = neg[:n_signal - len(take_pos)]
        if len(take_neg) < n_signal - len(take_pos):  # top up from pos
            take_pos = pos[:n_signal - len(take_neg)]
        sig = np.concatenate([take_neg, take_pos]).astype(int)
    else:
        sig = feats[sig_mask][np.argsort(-np.abs(dY[sig_mask]))][:n_signal]
    null_mask = (np.abs(tY) < t_null) & (n_per[feats] >= null_min_cells)
    nulls = feats[null_mask][np.argsort(np.abs(dY[null_mask]))][:n_null]

    chosen = np.concatenate([sig, nulls]).astype(int)
    feat_gene_idx = pert_target[chosen]
    return dict(
        y_idx=y_idx, y_gene=str(gene_names[y_idx]), y_n_sig_perts=n_hits_y,
        y_ctrl_mean=float(mu_c[y_idx]), pert_idx=chosen,
        feat_gene_idx=feat_gene_idx,
        feat_genes=[str(g) for g in gene_names[feat_gene_idx]],
        is_null=[bool(p in set(nulls.tolist())) for p in chosen],
        delta=[float(delta[p, y_idx]) for p in chosen],
        welch_t=[float(t[p, y_idx]) for p in chosen],
        n_kd_cells=[int(n_per[p]) for p in chosen],
        on_target_delta=[float(on_target_delta[p]) for p in chosen],
        n_valid_perts=int(len(valid_idx)),
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def evaluate(stats: dict, sel: dict, model, *, n_rows: int, n_repeats: int,
             seed: int, device: str, sign_threshold: float) -> dict:
    ctrl = np.asarray(stats["ctrl_matrix"], dtype=np.float64)
    cols = np.asarray(sel["feat_gene_idx"], dtype=int)
    Xo = ctrl[:, cols]
    yo = ctrl[:, sel["y_idx"]]
    mu, sd = Xo.mean(0), Xo.std(0).clip(min=1e-9)
    Xs = (Xo - mu) / sd
    ys = (yo - yo.mean()) / max(yo.std(), 1e-9)
    n = Xs.shape[0]
    print(f"  observational controls: n={n}, p={Xs.shape[1]} (+Y)")

    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=min(n_rows, n), replace=False)
        if model is None:  # dry-run stub
            preds.append(BASELINES["marginal"](Xs[idx], ys[idx]))
        else:
            X_t = torch.from_numpy(Xs[idx].astype(np.float32)).unsqueeze(0).to(device)
            y_t = torch.from_numpy(ys[idx].astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                preds.append(model(X_t, y_t).squeeze(0).cpu().numpy())
    preds = np.stack(preds)
    tau_hat, tau_sd = preds.mean(0), preds.std(0)

    marginal = BASELINES["marginal"](Xs, ys)
    multivariate = BASELINES["multivariate"](Xs, ys)

    delta = np.asarray(sel["delta"])
    truth = -delta  # sign convention: knockdown Delta<0 => positive effect
    is_null = np.asarray(sel["is_null"])
    sig = np.abs(delta) >= sign_threshold

    def _sign_agree(est):
        if sig.sum() == 0:
            return float("nan")
        return float(np.mean(np.sign(est[sig]) == np.sign(truth[sig])))

    metrics = {
        "spearman_tau_vs_negdelta": {
            "fado": _spearman(tau_hat, truth),
            "marginal": _spearman(marginal, truth),
            "multivariate": _spearman(multivariate, truth),
        },
        "sign_agreement_absdelta_ge_threshold": {
            "threshold": sign_threshold, "n_features_above": int(sig.sum()),
            "fado": _sign_agree(tau_hat),
            "marginal": _sign_agree(marginal),
            "multivariate": _sign_agree(multivariate),
        },
        "null_decoy_mean_abs_tau": {
            "n_nulls": int(is_null.sum()),
            "fado": float(np.abs(tau_hat[is_null]).mean()) if is_null.any() else float("nan"),
            "marginal": float(np.abs(marginal[is_null]).mean()) if is_null.any() else float("nan"),
            "multivariate": float(np.abs(multivariate[is_null]).mean()) if is_null.any() else float("nan"),
        },
    }
    table = [
        {
            "gene": sel["feat_genes"][i],
            "is_null_decoy": bool(is_null[i]),
            "n_kd_cells": sel["n_kd_cells"][i],
            "on_target_knockdown_delta": sel["on_target_delta"][i],
            "delta_on_Y": float(delta[i]),
            "welch_t_on_Y": sel["welch_t"][i],
            "neg_delta (effect sign truth)": float(truth[i]),
            "tau_hat_mean": float(tau_hat[i]),
            "tau_hat_sd": float(tau_sd[i]),
            "marginal": float(marginal[i]),
            "multivariate": float(multivariate[i]),
        }
        for i in range(len(cols))
    ]
    return {"metrics": metrics, "per_feature": table,
            "n_control_cells_used": int(n)}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="FADO vs Perturb-seq CRISPRi knockdowns (sign/rank only)")
    ap.add_argument("--ckpt", default=None,
                    help="FADO checkpoint (required unless --dry-run)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--data-dir", default="/tmp/perturbseq_data")
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--out", default="causal_probe/results/perturbseq_eval.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic fixture; no download, ckpt optional")
    # selection knobs (documented defaults; all recorded in JSON)
    ap.add_argument("--n-signal", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--null-min-cells", type=int, default=100)
    ap.add_argument("--min-knockdown", type=float, default=-0.5,
                    help="required on-target Delta (control sds) to accept a "
                         "perturbation as a real knockdown")
    ap.add_argument("--min-ctrl-nonzero-frac", type=float, default=0.25)
    ap.add_argument("--min-ctrl-mean", type=float, default=0.25)
    ap.add_argument("--y-expr-quantile", type=float, default=0.90)
    ap.add_argument("--t-signal", type=float, default=5.0)
    ap.add_argument("--t-null", type=float, default=2.0)
    ap.add_argument("--sign-threshold", type=float, default=0.1)
    ap.add_argument("--exclude-gene-prefixes", default="MT-",
                    help="comma-separated gene-name prefixes excluded from Y "
                         "and features (mitochondrial stress/QC axis)")
    ap.add_argument("--sign-balance", action="store_true",
                    help="pick strongest signal features from BOTH Delta "
                         "signs (default: top |Delta| regardless of sign, "
                         "which on this data is all one sign)")
    args = ap.parse_args()

    if args.n_signal + args.n_null + 1 > 15:
        raise SystemExit("n_signal + n_null + Y must be <= 15 (FADO envelope)")
    if not args.dry_run and args.ckpt is None:
        raise SystemExit("--ckpt is required unless --dry-run")

    if args.dry_run:
        print("=== Perturb-seq eval: DRY RUN on SYNTHETIC fixture ===")
        stats = _synthetic_stats(seed=args.seed)
    else:
        print("=== Perturb-seq eval: Replogle 2022 K562 essential CRISPRi ===")
        data_dir = Path(args.data_dir)
        h5ad = data_dir / H5AD_NAME
        cache = data_dir / STATS_CACHE
        if not h5ad.exists() and not cache.exists():
            raise SystemExit(
                f"{h5ad} not found. Download ({H5AD_SIZE_MB} MB):\n"
                f"  mkdir -p {data_dir} && curl -L -o {h5ad} '{H5AD_URL}'")
        stats = _compute_stats(h5ad, cache)

    sel = select_genes(
        stats, n_signal=args.n_signal, n_null=args.n_null,
        min_cells=args.min_cells, null_min_cells=args.null_min_cells,
        min_knockdown=args.min_knockdown,
        min_ctrl_nonzero_frac=args.min_ctrl_nonzero_frac,
        min_ctrl_mean=args.min_ctrl_mean,
        y_expr_quantile=args.y_expr_quantile,
        t_signal=args.t_signal, t_null=args.t_null,
        exclude_gene_prefixes=tuple(
            p for p in args.exclude_gene_prefixes.split(",") if p),
        sign_balance=args.sign_balance)

    print(f"  Y = {sel['y_gene']} (ctrl mean log-expr {sel['y_ctrl_mean']:.2f}, "
          f"significantly hit by {sel['y_n_sig_perts']} perturbations)")
    print(f"  valid perturbations after filters: {sel['n_valid_perts']}")
    print(f"  features: {sel['feat_genes']}")
    print(f"  null decoys: "
          f"{[g for g, isn in zip(sel['feat_genes'], sel['is_null']) if isn]}")

    model = _load_any_model(args.ckpt, device=args.device) if args.ckpt else None
    if model is None:
        print("  [dry-run] no --ckpt: stub model (marginal baseline) "
              "stands in for FADO")

    res = evaluate(stats, sel, model, n_rows=args.n_rows,
                   n_repeats=args.n_repeats, seed=args.seed,
                   device=args.device, sign_threshold=args.sign_threshold)

    hdr = (f"{'gene':<10}{'null?':>6}{'Delta_Y':>9}{'t':>8}{'-Delta':>8}"
           f"{'tau_hat':>9}{'+/-':>7}{'marg':>8}{'multi':>8}")
    print("\n" + hdr)
    for r in res["per_feature"]:
        print(f"{r['gene']:<10}{str(r['is_null_decoy']):>6}"
              f"{r['delta_on_Y']:>+9.3f}{r['welch_t_on_Y']:>8.1f}"
              f"{r['neg_delta (effect sign truth)']:>+8.3f}"
              f"{r['tau_hat_mean']:>+9.3f}{r['tau_hat_sd']:>7.3f}"
              f"{r['marginal']:>+8.3f}{r['multivariate']:>+8.3f}")
    m = res["metrics"]
    print(f"\n  Spearman(tau_hat, -Delta): fado={m['spearman_tau_vs_negdelta']['fado']:+.3f}  "
          f"marginal={m['spearman_tau_vs_negdelta']['marginal']:+.3f}  "
          f"multivariate={m['spearman_tau_vs_negdelta']['multivariate']:+.3f}")
    sa = m["sign_agreement_absdelta_ge_threshold"]
    print(f"  sign agreement (|Delta|>={sa['threshold']}, n={sa['n_features_above']}): "
          f"fado={sa['fado']:.3f}  marginal={sa['marginal']:.3f}  "
          f"multivariate={sa['multivariate']:.3f}")
    nd = m["null_decoy_mean_abs_tau"]
    print(f"  null-decoy mean |tau_hat| (n={nd['n_nulls']}): fado={nd['fado']:.3f}  "
          f"marginal={nd['marginal']:.3f}  multivariate={nd['multivariate']:.3f}")

    out = {
        "STATUS": "EXPLORATORY SPIKE — sign/rank comparison only, "
                  "no magnitude claims",
        "synthetic_fixture": bool(stats["synthetic"]),
        "tau_source": ("stub_marginal_baseline (dry-run, no ckpt)"
                       if model is None else "fado"),
        "dataset": {
            "name": "ReplogleWeissman2022_K562_essential (scPerturb)",
            "url": H5AD_URL, "size_mb": H5AD_SIZE_MB,
            "intervention": "CRISPRi knockdown (do(X_i ~= low))",
        } if not args.dry_run else {"name": "synthetic linear-SCM fixture"},
        "ckpt": args.ckpt, "seed": args.seed,
        "n_rows": args.n_rows, "n_repeats": args.n_repeats,
        "sign_convention": "truth sign = sign(-Delta_i); Delta_i = "
                           "(mean_kd(logY) - mean_ctrl(logY)) / sd_ctrl(logY)",
        "selection": {
            k: v for k, v in vars(args).items()
            if k in ("n_signal", "n_null", "min_cells", "null_min_cells",
                     "min_knockdown", "min_ctrl_nonzero_frac", "min_ctrl_mean",
                     "y_expr_quantile", "t_signal", "t_null", "sign_threshold",
                     "exclude_gene_prefixes", "sign_balance")
        },
        "selected": {
            "Y": sel["y_gene"], "Y_ctrl_mean_log_expr": sel["y_ctrl_mean"],
            "Y_n_significant_perturbations": sel["y_n_sig_perts"],
            "n_valid_perturbations": sel["n_valid_perts"],
            "features": sel["feat_genes"], "is_null_decoy": sel["is_null"],
        },
        **res,
        "caveats": CAVEATS,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
