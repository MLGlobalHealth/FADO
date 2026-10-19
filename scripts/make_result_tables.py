#!/usr/bin/env python
"""Regenerate the five real-data LaTeX table-input files from result JSONs.

Each output is a bare ``\\begin{tabular}...\\end{tabular}`` (no table float, no
caption -- those live in main.tex), matching the auto-generated style of
``paper/tables/acic_fp.tex``: first two lines are comments (source json + label),
booktabs rules, every number wrapped in ``$...$``, ``—`` for N/A.

Run:  uv run python scripts/make_result_tables.py

Sources (all under causal_probe/results/realdata_2026-09-29/):
  1. bootstrap_coverage.json               -> uncertainty_coverage.tex
  2. uncertainty_estimator_compare.json
     + causalpfn_uncertainty_compare.json  -> uncertainty_quality.tex
  3. rct_negative_controls_full.json
     + star_negcontrols_full.json          -> rct_negative_controls.tex
  4. ihdp_decoy_full.json                   -> ihdp_decoy_fp.tex
  5. causal_chambers_full_L.json           -> chambers_perfeature.tex
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "causal_probe", "results", "realdata_2026-09-29")
TABLES = os.path.join(REPO, "paper", "tables")

NA = "—"  # em dash for N/A, matches acic_fp.tex


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def fmt(x, nd=3):
    """Format a number inside $...$, or NA for None/NaN."""
    if x is None:
        return NA
    try:
        if isinstance(x, float) and math.isnan(x):
            return NA
    except TypeError:
        return NA
    return f"${x:.{nd}f}$"


def write(fname, lines):
    path = os.path.join(TABLES, fname)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# 1. uncertainty_coverage.tex
# ---------------------------------------------------------------------------
def make_coverage():
    src = "bootstrap_coverage.json"
    d = load(src)["results"]
    # regime key -> (display name, parenthetical tag)
    regimes = [
        ("linear:laplace", "Linear", "(identifiable)"),
        ("linear:gaussian", "Linear", "(non-ident.)"),
        ("hidden:laplace", "Hidden conf.", "(non-ident.)"),
        ("mlp:laplace", "MLP", ""),
        ("mixture:laplace", "Mixture", ""),
    ]
    lines = [
        f"% Auto-generated from causal_probe/results/realdata_2026-09-29/{src}",
        "% tab:uncertainty-coverage",
        # Regime | 4 nominal (all) | cause@95 | null@95 | spearman
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Empirical coverage (all features), nominal} & \multicolumn{2}{c}{Coverage@95} & \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r"Regime & $50$ & $80$ & $90$ & $95$ & Cause & Null & $\rho(\sigma,\mathrm{err})$ \\",
        r"\midrule",
    ]
    for key, name, tag in regimes:
        v = d[key]
        cov = v["coverage"]
        label = f"{name} {tag}".strip()
        row = [
            label,
            fmt(cov["50"]["all"]["coverage"]),
            fmt(cov["80"]["all"]["coverage"]),
            fmt(cov["90"]["all"]["coverage"]),
            fmt(cov["95"]["all"]["coverage"]),
            fmt(cov["95"]["cause"]["coverage"]),
            fmt(cov["95"]["null"]["coverage"]),
            fmt(v["spearman_std_err"]),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write("uncertainty_coverage.tex", lines), 8  # 8 columns


# ---------------------------------------------------------------------------
# 2. uncertainty_quality.tex
# ---------------------------------------------------------------------------
def make_quality():
    est_src = "uncertainty_estimator_compare.json"
    pfn_src = "causalpfn_uncertainty_compare.json"
    est = load(est_src)["results"]
    pfn = load(pfn_src)["results"]
    # FADO columns come from the estimator json's `bootstrap` block (same
    # row-bootstrap estimator as the causalpfn json's `fado_bootstrap_sigma`,
    # but computed over n_scms=200 rather than 100; values differ by <0.02).
    regimes = [
        ("linear:laplace", "Linear (ident.)"),
        ("linear:gaussian", "Linear (non-ident.)"),
        ("hidden:laplace", "Hidden conf."),
        ("mlp:laplace", "MLP"),
        ("mixture:laplace", "Mixture"),
    ]
    lines = [
        f"% Auto-generated from causal_probe/results/realdata_2026-09-29/{est_src}",
        f"%   and causal_probe/results/realdata_2026-09-29/{pfn_src}",
        "% tab:uncertainty-quality",
        "% FADO cols: estimator json `bootstrap` (n_scms=200). CausalPFN cols: causalpfn json (n_scms=100).",
        "% mixture:laplace + linear:gaussian are absent from the CausalPFN run (— in those cells).",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{FADO bootstrap} & \multicolumn{2}{c}{$K{=}5$ ensemble} & \multicolumn{2}{c}{CausalPFN native} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Regime & $\rho$ & sel.\ gain & $\rho$ & sel.\ gain & $\rho$ & sel.\ gain \\",
        r"\midrule",
    ]
    for key, name in regimes:
        e = est.get(key)
        p = pfn.get(key)
        if e is not None:
            fado_rho = fmt(e["bootstrap"]["spearman_sigma_err"])
            fado_gain = fmt(e["bootstrap"]["selective_gain"])
            ens_rho = fmt(e["ensemble"]["spearman_sigma_err"])
            ens_gain = fmt(e["ensemble"]["selective_gain"])
        else:
            fado_rho = fado_gain = ens_rho = ens_gain = NA
        if p is not None:
            pfn_rho = fmt(p["causalpfn_native"]["spearman_sigma_err"])
            pfn_gain = fmt(p["causalpfn_native"]["selective_gain"])
        else:
            pfn_rho = pfn_gain = NA
        row = [name, fado_rho, fado_gain, ens_rho, ens_gain, pfn_rho, pfn_gain]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write("uncertainty_quality.tex", lines), 7


# ---------------------------------------------------------------------------
# 3. rct_negative_controls.tex
# ---------------------------------------------------------------------------
def make_rct():
    rct_src = "rct_negative_controls_full.json"
    star_src = "star_negcontrols_full.json"
    rct = load(rct_src)
    star = load(star_src)
    methods = ["fado", "shap", "permutation", "lingam", "marginal"]
    headers = {
        "fado": "FADO",
        "shap": "SHAP",
        "permutation": "Perm.",
        "lingam": "LiNGAM",
        "marginal": "Marginal",
    }

    def dataset_row(label, fado_val, comp_summ):
        cells = [label]
        for m in methods:
            if m == "fado":
                cells.append(fmt(fado_val))
            else:
                v = comp_summ.get(m)
                cells.append(fmt(v) if v is not None else NA)
        return " & ".join(cells) + r" \\"

    ds = rct["datasets"]
    rows = []
    # Hillstrom
    h = ds["hillstrom"]["summary"]
    rows.append(dataset_row(
        "Hillstrom",
        h["mean_abs_tau_hat_T"],
        {m: h["comparators"][m]["mean_abs_tau_hat_T"] for m in methods if m != "fado"},
    ))
    # LaLonde
    l = ds["lalonde"]["summary"]
    rows.append(dataset_row(
        "LaLonde",
        l["mean_abs_tau_hat_T"],
        {m: l["comparators"][m]["mean_abs_tau_hat_T"] for m in methods if m != "fado"},
    ))
    # STAR -- different schema: FADO = negative_control_MAE, comparators in
    # negative_control_MAE_comparators (keyed identically).
    rows.append(dataset_row(
        "STAR",
        star["negative_control_MAE"],
        {m: star["negative_control_MAE_comparators"].get(m) for m in methods if m != "fado"},
    ))
    # Criteo -- clean-six subset: controls with top_freq < 0.9. The six
    # near-constant binary features (top_freq >= 0.9) are degenerate as pseudo-
    # treatments and inflate every median-cut estimator; FADO's full-12 mean is
    # 0.117 vs 0.027 on the clean six.
    cr_controls = ds["criteo"]["controls"]
    clean = [c for c in cr_controls if c["top_freq"] < 0.9]
    fado_clean = sum(abs(c["tau_hat_T_mean"]) for c in clean) / len(clean)
    comp_clean = {}
    for m in methods:
        if m == "fado":
            continue
        vals = [abs(c["comparators"][m]["tau_hat_T"]) for c in clean
                if c["comparators"][m]["tau_hat_T"] is not None]
        comp_clean[m] = (sum(vals) / len(vals)) if vals else None
    rows.append(dataset_row(
        f"Criteo (clean {len(clean)})",
        fado_clean,
        comp_clean,
    ))

    lines = [
        f"% Auto-generated from causal_probe/results/realdata_2026-09-29/{rct_src}",
        f"%   STAR row from causal_probe/results/realdata_2026-09-29/{star_src} (different schema: negative_control_MAE)",
        "% tab:rct-negcontrols",
        "% Criteo = clean-six subset (controls with top_freq < 0.9); full-12 FADO mean is 0.117, inflated by near-constant binary pseudo-treatments.",
        "% Values are mean |tau_hat_T| over negative-control pseudo-treatments (lower is better; all should be ~0).",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Dataset & " + " & ".join(headers[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    lines += rows
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write("rct_negative_controls.tex", lines), 1 + len(methods)


# ---------------------------------------------------------------------------
# 4. ihdp_decoy_fp.tex  (companion to acic_fp.tex)
# ---------------------------------------------------------------------------
def make_ihdp():
    src = "ihdp_decoy_full.json"
    d = load(src)
    # IHDP decoy JSON has FP families Noise/Proxy/Leak (NO conf-child column,
    # unlike acic_fp.tex). Methods: F (fado), Mg (marginal), Mv (multivariate),
    # plus the oracle do-Shapley (dS) and Causal-Shapley (cS) column groups --
    # these exist here because we own the IHDP-decoy generator.
    families = ["noise", "proxy", "leak"]
    fam_hdr = {"noise": "Noise", "proxy": "Proxy", "leak": "Leak"}
    # within each family, these methods in order
    methods = [
        ("fado", "F"),
        ("marginal", "Mg"),
        ("multivariate", "Mv"),
        ("do_shapley", "dS"),
        ("causal_shapley", "cS"),
    ]
    row_label = {"linear": "Linear", "mixed": "Mixed nonlinear",
                 "mixed_nonlinear": "Mixed nonlinear", "nonlinear": "Mixed nonlinear",
                 "interaction": "Interaction"}

    def key(method, fam):
        return f"{method}_fp_{fam}"

    n_meth = len(methods)
    n_fam = len(families)
    ncols = 2 + n_fam * n_meth  # Regime + Family + families*methods

    # column spec: ll then r's
    colspec = "ll" + "r" * (n_fam * n_meth)
    lines = [
        f"% Auto-generated from causal_probe/results/realdata_2026-09-29/{src}",
        "% tab:ihdp-decoy-fp",
        "% Companion to acic_fp.tex. FP families Noise/Proxy/Leak (IHDP has no conf-child column).",
        "% Methods: F=FADO, Mg=marginal, Mv=multivariate, dS=do-Shapley (oracle), cS=Causal-Shapley (oracle).",
        "% False-positive attribution on decoy features; lower is better (0 = ignores the decoy).",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]
    # top header: family multicolumn groups
    top = ["", ""]
    for fam in families:
        top.append(rf"\multicolumn{{{n_meth}}}{{c}}{{{fam_hdr[fam]}}}")
    lines.append(" & ".join(top) + r" \\")
    # cmidrules under each family group
    cmids = []
    start = 3
    for _ in families:
        end = start + n_meth - 1
        cmids.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cmids))
    # sub-header
    sub = ["Regime", "Family"] + [abbr for _, abbr in methods] * n_fam
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    def emit_block(entries, regime_name):
        n = len(entries)
        order = ["linear", "mixed_nonlinear", "mixed", "nonlinear", "interaction"]
        entries = sorted(entries, key=lambda e: order.index(e["family"]) if e["family"] in order else 99)
        for i, e in enumerate(entries):
            fam_label = row_label.get(e["family"], e["family"].capitalize())
            if i == 0:
                first = rf"\multirow{{{n}}}{{*}}{{{regime_name}}}"
            else:
                first = ""
            cells = [first, fam_label]
            for fam in families:
                for method, _ in methods:
                    cells.append(fmt(e.get(key(method, fam))))
            lines.append(" & ".join(cells) + r" \\")

    # drop the trailing per-group SUMMARY entry (no "family" key)
    no_conf = [e for e in d["no_confounding"] if "family" in e]
    hidden = [e for e in d["hidden_confounder"] if "family" in e]
    emit_block(no_conf, "No confounding")
    lines.append(r"\midrule")
    emit_block(hidden, "Hidden confounder")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write("ihdp_decoy_fp.tex", lines), ncols


# ---------------------------------------------------------------------------
# 5. chambers_perfeature.tex
# ---------------------------------------------------------------------------
def make_chambers():
    src = "causal_chambers_full_L.json"
    d = load(src)
    # lt_vis1 has the clearest decoy story: ir_2 is the corr-0.9 DAG-null decoy.
    cfg = next(c for c in d["configs"] if c["label"] == "lt_vis1")
    features = cfg["features"]
    tau_true = cfg["tau_true"]
    tau_hat = cfg["tau_hat"]  # FADO
    comp = cfg["comparators"]
    truth = {t["feature"]: t for t in cfg["truth"]}

    methods = [
        ("shap", "SHAP"),
        ("marginal", "Marginal"),
        ("backdoor_linear", "Backdoor"),
    ]

    lines = [
        f"% Auto-generated from causal_probe/results/realdata_2026-09-29/{src}",
        "% tab:chambers-perfeature",
        "% Config lt_vis1 (outcome vis_1). ir_2 is the corr-0.9 DAG-null decoy.",
        r"% DAG-null decoy rows (not a DAG parent of y AND tau_true 0) flagged with $\dagger$: current, angle_1, ir_2.",
        "% Backdoor = backdoor_linear graph-oracle skyline.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Feature & $\tau_{\mathrm{true}}$ & FADO & " + " & ".join(h for _, h in methods) + r" \\",
        r"\midrule",
    ]
    for i, feat in enumerate(features):
        # DAG-null decoy: not a DAG parent of y AND zero true effect (current,
        # angle_1, ir_2). pol_1/pol_2 are intervened with small nonzero tau_true,
        # so they are not flagged as nulls.
        is_null = (not truth[feat]["dag_parent_of_y"]) and abs(tau_true[i]) < 1e-9
        name = feat.replace("_", r"\_")
        if is_null:
            name = name + r"$^\dagger$"
        cells = [name, fmt(tau_true[i]), fmt(tau_hat[i])]
        for m, _ in methods:
            cells.append(fmt(comp[m]["tau"][i]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write("chambers_perfeature.tex", lines), 2 + 1 + len(methods)


def main():
    outs = []
    outs.append(make_coverage())
    outs.append(make_quality())
    outs.append(make_rct())
    outs.append(make_ihdp())
    outs.append(make_chambers())
    print("\nColumn counts (declared):")
    for path, ncols in outs:
        print(f"  {os.path.basename(path)}: {ncols} columns")


if __name__ == "__main__":
    main()
