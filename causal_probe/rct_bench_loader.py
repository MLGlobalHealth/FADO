"""Load an RCT_Bench trial into (X, T, Y) for causal-method evaluation.

Naming contract (Bingkai Wang et al., RCT_Bench):
  Treatment            randomized arm (control listed first when identifiable)
  YP_*                 primary outcome(s)  -- we use the first *usable* one
  YS_*                 secondary outcomes  (ignored)
  X_*                  baseline covariates (pre-treatment) -> adjustment set
  Crossover_*, Assessment_Window, Participant_ID  -> design fields, excluded

Design decisions:
  * Treatment binarization: arms are ordered with the control arm first (per the
    benchmark's convention). We form a single contrast = control (T=0) vs the
    first active arm (T=1) and DROP any further arms. This keeps every trial a
    clean two-arm ATE; multi-arm trials are noted in the output.
  * Primary outcome: the first YP_* column that is usable -- numeric with data,
    or a binary categorical (Yes/No, yes/no, present/absent, ...) which we
    0/1-encode. Rows with a missing primary are dropped.
  * Covariates: X_* only. Non-numeric X are one-hot/label encoded; missing values
    are median- (numeric) or mode- (categorical) imputed. Zero-variance columns
    dropped.
  * Ground truth = the RCT unadjusted difference in means of the RAW primary
    outcome between the two arms (unbiased under randomization).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DESIGN_FIELDS = {"Crossover_Sequence", "Crossover_Period", "Assessment_Window",
                 "Participant_ID"}

# common binary categorical outcome vocabularies -> 1 (event) / 0 (non-event)
_POS = {"yes", "y", "true", "1", "present", "positive", "success", "event",
        "improved", "responder", "alive", "colonized", "adhered", "infected"}
_NEG = {"no", "n", "false", "0", "absent", "negative", "failure", "none",
        "not improved", "non-responder", "dead", "death"}


@dataclass
class TrialData:
    trial: str
    X: np.ndarray                     # (n, p) numeric covariates, imputed
    T: np.ndarray                     # (n,) in {0,1}
    Y: np.ndarray                     # (n,) raw primary outcome (numeric)
    x_names: list[str]
    primary: str
    primary_type: str                 # "continuous" | "binary"
    control_arm: str
    active_arm: str
    n_arms_original: int
    dropped_arms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.T)

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def rct_ate_raw(self) -> float:
        """Unadjusted difference in means (T=1 minus T=0) on the raw outcome."""
        return float(self.Y[self.T == 1].mean() - self.Y[self.T == 0].mean())


def _encode_binary_categorical(s: pd.Series) -> Optional[np.ndarray]:
    """Map a 2-level categorical (Yes/No, ...) to {0,1}; None if not binary."""
    vals = s.dropna().astype(str).str.strip().str.lower()
    uniq = sorted(vals.unique())
    if len(uniq) != 2:
        return None
    a, b = uniq
    # decide which level is the "1"
    if a in _POS and b in _NEG:
        pos = a
    elif b in _POS and a in _NEG:
        pos = b
    else:
        pos = uniq[1]  # fallback: lexicographically-later level = 1
    out = np.full(len(s), np.nan)
    m = s.notna()
    out[m.values] = (s[m].astype(str).str.strip().str.lower() == pos).astype(float).values
    return out


def _pick_primary(df: pd.DataFrame) -> tuple[Optional[str], Optional[np.ndarray], Optional[str]]:
    """Return (colname, numeric_values, type) for the first usable YP_* column."""
    yp = [c for c in df.columns if c.startswith("YP_")]
    for c in yp:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() >= 10:
            uniq = np.unique(num.dropna().values)
            typ = "binary" if set(uniq.tolist()) <= {0.0, 1.0} else "continuous"
            return c, num.values, typ
        enc = _encode_binary_categorical(df[c])
        if enc is not None and np.isfinite(enc).sum() >= 10:
            return c, enc, "binary"
    return None, None, None


def _encode_covariates(dfX: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for c in dfX.columns:
        num = pd.to_numeric(dfX[c], errors="coerce")
        if num.notna().sum() >= max(5, int(0.5 * len(dfX))):
            v = num.values.astype(float)
            if np.nanstd(v) == 0 or not np.isfinite(np.nanstd(v)):
                continue
            med = np.nanmedian(v)
            v = np.where(np.isfinite(v), v, med)
            cols.append(v); names.append(c)
        else:  # low-cardinality categorical -> label encode (imputed to mode)
            s = dfX[c].astype("category")
            if s.nunique(dropna=True) < 2 or s.nunique(dropna=True) > 20:
                continue
            mode = s.mode(dropna=True)
            s = s.cat.add_categories(["__na__"]).fillna(mode.iloc[0] if len(mode) else "__na__")
            codes = s.cat.codes.values.astype(float)
            if np.std(codes) == 0:
                continue
            cols.append(codes); names.append(c)
    if not cols:
        return np.zeros((len(dfX), 0)), []
    return np.column_stack(cols), names


def load_trial(path: str, min_n_per_arm: int = 5) -> Optional[TrialData]:
    trial = re.sub(r"\.csv$", "", path.split("/")[-1])
    df = pd.read_csv(path, low_memory=False)
    if "Treatment" not in df.columns:
        return None

    primary, yvals, ptype = _pick_primary(df)
    if primary is None:
        return None
    df = df.assign(_Y=yvals)

    arms = df["Treatment"].dropna().astype(str)
    order = arms.value_counts().index.tolist()  # by frequency; control is usually large
    # Prefer explicit control naming if present, else first-listed unique in file order
    file_order = list(dict.fromkeys(df["Treatment"].dropna().astype(str).tolist()))
    ctrl = None
    for a in file_order:
        if re.search(r"\b(control|placebo|sham|usual care|standard|no e-?mail)\b", a, re.I):
            ctrl = a; break
    if ctrl is None:
        ctrl = file_order[0]
    actives = [a for a in file_order if a != ctrl]
    if not actives:
        return None
    active = actives[0]
    dropped = actives[1:]

    sub = df[df["Treatment"].astype(str).isin([ctrl, active])].copy()
    sub = sub[sub["_Y"].notna()]
    T = (sub["Treatment"].astype(str) == active).astype(int).values
    Y = sub["_Y"].values.astype(float)
    if (T == 1).sum() < min_n_per_arm or (T == 0).sum() < min_n_per_arm:
        return None

    xcols = [c for c in df.columns if c.startswith("X_") and c not in DESIGN_FIELDS]
    X, xnames = _encode_covariates(sub[xcols]) if xcols else (np.zeros((len(sub), 0)), [])

    notes = []
    if len(dropped):
        notes.append(f"multi-arm ({len(file_order)} arms); kept {ctrl!r} vs {active!r}")
    if X.shape[1] == 0:
        notes.append("no usable covariates")
    return TrialData(trial=trial, X=X, T=T, Y=Y, x_names=xnames, primary=primary,
                     primary_type=ptype, control_arm=ctrl, active_arm=active,
                     n_arms_original=len(file_order), dropped_arms=dropped, notes=notes)
