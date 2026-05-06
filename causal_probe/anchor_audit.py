"""Anchor-pair audit: compare the symmetric (-1, +1) contrast against the
range of the empirical response curve at fixed anchors.

Diagnostic only — no model retraining. For every (X_i, Y) pair this gives
us a quick read on whether the symmetric scalar contrast is a faithful
summary of the per-feature dependence, or whether a non-symmetric anchor
would reveal a stronger signal (a U-shape, a saturating effect, a
threshold, etc.).

Outputs per feature:

    r_obs(t_k)   = (E_hat[Y | X_std ≈ t_k] - E[Y]) / std(Y)
    contrast_sym = r_obs(+1) - r_obs(-1)
    range        = max_k r_obs(t_k) - min_k r_obs(t_k)
    asymmetry    = max_k |r_obs(t_k) + r_obs(-t_k)| / 2     (only over t > 0)

`asymmetry` is loosely "how far is the curve from being odd-symmetric".
A symmetric contrast head that anchors on (-a, +a) collapses any
even-symmetric component of the curve; a large `range` with a small
`contrast_sym` is the signature.

Estimation is Nadaraya-Watson with a Gaussian kernel (default bandwidth
0.4 in standardized X units), which is plenty for the diagnostic — we
are looking for first-order shape, not last-decimal calibration.

This module deliberately uses observational E[Y|X_i=t] rather than do-
intervention. For RCT covariates other than the treatment, that means
the audit reflects observational shape, not necessarily the causal
shape. The relevant diagnostic question is still "could the symmetric
contrast have been a poor summary on this dataset?" which the
observational shape answers directly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import numpy as np


T_GRID = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
K = T_GRID.shape[0]


@dataclass
class FeatureAudit:
    name: str
    n: int
    r_curve: list[float]          # length-K, r_obs(t_k) on the standardized grid
    n_eff: list[float]             # per-anchor effective sample size
    rank_curve: list[float]        # rank-based crosscheck at quantiles 0.1..0.9
    contrast_sym: float           # r(+1) - r(-1)
    contrast_05_15: float         # r(+1.5) - r(+0.5), checks asymmetric pair
    contrast_neg2_pos2: float     # r(+2) - r(-2)
    range_curve: float            # max - min over t_k (Gaussian anchors)
    range_rank: float             # max - min over rank quantiles (robust crosscheck)
    asymmetry: float              # max_k |r(t) + r(-t)| / 2 (k>0)
    abs_skew: float               # |E[X^3] / std^3| of the standardized feature
    kurtosis: float               # excess kurtosis of the standardized feature
    n_eff_min: float              # smallest n_eff across the standardized anchors


def _kernel_response_curve(
    X: np.ndarray, Y: np.ndarray, anchors: np.ndarray, *,
    bandwidth: float = 0.4, min_eff: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nadaraya-Watson estimate of E[Y_std | X_std = t] at each anchor.

    Returns (curve, n_eff_per_anchor). Anchors where the effective sample
    size falls below `min_eff` get NaN — the reading there would be
    dominated by a handful of points and is not reliable, especially for
    heavy-tailed features.
    """
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError(f"expected 1-D X, Y; got {X.shape}, {Y.shape}")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same length; got {X.shape}, {Y.shape}")
    sx = X.std()
    sy = Y.std()
    if sx <= 1e-9 or sy <= 1e-9:
        return np.full_like(anchors, np.nan), np.zeros_like(anchors)
    Xs = (X - X.mean()) / sx
    Ys = (Y - Y.mean()) / sy
    out = np.full_like(anchors, np.nan, dtype=np.float64)
    n_eff = np.zeros_like(anchors, dtype=np.float64)
    for k, t in enumerate(anchors):
        w = np.exp(-0.5 * ((Xs - t) / bandwidth) ** 2)
        s = float(w.sum())
        n_eff[k] = s
        if s < min_eff:
            continue
        out[k] = float((w * Ys).sum() / s)
    return out, n_eff


def _rank_response_curve(
    X: np.ndarray, Y: np.ndarray, *,
    quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    half_width: float = 0.075,
) -> np.ndarray:
    """Rank-based response curve: bin X by empirical quantile, take mean
    of standardized Y within the bin around each given quantile. Returns
    a length-len(quantiles) array. NaN where the bin is empty.

    This is a robust crosscheck for heavy-tailed features where the
    Gaussian-anchor kernel is unreliable at +/-2 because the 'tail' is
    really one or two outliers.
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same length; got {X.shape}, {Y.shape}")
    sy = Y.std()
    if sy <= 1e-9:
        return np.full(len(quantiles), np.nan)
    Ys = (Y - Y.mean()) / sy
    ranks = (np.argsort(np.argsort(X)) + 0.5) / X.shape[0]
    out = np.full(len(quantiles), np.nan)
    for k, q in enumerate(quantiles):
        in_bin = (ranks >= q - half_width) & (ranks <= q + half_width)
        if in_bin.sum() < 25:
            continue
        out[k] = float(Ys[in_bin].mean())
    return out


def audit_feature(
    name: str, X: np.ndarray, Y: np.ndarray, *,
    bandwidth: float = 0.4, min_eff: float = 25.0,
) -> FeatureAudit:
    """Audit a single (X, Y) pair. X and Y must be 1-D, same length."""
    Xf = np.asarray(X, dtype=np.float64).ravel()
    Yf = np.asarray(Y, dtype=np.float64).ravel()
    n = int(Xf.shape[0])

    curve, n_eff_arr = _kernel_response_curve(
        Xf, Yf, T_GRID, bandwidth=bandwidth, min_eff=min_eff,
    )
    # Asymmetric anchor pair r(+1.5) - r(+0.5) requires interpolation off
    # the fixed grid; use the same kernel.
    extra_anchors = np.array([-2.0, -1.5, -0.5, 0.5, 1.5, 2.0])
    curve_extra, _ = _kernel_response_curve(
        Xf, Yf, extra_anchors, bandwidth=bandwidth, min_eff=min_eff,
    )
    r05 = float(curve_extra[3])  # +0.5
    r15 = float(curve_extra[4])  # +1.5
    rm2 = float(curve_extra[0])
    rp2 = float(curve_extra[5])

    # Rank-based crosscheck at quantiles 0.1, 0.3, 0.5, 0.7, 0.9 — robust to
    # heavy tails (every bin has by-construction equal count).
    rank_curve = _rank_response_curve(Xf, Yf)

    # Contrast and range computed over the available (non-NaN) entries.
    finite_mask = np.isfinite(curve)
    n_eff_min = float(np.min(n_eff_arr))
    if finite_mask.sum() < 3:
        # Too few good anchors to talk about shape.
        return FeatureAudit(
            name=name, n=n, r_curve=curve.tolist(), n_eff=n_eff_arr.tolist(),
            rank_curve=rank_curve.tolist(),
            contrast_sym=float("nan"),
            contrast_05_15=float("nan"),
            contrast_neg2_pos2=float("nan"),
            range_curve=float("nan"),
            range_rank=float(np.nanmax(rank_curve) - np.nanmin(rank_curve)) if np.any(np.isfinite(rank_curve)) else float("nan"),
            asymmetry=float("nan"),
            abs_skew=float("nan"),
            kurtosis=float("nan"),
            n_eff_min=n_eff_min,
        )
    rng_curve = float(np.nanmax(curve) - np.nanmin(curve))
    rng_rank = float(np.nanmax(rank_curve) - np.nanmin(rank_curve)) if np.any(np.isfinite(rank_curve)) else float("nan")
    contrast = float(curve[3] - curve[1]) if (finite_mask[1] and finite_mask[3]) else float("nan")
    contrast_05_15 = (r15 - r05) if (np.isfinite(r05) and np.isfinite(r15)) else float("nan")
    contrast_neg2_pos2 = (rp2 - rm2) if (np.isfinite(rm2) and np.isfinite(rp2)) else float("nan")
    pos_pairs = [(1, 3), (0, 4)]
    asymm_vals = []
    for ineg, ipos in pos_pairs:
        if finite_mask[ineg] and finite_mask[ipos]:
            asymm_vals.append(abs(curve[ineg] + curve[ipos]) / 2.0)
    asymmetry = float(np.max(asymm_vals)) if asymm_vals else float("nan")

    sx = Xf.std()
    Xs = (Xf - Xf.mean()) / (sx + 1e-12)
    abs_skew = float(np.abs(np.mean(Xs ** 3)))
    kurtosis = float(np.mean(Xs ** 4) - 3.0)

    return FeatureAudit(
        name=name, n=n, r_curve=curve.tolist(), n_eff=n_eff_arr.tolist(),
        rank_curve=rank_curve.tolist(),
        contrast_sym=contrast,
        contrast_05_15=contrast_05_15,
        contrast_neg2_pos2=contrast_neg2_pos2,
        range_curve=rng_curve,
        range_rank=rng_rank,
        asymmetry=asymmetry,
        abs_skew=abs_skew, kurtosis=kurtosis,
        n_eff_min=n_eff_min,
    )


def audit_table(
    items: Iterable[tuple[str, np.ndarray, np.ndarray]], *,
    bandwidth: float = 0.4, min_eff: float = 25.0,
) -> list[dict]:
    """Run audit_feature over an iterable of (name, X, Y); return list of dicts."""
    out = []
    for name, X, Y in items:
        a = audit_feature(name, X, Y, bandwidth=bandwidth, min_eff=min_eff)
        out.append(asdict(a))
    return out


def flag_potentially_missed(
    rows: list[dict], *,
    contrast_floor: float = 0.05, range_floor: float = 0.20,
    require_rank_corroboration: bool = True,
) -> list[dict]:
    """Return audit rows where the symmetric contrast looks small but the
    response range is large — the candidates for "missed by the symmetric
    pair". When `require_rank_corroboration` is True, we additionally
    require the rank-based curve range (heavy-tail-robust) to also be large,
    which filters out artifacts of a single tail outlier.
    """
    flagged = []
    for r in rows:
        c = r["contrast_sym"]
        rng = r["range_curve"]
        rng_rank = r.get("range_rank", float("nan"))
        if not np.isfinite(c) or not np.isfinite(rng):
            continue
        if abs(c) > contrast_floor or rng < range_floor:
            continue
        if require_rank_corroboration:
            # Use the rank-based range as the primary evidence of variation
            # — it is robust to heavy tails by construction (each bin has
            # equal count). Without rank corroboration, an apparent
            # Gaussian-anchor range is often driven by one tail outlier.
            if not (np.isfinite(rng_rank) and rng_rank >= range_floor):
                continue
        flagged.append(r)
    flagged.sort(key=lambda r: r["range_curve"], reverse=True)
    return flagged
