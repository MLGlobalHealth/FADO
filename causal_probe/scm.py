"""Linear non-Gaussian SCM generator with exact total-effect labels.

Model: Z = B Z + eps where Z = (X_1, ..., X_p, Y) under a sampled topological
order, B is strictly triangular under that order (DAG), and eps is Laplace or
a Gaussian-mixture non-Gaussian noise.

For acyclic B the total-effect matrix is A = (I - B)^{-1} and the per-feature
intervention contrast (+1 vs -1 in standardized scale) is

    tau_i = A[y_idx, i] * 2 * std(X_i) / std(Y)

where std(.) uses population covariance Cov(Z) = A diag(Var(eps)) A^T.

The generator returns standardized (X, y) and the tau vector with labels
in the ORIGINAL feature order (pre-standardization permutation is handled
in training to force column-permutation equivariance).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _sample_laplace(shape, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean unit-variance Laplace noise."""
    # Laplace(0, b) has variance 2 b^2. For unit variance, b = 1/sqrt(2).
    return rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=shape)


def _sample_mixture(shape, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean unit-variance mixture noise (exponential + reflected)."""
    signs = rng.choice([-1.0, 1.0], size=shape)
    mag = rng.exponential(scale=1.0, size=shape)
    raw = signs * mag
    return raw / np.std(raw) if np.std(raw) > 0 else raw


def _sample_heavy(shape, rng: np.random.Generator) -> np.ndarray:
    """Per-column random heavy/skewed unit-variance noise.

    For each column j, sample a family from a wide pool covering symmetric
    heavy tails, asymmetric/skewed distributions, and bimodal mixtures —
    then center to mean 0 and rescale to unit sample variance. Designed
    so the training prior covers the kind of marginal asymmetry / heavy
    tail seen in real-world cause-effect pairs.
    """
    if isinstance(shape, int):
        n, N = shape, 1
        out = np.empty(n, dtype=np.float64)
    else:
        n, N = shape
        out = np.empty((n, N), dtype=np.float64)
    families = (
        "laplace", "gaussian", "student_t3", "lognormal_signed",
        "exp_signed", "gauss_mixture_bimodal", "skewnormal",
    )
    for j in range(N):
        fam = families[int(rng.integers(0, len(families)))]
        if fam == "laplace":
            raw = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=n)
        elif fam == "gaussian":
            raw = rng.standard_normal(size=n)
        elif fam == "student_t3":
            raw = rng.standard_t(df=3, size=n)
        elif fam == "lognormal_signed":
            sigma = float(rng.uniform(0.4, 0.9))
            raw = rng.lognormal(mean=0.0, sigma=sigma, size=n)
            if rng.random() < 0.5:
                raw = -raw
        elif fam == "exp_signed":
            sign = rng.choice([-1.0, 1.0], size=n)
            mag = rng.exponential(scale=1.0, size=n)
            raw = sign * mag
        elif fam == "gauss_mixture_bimodal":
            sep = float(rng.uniform(1.0, 2.5))
            comp = rng.random(n) < 0.5
            raw = np.where(
                comp, rng.normal(-sep, 0.5, size=n), rng.normal(sep, 0.5, size=n),
            )
        elif fam == "skewnormal":
            # Closed-form skewed via |Z1| * sign(Z2_corr) trick:
            # X = alpha*|U| + V, with alpha controlling skew.
            alpha = float(rng.uniform(-1.5, 1.5))
            u = rng.standard_normal(size=n)
            v = rng.standard_normal(size=n)
            raw = alpha * np.abs(u) + v
        else:
            raw = rng.standard_normal(size=n)
        raw = raw - raw.mean()
        s = float(np.std(raw))
        col = raw / s if s > 0 else raw
        if out.ndim == 1:
            out[:] = col
        else:
            out[:, j] = col
    return out


@dataclass(frozen=True)
class SCMSample:
    X: np.ndarray          # (n, p) standardized feature matrix
    y: np.ndarray          # (n,) standardized target
    tau: np.ndarray        # (p,) population total-effect contrast labels
    y_idx_in_Z: int        # position of Y in the latent Z vector
    feature_to_Z: np.ndarray  # (p,) mapping feature-index -> Z-index
    B: np.ndarray          # strictly triangular structural matrix (Z x Z)
    std_Z: np.ndarray      # (p+1,) population std per Z node


class LinearNonGaussianSCM:
    """Random linear non-Gaussian DAG on p features + Y.

    Parameters
    ----------
    p : int
        Number of features (outcome Y is an extra node).
    edge_prob : float
        Probability that a precedes-in-order pair gets a nonzero edge.
    weight_lo, weight_hi : float
        Uniform magnitude range for nonzero B entries.
    noise : {"laplace", "mixture"}
        Non-Gaussian noise family. "gaussian" is also supported for
        the identifiability negative control.
    rng : np.random.Generator
        Reproducibility.
    """

    def __init__(
        self,
        p: int,
        rng: np.random.Generator,
        *,
        edge_prob: float = 0.35,
        weight_lo: float = 0.5,
        weight_hi: float = 2.0,
        noise: str = "laplace",
        y_position: Optional[int] = None,
    ) -> None:
        if p < 2:
            raise ValueError("need at least 2 features")
        self.p = p
        self.noise = noise
        N = p + 1  # features + Y
        # 1. Sample a topological order over the N nodes.
        self._topo_order = rng.permutation(N)
        # 2. Sample Y's position in that order. y_position gives its RANK;
        #    if None, uniform over [1, N-1] (exclude pure-source to keep
        #    tau non-trivial on at least one ancestor).
        if y_position is None:
            y_position = int(rng.integers(0, N))
        y_position = int(max(0, min(y_position, N - 1)))
        # Y occupies rank y_position in _topo_order; the p features fill
        # the other ranks, so Y is node _topo_order[y_position].
        self.y_idx_in_Z = int(self._topo_order[y_position])
        self._y_position = y_position

        # Feature-to-Z mapping: preserves original feature order 0..p-1;
        # every feature is a distinct non-Y node of Z.
        non_y_nodes = [int(k) for k in self._topo_order if int(k) != self.y_idx_in_Z]
        self.feature_to_Z = np.asarray(non_y_nodes[: p], dtype=int)
        # We still want the p features to map to Z-indices in a deterministic
        # order (feature 0 -> first non-Y topologically-earliest node, etc.)
        # to make labels well-defined. Sort by topological rank.
        rank = {int(node): i for i, node in enumerate(self._topo_order.tolist())}
        self.feature_to_Z = np.asarray(
            sorted(non_y_nodes, key=lambda k: rank[int(k)]), dtype=int
        )

        # 3. Sample structural matrix B with entries only for parent-earlier
        #    edges under _topo_order, then check no self-loops.
        B = np.zeros((N, N), dtype=np.float64)
        for rank_c, child in enumerate(self._topo_order.tolist()):
            for rank_p in range(rank_c):
                parent = int(self._topo_order[rank_p])
                if rng.random() < edge_prob:
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    mag = rng.uniform(weight_lo, weight_hi)
                    B[int(child), parent] = sign * mag
        self.B = B
        # 4. Noise variance is unit per node.
        self.eps_var = np.ones(N, dtype=np.float64)

        # 5. A = (I - B)^{-1}. For strictly lower-triangular B under a topo
        #    order, (I - B) is triangular and trivially invertible.
        I = np.eye(N, dtype=np.float64)
        try:
            self.A = np.linalg.solve(I - B, I)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"(I - B) singular — not a valid DAG? {e}")

        # 6. Population covariance and per-node std.
        #    Cov(Z) = A diag(eps_var) A^T.
        cov_Z = self.A @ np.diag(self.eps_var) @ self.A.T
        var_Z = np.clip(np.diag(cov_Z), a_min=1e-12, a_max=None)
        self.std_Z = np.sqrt(var_Z)

        # 7. tau_i = A[y_idx, i_idx] * 2 * std(X_i) / std(Y), ordered by
        #    feature index (feature 0 -> Z[feature_to_Z[0]], ...).
        std_y = float(self.std_Z[self.y_idx_in_Z])
        tau = np.empty(p, dtype=np.float64)
        for i, zi in enumerate(self.feature_to_Z.tolist()):
            total = float(self.A[self.y_idx_in_Z, int(zi)])
            tau[i] = total * 2.0 * float(self.std_Z[int(zi)]) / std_y
        self.tau = tau

    @classmethod
    def from_adjacency(
        cls,
        B: np.ndarray,
        y_idx_in_Z: int,
        topo_order: np.ndarray,
        *,
        eps_var: Optional[np.ndarray] = None,
        noise: str = "laplace",
    ) -> "LinearNonGaussianSCM":
        """Build an SCM directly from a structural matrix B + topo order.

        For real-DAG benchmarks (e.g.\\ Sachs) where B is fixed by the
        published structure rather than randomly sampled. Sets the same
        attributes ``__init__`` does, so future additions to the init
        contract should be mirrored here.
        """
        N = B.shape[0]
        if B.shape != (N, N):
            raise ValueError(f"B must be square, got {B.shape}")
        topo_order = np.asarray(topo_order, dtype=int)
        if topo_order.shape != (N,):
            raise ValueError(f"topo_order must have shape ({N},), got {topo_order.shape}")
        if eps_var is None:
            eps_var = np.ones(N, dtype=np.float64)

        instance = cls.__new__(cls)
        instance.p = N - 1
        instance.noise = noise
        instance._topo_order = topo_order
        instance.y_idx_in_Z = int(y_idx_in_Z)
        instance._y_position = int(np.where(topo_order == y_idx_in_Z)[0][0])

        non_y = [int(k) for k in topo_order.tolist() if int(k) != instance.y_idx_in_Z]
        rank = {int(node): i for i, node in enumerate(topo_order.tolist())}
        instance.feature_to_Z = np.asarray(
            sorted(non_y, key=lambda k: rank[int(k)]), dtype=int
        )
        instance.B = B.astype(np.float64, copy=True)
        instance.eps_var = np.asarray(eps_var, dtype=np.float64)

        I = np.eye(N, dtype=np.float64)
        try:
            instance.A = np.linalg.solve(I - instance.B, I)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"(I - B) singular - not a valid DAG? {e}")

        cov_Z = instance.A @ np.diag(instance.eps_var) @ instance.A.T
        var_Z = np.clip(np.diag(cov_Z), a_min=1e-12, a_max=None)
        instance.std_Z = np.sqrt(var_Z)

        std_y = float(instance.std_Z[instance.y_idx_in_Z])
        tau = np.empty(instance.p, dtype=np.float64)
        for i, zi in enumerate(instance.feature_to_Z.tolist()):
            total = float(instance.A[instance.y_idx_in_Z, int(zi)])
            tau[i] = total * 2.0 * float(instance.std_Z[int(zi)]) / std_y
        instance.tau = tau
        return instance

    def _draw_eps(self, n: int, rng: np.random.Generator) -> np.ndarray:
        N = self.p + 1
        if self.noise == "laplace":
            return _sample_laplace((n, N), rng)
        if self.noise == "mixture":
            return _sample_mixture((n, N), rng)
        if self.noise == "gaussian":
            return rng.standard_normal((n, N))
        if self.noise == "heavy":
            return _sample_heavy((n, N), rng)
        raise ValueError(f"unknown noise {self.noise!r}")

    def sample(self, n: int, rng: np.random.Generator) -> SCMSample:
        """Draw n observational rows and return a standardized SCMSample.

        Standardization uses population stds (self.std_Z), not sample stds,
        to keep tau labels well-calibrated.
        """
        N = self.p + 1
        eps = self._draw_eps(n, rng)
        # Topological forward sim: Z_c = B[c, :parents] Z_parents + eps_c.
        Z = np.zeros((n, N), dtype=np.float64)
        for rank, node in enumerate(self._topo_order.tolist()):
            parents = [int(self._topo_order[r]) for r in range(rank)]
            if parents:
                Z[:, int(node)] = Z[:, parents] @ self.B[int(node), parents] + eps[:, int(node)]
            else:
                Z[:, int(node)] = eps[:, int(node)]
        # Standardize by population std.
        Z_std = Z / self.std_Z.reshape(1, -1)
        y = Z_std[:, self.y_idx_in_Z].copy()
        X = Z_std[:, self.feature_to_Z].copy()
        return SCMSample(
            X=X, y=y, tau=self.tau.copy(),
            y_idx_in_Z=self.y_idx_in_Z,
            feature_to_Z=self.feature_to_Z.copy(),
            B=self.B.copy(),
            std_Z=self.std_Z.copy(),
        )


def sample_random_scms(
    n_scms: int,
    p: int,
    rng: np.random.Generator,
    **scm_kwargs,
) -> list[LinearNonGaussianSCM]:
    """Convenience: sample n_scms random linear-non-Gaussian DAGs."""
    out = []
    for _ in range(n_scms):
        seed_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        out.append(LinearNonGaussianSCM(p=p, rng=seed_rng, **scm_kwargs))
    return out
