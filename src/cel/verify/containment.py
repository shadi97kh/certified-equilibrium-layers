"""Numerical audit of the certified off-target containment bound (Contribution 2).

The propagation operator is the personalized-PageRank / APPNP operator

    P_alpha = alpha (I - (1 - alpha) S)^{-1} = alpha * sum_{k>=0} (1-alpha)^k S^k,

with S the symmetric-normalized GRN adjacency (eigenvalues in [-1, 1], so the
spectral norm is at most 1; this module spectral-norm-clips S to enforce that by
construction). Because S^k h is supported within k hops of the seed support, the
part of P_alpha h that has travelled at least r hops obeys

    || (P_alpha h)|_{dist >= r} ||  <=  (1 - alpha)^r * || h ||.

That is the certified decay envelope this module verifies on a real STRING
subgraph. It also reports the spectral gap and an effective-resistance distance
from the seed support, the two ingredients of the planned spectral upgrade bound.

This module is pure numpy / scipy (no torch) so it runs in the lightweight CI
image. The torch propagation operator used by the model lives in
:mod:`cel.models.propagation_head` and implements the same power iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path


def operator_norm(S: sp.spmatrix, iters: int = 200, seed: int = 0) -> float:
    """Estimate the spectral norm ||S||_2 by power iteration on the symmetric S.

    For a symmetric matrix the dominant-magnitude eigenvalue equals the spectral
    norm. Returns 0.0 for an all-zero operator.
    """
    n = S.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        return 0.0
    v /= nrm
    sigma = 0.0
    for _ in range(iters):
        w = S @ v
        sigma = float(np.linalg.norm(w))
        if sigma <= 1e-12:
            return 0.0
        v = w / sigma
    return sigma


def spectral_clip(S: sp.spmatrix, margin: float = 1e-6) -> tuple[sp.csr_matrix, float]:
    """Scale S so its spectral norm is at most 1 (containment by construction).

    Returns the (possibly unchanged) operator and the estimated pre-clip norm.
    A proper symmetric-normalized adjacency already has norm <= 1, so this is a
    safety net that leaves it untouched.
    """
    sigma = operator_norm(S)
    scale = 1.0 / max(1.0, sigma * (1.0 + margin))
    clipped = (S * scale).tocsr() if scale < 1.0 else S.tocsr()
    return clipped, sigma


def propagate(S: sp.spmatrix, h: np.ndarray, alpha: float, num_steps: int) -> np.ndarray:
    """Apply the APPNP operator P_alpha to h via ``num_steps`` power iterations.

    z_0 = h; z_{t+1} = (1 - alpha) S z_t + alpha h. Converges to P_alpha h.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    h = np.asarray(h, dtype=float)
    z = h.copy()
    for _ in range(num_steps):
        z = (1.0 - alpha) * (S @ z) + alpha * h
    return z


def dense_propagation_operator(S: sp.spmatrix, alpha: float) -> np.ndarray:
    """Materialize the exact dense P_alpha = alpha (I - (1-alpha) S)^{-1}.

    This is the closed form whose power-iteration application is :func:`propagate`.
    It is invertible for alpha in (0, 1) and ||S||_2 <= 1 (the eigenvalues of
    I - (1-alpha) S lie in [alpha, 2-alpha] > 0), so it serves as the KNOWN
    invertible mixing operator P_alpha. Intended for small graphs; for large
    graphs use :func:`propagate`.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = S.shape[0]
    dense = S.toarray() if sp.issparse(S) else np.asarray(S, dtype=float)
    return alpha * np.linalg.inv(np.eye(n) - (1.0 - alpha) * dense)


def hop_distances(S: sp.spmatrix, seed_idx: np.ndarray) -> np.ndarray:
    """Unweighted shortest-path hop distance from the seed support to every node.

    Unreachable nodes get ``np.inf``. Distance is taken over the binary support
    of S (an edge exists wherever S is nonzero).
    """
    binary = (S != 0).astype(np.int8)
    dists = shortest_path(binary, method="D", unweighted=True, indices=np.asarray(seed_idx))
    return np.asarray(dists).min(axis=0)


def seed_vector(n: int, seed_idx: np.ndarray) -> np.ndarray:
    """Indicator vector over the seed support nodes."""
    h = np.zeros(n, dtype=float)
    h[np.asarray(seed_idx)] = 1.0
    return h


def spectral_gap(S: sp.spmatrix) -> float:
    """Spectral gap of S: second-smallest eigenvalue of the normalized Laplacian.

    L = I - S has eigenvalues 1 - lambda_i(S). The smallest is 0 (for a
    connected graph); the second smallest, 1 - lambda_2(S), is the gap.
    """
    n = S.shape[0]
    lap = sp.identity(n, format="csr") - S
    if n <= 512:
        evals = np.linalg.eigvalsh(lap.toarray())
    else:  # pragma: no cover - large-graph path not hit by the fixture tests
        from scipy.sparse.linalg import eigsh

        evals = eigsh(lap.astype(float), k=2, which="SM", return_eigenvectors=False)
    evals = np.sort(np.real(evals))
    return float(evals[1]) if len(evals) > 1 else 0.0


def effective_resistance_from_seed(S: sp.spmatrix, seed_idx: np.ndarray) -> np.ndarray:
    """Effective resistance from the seed support to every node.

    Uses the combinatorial Laplacian L = D - A of the weighted graph and its
    Moore-Penrose pseudoinverse; per node returns the minimum effective
    resistance to any seed node (0 for seed nodes themselves). This is the
    distance the planned effective-resistance upgrade bound is built on.
    """
    n = S.shape[0]
    a = S.tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    lap = (sp.diags(deg) - a).toarray()
    pinv = np.linalg.pinv(lap)
    diag = np.diag(pinv)
    seed_idx = np.asarray(seed_idx)
    # R(u, v) = pinv[u,u] + pinv[v,v] - 2 pinv[u,v]; take min over seed nodes.
    res = np.full(n, np.inf)
    for s in seed_idx:
        r_s = diag + diag[s] - 2.0 * pinv[:, s]
        res = np.minimum(res, r_s)
    res[seed_idx] = 0.0
    return np.maximum(res, 0.0)


@dataclass(frozen=True)
class ContainmentReport:
    """Result of a containment audit on one (graph, seed support, alpha)."""

    alpha: float
    hops: np.ndarray  # r = 0, 1, ..., max_r
    empirical: np.ndarray  # ||(P_alpha h)|_{dist>=r}|| / ||h||
    envelope: np.ndarray  # (1 - alpha)^r
    holds: bool
    spectral_norm: float  # pre-clip ||S||_2
    spectral_gap: float
    eff_resistance: np.ndarray  # per-node effective resistance from seed support
    num_steps: int

    @property
    def max_violation(self) -> float:
        """Largest amount by which the empirical decay exceeds the envelope."""
        return float(np.max(self.empirical - self.envelope))


def verify_containment(
    S: sp.spmatrix,
    seed_idx: np.ndarray,
    alpha: float,
    num_steps: int | None = None,
    max_r: int | None = None,
    tol: float = 1e-9,
) -> ContainmentReport:
    """Verify the (1-alpha)^r decay envelope and compute the spectral quantities.

    Args:
        S: symmetric-normalized GRN adjacency (scipy sparse).
        seed_idx: indices of the seed support nodes.
        alpha: teleport probability in (0, 1).
        num_steps: power-iteration steps. Must be >= max_r; defaults to a value
            that both converges and covers max_r.
        max_r: largest hop radius to check. Defaults to the graph's reachable
            diameter from the seed support.
        tol: numerical slack when asserting the envelope holds.
    """
    n = S.shape[0]
    seed_idx = np.asarray(seed_idx)
    clipped, sigma = spectral_clip(S)

    dist = hop_distances(S, seed_idx)
    finite = dist[np.isfinite(dist)]
    reach = int(finite.max()) if finite.size else 0
    if max_r is None:
        max_r = reach
    if num_steps is None:
        num_steps = max(max_r, 32)
    if num_steps < max_r:
        raise ValueError("num_steps must be at least max_r for the envelope to hold")

    h = seed_vector(n, seed_idx)
    hnorm = float(np.linalg.norm(h))
    p = propagate(clipped, h, alpha, num_steps)

    radii = np.arange(max_r + 1)
    empirical = np.array([float(np.linalg.norm(p[dist >= r])) / hnorm for r in radii])
    envelope = (1.0 - alpha) ** radii
    holds = bool(np.all(empirical <= envelope + tol))

    return ContainmentReport(
        alpha=float(alpha),
        hops=radii,
        empirical=empirical,
        envelope=envelope,
        holds=holds,
        spectral_norm=float(sigma),
        spectral_gap=spectral_gap(clipped),
        eff_resistance=effective_resistance_from_seed(clipped, seed_idx),
        num_steps=num_steps,
    )


def plot_containment(report: ContainmentReport, out_path: str) -> str:
    """Plot empirical decay against the certified envelope; save to ``out_path``.

    matplotlib is imported lazily (Agg backend) so importing this module never
    requires it. Returns the path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(report.hops, report.envelope, "k--", label=r"certified $(1-\alpha)^r$")
    ax.semilogy(report.hops, np.maximum(report.empirical, 1e-12), "o-", label="empirical decay")
    ax.set_xlabel("hop distance r from seed support")
    ax.set_ylabel("off-target signal fraction")
    ax.set_title(
        f"containment: alpha={report.alpha:.3f}, gap={report.spectral_gap:.3g}, "
        f"holds={report.holds}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
