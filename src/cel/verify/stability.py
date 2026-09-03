"""Theorem 2 stability primitives: P_alpha contraction on signed, directed GRNs.

P_alpha = alpha (I - (1-alpha) S)^{-1} is the resolvent / personalized-PageRank
operator. Its Neumann series alpha * sum_k (1-alpha)^k S^k converges (the operator
is well defined) iff (1-alpha) rho(S) < 1, where rho(S) is the spectral radius.

The L2 Lipschitz constant of P_alpha is its spectral norm ||P_alpha||_2. The
geometric-series bound gives

    ||P_alpha||_2 <= alpha / (1 - (1-alpha) ||S||_2),   when (1-alpha) ||S||_2 < 1,

with ||S||_2 the largest singular value. For a SYMMETRIC S (undirected graphs)
||S||_2 = rho(S), so the rho-based constant alpha / (1 - (1-alpha) rho(S)) is the
exact, tight L2 bound. For a SIGNED, DIRECTED (non-normal) S, ||S||_2 >= rho(S):
the rho-based constant need not bound ||P_alpha||_2, and the valid certified
constant uses ||S||_2. This module exposes both so the gap (the dependence on the
directed / negative-edge spectrum) can be measured on real graphs.

Pure numpy / scipy. Random probe vectors are used only to MEASURE the operator
empirically; they construct no data and live in the verification layer (this
module is not on any reported-data path).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def spectral_radius(S: sp.spmatrix) -> float:
    """rho(S) = max |eigenvalue|."""
    n = S.shape[0]
    if n == 0:
        return 0.0
    if n <= 256:
        vals = np.linalg.eigvals(S.toarray() if sp.issparse(S) else np.asarray(S))
    else:
        vals = sla.eigs(
            S.tocsc().astype(float), k=1, which="LM", return_eigenvectors=False, maxiter=5000
        )
    return float(np.max(np.abs(vals)))


def operator_2norm(S: sp.spmatrix) -> float:
    """||S||_2 = largest singular value."""
    n = S.shape[0]
    if n == 0:
        return 0.0
    if n <= 256:
        return float(np.linalg.norm(S.toarray() if sp.issparse(S) else np.asarray(S), 2))
    return float(sla.svds(S.astype(float), k=1, return_singular_vectors=False)[0])


def inf_operator_norm(S: sp.spmatrix) -> float:
    """||S||_inf = max absolute row sum.

    For a row-normalized signed transition this is <= 1, so the inf-norm certified
    constant alpha / (1 - (1-alpha) ||S||_inf) is <= 1 for all alpha: P_alpha is
    non-expansive in the inf-norm even when its L2 norm exceeds 1.
    """
    if S.shape[0] == 0:
        return 0.0
    return float(np.asarray(abs(S).sum(axis=1)).ravel().max())


def neumann_converges(rho: float, alpha: float) -> bool:
    """True iff the Neumann series for P_alpha converges: (1-alpha) rho(S) < 1."""
    return (1.0 - alpha) * rho < 1.0


def certified_constant(spec: float, alpha: float) -> float:
    """Certified Lipschitz constant alpha / (1 - (1-alpha) spec).

    ``spec`` is rho(S) (tight for symmetric S) or ||S||_2 (the valid L2 bound).
    Returns +inf when the geometric series does not converge for this ``spec``.
    """
    denom = 1.0 - (1.0 - alpha) * spec
    if denom <= 0.0:
        return float("inf")
    return alpha / denom


def _resolvent_lu(S: sp.spmatrix, alpha: float):
    n = S.shape[0]
    a = (sp.eye(n, format="csc") - (1.0 - alpha) * S.tocsc()).tocsc()
    return sla.splu(a)


def apply_p_alpha(S: sp.spmatrix, alpha: float, d: np.ndarray) -> np.ndarray:
    """Apply P_alpha to a vector via a sparse solve (no dense inverse)."""
    lu = _resolvent_lu(S, alpha)
    return alpha * lu.solve(np.asarray(d, dtype=float))


def resolvent_2norm(S: sp.spmatrix, alpha: float, iters: int = 80, seed: int = 0) -> float:
    """||P_alpha||_2 (the true L2 Lipschitz constant) via power iteration on P^T P.

    Uses an LU factorization of (I - (1-alpha) S) and its transpose, so each step
    is two sparse solves; exact up to power-iteration convergence.
    """
    n = S.shape[0]
    if n == 0:
        return 0.0
    lu = _resolvent_lu(S, alpha)
    lu_t = _resolvent_lu(S.T, alpha)  # I - (1-alpha) S^T
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        return 0.0
    v /= nv
    for _ in range(iters):
        w = alpha * lu.solve(v)  # P_alpha v
        z = alpha * lu_t.solve(w)  # P_alpha^T (P_alpha v)
        nz = np.linalg.norm(z)
        if nz <= 1e-300:
            return 0.0
        v = z / nz
    return float(np.linalg.norm(alpha * lu.solve(v)))


def measure_lipschitz(
    S: sp.spmatrix, alpha: float, n_samples: int = 200, seed: int = 0, ord: float = 2
) -> dict:
    """Empirical max contraction ratio ||P_alpha d|| / ||d|| over random directions.

    ``ord`` selects the vector norm (2 for L2, np.inf for the inf-norm). Returns the
    max and mean ratio; the max is a lower bound on the operator norm in that norm.
    """
    n = S.shape[0]
    lu = _resolvent_lu(S, alpha)
    rng = np.random.default_rng(seed + 1)
    ratios = []
    for _ in range(n_samples):
        d = rng.standard_normal(n)
        nd = np.linalg.norm(d, ord)
        if nd == 0.0:
            continue
        pd = alpha * lu.solve(d)
        ratios.append(float(np.linalg.norm(pd, ord) / nd))
    arr = np.asarray(ratios) if ratios else np.zeros(1)
    return {
        "max_ratio": float(arr.max()),
        "mean_ratio": float(arr.mean()),
        "n_samples": len(ratios),
    }


def _abs_symmetric_weight(S: sp.spmatrix) -> sp.csr_matrix:
    absS = abs(S).tocsr()
    return ((absS + absS.T) * 0.5).tocsr()


def dirichlet_energy(S: sp.spmatrix, x: np.ndarray) -> float:
    """Graph Dirichlet energy 0.5 sum_ij W_ij (x_i - x_j)^2 with W = |S| symmetrized.

    Equals x^T (D - W) x, the smoothness penalty whose decay signals oversmoothing.
    """
    x = np.asarray(x, dtype=float)
    w = _abs_symmetric_weight(S)
    deg = np.asarray(w.sum(axis=1)).ravel()
    lx = deg * x - (w @ x)
    return float(x @ lx)


def dirichlet_smoothing(S: sp.spmatrix, alpha: float, n_probes: int = 16, seed: int = 0) -> dict:
    """Mean Dirichlet energy of P_alpha x vs x over random probes (oversmoothing)."""
    n = S.shape[0]
    lu = _resolvent_lu(S, alpha)
    rng = np.random.default_rng(seed + 2)
    e_in, e_out = [], []
    for _ in range(n_probes):
        x = rng.standard_normal(n)
        px = alpha * lu.solve(x)
        e_in.append(dirichlet_energy(S, x))
        e_out.append(dirichlet_energy(S, px))
    ein = float(np.mean(e_in)) if e_in else 0.0
    eout = float(np.mean(e_out)) if e_out else 0.0
    return {"dirichlet_in": ein, "dirichlet_out": eout, "ratio": (eout / ein if ein > 0 else None)}


def signed_spectral_gap(S: sp.spmatrix, k: int = 6) -> dict:
    """Spectral radius, second-largest |eigenvalue|, gap, and the most negative eig.

    The most-negative real eigenvalue reflects the negative-edge (repression)
    spectrum that drives the directed / signed bound.
    """
    n = S.shape[0]
    if n <= 256:
        vals = np.linalg.eigvals(S.toarray() if sp.issparse(S) else np.asarray(S))
    else:
        vals = sla.eigs(
            S.tocsc().astype(float),
            k=min(k, n - 2),
            which="LM",
            return_eigenvectors=False,
            maxiter=5000,
        )
    mags = np.sort(np.abs(vals))[::-1]
    rho = float(mags[0]) if len(mags) else 0.0
    second = float(mags[1]) if len(mags) > 1 else 0.0
    return {
        "rho": rho,
        "second_abs": second,
        "spectral_gap": rho - second,
        "min_real_eig": float(np.min(np.real(vals))) if len(vals) else 0.0,
    }


def symmetric_normalized_variant(S: sp.spmatrix) -> sp.csr_matrix:
    """Symmetric-normalized D^-1/2 |A| D^-1/2 from |S| symmetrized (undirected case).

    Drops direction and sign to give the symmetric reference operator on the same
    real edges (eigenvalues in [-1, 1], ||.||_2 = rho).
    """
    a = _abs_symmetric_weight(S)
    deg = np.asarray(a.sum(axis=1)).ravel()
    dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    d = sp.diags(dinv)
    return (d @ a @ d).tocsr()
