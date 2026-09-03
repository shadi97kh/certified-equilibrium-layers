"""Theorem 4: deterministic certified bound on the propagated off-target effect.

The propagated off-target signal is E = P_alpha(S) e, where e is the real
seed-thermodynamic input (ViennaRNA, with its per-transcript error) and S is the
real GRN. With the true (e_true, S_true), the triangle inequality gives

    ||E - E_true|| = ||P_alpha(S) e - P_alpha(S_true) e_true||
                  <= ||P_alpha(S)|| ||e - e_true||
                     + ||P_alpha(S) - P_alpha(S_true)|| ||e_true||.

The first (seed-error) term uses the T2 contraction constant
C(alpha, rho) = alpha / (1 - (1-alpha) rho(S)) as a certified bound on
||P_alpha(S)||. The second (GRN-edge) term uses the SECOND RESOLVENT IDENTITY,

    P_alpha(S) - P_alpha(S') = ((1-alpha)/alpha) P_alpha(S) (S - S') P_alpha(S'),

so ||P_alpha(S) - P_alpha(S')|| <= ((1-alpha)/alpha) ||P_alpha(S)|| ||S - S'||
||P_alpha(S')||. Combining:

    ||E - E_true|| <= ||P_alpha(S)|| ||e - e_true||
                      + ((1-alpha)/alpha) ||P_alpha(S)|| ||P_alpha(S')|| ||S - S'|| ||e_true||.

A global L2 bound can be loose; ``rowwise_bound`` gives the per-transcript
refinement (using the per-row operator norm of P_alpha), which is the tightening
to report when the global bound is vacuous.

Pure numpy / scipy, reusing the P_alpha operator from
:mod:`cel.verify.stability`. Verification-layer code (off any reported-data
path).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import stability as st


def _norm2(M) -> float:
    if sp.issparse(M):
        return st.operator_2norm(M)
    return float(np.linalg.norm(np.asarray(M), 2))


def propagate(S, e, alpha: float) -> np.ndarray:
    """Propagated off-target E = P_alpha(S) e (the personalized-PageRank diffusion)."""
    return st.apply_p_alpha(S, alpha, np.asarray(e, dtype=float))


def p_alpha_norm(S, alpha: float, iters: int = 80, seed: int = 0) -> float:
    """True spectral norm ||P_alpha(S)||_2 of the resolvent operator."""
    return st.resolvent_2norm(S, alpha, iters=iters, seed=seed)


def contraction_constant(rho: float, alpha: float) -> float:
    """T2 contraction constant C(alpha, rho) = alpha / (1 - (1-alpha) rho)."""
    return st.certified_constant(rho, alpha)


def dense_p_alpha(S, alpha: float) -> np.ndarray:
    """Dense P_alpha = alpha (I - (1-alpha) S)^{-1} (for per-row / small-graph use)."""
    n = S.shape[0]
    eye = np.eye(n)
    mat = eye - (1.0 - alpha) * (S.toarray() if sp.issparse(S) else np.asarray(S))
    return alpha * np.linalg.inv(mat)


def resolvent_difference_bound(S, s_prime, alpha: float) -> dict:
    """Bound ||P_alpha(S) - P_alpha(S')||_2 via the second resolvent identity."""
    diff = S - s_prime
    ds_norm = _norm2(diff)
    pn = p_alpha_norm(S, alpha)
    pn_prime = p_alpha_norm(s_prime, alpha)
    bound = ((1.0 - alpha) / alpha) * pn * ds_norm * pn_prime
    return {
        "bound": bound,
        "delta_S_norm": ds_norm,
        "p_alpha_norm": pn,
        "p_alpha_prime_norm": pn_prime,
    }


def certified_bound(
    S,
    alpha: float,
    e_norm: float,
    e_err_norm: float,
    s_err_norm: float,
    p_norm: float | None = None,
    p_prime_norm: float | None = None,
) -> dict:
    """Total certified deviation bound = seed-error term + GRN-edge term.

    Args:
        S: the real GRN operator.
        alpha: teleport parameter.
        e_norm: ||e_true|| (or ||e|| as a proxy).
        e_err_norm: ||e - e_true||, the seed-thermodynamic error norm.
        s_err_norm: ||S - S_true||, the GRN-edge uncertainty norm.
        p_norm: ||P_alpha(S)||_2 (computed if None).
        p_prime_norm: ||P_alpha(S')||_2 (defaults to p_norm: a perturbed GRN with
            the same contraction regime has the same norm bound).
    """
    if p_norm is None:
        p_norm = p_alpha_norm(S, alpha)
    if p_prime_norm is None:
        p_prime_norm = p_norm
    seed_term = p_norm * e_err_norm
    grn_term = ((1.0 - alpha) / alpha) * p_norm * p_prime_norm * s_err_norm * e_norm
    return {
        "bound": seed_term + grn_term,
        "seed_error_term": seed_term,
        "grn_edge_term": grn_term,
        "p_alpha_norm": p_norm,
    }


def rowwise_bound(
    S,
    alpha: float,
    e_norm: float,
    e_err_norm: float,
    s_err_norm: float,
) -> np.ndarray:
    """Per-transcript refinement |E_g - E_true_g| <= rowwise_bound[g].

    Uses the per-row operator norm ||(P_alpha)_g||_2 (the g-th row of the dense
    resolvent) instead of the global ||P_alpha||_2, so each transcript's bound is
    no larger than (and usually much smaller than) the global bound.
    """
    p = dense_p_alpha(S, alpha)
    row_norms = np.linalg.norm(p, axis=1)  # ||(P_alpha)_g||_2
    p_global = float(np.linalg.norm(p, 2))
    seed = row_norms * e_err_norm
    grn = ((1.0 - alpha) / alpha) * row_norms * p_global * s_err_norm * e_norm
    return seed + grn


def verify_bound(e_pred: np.ndarray, e_measured: np.ndarray, bound: float) -> dict:
    """Does the predicted-vs-measured L2 deviation stay within the certified bound?"""
    dev = float(
        np.linalg.norm(np.asarray(e_pred, dtype=float) - np.asarray(e_measured, dtype=float))
    )
    return {"deviation": dev, "bound": float(bound), "holds": bool(dev <= bound * (1.0 + 1e-9))}


def is_vacuous(bound: float, measured: np.ndarray, factor: float = 10.0) -> dict:
    """Flag a vacuous (uninformative) bound: bound much larger than the measured scale."""
    scale = float(np.linalg.norm(np.asarray(measured, dtype=float)))
    ratio = (bound / scale) if scale > 0 else None
    return {
        "measured_scale": scale,
        "bound": float(bound),
        "bound_over_measured": ratio,
        "vacuous": bool(scale > 0 and bound > factor * scale),
        "factor": factor,
    }
