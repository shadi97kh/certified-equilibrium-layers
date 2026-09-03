"""Symmetric-normalized certificate (Project 2 C5) via degree concentration.

For the symmetric-normalized operator S(A) = D_A^{-1/2} A D_A^{-1/2} the degree
normalization D_A couples the edges, so the clean per-edge matrix-Bernstein
decomposition of the un-normalized case (cel.certification.certificate) does
not apply directly. We handle the coupling with a concentration-of-the-
normalized-adjacency argument in the style of Le, Levina, and Vershynin
(Random Structures and Algorithms, 2017): condition on the high-probability
event that the random degrees concentrate around their expectations, then
propagate that into the resolvent bound.

Assumption (minimum expected degree). Let dbar_i = sum_j c_ij be the expected
degree and var_i = sum_j c_ij(1-c_ij). With probability at least 1 - delta/2 the
degrees concentrate: |deg_i(A) - dbar_i| <= t_i for all i, where (scalar
Bernstein, union over n nodes)

    t_i = sqrt(2 var_i log(2n / (delta/2))) + (2/3) log(2n / (delta/2)).

The relative degree fluctuation is rho = max_i t_i / dbar_i. The certificate is
valid only when rho < 1 (degrees concentrate, every node stays connected with
high probability). This is the finite-sample analogue of the Le-Levina-Vershynin
minimum-expected-degree condition dbar_min >= C log n; as the GRN is thresholded
sparser, dbar_min falls, rho crosses 1, and the certificate breaks down.

Bound. Split S(A) - S(A_bar) (with A_bar = E[A] the confidence matrix) as

    S(A) - S(A_bar) = [S(A) - S0(A)] + [S0(A) - S(A_bar)],
    S0(A) = D_bar^{-1/2} A D_bar^{-1/2}  (A normalized by EXPECTED degrees).

The second (numerator) term is D_bar^{-1/2}(A - A_bar)D_bar^{-1/2}, bounded by
b_num / dbar_min where b_num is the un-normalized matrix-Bernstein bound on
||A - A_bar||_op (reused from certificate.py) and dbar_min is the minimum
expected degree. The first (degree) term equals E S0(A) E - S0(A) with
E = diag((deg_i/dbar_i)^{-1/2}); on the concentration event ||E - I|| <= eta and
||E|| <= (1-rho)^{-1/2} with eta = (1-rho)^{-1/2} - 1, giving

    ||S(A) - S(A_bar)||_op <= eta ((1-rho)^{-1/2} + 1)(1 + b_num/dbar_min) + b_num/dbar_min =: B_sym.

Resolvent conversion. Because symmetric normalization guarantees ||S||_op <= 1
for ANY graph (eigenvalues in [-1, 1]), ||P_alpha||_op = 1 always, so

    radius_sym(delta) = ((1-alpha)/alpha) * B_sym * ||x0||_2

holds with probability at least 1 - delta (degree event delta/2 plus numerator
event delta/2). Unlike the un-normalized certificate this needs no spectral
matching assumption; it needs the minimum-expected-degree assumption instead.

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .certificate import bernstein_b, uniform_edge_bound, variance_proxy
from .noise_model import EdgeBernoulliNoise


def expected_degrees(C: sp.spmatrix) -> np.ndarray:
    """Expected degrees dbar_i = sum_j c_ij (row sums of the confidence matrix)."""
    return np.asarray(C.sum(axis=1)).ravel()


def degree_variances(C: sp.spmatrix) -> np.ndarray:
    """Per-node degree variance var_i = sum_j c_ij(1 - c_ij)."""
    dbar = expected_degrees(C)
    sumsq = np.asarray(C.multiply(C).sum(axis=1)).ravel()
    return dbar - sumsq


def degree_deviation_bounds(C: sp.spmatrix, delta_deg: float) -> np.ndarray:
    """Scalar-Bernstein degree-deviation bounds t_i (union over n nodes)."""
    n = C.shape[0]
    log_term = math.log(2.0 * n / delta_deg)
    var = degree_variances(C)
    return np.sqrt(2.0 * var * log_term) + (2.0 / 3.0) * log_term


def min_degree_diagnostic(noise: EdgeBernoulliNoise, delta: float, llv_const: float = 1.0) -> dict:
    """Minimum-expected-degree diagnostic and the degree-concentration verdict."""
    c = noise.mean_adjacency()
    n = c.shape[0]
    dbar = expected_degrees(c)
    active = dbar > 0
    n_active = int(active.sum())
    delta_deg = delta / 2.0
    llv_threshold = llv_const * math.log(max(n, 2))

    if n_active == 0:
        return {
            "n_nodes": n,
            "n_active": 0,
            "expected_min_degree": 0.0,
            "mean_expected_degree": 0.0,
            "degree_rel_fluctuation": math.inf,
            "llv_min_degree_threshold": llv_threshold,
            "llv_condition_holds": False,
            "assumption_holds": False,
            "delta_deg": delta_deg,
        }

    t = degree_deviation_bounds(c, delta_deg)
    dbar_min = float(dbar[active].min())
    rho = float((t[active] / dbar[active]).max())
    return {
        "n_nodes": n,
        "n_active": n_active,
        "expected_min_degree": dbar_min,
        "mean_expected_degree": float(dbar[active].mean()),
        "degree_rel_fluctuation": rho,
        "llv_min_degree_threshold": llv_threshold,
        "llv_condition_holds": bool(dbar_min >= llv_threshold),
        "assumption_holds": bool(rho < 1.0),
        "delta_deg": delta_deg,
    }


@dataclass(frozen=True)
class NormalizedCertificate:
    """Symmetric-normalized certified radius via degree concentration.

    When ``assumption_holds`` is False (rho >= 1, the degree-concentration event
    cannot be guaranteed) the certificate is invalid and the radii are +inf.
    """

    radius_sym: float
    structure_bound: float  # B_sym, bound on ||S(A) - S(A_bar)||_op
    b_num: float  # un-normalized Bernstein bound on ||A - A_bar||_op
    term_numerator: float  # b_num / dbar_min
    term_degree: float  # degree-perturbation contribution
    eta: float  # (1-rho)^{-1/2} - 1
    expected_min_degree: float
    degree_rel_fluctuation: float  # rho
    assumption_holds: bool
    llv_min_degree_threshold: float
    llv_condition_holds: bool
    n_nodes: int
    n_active: int
    alpha: float
    delta: float
    x0_norm: float
    v: float
    L: float


def normalized_certificate(
    x0: np.ndarray,
    noise: EdgeBernoulliNoise,
    alpha: float,
    delta: float,
    llv_const: float = 1.0,
) -> NormalizedCertificate:
    """Symmetric-normalized certificate under the minimum-expected-degree assumption."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    c = noise.mean_adjacency()
    n = c.shape[0]
    diag = min_degree_diagnostic(noise, delta, llv_const)
    v, _ = variance_proxy(noise)
    l_edge = uniform_edge_bound(noise)
    b_num = bernstein_b(delta / 2.0, v, l_edge, n)
    x0_norm = float(np.linalg.norm(np.asarray(x0, dtype=float)))
    rho = diag["degree_rel_fluctuation"]
    dbar_min = diag["expected_min_degree"]

    common = {
        "b_num": b_num,
        "expected_min_degree": dbar_min,
        "degree_rel_fluctuation": rho,
        "assumption_holds": diag["assumption_holds"],
        "llv_min_degree_threshold": diag["llv_min_degree_threshold"],
        "llv_condition_holds": diag["llv_condition_holds"],
        "n_nodes": n,
        "n_active": diag["n_active"],
        "alpha": alpha,
        "delta": delta,
        "x0_norm": x0_norm,
        "v": v,
        "L": l_edge,
    }

    if not diag["assumption_holds"]:
        # Degrees are not guaranteed to concentrate (rho >= 1): no finite radius.
        return NormalizedCertificate(
            radius_sym=math.inf,
            structure_bound=math.inf,
            term_numerator=(b_num / dbar_min if dbar_min > 0 else math.inf),
            term_degree=math.inf,
            eta=math.inf,
            **common,
        )

    eta = (1.0 - rho) ** -0.5 - 1.0
    term_numerator = b_num / dbar_min
    s0_norm_bound = 1.0 + term_numerator  # ||D_bar^{-1/2} A D_bar^{-1/2}|| <= 1 + b_num/dbar_min
    term_degree = eta * ((1.0 - rho) ** -0.5 + 1.0) * s0_norm_bound
    structure_bound = term_degree + term_numerator
    radius_sym = (1.0 - alpha) / alpha * structure_bound * x0_norm

    return NormalizedCertificate(
        radius_sym=radius_sym,
        structure_bound=structure_bound,
        term_numerator=term_numerator,
        term_degree=term_degree,
        eta=eta,
        **common,
    )
