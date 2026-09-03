"""Tests for the Theorem 2 stability primitives (cel.verify.stability)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cel.verify import stability as st


def _directed_broadcast(k: int = 6) -> sp.csr_matrix:
    """A row-normalized signed directed star: node 0 broadcasts to k leaves.

    Strictly lower-triangular, so rho(S) = 0, but column 0 has k unit entries, so
    ||S||_2 = sqrt(k): a clean non-normal (||S||_2 >> rho) signed operator.
    """
    rows = list(range(1, k + 1))
    cols = [0] * k
    data = [1.0] * (k - 1) + [-1.0]  # one repression edge
    return sp.coo_matrix((data, (rows, cols)), shape=(k + 1, k + 1)).tocsr()


def _symmetric_graph() -> sp.csr_matrix:
    # Symmetric-normalized variant of a small connected signed graph.
    base = _directed_broadcast(5)
    return st.symmetric_normalized_variant(base)


def test_neumann_and_certified_constant_formula() -> None:
    assert st.neumann_converges(rho=1.0, alpha=0.3) is True  # (1-0.3)*1 < 1
    assert st.neumann_converges(rho=2.0, alpha=0.3) is False  # (1-0.3)*2 = 1.4
    # alpha / (1 - (1-alpha) spec)
    assert np.isclose(st.certified_constant(1.0, 0.5), 0.5 / (1 - 0.5 * 1.0))
    assert st.certified_constant(2.0, 0.1) == float("inf")  # (1-0.1)*2 > 1


def test_directed_is_non_normal_norm_exceeds_rho() -> None:
    S = _directed_broadcast(6)
    rho = st.spectral_radius(S)
    norm2 = st.operator_2norm(S)
    assert rho < 1e-6  # strictly lower triangular -> nilpotent -> rho = 0
    assert np.isclose(norm2, np.sqrt(6), rtol=1e-6)  # ||S||_2 = sqrt(k)
    assert st.inf_operator_norm(S) <= 1.0 + 1e-9  # row substochastic


def test_resolvent_2norm_matches_dense() -> None:
    S = _directed_broadcast(5)
    n = S.shape[0]
    for alpha in (0.3, 0.6):
        dense = alpha * np.linalg.inv(np.eye(n) - (1 - alpha) * S.toarray())
        true = float(np.linalg.norm(dense, 2))
        est = st.resolvent_2norm(S, alpha, iters=300, seed=0)
        assert np.isclose(est, true, rtol=1e-2)


def test_directed_violates_l2_rho_but_norm_bound_holds() -> None:
    S = _directed_broadcast(6)
    rho, norm2 = st.spectral_radius(S), st.operator_2norm(S)
    alpha = 0.5
    op = st.resolvent_2norm(S, alpha, iters=300, seed=0)
    c_rho = st.certified_constant(rho, alpha)
    c_norm = st.certified_constant(norm2, alpha)
    assert op > c_rho  # rho-based L2 constant is exceeded (non-normality)
    if np.isfinite(c_norm):
        assert op <= c_norm * 1.02  # the ||S||_2-based constant is a valid bound
    # The sampled ratio is only a lower bound on the true operator norm.
    measured = st.measure_lipschitz(S, alpha, n_samples=200, seed=0)["max_ratio"]
    assert measured <= op + 1e-6


def test_symmetric_l2_rho_bound_is_tight() -> None:
    S = _symmetric_graph()
    assert abs(st.operator_2norm(S) - st.spectral_radius(S)) < 1e-6  # ||S||_2 = rho
    rho = st.spectral_radius(S)
    for alpha in (0.2, 0.5, 0.8):
        op = st.resolvent_2norm(S, alpha, iters=300, seed=0)
        c_rho = st.certified_constant(rho, alpha)
        assert op <= c_rho * 1.02  # never exceeded for the symmetric operator


def test_dirichlet_energy_and_smoothing() -> None:
    S = _symmetric_graph()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(S.shape[0])
    assert st.dirichlet_energy(S, x) >= 0.0
    assert st.dirichlet_energy(S, np.ones(S.shape[0])) == 0.0  # constant has zero energy
    # Smaller alpha (more diffusion) smooths more: lower Dirichlet energy ratio.
    low = st.dirichlet_smoothing(S, 0.1, n_probes=8, seed=0)["ratio"]
    high = st.dirichlet_smoothing(S, 0.9, n_probes=8, seed=0)["ratio"]
    assert low < high


def test_signed_spectral_gap_reports_negative_spectrum() -> None:
    S = _directed_broadcast(6)
    gap = st.signed_spectral_gap(S)
    assert {"rho", "second_abs", "spectral_gap", "min_real_eig"} <= set(gap)
    assert gap["spectral_gap"] >= -1e-9


def test_symmetric_variant_is_symmetric() -> None:
    S = st.symmetric_normalized_variant(_directed_broadcast(5))
    assert abs(S - S.T).max() < 1e-9
