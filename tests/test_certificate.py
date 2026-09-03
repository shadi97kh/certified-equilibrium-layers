"""Tests for the matrix-Bernstein certificate. Pure numpy, runs in CI."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cel.data import load_string_confidence
from cel.verify.containment import dense_propagation_operator
from certification import (
    EdgeBernoulliNoise,
    bernstein_b,
    bernstein_certificate,
    bernstein_radius,
    monte_carlo_certificate,
    p_alpha_eigenvalue_range,
)
from certification.noise_model import normalize_adjacency

FIXTURE = Path(__file__).parent / "fixtures" / "string_tiny.txt"


def _fixture_noise() -> EdgeBernoulliNoise:
    nodes, c = load_string_confidence(FIXTURE, threshold=700)
    return EdgeBernoulliNoise(C=c, nodes=nodes)


def test_per_edge_variance_is_c_times_one_minus_c() -> None:
    noise = _fixture_noise()
    cert = bernstein_certificate(np.ones(noise.n_nodes), noise, alpha=0.5, delta=0.1)
    for (i, j), var in zip(cert.edges, cert.per_edge_variance, strict=True):
        c = noise.C[i, j]
        assert var == pytest.approx(c * (1.0 - c))


def test_variance_proxy_is_max_node_incident_variance() -> None:
    noise = _fixture_noise()
    cert = bernstein_certificate(np.ones(noise.n_nodes), noise, alpha=0.5, delta=0.1)
    # v = max_i sum_{j~i} c_ij(1-c_ij), the operator norm of the diagonal variance.
    node_var = np.zeros(noise.n_nodes)
    for (i, j), var in zip(cert.edges, cert.per_edge_variance, strict=True):
        node_var[i] += var
        node_var[j] += var
    assert cert.v == pytest.approx(node_var.max())
    assert np.allclose(cert.node_variance, node_var)


def test_uniform_edge_bound() -> None:
    noise = _fixture_noise()
    cert = bernstein_certificate(np.ones(noise.n_nodes), noise, alpha=0.5, delta=0.1)
    c = sp.triu(noise.C, k=1).tocoo().data
    expected_l = float(np.maximum(c, 1.0 - c).max())
    assert abs(cert.L - expected_l) < 1e-12


def test_b_closed_form() -> None:
    noise = _fixture_noise()
    cert = bernstein_certificate(np.ones(noise.n_nodes), noise, alpha=0.5, delta=0.1)
    log_term = math.log(2.0 * cert.n_nodes / cert.delta)
    expected = math.sqrt(2.0 * cert.v * log_term) + (2.0 * cert.L / 3.0) * log_term
    assert cert.b == pytest.approx(expected)
    assert cert.b == pytest.approx(bernstein_b(cert.delta, cert.v, cert.L, cert.n_nodes))


def test_radius_uses_correct_resolvent_constant() -> None:
    noise = _fixture_noise()
    alpha = 0.4
    x0 = np.random.default_rng(0).standard_normal(noise.n_nodes)
    cert = bernstein_certificate(x0, noise, alpha=alpha, delta=0.1)
    x0_norm = float(np.linalg.norm(x0))
    correct = (1.0 - alpha) / alpha * cert.b * x0_norm
    forbidden = (2.0 - alpha) / alpha * cert.b * x0_norm  # the constant the spec rules out
    assert cert.radius == pytest.approx(correct)
    # The forbidden (2-alpha)/alpha constant would give a strictly larger radius.
    assert cert.radius != pytest.approx(forbidden)
    assert cert.radius < forbidden


def test_eigenvalue_range_and_op_norm() -> None:
    alpha = 0.5
    lo, hi = p_alpha_eigenvalue_range(alpha)
    assert lo == pytest.approx(alpha / (2.0 - alpha))
    assert hi == 1.0
    cert = bernstein_certificate(np.ones(4), _fixture_noise(), alpha=alpha, delta=0.1)
    assert cert.eig_min == pytest.approx(alpha / (2.0 - alpha))
    assert cert.eig_max == 1.0
    assert cert.p_alpha_op_norm == 1.0
    with pytest.raises(ValueError, match="alpha"):
        p_alpha_eigenvalue_range(1.5)


def test_p_alpha_eigenvalues_match_predicted_range() -> None:
    # On a symmetric-normalized operator (eigenvalues in [-1, 1]) the predicted
    # P_alpha eigenvalue range [alpha/(2-alpha), 1] holds and ||P_alpha||=1.
    noise = _fixture_noise()
    alpha = 0.5
    s_bar = normalize_adjacency(noise.mean_adjacency(), "sym")
    p = dense_propagation_operator(s_bar, alpha)
    ev = np.linalg.eigvalsh((p + p.T) / 2.0)
    assert ev.max() == pytest.approx(1.0, abs=1e-6)
    assert ev.min() >= alpha / (2.0 - alpha) - 1e-6


def test_resolvent_identity_holds() -> None:
    # Algebraic identity that pins the (1-alpha)/alpha constant.
    noise = _fixture_noise()
    alpha = 0.5
    rng = np.random.default_rng(1)
    m1 = normalize_adjacency(noise.sample_adjacency(rng), "sym")
    m2 = normalize_adjacency(noise.mean_adjacency(), "sym")
    p1 = dense_propagation_operator(m1, alpha)
    p2 = dense_propagation_operator(m2, alpha)
    lhs = p1 - p2
    rhs = (1.0 - alpha) / alpha * p1 @ (m1.toarray() - m2.toarray()) @ p2
    assert np.allclose(lhs, rhs, atol=1e-8)


def test_radius_and_b_monotone_in_delta() -> None:
    noise = _fixture_noise()
    cert = bernstein_certificate(np.ones(noise.n_nodes), noise, alpha=0.5, delta=0.1)
    bs = [cert.b_at(d) for d in (0.2, 0.1, 0.05, 0.01)]
    rs = [cert.radius_at(d) for d in (0.2, 0.1, 0.05, 0.01)]
    assert bs == sorted(bs)  # smaller delta => larger b
    assert rs == sorted(rs)
    assert bernstein_radius(0.1, cert.v, cert.L, cert.n_nodes, 0.5, cert.x0_norm) == pytest.approx(
        cert.radius
    )


def test_bernstein_bound_holds_empirically() -> None:
    # The high-prob structure bound: emp (1-delta) quantile ||A - A_bar||_op <= b.
    noise = _fixture_noise()
    delta = 0.1
    cert = bernstein_certificate(np.ones(noise.n_nodes), noise, alpha=0.5, delta=delta)
    a_bar = noise.mean_adjacency().toarray()
    rng = np.random.default_rng(0)
    ops = np.array(
        [np.linalg.norm(noise.sample_adjacency(rng).toarray() - a_bar, 2) for _ in range(4000)]
    )
    assert cert.b >= np.quantile(ops, 1.0 - delta)


def test_end_to_end_radius_covers_prediction_deviation() -> None:
    # Single-edge graph keeps ||A||_op <= 1 (un-normalized), so P_alpha is well
    # defined and ||P_alpha||=1 holds; the certified radius must cover the actual
    # prediction deviation with probability at least 1 - delta.
    c = sp.csr_matrix(np.array([[0.0, 0.7], [0.7, 0.0]]))
    noise = EdgeBernoulliNoise(C=c)
    alpha, delta = 0.5, 0.1
    x0 = np.array([1.0, -0.5])
    cert = bernstein_certificate(x0, noise, alpha=alpha, delta=delta)
    y_plugin = dense_propagation_operator(noise.mean_adjacency(), alpha) @ x0
    rng = np.random.default_rng(0)
    devs = []
    for _ in range(2000):
        a = noise.sample_adjacency(rng)
        y = dense_propagation_operator(a, alpha) @ x0
        devs.append(float(np.linalg.norm(y - y_plugin)))
    coverage = float(np.mean(np.array(devs) <= cert.radius))
    assert coverage >= 1.0 - delta


def test_certain_edge_has_zero_variance() -> None:
    c = sp.csr_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.5, 0.0]]))
    noise = EdgeBernoulliNoise(C=c)
    cert = bernstein_certificate(np.ones(3), noise, alpha=0.5, delta=0.1)
    var_by_edge = dict(zip(cert.edges, cert.per_edge_variance, strict=True))
    assert var_by_edge[(0, 1)] == pytest.approx(0.0)  # c = 1 => no variance
    assert var_by_edge[(1, 2)] == pytest.approx(0.25)  # c = 0.5


def test_monte_carlo_certificate() -> None:
    rng = np.random.default_rng(0)
    n, d = 400, 4
    truth = np.array([0.5, -0.5, 0.0, 1.0])
    samples = truth + 0.2 * rng.standard_normal((n, d))
    mc = monte_carlo_certificate(samples, delta=0.05, value_radius=2.0)

    assert mc.y_bar.shape == (d,)
    assert np.all(mc.mean_ci_low <= mc.y_bar) and np.all(mc.y_bar <= mc.mean_ci_high)
    assert np.all(mc.median_ci_low <= mc.y_median) and np.all(mc.y_median <= mc.median_ci_high)
    # The Hoeffding interval contains the truth (it holds with high probability).
    assert np.all(mc.mean_ci_low <= truth) and np.all(truth <= mc.mean_ci_high)


def test_monte_carlo_halfwidth_shrinks_with_samples() -> None:
    rng = np.random.default_rng(1)
    small = rng.standard_normal((50, 3))
    large = rng.standard_normal((5000, 3))
    hw_small = monte_carlo_certificate(small, delta=0.05, value_radius=1.0).mean_halfwidth
    hw_large = monte_carlo_certificate(large, delta=0.05, value_radius=1.0).mean_halfwidth
    assert np.all(hw_large < hw_small)


def test_monte_carlo_guards() -> None:
    with pytest.raises(ValueError, match="delta"):
        monte_carlo_certificate(np.zeros((10, 2)), delta=0.0)
    with pytest.raises(ValueError, match="at least 2"):
        monte_carlo_certificate(np.zeros((1, 2)), delta=0.1)
