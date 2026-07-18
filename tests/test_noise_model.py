"""Tests for the inhomogeneous random-subgraph noise model and smoothed predictor.

Pure numpy, runs in CI. Uses the committed STRING fixture graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cel.data import load_string_confidence
from certification import (
    EdgeBernoulliNoise,
    normalize_adjacency,
    smoothed_prediction,
)
from certification.smoothed import apply_p_alpha

FIXTURE = Path(__file__).parent / "fixtures" / "string_tiny.txt"


def _fixture_noise() -> EdgeBernoulliNoise:
    nodes, c = load_string_confidence(FIXTURE, threshold=700)
    return EdgeBernoulliNoise(C=c, nodes=nodes)


def test_string_confidence_matrix() -> None:
    nodes, c = load_string_confidence(FIXTURE, threshold=700)
    dense = c.toarray()
    assert len(nodes) == 4
    assert np.allclose(dense, dense.T)  # symmetric
    assert dense.min() >= 0.0 and dense.max() <= 1.0  # confidences in [0, 1]
    # combined_score / 1000 on the canonicalized support.
    assert dense[0, 1] == pytest.approx(0.9)  # max(880, 900)
    assert dense[1, 2] == pytest.approx(0.8)
    assert dense[2, 3] == pytest.approx(0.75)
    assert dense[0, 3] == pytest.approx(0.72)


def test_mean_adjacency_is_confidence() -> None:
    noise = _fixture_noise()
    assert (noise.mean_adjacency() != noise.C).nnz == 0  # A_bar == C exactly


def test_sample_is_binary_symmetric_subset() -> None:
    noise = _fixture_noise()
    a = noise.sample_adjacency(np.random.default_rng(1))
    dense = a.toarray()
    assert set(np.unique(dense).tolist()) <= {0.0, 1.0}  # binary inclusion
    assert np.allclose(dense, dense.T)  # symmetric
    # Support is a subset of the candidate edges.
    cand = (noise.C.toarray() > 0).astype(float)
    assert np.all((dense > 0) <= (cand > 0))


def test_inclusion_frequency_matches_confidence() -> None:
    noise = _fixture_noise()
    rng = np.random.default_rng(0)
    n = 8000
    acc = np.zeros(noise.C.shape)
    for _ in range(n):
        acc += noise.sample_adjacency(rng).toarray()
    freq = acc / n
    c = noise.C.toarray()
    tri = np.triu(c, 1)
    for i, j in zip(*np.nonzero(tri), strict=True):
        assert abs(freq[i, j] - tri[i, j]) < 0.03  # E[A_ij] = c_ij


def test_certain_edge_always_included() -> None:
    # c_ij = 1 => the edge is always present.
    c = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    noise = EdgeBernoulliNoise(C=c)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert noise.sample_adjacency(rng).toarray()[0, 1] == 1.0


def test_normalization_schemes() -> None:
    noise = _fixture_noise()
    c = noise.C
    row = normalize_adjacency(c, "row").toarray()
    assert np.allclose(row.sum(axis=1), 1.0)  # row-stochastic
    sym = normalize_adjacency(c, "sym").toarray()
    assert np.allclose(sym, sym.T)  # symmetric normalization
    assert np.allclose(normalize_adjacency(c, "none").toarray(), c.toarray())
    with pytest.raises(ValueError, match="scheme"):
        normalize_adjacency(c, "bogus")


def test_normalization_handles_isolated_node() -> None:
    # A node with no edges must not produce NaN/inf rows.
    a = sp.csr_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    for scheme in ("row", "sym"):
        m = normalize_adjacency(a, scheme).toarray()
        assert np.isfinite(m).all()
        assert np.allclose(m[2], 0.0)  # isolated node has an all-zero row


def test_invalid_confidence_and_shape_raise() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        EdgeBernoulliNoise(C=sp.csr_matrix(np.array([[0.0, 1.5], [1.5, 0.0]])))
    with pytest.raises(ValueError, match="square"):
        EdgeBernoulliNoise(C=sp.csr_matrix(np.zeros((2, 3))))


def test_from_string_matches_loader() -> None:
    noise = EdgeBernoulliNoise.from_string(FIXTURE, threshold=700)
    _, c = load_string_confidence(FIXTURE, threshold=700)
    assert np.allclose(noise.C.toarray(), c.toarray())
    assert noise.nodes is not None


def test_smoothed_prediction_centers_and_gap() -> None:
    noise = _fixture_noise()
    x0 = np.random.default_rng(2).standard_normal(noise.n_nodes)
    sm = smoothed_prediction(
        x0, noise, alpha=0.2, scheme="row", n_samples=2000, rng=np.random.default_rng(3)
    )
    assert sm.y_bar.shape == (noise.n_nodes,)
    assert sm.y_median.shape == (noise.n_nodes,)
    assert sm.y_plugin.shape == (noise.n_nodes,)
    assert np.isfinite(sm.y_bar).all() and np.isfinite(sm.y_plugin).all()

    # y_plugin is exactly the plug-in mean-graph prediction P_alpha(S(A_bar)) x0.
    expected_plugin = apply_p_alpha(
        normalize_adjacency(noise.mean_adjacency(), "row"), x0, alpha=0.2, method="dense"
    )
    assert np.allclose(sm.y_plugin, expected_plugin)

    # E_A[P_alpha(S(A))] != P_alpha(S(A_bar)) because P_alpha is nonlinear in A.
    assert sm.plugin_gap > 1e-3
    assert sm.plugin_gap == pytest.approx(float(np.linalg.norm(sm.y_bar - sm.y_plugin)))


def test_smoothed_prediction_sym_scheme_runs() -> None:
    noise = _fixture_noise()
    x0 = np.ones(noise.n_nodes)
    sm = smoothed_prediction(
        x0, noise, alpha=0.2, scheme="sym", n_samples=500, rng=np.random.default_rng(0)
    )
    assert sm.scheme == "sym"
    assert np.isfinite(sm.y_std).all()


def test_smoothed_mean_converges_to_plugin_when_confidence_is_certain() -> None:
    # If every candidate edge has confidence 1, the ensemble is deterministic and
    # y_bar equals y_plugin (the only nonlinearity source, edge randomness, is gone).
    c = sp.csr_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    noise = EdgeBernoulliNoise(C=c)
    x0 = np.array([1.0, -0.5, 0.25])
    sm = smoothed_prediction(
        x0, noise, alpha=0.3, scheme="row", n_samples=64, rng=np.random.default_rng(0)
    )
    assert np.allclose(sm.y_bar, sm.y_plugin)
    assert sm.plugin_gap < 1e-9
