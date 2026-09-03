"""Tests for the containment audit library (pure numpy/scipy, no torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cel.data import load_string_grn
from cel.verify import containment as C

FIXTURE = Path(__file__).parent / "fixtures" / "string_tiny.txt"


def _fixture_S(cache_root: Path) -> sp.csr_matrix:
    return load_string_grn(FIXTURE, cache_root=cache_root, threshold=700).S


def _random_sym_norm_graph(n: int, p: float, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    w = rng.random((n, n)) * (rng.random((n, n)) < p)
    w = np.triu(w, 1)
    w = w + w.T  # symmetric, zero diagonal
    deg = w.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    s = dinv[:, None] * w * dinv[None, :]
    return sp.csr_matrix(s)


def test_operator_norm_of_normalized_adjacency_at_most_one(cache_root: Path) -> None:
    s = _fixture_S(cache_root)
    assert C.operator_norm(s) <= 1.0 + 1e-6


def test_spectral_clip_caps_norm() -> None:
    # Scale a normalized operator up so its norm exceeds 1, then clip it back.
    base = _random_sym_norm_graph(8, 0.5, seed=1)
    inflated = base * 5.0
    assert C.operator_norm(inflated) > 1.0
    clipped, sigma = C.spectral_clip(inflated)
    assert sigma > 1.0
    assert C.operator_norm(clipped) <= 1.0 + 1e-6


def test_envelope_holds_on_fixture(cache_root: Path) -> None:
    s = _fixture_S(cache_root)
    for alpha in (0.05, 0.1, 0.25, 0.5, 0.9):
        report = C.verify_containment(s, seed_idx=np.array([0]), alpha=alpha)
        assert report.holds
        # Empirical decay never exceeds the certified (1-alpha)^r envelope.
        assert np.all(report.empirical <= report.envelope + 1e-9)
        # r = 0 is the full signal fraction and must not amplify.
        assert report.empirical[0] <= 1.0 + 1e-9


def test_envelope_holds_on_random_graphs() -> None:
    for seed in range(5):
        s = _random_sym_norm_graph(12, 0.4, seed=seed)
        rng = np.random.default_rng(seed)
        seed_idx = rng.choice(12, size=2, replace=False)
        report = C.verify_containment(s, seed_idx=seed_idx, alpha=0.2)
        assert report.holds, f"envelope violated on random graph seed={seed}"


def test_no_amplification(cache_root: Path) -> None:
    # ||P_alpha h|| <= ||h|| since ||P_alpha||_2 <= 1.
    s = _fixture_S(cache_root)
    rng = np.random.default_rng(0)
    h = rng.standard_normal(s.shape[0])
    p = C.propagate(s, h, alpha=0.3, num_steps=128)
    assert np.linalg.norm(p) <= np.linalg.norm(h) + 1e-6


def test_alpha_out_of_range_raises(cache_root: Path) -> None:
    s = _fixture_S(cache_root)
    h = np.ones(s.shape[0])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            C.propagate(s, h, alpha=bad, num_steps=4)


def test_spectral_gap_positive_for_connected_fixture(cache_root: Path) -> None:
    s = _fixture_S(cache_root)
    gap = C.spectral_gap(s)
    assert gap > 0.0  # the fixture component is connected


def test_effective_resistance_nonneg_and_seed_zero(cache_root: Path) -> None:
    s = _fixture_S(cache_root)
    res = C.effective_resistance_from_seed(s, seed_idx=np.array([0]))
    assert res.shape == (s.shape[0],)
    assert np.all(res >= 0.0)
    assert res[0] == 0.0  # zero resistance to itself
    assert np.all(np.isfinite(res))  # connected graph
