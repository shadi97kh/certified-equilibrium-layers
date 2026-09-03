"""Tests for Project 2 (certification). Pure numpy, runs in CI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from cel.data import load_string_grn
from certification import RandomEdgeNoise, certify_l2_region, empirical_coverage

FIXTURE = Path(__file__).parent / "fixtures" / "string_tiny.txt"


def _graph(cache_root: Path):
    return load_string_grn(FIXTURE, cache_root=cache_root, threshold=700).S


def test_noise_preserves_symmetry_nonneg_and_contained(cache_root: Path) -> None:
    S = _graph(cache_root)
    noise = RandomEdgeNoise(base_std=0.2)
    sp_ = noise.perturb(S, np.random.default_rng(0))
    assert sp.issparse(sp_)
    assert abs(sp_ - sp_.T).max() < 1e-12  # symmetric
    assert sp_.min() >= 0.0  # nonnegative weights
    assert abs(sp_).max() <= 1.0 + 1e-9  # spectral-clipped: containment preserved


def test_radius_monotone_in_delta_and_analytic_dominates(cache_root: Path) -> None:
    S = _graph(cache_root)
    x0 = np.random.default_rng(0).standard_normal(S.shape[0])
    noise = RandomEdgeNoise(base_std=0.15)
    radii = []
    for delta in (0.2, 0.1, 0.05, 0.01):
        res = certify_l2_region(
            x0,
            S,
            noise,
            alpha=0.2,
            n_samples=600,
            delta=delta,
            gamma=0.05,
            rng=np.random.default_rng(1),
        )
        assert res.radius_mc >= 0.0
        assert res.radius_analytic >= res.radius_mc  # perturbation bound dominates
        assert not res.insufficient_samples
        radii.append(res.radius_mc)
    # delta decreases across the loop (0.2 -> 0.01); a stronger guarantee (smaller
    # delta) gives a larger radius, so the radii are non-decreasing.
    assert radii == sorted(radii)


def test_certificate_covers_out_of_sample(cache_root: Path) -> None:
    S = _graph(cache_root)
    x0 = np.random.default_rng(0).standard_normal(S.shape[0])
    noise = RandomEdgeNoise(base_std=0.15)
    delta = 0.1
    res = certify_l2_region(
        x0,
        S,
        noise,
        alpha=0.2,
        n_samples=800,
        delta=delta,
        gamma=0.05,
        rng=np.random.default_rng(1),
    )
    cov = empirical_coverage(
        x0,
        S,
        noise,
        alpha=0.2,
        radius=res.radius_mc,
        n_test=1500,
        rng=np.random.default_rng(7),
    )
    # The certified region holds out of sample at the nominal level (with slack).
    assert cov >= 1.0 - delta - 0.02


def test_zero_noise_gives_zero_radius(cache_root: Path) -> None:
    S = _graph(cache_root)
    x0 = np.ones(S.shape[0])
    res = certify_l2_region(
        x0,
        S,
        RandomEdgeNoise(base_std=0.0),
        alpha=0.2,
        n_samples=200,
        delta=0.05,
        gamma=0.05,
        rng=np.random.default_rng(0),
    )
    assert res.radius_mc == 0.0  # no perturbation, no deviation


def test_larger_noise_gives_larger_radius(cache_root: Path) -> None:
    S = _graph(cache_root)
    x0 = np.random.default_rng(0).standard_normal(S.shape[0])

    def radius(std: float) -> float:
        return certify_l2_region(
            x0,
            S,
            RandomEdgeNoise(base_std=std),
            alpha=0.2,
            n_samples=500,
            delta=0.1,
            gamma=0.05,
            rng=np.random.default_rng(3),
        ).radius_mc

    assert radius(0.3) > radius(0.05)


def test_committed_certification_json() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "aaai" / "certified_region.json"
    result = json.loads(path.read_text())
    assert result["all_calibrated"] is True
    for d in result["per_delta"]:
        assert d["analytic_dominates"] is True
        assert d["covers_nominal"] is True
