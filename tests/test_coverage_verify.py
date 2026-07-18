"""Regression test for the AAAI coverage verification. Pure numpy, runs in CI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from cel.verify.containment import dense_propagation_operator
from certification import EdgeBernoulliNoise, bernstein_certificate

COVERAGE_JSON = Path(__file__).resolve().parents[1] / "results" / "aaai" / "coverage.json"


def _matching(confidences: np.ndarray) -> sp.csr_matrix:
    k = len(confidences)
    rows, cols, data = [], [], []
    for e, c in enumerate(confidences):
        rows += [2 * e, 2 * e + 1]
        cols += [2 * e + 1, 2 * e]
        data += [float(c), float(c)]
    return sp.coo_matrix((data, (rows, cols)), shape=(2 * k, 2 * k)).tocsr()


def test_certificate_not_violated_on_matching() -> None:
    # The certified radius must hold for at least 1 - delta of the draws.
    noise = EdgeBernoulliNoise(C=_matching(np.array([0.9, 0.6, 0.75, 0.5])))
    alpha = 0.5
    x0 = np.random.default_rng(0).standard_normal(noise.n_nodes)
    cert = bernstein_certificate(x0, noise, alpha, delta=0.05)
    y_plugin = dense_propagation_operator(noise.mean_adjacency(), alpha) @ x0

    rng = np.random.default_rng(1)
    n = 3000
    devs = np.empty(n)
    max_spec = 0.0
    for t in range(n):
        a = noise.sample_adjacency(rng)
        max_spec = max(max_spec, float(np.linalg.norm(a.toarray(), 2)))
        devs[t] = float(np.linalg.norm(dense_propagation_operator(a, alpha) @ x0 - y_plugin))

    assert max_spec <= 1.0 + 1e-9  # matching keeps ||A||_op <= 1 (regime assumption)
    for delta in (0.2, 0.1, 0.05, 0.01):
        coverage = float(np.mean(devs <= cert.radius_at(delta)))
        assert coverage >= 1.0 - delta  # violation rate <= delta


def test_committed_coverage_json_passes() -> None:
    result = json.loads(COVERAGE_JSON.read_text())
    assert result["verdict"] == "PASS"
    assert result["any_violation"] is False
    assert result["max_excess_violation"] <= 1e-9  # violation_rate - delta never positive
    assert result["max_observed_spectral_norm"] <= 1.0 + 1e-9  # matching regime held
    for cfg in result["configs"]:
        for d in cfg["per_delta"]:
            assert d["empirical_coverage"] >= d["nominal_coverage"]
            assert d["violated"] is False
