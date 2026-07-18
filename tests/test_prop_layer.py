"""Monotone equilibrium propagation layer: fixed point, APPNP limit, exact IFT gradient.

Uses a small fixed row-normalized operator for speed; the real-TRRUST run is the committed
ift_check.json. Needs torch; skipped otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("torch")
import torch  # noqa: E402

from cel.models.prop_layer import (  # noqa: E402
    MonotoneEquilibriumPropagation,
    analytic_propagate,
)


def _small_operator():
    a = torch.tensor(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
        ],
        dtype=torch.float64,
    )
    return a / a.sum(1, keepdim=True)  # row-normalized -> spectral radius 1


def test_fixed_point_and_appnp_limit() -> None:
    s = _small_operator()
    e = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=torch.float64)
    layer = MonotoneEquilibriumPropagation(s, alpha_init=0.4)
    eq = layer(e)
    alpha, w = float(layer.alpha()), float(layer.w())
    beta = (1.0 - alpha) * w
    # E solves the equilibrium to machine precision.
    assert float(torch.linalg.norm(eq - (alpha * e + beta * (s @ eq)))) < 1e-9
    # and equals the closed-form resolvent (APPNP limit).
    assert float(torch.linalg.norm(eq - analytic_propagate(s, e, alpha, w))) < 1e-9


def test_ift_gradient_matches_finite_diff() -> None:
    s = _small_operator()
    e = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=torch.float64, requires_grad=True)
    layer = MonotoneEquilibriumPropagation(s, alpha_init=0.4)
    loss = 0.5 * (layer(e) ** 2).sum()
    loss.backward()

    def fd(t, k, h=1e-6):
        with torch.no_grad():
            t[k] += h
            lp = float(0.5 * (layer(e).detach() ** 2).sum())
            t[k] -= 2 * h
            lm = float(0.5 * (layer(e).detach() ** 2).sum())
            t[k] += h
        return (lp - lm) / (2 * h)

    def rel(a, b):
        return abs(a - b) / (abs(b) + 1e-12)

    assert rel(float(layer.alpha_raw.grad), fd(layer.alpha_raw.data.view(-1), 0)) < 1e-5
    assert rel(float(layer.w_raw.grad), fd(layer.w_raw.data.view(-1), 0)) < 1e-5
    assert rel(float(e.grad[2]), fd(e.data, 2)) < 1e-5


def test_committed_ift_check_pass() -> None:
    path = REPO / "results" / "cascade_ot" / "ift_check.json"
    if not path.exists():
        pytest.skip("ift_check.json not built yet")
    j = json.loads(path.read_text())
    c = j["checks"]
    assert j["verdict"] == "PASS"
    assert c["fixed_point_machine_precision"] and c["appnp_agreement"]
    assert c["ift_grad_pass"] and c["unique_fixed_point"]
    assert c["ift_grad_vs_finite_diff_max_rel_error"] < 1e-5
