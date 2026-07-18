"""Tests for the KKT mass-action projection layers (cel.models.kkt_projection)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cel.models.kkt_projection import (  # noqa: E402
    KKTLinearProjection,
    MassActionProjection,
    fit_mass_action_line,
    hill_occupancy,
    mass_action_line,
    mass_action_residual,
    newton_hard_projection,
)


def test_kkt_linear_projection_satisfies_constraint() -> None:
    torch.manual_seed(0)
    n, m = 50, 3
    u = torch.randn(n).double()
    a_mat = torch.randn(m, n).double()
    b = torch.randn(m).double()
    up = KKTLinearProjection()(u, a_mat, b)
    assert float((a_mat @ up - b).abs().max()) < 1e-10  # constraint satisfied exactly
    # The correction lies in the row space of A (orthogonal / min-norm projection).
    correction = u - up
    null_component = correction - a_mat.T @ torch.linalg.solve(a_mat @ a_mat.T, a_mat @ correction)
    assert float(null_component.abs().max()) < 1e-8


def test_mass_action_projection_residual_and_retained_dof() -> None:
    torch.manual_seed(0)
    n = 200
    dG = (-8.0 + 4.0 * torch.rand(n)).double()
    y = torch.sigmoid(-(dG + 6.0) / 0.8).clamp(0.02, 0.98)
    a, b = fit_mass_action_line(dG, y)
    assert b < 0  # Boltzmann slope: lower dG (stronger binding) -> more knockdown

    proj = MassActionProjection(a, b)
    s = torch.randn(n).double()
    sp = proj(s, dG)
    assert proj.residual(sp, dG) < 1e-8  # constraint at machine precision
    assert mass_action_residual(s, dG, a, b) > 1e-3  # the raw prediction violates it

    # The component orthogonal to [1, dG] (the feature signal) is preserved.
    z = mass_action_line(dG)
    p = z @ torch.linalg.solve(z.T @ z, z.T)
    orth_before, orth_after = s - p @ s, sp - p @ sp
    assert float((orth_before - orth_after).abs().max()) < 1e-8


def test_mass_action_projection_is_differentiable() -> None:
    torch.manual_seed(0)
    n = 30
    dG = -8.0 + 4.0 * torch.rand(n)
    proj = MassActionProjection(-10.0, -1.0)
    s = torch.randn(n, requires_grad=True)
    loss = (proj(s, dG) ** 2).sum()
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_newton_hard_projection_converges() -> None:
    torch.manual_seed(0)
    n = 100
    dG = (-8.0 + 4.0 * torch.rand(n)).double()
    target = hill_occupancy(dG, mu=-6.0, tau=1.2).clamp(0.15, 0.85)
    s = (0.5 * torch.randn(n)).double()
    u = newton_hard_projection(s, lambda z: torch.sigmoid(z) - target)
    assert float((torch.sigmoid(u) - target).abs().max()) < 1e-8  # exact nonlinear pin


def test_fit_mass_action_line_recovers_slope() -> None:
    torch.manual_seed(1)
    n = 500
    dG = torch.randn(n).double()  # centered so the logit stays in the non-saturated range
    # knockdown generated from logit = 0.5 - 1.5*dG, so the fit recovers (0.5, -1.5).
    y = torch.sigmoid(0.5 - 1.5 * dG).clamp(0.01, 0.99)
    a, b = fit_mass_action_line(dG, y)
    assert abs(b - (-1.5)) < 0.1 and abs(a - 0.5) < 0.1
