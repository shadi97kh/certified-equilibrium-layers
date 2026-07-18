"""Monotone equilibrium propagation layer: differentiable P_alpha with exact IFT backward.

Recasts the analytic personalized-PageRank / APPNP propagation P_alpha as an implicit
(monDEQ / IGNN-style) equilibrium layer that solves the fixed point

    E = alpha * e + (1 - alpha) * w * S @ E

for E, where S is the real (row-normalized signed) TRRUST operator, alpha in (0, 1) is the
restart probability, and w in (0, 1] is a learnable edge-weight scale. The map is a contraction
whenever (1 - alpha) * w * rho(S) < 1; since the row-normalized S has spectral radius rho(S) <= 1
and alpha > 0, the fixed point is UNIQUE even though P_alpha is highly non-normal
(||P_alpha|| >> rho) -- this is the monotone-operator guarantee of well-posedness.

Forward: a fixed-point / Anderson iteration to machine precision. Backward: the implicit
function theorem, NOT unrolling -- at the fixed point E*, differentiating E* = alpha e +
(1-alpha) w S E* gives the adjoint system (I - beta S^T) y = grad_E with beta = (1-alpha) w, and
the parameter gradients follow in closed form. Because the fixed point is linear, this IFT
gradient is EXACT (matches finite differences to numerical precision).

The equilibrium equals the analytic resolvent E* = alpha (I - beta S)^{-1} e, so with w = 1 it
reproduces the closed-form P_alpha propagation (the APPNP limit). Needs torch; cel.models
guards the import so the torch-free path is unaffected.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_MAX_ITER = 2000
_TOL = 1e-12


def _solve_fixed_point(
    s: Tensor, e: Tensor, beta: float, max_iter: int = _MAX_ITER, tol: float = _TOL
):
    """Iterate E = e_scaled + beta S E to convergence; returns (E, residual)."""
    eq = e
    eq_prev = torch.zeros_like(e)
    for _ in range(max_iter):
        eq = e + beta * (s @ eq_prev)
        if torch.linalg.norm(eq - eq_prev) <= tol * (1.0 + torch.linalg.norm(eq)):
            break
        eq_prev = eq
    residual = float(torch.linalg.norm(eq - (e + beta * (s @ eq))))
    return eq, residual


class _Equilibrium(torch.autograd.Function):
    """E = alpha e + beta S E (beta = (1-alpha) w) with an exact IFT backward."""

    @staticmethod
    def forward(ctx, e: Tensor, alpha: Tensor, w: Tensor, s: Tensor) -> Tensor:
        beta = float((1.0 - alpha) * w)
        eq, residual = _solve_fixed_point(s, alpha * e, beta)
        ctx.save_for_backward(eq, e, alpha, w)
        ctx.s = s
        ctx.residual = residual
        return eq

    @staticmethod
    def backward(ctx, grad_e: Tensor):
        eq, e, alpha, w = ctx.saved_tensors
        s = ctx.s
        beta = float((1.0 - alpha) * w)
        # Adjoint fixed point: (I - beta S^T) y = grad_e  ->  y = grad_e + beta S^T y.
        y, _res = _solve_fixed_point(s.transpose(-1, -2), grad_e, beta)
        # dE/de = alpha (I - beta S)^{-1}; so grad wrt e is alpha * y.
        g_e = alpha * y
        se = s @ eq
        # E = alpha e + (1-alpha) w S E. dbeta/dalpha = -w, d(alpha e)/dalpha = e.
        g_alpha = (y * (e - w * se)).sum()
        # dbeta/dw = (1-alpha).
        g_w = (1.0 - alpha) * (y * se).sum()
        return g_e, g_alpha, g_w, None


class MonotoneEquilibriumPropagation(nn.Module):
    """Learnable-alpha / edge-scale equilibrium layer for E = alpha e + (1-alpha) w S E."""

    def __init__(self, s: Tensor, alpha_init: float = 0.5):
        super().__init__()
        self.register_buffer("S", s)
        # Unconstrained params mapped through sigmoid so alpha in (0,1), w in (0,1] (monotone repair).
        import math

        self.alpha_raw = nn.Parameter(
            torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=s.dtype)
        )
        self.w_raw = nn.Parameter(torch.tensor(4.0, dtype=s.dtype))  # sigmoid(4) ~ 0.982

    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_raw)

    def w(self) -> Tensor:
        return torch.sigmoid(self.w_raw)

    def forward(self, e: Tensor) -> Tensor:
        return _Equilibrium.apply(e, self.alpha(), self.w(), self.S)


def analytic_propagate(s: Tensor, e: Tensor, alpha: float, w: float = 1.0) -> Tensor:
    """Closed-form equilibrium E* = alpha (I - (1-alpha) w S)^{-1} e (the APPNP limit)."""
    n = s.shape[-1]
    beta = (1.0 - alpha) * w
    a = torch.eye(n, dtype=s.dtype, device=s.device) - beta * s
    return torch.linalg.solve(a, alpha * e)
