"""KKT orthogonal-projection layers for hard mass-action / thermodynamic constraints.

Theorem 3 ties the predicted off-target knockdown to the real seed-duplex binding
free energy dG via a mass-action (Langmuir / Hill) occupancy law, and enforces it
with a differentiable KKT projection applied at train and test.

Linear case (KKT-hPINN, Chen, Constante-Flores, Li 2024). For a linear equality
A u = b the orthogonal projection onto {u : A u = b} is closed-form:

    u_proj = u - A^T (A A^T)^{-1} (A u - b).

The mass-action law is linear in logit space: logit(occupancy) = a + b*dG with
a = mu / tau and b = -1 / tau (the Boltzmann slope). The constraint pins the
predictions' trend on [1, dG] (their ordinary-least-squares slope and intercept)
to the calibrated mass-action line, while the component of the prediction
ORTHOGONAL to [1, dG] (the feature-driven signal) is left untouched. So the
thermodynamic relationship is enforced exactly without discarding the feature
information, which is why accuracy is preserved.

Nonlinear case (KKT-Hardnet, Iftakher et al. 2025). When the occupancy law must
hold exactly and nonlinearly, ``newton_hard_projection`` projects onto the
nonlinear manifold {u : c(u) = 0} by Newton / SQP steps on the KKT system.

torch only (this module is not on any reported-data path).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class KKTLinearProjection(nn.Module):
    """Closed-form orthogonal projection onto {u : A u = b} (KKT-hPINN)."""

    def forward(self, u: Tensor, A: Tensor, b: Tensor) -> Tensor:
        """Project ``u`` [N] onto {u : A u = b}, with A [m, N] and b [m]."""
        au = A @ u  # [m]
        aat = A @ A.transpose(-1, -2)  # [m, m]
        lam = torch.linalg.solve(aat, au - b)  # [m]
        return u - A.transpose(-1, -2) @ lam  # [N]


def mass_action_line(dG: Tensor) -> Tensor:
    """Design matrix Z = [1, dG] (N x 2) for the mass-action logit trend."""
    return torch.stack([torch.ones_like(dG), dG], dim=-1)


def fit_mass_action_line(dG: Tensor, y: Tensor, eps: float = 1e-3) -> tuple[float, float]:
    """Calibrate the mass-action line (a, b) from real (dG, knockdown) data.

    Fits logit(y) ~ a + b*dG by ordinary least squares; ``b`` is the empirical
    Boltzmann slope (expected negative: stronger binding, lower dG, more knockdown).
    """
    z = dG.double()
    yc = y.double().clamp(eps, 1.0 - eps)
    s = torch.log(yc / (1.0 - yc))
    zmat = mass_action_line(z)  # [N, 2]
    coeffs = torch.linalg.solve(zmat.T @ zmat, zmat.T @ s)  # [2]
    return float(coeffs[0]), float(coeffs[1])


def mass_action_residual(s: Tensor, dG: Tensor, a: float, b: float) -> float:
    """Max |OLS-trend(s on [1, dG]) - (a, b)|, in float64 (target < 1e-8).

    Uses a QR basis of [1, dG] so the fit is stable even when dG is large / tightly
    spaced (which makes Z^T Z ill conditioned).
    """
    z = mass_action_line(dG.double())
    q, r = torch.linalg.qr(z)
    coeffs = torch.linalg.solve(r, q.T @ s.double())  # (Z^T Z)^{-1} Z^T s = R^{-1} Q^T s
    target = torch.tensor([a, b], dtype=torch.float64)
    return float(torch.max(torch.abs(coeffs - target)))


class MassActionProjection(nn.Module):
    """Pin the predicted-knockdown logit trend on dG to the mass-action line (a, b).

    The KKT-hPINN projection enforces, exactly, that the ordinary-least-squares fit
    of the predicted logits on [1, dG] equals the calibrated mass-action (a, b);
    the orthogonal-to-[1, dG] feature residual is preserved. ``a, b`` come from
    :func:`fit_mass_action_line` on the real training data.
    """

    def __init__(self, a: float, b: float):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))
        self.register_buffer("b", torch.tensor(float(b)))
        self.proj = KKTLinearProjection()

    def forward(self, s: Tensor, dG: Tensor) -> Tensor:
        # Stable orthogonal projection via a QR basis of [1, dG]:
        # s_proj = (I - Q Q^T) s + Q (R @ (a, b)) = (I - P) s + Z (a, b).
        z = mass_action_line(dG)  # [N, 2]
        q, r = torch.linalg.qr(z)
        coeffs = torch.stack([self.a, self.b]).to(s.dtype)  # [2]
        return s - q @ (q.transpose(-1, -2) @ s) + q @ (r @ coeffs)

    def residual(self, s: Tensor, dG: Tensor) -> float:
        return mass_action_residual(s, dG, float(self.a), float(self.b))


def hill_occupancy(dG: Tensor, mu: float, tau: float) -> Tensor:
    """Mass-action / Langmuir occupancy theta = 1 / (1 + exp((dG - mu) / tau))."""
    return torch.sigmoid(-(dG - mu) / tau)


def _jacobian(constraint_fn, u: Tensor) -> Tensor:
    u_req = u.clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(lambda z: constraint_fn(z).double(), u_req).detach()


def newton_hard_projection(
    u0: Tensor,
    constraint_fn,
    iters: int = 200,
    tol: float = 1e-12,
    damping: float = 1e-2,
) -> Tensor:
    """Project u0 onto the nonlinear manifold {u : c(u) = 0} (KKT-Hardnet).

    Solves min ||u - u0||^2 s.t. c(u) = 0 by adaptive Levenberg-Marquardt SQP steps
    on the linearized KKT system:

        u_try = u0 - J^T (J J^T + lambda I)^{-1} (c(u) + J (u0 - u)).

    ``lambda`` is shrunk after every step that reduces ||c|| and grown otherwise,
    so near the manifold it tends to 0 and the step becomes the exact KKT / Gauss-
    Newton projection, driving the residual to machine precision (when reachable).
    Returns the projected u in float64.
    """
    u = u0.detach().double().clone()
    base = u0.detach().double()
    lam = float(damping)
    cur = float(torch.max(torch.abs(constraint_fn(u).double())))
    for _ in range(iters):
        if cur < tol:
            break
        c = constraint_fn(u).double().detach()
        jac = _jacobian(constraint_fn, u)
        m = jac.shape[0]
        rhs = c + jac @ (base - u)
        accepted = False
        for _ls in range(30):
            jjt = jac @ jac.transpose(-1, -2) + lam * torch.eye(m, dtype=torch.float64)
            step = torch.linalg.solve(jjt, rhs)
            u_try = base - jac.transpose(-1, -2) @ step
            new = float(torch.max(torch.abs(constraint_fn(u_try).double())))
            if new < cur:
                u, cur, lam = u_try, new, max(lam * 0.3, 1e-14)
                accepted = True
                break
            lam *= 4.0
        if not accepted:
            break
    return u
