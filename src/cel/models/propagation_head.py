"""APPNP / personalized-PageRank propagation head with certified containment.

Implements the propagation operator

    P_alpha = alpha (I - (1 - alpha) S)^{-1}

over the symmetric-normalized GRN adjacency S, via ``num_steps`` power-iteration
steps (no dense inverse). The teleport probability alpha is the single
interpretable locality knob: it is parameterized as ``sigmoid(theta)`` so any
learned value stays in the open interval (0, 1).

Containment is enforced by construction, not by a loss penalty (OUTLINE.md hard
rule 2). S is spectral-norm-clipped to ||S||_2 <= 1; the geometric series then
gives

    ||P_alpha||_2 <= alpha * sum_{k>=0} (1-alpha)^k ||S||_2^k <= 1.

The forward pass asserts the post-clip spectral norm is within tolerance of 1.
The same operator and decay envelope are audited numerically in
:mod:`cel.verify.containment`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"alpha_init must be in (0, 1), got {p}")
    return float(torch.logit(torch.tensor(p)).item())


class PropagationHead(nn.Module):
    """Bounded-alpha APPNP propagation with a spectral-norm containment clip.

    Args:
        num_steps: number of power-iteration steps (the propagation horizon K).
        alpha_init: initial teleport probability in (0, 1).
        learnable_alpha: if True, alpha is trained through ``theta``.
        clip: if True, spectral-norm-clip S inside forward so ||S||_2 <= 1.
        clip_iters: power-iteration steps used to estimate ||S||_2.
    """

    def __init__(
        self,
        num_steps: int = 10,
        alpha_init: float = 0.1,
        learnable_alpha: bool = True,
        clip: bool = True,
        clip_iters: int = 100,
    ):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        self.num_steps = num_steps
        self.clip = clip
        self.clip_iters = clip_iters
        theta0 = torch.tensor(_logit(alpha_init))
        self.theta = nn.Parameter(theta0, requires_grad=learnable_alpha)

    @property
    def alpha(self) -> Tensor:
        """Teleport probability sigmoid(theta), guaranteed in (0, 1)."""
        return torch.sigmoid(self.theta)

    @staticmethod
    def _build_sparse(edge_index: Tensor, edge_weight: Tensor, n: int) -> Tensor:
        return torch.sparse_coo_tensor(edge_index, edge_weight, (n, n)).coalesce()

    @staticmethod
    def spectral_norm(s_sparse: Tensor, iters: int = 100, generator=None) -> Tensor:
        """Estimate ||S||_2 by power iteration (S symmetric => dominant |eigenvalue|)."""
        n = s_sparse.shape[0]
        v = torch.randn(n, 1, generator=generator, dtype=s_sparse.dtype, device=s_sparse.device)
        v = v / v.norm().clamp_min(1e-12)
        sigma = s_sparse.new_zeros(())
        for _ in range(iters):
            w = torch.sparse.mm(s_sparse, v)
            sigma = w.norm()
            if sigma <= 1e-12:
                return sigma
            v = w / sigma
        return sigma

    def _clip_weights(self, edge_index: Tensor, edge_weight: Tensor, n: int) -> Tensor:
        """Scale edge weights so the operator's spectral norm is at most 1.

        The scale is a constant w.r.t. autograd (estimated under no_grad), so
        gradients still flow linearly to edge_weight but not through the norm
        estimate.
        """
        with torch.no_grad():
            s_sparse = self._build_sparse(edge_index, edge_weight, n)
            sigma = self.spectral_norm(s_sparse, self.clip_iters)
            scale = 1.0 / torch.clamp(sigma * (1.0 + 1e-6), min=1.0)
        return edge_weight * scale

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int | None = None,
        assert_contained: bool = True,
    ) -> Tensor:
        """Propagate node signal h [N] or [N, F] through P_alpha.

        S is built from (edge_index, edge_weight), spectral-norm-clipped when
        ``self.clip`` is set, and applied via power iteration. With containment
        enforced, ||P_alpha||_2 <= 1, so propagation never amplifies the signal.
        """
        n = num_nodes if num_nodes is not None else int(edge_index.max()) + 1
        squeeze = h.dim() == 1
        x = h.unsqueeze(-1) if squeeze else h

        w = self._clip_weights(edge_index, edge_weight, n) if self.clip else edge_weight
        s_sparse = self._build_sparse(edge_index, w, n)

        if assert_contained:
            with torch.no_grad():
                sigma = float(self.spectral_norm(s_sparse, self.clip_iters))
            assert sigma <= 1.0 + 1e-4, f"||S||_2={sigma:.6f} exceeds 1; containment broken"

        alpha = self.alpha
        z = x
        for _ in range(self.num_steps):
            z = (1.0 - alpha) * torch.sparse.mm(s_sparse, z) + alpha * x
        return z.squeeze(-1) if squeeze else z

    def operator_norm(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int | None = None,
        iters: int = 100,
        generator=None,
    ) -> float:
        """Empirically estimate ||P_alpha||_2 by power iteration on the forward map.

        Provably bounded by 1 under the spectral clip; used by the containment
        tests to confirm the bound holds across random weights.
        """
        n = num_nodes if num_nodes is not None else int(edge_index.max()) + 1
        v = torch.randn(n, generator=generator)
        v = v / v.norm().clamp_min(1e-12)
        sigma = 0.0
        with torch.no_grad():
            for _ in range(iters):
                w = self.forward(v, edge_index, edge_weight, n, assert_contained=False)
                sigma = float(w.norm())
                if sigma <= 1e-12:
                    return 0.0
                v = w / sigma
        return sigma
