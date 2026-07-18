"""Magnetic personalized-PageRank propagation on signed, directed GRNs.

``MagneticPPR`` is the magnetic-Laplacian analogue of APPNP. It runs ``K``
truncated personalized-PageRank power-iteration steps with teleport probability
``alpha``, where each propagation step is a magnetic signed Laplacian convolution
(``MSConv`` from torch_geometric_signed_directed, He et al. 2022, MSGNN):

    h^{0}     = transform(x)                         (real teleport target)
    h^{k+1}   = (1 - alpha) * MSConv(h^{k}) + alpha * h^{0},   k = 0 .. K-1

The magnetic signed Laplacian L_q^sigma is Hermitian, so (1-alpha)(I - L_q^sigma)
has spectral radius < 1 and, being normal, its operator 2-norm equals that
spectral radius. This is exactly what tames the non-normality of the real
directed transition S (whose personalized-PageRank operator P_alpha is non-normal
with ||P_alpha||_2 > 1 even when rho(S) <= 1): the verification of that contrast
lives in tests/test_propagation.py.

This module needs torch and torch_geometric_signed_directed; it is gated out of
the lightweight (numpy/scipy) CI image. It lives in the shared core so the
T2 experiments can import it as cel.propagation.MagneticPPR.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric_signed_directed.nn import MSConv


class MagneticPPR(nn.Module):
    """APPNP-style PPR propagation on the magnetic signed Laplacian.

    Args:
        in_channels: input feature width.
        out_channels: output feature width (and the propagated state width).
        K: number of truncated PPR power-iteration steps.
        q: magnetic charge / potential of L_q^sigma (0 to 0.25; 0.25 is the
            MSGNN default that maximally separates the two edge directions).
        alpha: teleport probability in (0, 1).
        normalization: magnetic-Laplacian normalization passed to MSConv.
        trainable_q: whether MSConv learns q.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 10,
        q: float = 0.25,
        alpha: float = 0.15,
        normalization: str = "sym",
        trainable_q: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = int(K)
        self.alpha = float(alpha)
        self.q = float(q)
        self.lin = nn.Linear(in_channels, out_channels)
        # Single-hop magnetic signed Laplacian conv (Chebyshev order 1) used as the
        # per-step propagation primitive; the K hops come from the PPR loop below.
        self.prop = MSConv(
            out_channels,
            out_channels,
            K=1,
            q=q,
            trainable_q=trainable_q,
            normalization=normalization,
            bias=False,
        )

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.prop.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Propagate real features ``x`` through K magnetic PPR steps.

        Returns the complex propagated state as ``(h_real, h_imag)``.
        """
        h0 = self.lin(x)
        h_real, h_imag = h0, torch.zeros_like(h0)
        for _ in range(self.K):
            p_real, p_imag = self.prop(h_real, h_imag, edge_index, edge_weight)
            h_real = (1.0 - self.alpha) * p_real + self.alpha * h0
            h_imag = (1.0 - self.alpha) * p_imag
        return h_real, h_imag
