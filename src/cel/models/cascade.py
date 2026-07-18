"""The full CASCADE model: CCMP backbone followed by the certified propagation head.

Pipeline (per transcript node of the GRN):

    features --CCMP backbone--> x0 (direct seed-binding repression)
            --propagation head--> predicted off-target log2 fold-change.

The CCMP backbone (:class:`cel.models.ccmp_backbone.CCMPBackbone`) maps each
transcript's K=9 concept feature blocks to a scalar direct repression x0,
preserving the L_k = 0 isolation. The propagation head
(:class:`cel.models.propagation_head.PropagationHead`) diffuses x0 over the
GRN with the bounded-alpha personalized-PageRank operator P_alpha, whose
operator norm is at most 1 by construction, so the diffused prediction is
certifiably contained (Contribution 2).

The full prediction is therefore ``y = P_alpha x0``: the direct seed effect plus
its certified-decay ripple through the regulatory graph.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from .ccmp_backbone import CCMPBackbone
from .propagation_head import PropagationHead


class CASCADE(nn.Module):
    """CCMP backbone (direct repression x0) composed with the propagation head.

    Args:
        backbone: the per-concept CCMP backbone producing x0 per transcript.
        propagation: the APPNP propagation head diffusing x0 over the GRN.
    """

    def __init__(self, backbone: CCMPBackbone, propagation: PropagationHead):
        super().__init__()
        self.backbone = backbone
        self.propagation = propagation

    @property
    def alpha(self) -> Tensor:
        """The propagation teleport probability (the interpretable locality knob)."""
        return self.propagation.alpha

    def direct_repression(self, features: Mapping[str, Tensor] | Tensor) -> Tensor:
        """Per-transcript direct seed-binding repression x0 [N] (pre-propagation)."""
        return self.backbone(features)

    def forward(
        self,
        features: Mapping[str, Tensor] | Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int | None = None,
        assert_contained: bool = True,
    ) -> Tensor:
        """Predict per-transcript off-target log2FC: y = P_alpha x0.

        Args:
            features: per-transcript concept blocks (dict) or the concatenated
                feature matrix [N, total_dim].
            edge_index, edge_weight: the symmetric-normalized GRN.
            num_nodes: node count N (inferred from edge_index if omitted).
            assert_contained: assert the propagation spectral norm stays <= 1.
        """
        x0 = self.backbone(features)
        return self.propagation(
            x0, edge_index, edge_weight, num_nodes=num_nodes, assert_contained=assert_contained
        )

    def get_per_concept_embeddings(
        self, features: Mapping[str, Tensor] | Tensor
    ) -> dict[str, Tensor]:
        """Delegate to the backbone: the K separated per-concept embeddings."""
        return self.backbone.get_per_concept_embeddings(features)

    def get_concept_scores(self, features: Mapping[str, Tensor] | Tensor) -> Tensor:
        """Delegate to the backbone: the [N, K] concept bottleneck."""
        return self.backbone.get_concept_scores(features)
