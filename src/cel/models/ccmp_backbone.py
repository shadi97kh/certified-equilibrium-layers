"""CCMP backbone: FiLM-conditioned per-concept heterogeneous message passing.

This is the CASCADE backbone. It reuses the CCMP ingredients from the prior
implementation (a FiLM generator, a FiLM-conditioned HeteroSAGE message-passing
layer, and a per-concept scalar head; Perez et al. 2018 for FiLM, Hamilton et
al. 2017 for GraphSAGE) and adapts them to the K=9 concept partition produced
by :mod:`cel.features`.

Each concept channel k owns:

- its own input feature block x_k (the non-overlapping block from the features
  layer),
- its own concept identity embedding e_k that drives FiLM modulation,
- its own independent FiLM-conditioned HeteroSAGE stream over a small
  heterogeneous graph of three node types (siRNA, mRNA, interaction).

The streams are independent modules and each reads only its own input block, so
the per-concept embedding z_k is a function of x_k (and e_k) alone. An
intervention on concept k's input therefore cannot alter any other concept's
embedding z_j. That is the by-construction L_k = 0 isolation property; the
prior CCMP obtained it by giving each concept its own FiLM pass over a shared
backbone, and CASCADE strengthens it to fully separated per-concept streams.
The numerical leakage audit in the tests confirms it holds bit-for-bit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cel.features import CONCEPT_SPECS


def default_concept_dims() -> dict[str, int]:
    """Per-concept input widths from the features layer (concept -> block width)."""
    return {name: len(feats) for name, feats in CONCEPT_SPECS}


class FiLMGenerator(nn.Module):
    """Generate FiLM parameters (gamma, beta) per node type from a concept embedding.

    Follows the prior CCMP generator: gamma is centered at 1 and beta at 0 with
    small scale, so an untrained stream starts close to an identity modulation.
    """

    def __init__(self, concept_dim: int, hidden_dim: int, n_node_types: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_node_types = n_node_types
        self.net = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 2 * hidden_dim * n_node_types),
        )

    def forward(self, concept_emb: Tensor) -> list[tuple[Tensor, Tensor]]:
        params = self.net(concept_emb)
        d = self.hidden_dim
        out: list[tuple[Tensor, Tensor]] = []
        for t in range(self.n_node_types):
            off = t * 2 * d
            gamma = 1.0 + 0.1 * params[..., off : off + d]
            beta = 0.1 * params[..., off + d : off + 2 * d]
            out.append((gamma, beta))
        return out


class FiLMHeteroSAGELayer(nn.Module):
    """One FiLM-conditioned heterogeneous GraphSAGE step over (siRNA, mRNA, interaction).

    Messages flow siRNA/mRNA -> interaction and interaction -> siRNA/mRNA via
    typed projections, each node is updated and LayerNorm-ed, then FiLM-modulated
    as h = gamma * h + beta. Identical in structure to the prior CCMP layer.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj_s2i = nn.Linear(dim, dim)
        self.proj_m2i = nn.Linear(dim, dim)
        self.update_inter = nn.Linear(dim + dim, dim)
        self.proj_i2s = nn.Linear(dim, dim)
        self.update_sirna = nn.Linear(dim + dim, dim)
        self.proj_i2m = nn.Linear(dim, dim)
        self.update_mrna = nn.Linear(dim + dim, dim)
        self.ln_s = nn.LayerNorm(dim)
        self.ln_m = nn.LayerNorm(dim)
        self.ln_i = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, s: Tensor, m: Tensor, i: Tensor, film: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        neighbor_agg = (F.relu(self.proj_s2i(s)) + F.relu(self.proj_m2i(m))) / 2.0
        inter_h = F.relu(self.update_inter(torch.cat([i, neighbor_agg], dim=-1)))
        sirna_h = F.relu(self.update_sirna(torch.cat([s, F.relu(self.proj_i2s(i))], dim=-1)))
        mrna_h = F.relu(self.update_mrna(torch.cat([m, F.relu(self.proj_i2m(i))], dim=-1)))

        sirna_h = self.ln_s(sirna_h)
        mrna_h = self.ln_m(mrna_h)
        inter_h = self.ln_i(inter_h)

        (g_s, b_s), (g_m, b_m), (g_i, b_i) = film
        sirna_h = g_s * sirna_h + b_s
        mrna_h = g_m * mrna_h + b_m
        inter_h = g_i * inter_h + b_i

        return self.dropout(sirna_h), self.dropout(mrna_h), self.dropout(inter_h)


class ConceptHead(nn.Module):
    """Map a per-concept embedding to a scalar concept activation in [0, 1]."""

    def __init__(self, in_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class ConceptStream(nn.Module):
    """One independent FiLM-conditioned HeteroSAGE stream for a single concept.

    The concept's input block is lifted into three node-type embeddings, refined
    by ``n_layers`` FiLM-conditioned message-passing steps (FiLM driven by this
    stream's own concept embedding), and pooled by concatenation into the
    per-concept embedding z_k of width ``3 * hidden_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        concept_emb_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = 3 * hidden_dim
        self.enc_s = nn.Linear(in_dim, hidden_dim)
        self.enc_m = nn.Linear(in_dim, hidden_dim)
        self.enc_i = nn.Linear(in_dim, hidden_dim)
        # Concept identity: a learnable vector unique to this stream, used as the
        # FiLM conditioning signal (the analogue of CCMP's concept embedding).
        self.concept_emb = nn.Parameter(torch.randn(concept_emb_dim))
        self.layers = nn.ModuleList(
            FiLMHeteroSAGELayer(hidden_dim, dropout) for _ in range(n_layers)
        )
        self.films = nn.ModuleList(
            FiLMGenerator(concept_emb_dim, hidden_dim) for _ in range(n_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        s = F.relu(self.enc_s(x))
        m = F.relu(self.enc_m(x))
        i = F.relu(self.enc_i(x))
        emb = self.concept_emb.unsqueeze(0).expand(x.shape[0], -1)
        for layer, film in zip(self.layers, self.films, strict=True):
            s, m, i = layer(s, m, i, film(emb))
        return torch.cat([s, m, i], dim=-1)


class CCMPBackbone(nn.Module):
    """K=9 per-concept FiLM-conditioned message-passing backbone.

    Produces one separated embedding per concept and a per-transcript direct
    seed-binding repression score. Construction defaults to the concept layout
    of :mod:`cel.features`; pass ``concept_dims`` to override the per-concept
    input widths.
    """

    def __init__(
        self,
        concept_dims: Mapping[str, int] | None = None,
        concepts: Sequence[str] | None = None,
        hidden_dim: int = 32,
        concept_emb_dim: int = 32,
        n_layers: int = 2,
        head_hidden: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = dict(concept_dims) if concept_dims is not None else default_concept_dims()
        self.concepts: list[str] = list(concepts) if concepts is not None else list(dims)
        missing = [c for c in self.concepts if c not in dims]
        if missing:
            raise ValueError(f"no input dim given for concepts {missing}")

        self.streams = nn.ModuleDict(
            {
                name: ConceptStream(dims[name], hidden_dim, concept_emb_dim, n_layers, dropout)
                for name in self.concepts
            }
        )
        emb_dim = 3 * hidden_dim
        self.heads = nn.ModuleDict(
            {name: ConceptHead(emb_dim, head_hidden) for name in self.concepts}
        )
        # Strict concept bottleneck: the repression score reads only the K
        # concept scalars, never the high-dimensional per-concept embeddings.
        self.repression_readout = nn.Sequential(
            nn.Linear(len(self.concepts), head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Contiguous slices for splitting a concatenated feature matrix.
        self._slices: dict[str, slice] = {}
        cursor = 0
        for name in self.concepts:
            self._slices[name] = slice(cursor, cursor + dims[name])
            cursor += dims[name]
        self.total_dim = cursor

    @property
    def K(self) -> int:
        return len(self.concepts)

    def _as_blocks(self, features: Mapping[str, Tensor] | Tensor) -> dict[str, Tensor]:
        if isinstance(features, Tensor):
            if features.shape[-1] != self.total_dim:
                raise ValueError(f"expected last dim {self.total_dim}, got {features.shape[-1]}")
            return {name: features[..., sl] for name, sl in self._slices.items()}
        missing = [c for c in self.concepts if c not in features]
        if missing:
            raise ValueError(f"missing concept blocks {missing}")
        return {name: features[name] for name in self.concepts}

    def get_per_concept_embeddings(
        self, features: Mapping[str, Tensor] | Tensor
    ) -> dict[str, Tensor]:
        """Return the K separated per-concept embeddings {concept: z_k [B, 3*hidden]}.

        Each z_k is computed only from concept k's own input block, so the
        embeddings are mutually isolated (L_k = 0).
        """
        blocks = self._as_blocks(features)
        return {name: self.streams[name](blocks[name]) for name in self.concepts}

    def get_concept_scores(self, features: Mapping[str, Tensor] | Tensor) -> Tensor:
        """Return the per-concept scalar activations stacked as [B, K]."""
        embs = self.get_per_concept_embeddings(features)
        return torch.stack([self.heads[name](embs[name]) for name in self.concepts], dim=-1)

    def forward(self, features: Mapping[str, Tensor] | Tensor) -> Tensor:
        """Return per-transcript direct seed-binding repression scores [B].

        The score passes through the K-dimensional concept bottleneck, so it is a
        function of the concept scalars alone.
        """
        scores = self.get_concept_scores(features)
        return self.repression_readout(scores).squeeze(-1)
