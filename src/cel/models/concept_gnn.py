"""Concept GNN with an optional, leakage-free RNA-FM node-feature channel.

Wraps the K=9 disjoint-stream CCMP backbone (:mod:`cel.models.ccmp_backbone`).
With ``use_rnafm=True`` the 640-d RNA-FM embedding of the guide and 3'UTR seed-site
nodes is added as a SINGLE dedicated concept channel ("rnafm") with its own
independent message-passing stream. Because every concept (the 9 existing ones AND
the rnafm channel) owns a separate stream that reads only its own input block:

- the RNA-FM channel is assigned to exactly one concept and is never mixed with the
  others (no other concept's stream ever sees the RNA-FM block, and the RNA-FM stream
  never sees any other block), and
- the 9 existing concept embeddings are bit-for-bit unchanged whether or not the
  RNA-FM channel is present.

So the L_k = 0 isolation property of Theorem 1 (zero cross-channel leakage) is
preserved: an intervention on the RNA-FM block changes only the RNA-FM embedding, and
an intervention on any concept block leaves the RNA-FM embedding untouched. The
feature-dimension manifest is exposed and can be written to JSON.

Needs torch. ``cel.models`` guards the import so the lightweight (torch-free) path
still imports without this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from torch import Tensor, nn

from cel.embeddings.rnafm_extract import RNAFM_DIM
from cel.models.ccmp_backbone import CCMPBackbone, default_concept_dims

RNAFM_CONCEPT = "rnafm"  # the single concept that owns the RNA-FM channel


class ConceptGNN(nn.Module):
    """K=9 concept GNN, optionally extended by one disjoint RNA-FM concept channel.

    Args:
        use_rnafm: if True, add the RNA-FM embedding as its own concept channel.
        rnafm_dim: width of the RNA-FM block (default 640, the rna_fm_t12 dimension).
        hidden_dim / concept_emb_dim / n_layers / head_hidden / dropout: forwarded to
            the CCMP backbone (every concept stream, including rnafm, uses them).
    """

    def __init__(
        self,
        use_rnafm: bool = False,
        rnafm_dim: int = RNAFM_DIM,
        hidden_dim: int = 32,
        concept_emb_dim: int = 32,
        n_layers: int = 2,
        head_hidden: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_rnafm = bool(use_rnafm)
        self.rnafm_dim = int(rnafm_dim)
        base_dims = default_concept_dims()  # the 9 disjoint concept blocks
        self.base_concepts: list[str] = list(base_dims)
        dims = dict(base_dims)
        if self.use_rnafm:
            if RNAFM_CONCEPT in dims:
                raise ValueError(f"'{RNAFM_CONCEPT}' collides with a base concept name")
            dims[RNAFM_CONCEPT] = self.rnafm_dim
        self._dims: dict[str, int] = dims
        self.concepts: list[str] = list(dims)
        self.backbone = CCMPBackbone(
            concept_dims=dims,
            concepts=self.concepts,
            hidden_dim=hidden_dim,
            concept_emb_dim=concept_emb_dim,
            n_layers=n_layers,
            head_hidden=head_hidden,
            dropout=dropout,
        )

    @property
    def K(self) -> int:
        return self.backbone.K

    def assemble_blocks(
        self, concept_blocks: Mapping[str, Tensor], rnafm_emb: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Combine the 9 base concept blocks with the optional RNA-FM block.

        ``concept_blocks`` holds the base concept -> [B, dim] tensors; ``rnafm_emb`` is
        the [B, rnafm_dim] RNA-FM embedding of the guide and seed-site nodes (required
        iff use_rnafm). The RNA-FM block enters only under the ``rnafm`` key, so it is
        never concatenated into another concept's block.
        """
        missing = [c for c in self.base_concepts if c not in concept_blocks]
        if missing:
            raise ValueError(f"missing base concept blocks {missing}")
        blocks = {name: concept_blocks[name] for name in self.base_concepts}
        if self.use_rnafm:
            if rnafm_emb is None:
                raise ValueError("use_rnafm=True requires the RNA-FM embedding block rnafm_emb")
            if rnafm_emb.shape[-1] != self.rnafm_dim:
                raise ValueError(
                    f"rnafm_emb last dim {rnafm_emb.shape[-1]} != rnafm_dim {self.rnafm_dim}"
                )
            blocks[RNAFM_CONCEPT] = rnafm_emb
        elif rnafm_emb is not None:
            raise ValueError("rnafm_emb was given but use_rnafm=False")
        return blocks

    def get_per_concept_embeddings(
        self, concept_blocks: Mapping[str, Tensor], rnafm_emb: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Return the per-concept embeddings {concept: z_k}; each reads only its block."""
        return self.backbone.get_per_concept_embeddings(
            self.assemble_blocks(concept_blocks, rnafm_emb)
        )

    def get_concept_scores(
        self, concept_blocks: Mapping[str, Tensor], rnafm_emb: Tensor | None = None
    ) -> Tensor:
        return self.backbone.get_concept_scores(self.assemble_blocks(concept_blocks, rnafm_emb))

    def forward(
        self, concept_blocks: Mapping[str, Tensor], rnafm_emb: Tensor | None = None
    ) -> Tensor:
        return self.backbone(self.assemble_blocks(concept_blocks, rnafm_emb))

    def feature_manifest(self) -> dict:
        """Manifest of the per-concept feature dimensions and the RNA-FM channel."""
        return {
            "model": "ConceptGNN (K disjoint CCMP streams)",
            "use_rnafm": self.use_rnafm,
            "n_concepts": self.K,
            "base_concepts": self.base_concepts,
            "rnafm_concept": RNAFM_CONCEPT if self.use_rnafm else None,
            "rnafm_dim": self.rnafm_dim if self.use_rnafm else 0,
            "per_concept_dim": dict(self._dims),
            "total_feature_dim": int(sum(self._dims.values())),
            "leakage_free": True,
            "leakage_note": "each concept (including rnafm) has an independent stream reading "
            "only its own block; the RNA-FM channel is a single disjoint concept, never mixed, so "
            "L_k = 0 (zero cross-channel leakage) holds and the 9 base concepts are unchanged.",
        }

    def write_manifest(self, path: str | Path) -> dict:
        """Write the feature-dimension manifest to ``path`` as JSON; return it."""
        manifest = self.feature_manifest()
        Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
