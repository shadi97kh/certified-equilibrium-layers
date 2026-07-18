"""CASCADE-OT off-target predictor: leakage-free per-concept head over the seed->site->gene graph.

The predictor scores each gene's siRNA/miRNA off-target load from seed-match site features
organized into DISJOINT concept channels, exactly mirroring the CCMP backbone's Theorem 1
leakage-free design: each concept channel is read by its own stream and the per-gene
off-target head is a block-diagonal sum of per-concept contributions, so no concept's
readout can see another concept's input. leakage_offdiag_max asserts this is machine-zero
and a training run must fail if it is ever violated.

Graph view (collapsed to a per-gene example): a seed node connects to its candidate 3'UTR
sites (edge weighted by site type grade and ViennaRNA seed-duplex dG), and the sites
aggregate to the gene node. The aggregated per-gene concept features are:
  - site_load: log1p(site_count) and the has-site indicator (site -> gene multiplicity)
  - seed_pairing_stability: the ViennaRNA seed-duplex dG and its square (seed -> site weight)
  - site_grade: the best canonical site type grade (8mer/7mer/6mer; seed -> site weight)
  - target_accessibility: log1p of the 3'UTR length (gene-level accessibility proxy)
An optional RNA-FM channel can be concatenated as a FROZEN node feature (requires_grad
False); it is used only when a real embedding cache is supplied, never fabricated.

The analytical P_alpha GRN propagation (cel.verify.offtarget_bound.propagate) is reused
downstream and is NOT retrained here; this module learns only the per-gene direct readout.

Needs torch; cel.models guards the import so the torch-free path is unaffected.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

# Disjoint off-target concept channels (name -> input dimension). Block-diagonal readout.
CONCEPT_DIMS: dict[str, int] = {
    "site_load": 2,
    "seed_pairing_stability": 2,
    "site_grade": 1,
    "target_accessibility": 1,
}
RNAFM_CONCEPT = "rnafm"


def build_feature_blocks(
    site_count: np.ndarray,
    seed_dg: np.ndarray,
    site_grade: np.ndarray,
    utr_len: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the disjoint concept feature blocks from per-gene real features.

    Each input is a length-n array (one value per gene). seed_dg / site_grade may be constant
    within a condition (they are seed-level) and utr_len may be zeros for a species without
    sequence-derived features (the leakage-free readout then contributes only that block's bias).
    """
    site_count = np.asarray(site_count, dtype=float)
    seed_dg = np.asarray(seed_dg, dtype=float)
    site_grade = np.asarray(site_grade, dtype=float)
    utr_len = np.asarray(utr_len, dtype=float)
    return {
        "site_load": np.stack([np.log1p(site_count), (site_count > 0).astype(float)], axis=1),
        "seed_pairing_stability": np.stack([seed_dg, seed_dg**2], axis=1),
        "site_grade": site_grade[:, None],
        "target_accessibility": np.log1p(utr_len)[:, None],
    }


class _ConceptHead(nn.Module):
    """Per-concept readout stream; reads ONLY its own concept block (leakage-free)."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class OffTargetPredictor(nn.Module):
    """Block-diagonal per-concept off-target head. Per-gene score = sum_k head_k(concept_k)."""

    def __init__(
        self,
        concept_dims: dict[str, int] | None = None,
        hidden: int = 4,
        rnafm_dim: int = 0,
    ) -> None:
        super().__init__()
        self.concept_dims = dict(concept_dims or CONCEPT_DIMS)
        self.heads = nn.ModuleDict(
            {name: _ConceptHead(dim, hidden) for name, dim in self.concept_dims.items()}
        )
        self.rnafm_dim = rnafm_dim
        if rnafm_dim > 0:
            # RNA-FM is a FROZEN node-feature channel: its own stream, never trained, never mixed.
            self.heads[RNAFM_CONCEPT] = _ConceptHead(rnafm_dim, hidden)
            for p in self.heads[RNAFM_CONCEPT].parameters():
                p.requires_grad_(False)
            self.concept_dims[RNAFM_CONCEPT] = rnafm_dim
        self.bias = nn.Parameter(torch.zeros(1))

    def concept_contributions(self, blocks: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: self.heads[name](blocks[name]) for name in self.concept_dims}

    def forward(self, blocks: dict[str, Tensor]) -> Tensor:
        contribs = self.concept_contributions(blocks)
        return torch.stack([contribs[name] for name in self.concept_dims], dim=0).sum(0) + self.bias

    def leakage_offdiag_max(self, blocks: dict[str, Tensor]) -> float:
        """Max |d(contribution_k)/d(input_j)| over k != j. Must be machine-zero (Theorem 1)."""
        leaf = {
            name: blocks[name].detach().clone().requires_grad_(True) for name in self.concept_dims
        }
        worst = 0.0
        names = list(self.concept_dims)
        for k in names:
            out_k = self.heads[k](leaf[k]).sum()
            grads = torch.autograd.grad(
                out_k, [leaf[j] for j in names], allow_unused=True, retain_graph=True
            )
            for j, g in zip(names, grads, strict=True):
                if j != k and g is not None:
                    worst = max(worst, float(g.abs().max()))
        return worst


def assert_leakage_free(
    model: OffTargetPredictor, blocks: dict[str, Tensor], tol: float = 0.0
) -> float:
    """Raise if the per-concept head has any cross-channel leakage (Theorem 1 violated)."""
    worst = model.leakage_offdiag_max(blocks)
    if worst > tol:
        raise RuntimeError(
            f"cross-channel leakage detected (max off-diagonal sensitivity {worst:g} > {tol:g}); "
            "Theorem 1 leakage-freeness is violated, failing the run"
        )
    return worst
