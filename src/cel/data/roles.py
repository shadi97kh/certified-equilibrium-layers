"""Dataset role tags for the CASCADE / AAAI data layer.

Every real measured dataset carries one explicit role so downstream code (and
the paper) can never confuse what a dataset is for:

- ``efficacy_backbone``: on-target siRNA potency datasets (the OligoFormer
  nine-dataset suite). These measure on-target knockdown and DO NOT validate the
  off-target containment certificate.
- ``offtarget_certification``: measured off-target signatures (GSE184929 RNA-seq,
  the Jackson / Birmingham microarray sets, the SeedSeq union). These are the
  datasets the certificate is evaluated against.

Per the hard data policy (OUTLINE.md), the data layer holds only real measured
data. Random-graph certificate-soundness verification is not a dataset and lives
outside the data layer, under ``experiments/soundness/``.

The role lives both as a string constant and as a ``role`` attribute on each
loader's returned dataclass; :data:`DATASET_ROLES` is the single-source-of-truth
registry tabulated in docs/DATA_CARD.md.
"""

from __future__ import annotations

EFFICACY_BACKBONE = "efficacy_backbone"
OFFTARGET_CERTIFICATION = "offtarget_certification"

ROLES = frozenset({EFFICACY_BACKBONE, OFFTARGET_CERTIFICATION})

# Single source of truth: dataset name -> role. Tabulated in docs/DATA_CARD.md.
DATASET_ROLES: dict[str, str] = {
    # on-target efficacy backbone (OligoFormer nine-dataset suite)
    "huesken": EFFICACY_BACKBONE,
    "katoh": EFFICACY_BACKBONE,
    "mix": EFFICACY_BACKBONE,
    "shabalina": EFFICACY_BACKBONE,
    # off-target certification
    "gse184929": OFFTARGET_CERTIFICATION,
    "jackson2003": OFFTARGET_CERTIFICATION,
    "jackson2006": OFFTARGET_CERTIFICATION,
    "birmingham2006": OFFTARGET_CERTIFICATION,
    "seedseq": OFFTARGET_CERTIFICATION,
}


def dataset_role(name: str) -> str:
    """Return the role for a dataset name (KeyError if unknown)."""
    return DATASET_ROLES[name]


def datasets_with_role(role: str) -> list[str]:
    """List dataset names carrying ``role``."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLES)}")
    return [name for name, r in DATASET_ROLES.items() if r == role]
