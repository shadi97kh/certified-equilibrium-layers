"""Seed-match, thermodynamics, and per-concept feature extraction.

The public surface is the K=9 concept partition (:func:`featurize_pairs`,
:data:`CONCEPTS`, :func:`concept_layout`) plus the seed-match and thermodynamic
primitives the concepts are built from. The concept blocks are non-overlapping
by construction; this is what later guarantees the L_k = 0 isolation property.
"""

from .concepts import (
    CONCEPT_SPECS,
    CONCEPTS,
    ConceptFeatures,
    ConceptLayout,
    K,
    concept_layout,
    featurize_pair,
    featurize_pairs,
)
from .seedmatch import (
    SITE_GRADE,
    SITE_TYPES,
    Site,
    extract_seed,
    find_sites,
    revcomp,
    seed_match_score,
    seed_site_words,
)
from .thermo import (
    HAS_VIENNA,
    seed_pairing_stability,
    target_accessibility,
    thermo_asymmetry,
)

__all__ = [
    # concepts
    "CONCEPTS",
    "CONCEPT_SPECS",
    "K",
    "ConceptFeatures",
    "ConceptLayout",
    "concept_layout",
    "featurize_pair",
    "featurize_pairs",
    # seed match
    "Site",
    "SITE_TYPES",
    "SITE_GRADE",
    "extract_seed",
    "seed_site_words",
    "find_sites",
    "seed_match_score",
    "revcomp",
    # thermodynamics
    "HAS_VIENNA",
    "seed_pairing_stability",
    "target_accessibility",
    "thermo_asymmetry",
]
