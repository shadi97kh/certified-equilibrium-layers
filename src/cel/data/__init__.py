"""Data loaders and caching for CASCADE.

Each loader parses an open-access source into a small dataclass and caches the
parsed form under ``data/cache/`` (idempotent). See docs/DATA_CARD.md for
provenance and the rat-liver specificity caveat on GSE184929.
"""

from .cache import CacheKey, cache_dir_for, cached_frame, sha256_file
from .efficacy import (
    EFFICACY_SOURCES,
    EFFICACY_STUDIES,
    MIX_STUDIES,
    EfficacyDataset,
    default_efficacy_dir,
    load_efficacy_suite,
    load_harborth,
    load_huesken_efficacy,
    load_katoh,
    load_khvorova,
    load_mix,
    load_oligoformer_efficacy,
    load_reynolds,
    load_uitei,
    load_vickers,
    reverse_complement,
)
from .geo import (
    DEFAULT_CONTRASTS,
    GEOOffTargetDataset,
    load_gse184929,
)
from .grn import (
    TRRUST_URLS,
    SignedDirectedGRN,
    load_collectri,
    load_dorothea,
    load_string_undirected,
    load_trrust,
    signed_spectral_norm,
    signed_spectral_radius,
)
from .huesken import (
    HueskenDataset,
    canonical_split,
    load_huesken,
)
from .offtarget import (
    OFFTARGET_SOURCES,
    SEEDMATCHR_DATASETS,
    SEEDMATCHR_SOURCE,
    KSResult,
    OffTargetDataset,
    SeedMatchRDataset,
    load_birmingham2006,
    load_jackson2003,
    load_jackson2006,
    load_schlegel2022,
    load_seedseq,
    seedmatch_ks_test,
)
from .paths import cache_dir, data_dir, raw_dir, repo_root
from .reactome import load_reactome_graph
from .roles import (
    DATASET_ROLES,
    EFFICACY_BACKBONE,
    OFFTARGET_CERTIFICATION,
    dataset_role,
    datasets_with_role,
)
from .string_grn import (
    DEFAULT_THRESHOLD,
    StringGRN,
    load_string_confidence,
    load_string_grn,
)

__all__ = [
    # roles
    "EFFICACY_BACKBONE",
    "OFFTARGET_CERTIFICATION",
    "DATASET_ROLES",
    "dataset_role",
    "datasets_with_role",
    # cache primitives
    "CacheKey",
    "cache_dir_for",
    "cached_frame",
    "sha256_file",
    # paths
    "repo_root",
    "data_dir",
    "raw_dir",
    "cache_dir",
    # on-target efficacy backbone (OligoFormer suite)
    "EfficacyDataset",
    "load_oligoformer_efficacy",
    "EFFICACY_SOURCES",
    "MIX_STUDIES",
    # on-target efficacy backbone (REAL per-study loaders)
    "EFFICACY_STUDIES",
    "default_efficacy_dir",
    "load_efficacy_suite",
    "load_huesken_efficacy",
    "load_reynolds",
    "load_vickers",
    "load_harborth",
    "load_uitei",
    "load_khvorova",
    "load_katoh",
    "load_mix",
    "reverse_complement",
    # Huesken efficacy (one efficacy_backbone source)
    "HueskenDataset",
    "load_huesken",
    "canonical_split",
    # off-target certification loaders
    "OffTargetDataset",
    "load_jackson2003",
    "load_jackson2006",
    "load_birmingham2006",
    "load_seedseq",
    "OFFTARGET_SOURCES",
    # GSE184929 / Schlegel 2022 RNA-seq via SeedMatchR + the KS off-target metric
    "SeedMatchRDataset",
    "load_schlegel2022",
    "SEEDMATCHR_DATASETS",
    "SEEDMATCHR_SOURCE",
    "seedmatch_ks_test",
    "KSResult",
    # GSE184929 off-target
    "GEOOffTargetDataset",
    "load_gse184929",
    "DEFAULT_CONTRASTS",
    # STRING GRN
    "StringGRN",
    "load_string_grn",
    "load_string_confidence",
    "DEFAULT_THRESHOLD",
    # Signed/directed GRN (T2) + undirected STRING variant
    "SignedDirectedGRN",
    "load_trrust",
    "load_collectri",
    "load_dorothea",
    "load_string_undirected",
    "TRRUST_URLS",
    "signed_spectral_radius",
    "signed_spectral_norm",
    # Reactome GRN (tracked gap)
    "load_reactome_graph",
]
