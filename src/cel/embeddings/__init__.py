"""RNA foundation-model embeddings for CASCADE (gated off the lightweight import path).

The RNA-FM extractor (:mod:`cel.embeddings.rnafm_extract`) needs torch and the
``fm`` (rna-fm) package; both are imported lazily inside the functions so that merely
importing this package never requires them.
"""

from .equimrna_fuse import (
    CDS_EFFICACY_NODE,
    artifact_available,
    fuse_cds_efficacy,
    load_cds_embeddings,
)
from .rnafm_extract import (
    RNAFM_DIM,
    RNAFM_MAX_SEQ_LEN,
    extract_embeddings,
    require_rnafm,
    write_embeddings,
)

__all__ = [
    "CDS_EFFICACY_NODE",
    "RNAFM_DIM",
    "RNAFM_MAX_SEQ_LEN",
    "artifact_available",
    "extract_embeddings",
    "fuse_cds_efficacy",
    "load_cds_embeddings",
    "require_rnafm",
    "write_embeddings",
]
