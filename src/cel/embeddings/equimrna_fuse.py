"""EquiMRNA CDS-embedding concat fusion into the CASCADE efficacy readout (CDS targets only).

EquiMRNA embeds a coding sequence (CDS). For CDS-targeting siRNAs its CDS embedding is
concatenated into the on-target EFFICACY readout features (concat fusion). It is NEVER
fused into the off-target nodes (the guide nodes or the 3'UTR seed-site nodes): those carry
3'UTR / seed-thermodynamic signal, not CDS identity, so mixing a CDS embedding into the
off-target propagation would be biologically wrong and a cross-channel contamination. The
node-kind guard enforces this: a fusion call on an off-target node kind raises.

No public EquiMRNA weights exist, so this module NEVER downloads anything. It uses a LOCAL
.npy artifact that the user provides (default data/raw/embeddings/equimrna_cds.npy, with an
optional companion <name>.index.json mapping siRNA id to row). If the artifact is absent the
fusion is skipped cleanly (the base efficacy features are returned unchanged) and the skip
is logged.

REAL DATA ONLY: the EquiMRNA embedding must be a real local artifact; absent it, no
embedding is fabricated and nothing is fetched.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = REPO / "data" / "raw" / "embeddings" / "equimrna_cds.npy"
CDS_EFFICACY_NODE = "cds_efficacy"
OFFTARGET_NODE_KINDS = frozenset({"guide", "utr_seed_site", "utr_offtarget"})


def artifact_available(path: str | Path = DEFAULT_ARTIFACT) -> bool:
    """True iff the local EquiMRNA .npy artifact exists (no download is ever attempted)."""
    return Path(path).exists()


def load_cds_embeddings(path: str | Path = DEFAULT_ARTIFACT) -> tuple[np.ndarray, dict | None]:
    """Load the EquiMRNA CDS embedding matrix and its optional id->row index.

    Raises FileNotFoundError when the artifact is absent (no public weights exist; a local
    .npy must be supplied, never downloaded).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"EquiMRNA CDS embedding artifact not found at {p}. No public EquiMRNA weights "
            "exist; supply a local .npy (do NOT download). REAL DATA ONLY."
        )
    emb = np.load(p)
    index_path = p.with_suffix(".index.json")
    id_to_row = json.loads(index_path.read_text()) if index_path.exists() else None
    return emb, id_to_row


def _check_node_kind(node_kind: str) -> None:
    if node_kind in OFFTARGET_NODE_KINDS:
        raise ValueError(
            f"EquiMRNA CDS embedding must NOT be fused into off-target node '{node_kind}'; "
            "it is for the CDS-targeting on-target efficacy readout only."
        )
    if node_kind != CDS_EFFICACY_NODE:
        raise ValueError(f"unknown node kind {node_kind!r}; expected {CDS_EFFICACY_NODE!r}")


def fuse_cds_efficacy(
    base_features: np.ndarray,
    sirna_ids: Sequence,
    node_kind: str = CDS_EFFICACY_NODE,
    *,
    embeddings: np.ndarray | None = None,
    id_to_row: dict | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> tuple[np.ndarray, dict]:
    """Concat the EquiMRNA CDS embedding onto the efficacy features for CDS-targeting siRNAs.

    Returns (features, info). When the artifact is absent and no embeddings are passed, the
    base features are returned unchanged with info['fused'] = False (a clean, logged skip; no
    fabrication). Raises ValueError if node_kind is an off-target node (never fuse there).
    """
    _check_node_kind(node_kind)
    base = np.asarray(base_features, dtype=float)
    n = len(sirna_ids)
    if base.ndim != 2 or base.shape[0] != n:
        raise ValueError(f"base_features {base.shape} must be (n_sirnas={n}, d)")

    if embeddings is None:
        if not artifact_available(artifact_path):
            LOGGER.info(
                "EquiMRNA artifact absent at %s; skipping CDS efficacy fusion "
                "(no public weights, nothing fetched)",
                artifact_path,
            )
            return base, {
                "fused": False,
                "reason": "artifact_absent",
                "artifact_path": str(artifact_path),
                "n_cds": int(n),
                "equimrna_dim": 0,
            }
        embeddings, id_to_row = load_cds_embeddings(artifact_path)

    embeddings = np.asarray(embeddings, dtype=float)
    if id_to_row is not None:
        missing = [s for s in sirna_ids if str(s) not in id_to_row]
        if missing:
            raise KeyError(
                f"{len(missing)} siRNA ids absent from the EquiMRNA index, e.g. {missing[:3]}"
            )
        emb = embeddings[[id_to_row[str(s)] for s in sirna_ids]]
    else:
        if embeddings.shape[0] != n:
            raise ValueError(
                f"EquiMRNA embeddings have {embeddings.shape[0]} rows for {n} siRNAs and no "
                "id->row index was provided to align them"
            )
        emb = embeddings
    fused = np.hstack([base, emb])
    return fused, {
        "fused": True,
        "reason": "concat",
        "n_cds": int(n),
        "base_dim": int(base.shape[1]),
        "equimrna_dim": int(emb.shape[1]),
        "fused_dim": int(fused.shape[1]),
    }


def main() -> int:
    """Log EquiMRNA artifact availability and the resulting fusion mode (clean skip if absent)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if artifact_available():
        emb, idx = load_cds_embeddings()
        LOGGER.info(
            "[equimrna] artifact present at %s: %s embedding, index=%s; CDS efficacy fusion ENABLED",
            DEFAULT_ARTIFACT,
            tuple(emb.shape),
            "yes" if idx is not None else "positional",
        )
    else:
        LOGGER.info(
            "[equimrna] no artifact at %s; CDS efficacy fusion SKIPPED (no public weights, "
            "nothing fetched). Provide a local .npy to enable it.",
            DEFAULT_ARTIFACT,
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
