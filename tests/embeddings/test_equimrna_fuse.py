"""EquiMRNA CDS fusion: CDS-only concat, off-target guard, and clean skip when absent.

No public EquiMRNA weights exist, so in this environment the artifact is absent and the
fusion must skip cleanly and log. The concat logic and the off-target node guard are
checked with an in-memory / tmp .npy fixture (the fixture exercises the code path; it is
not a real EquiMRNA artifact).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.embeddings import equimrna_fuse as ef  # noqa: E402


def test_artifact_absent_in_this_env() -> None:
    # No public EquiMRNA weights: the default artifact must be absent (never downloaded).
    assert ef.artifact_available() is False
    with pytest.raises(FileNotFoundError):
        ef.load_cds_embeddings()


def test_offtarget_nodes_are_never_fused() -> None:
    base = np.zeros((3, 5))
    ids = ["s0", "s1", "s2"]
    emb = np.ones((3, 4))
    for kind in ("guide", "utr_seed_site", "utr_offtarget"):
        with pytest.raises(ValueError):
            ef.fuse_cds_efficacy(base, ids, node_kind=kind, embeddings=emb)
    with pytest.raises(ValueError):
        ef.fuse_cds_efficacy(base, ids, node_kind="nonsense", embeddings=emb)


def test_clean_skip_when_artifact_absent(caplog) -> None:
    base = np.arange(6.0).reshape(3, 2)
    ids = ["s0", "s1", "s2"]
    with caplog.at_level(logging.INFO, logger="cel.embeddings.equimrna_fuse"):
        out, info = ef.fuse_cds_efficacy(base, ids, node_kind=ef.CDS_EFFICACY_NODE)
    assert info["fused"] is False and info["reason"] == "artifact_absent"
    assert np.array_equal(out, base)  # unchanged, nothing fabricated
    assert any("skipping CDS efficacy fusion" in r.message for r in caplog.records)


def test_concat_fusion_cds_only_with_embeddings() -> None:
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    ids = ["sA", "sB"]
    emb = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    out, info = ef.fuse_cds_efficacy(base, ids, node_kind=ef.CDS_EFFICACY_NODE, embeddings=emb)
    assert info["fused"] is True and info["fused_dim"] == 5 and info["equimrna_dim"] == 3
    assert np.array_equal(out, np.hstack([base, emb]))


def test_load_and_fuse_from_tmp_npy_with_index(tmp_path: Path) -> None:
    art = tmp_path / "equimrna_cds.npy"
    emb = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # rows for r0, r1, r2
    np.save(art, emb)
    (tmp_path / "equimrna_cds.index.json").write_text(json.dumps({"gA": 2, "gB": 0}))

    assert ef.artifact_available(art) is True
    base = np.array([[5.0], [6.0]])
    out, info = ef.fuse_cds_efficacy(
        base, ["gA", "gB"], node_kind=ef.CDS_EFFICACY_NODE, artifact_path=art
    )
    # gA -> row 2 ([2,2]), gB -> row 0 ([0,0]); concatenated onto the base column.
    assert np.array_equal(out, np.array([[5.0, 2.0, 2.0], [6.0, 0.0, 0.0]]))
    assert info["fused"] is True

    with pytest.raises(KeyError):
        ef.fuse_cds_efficacy(base, ["gA", "missing"], artifact_path=art)


def test_positional_alignment_requires_matching_rows() -> None:
    base = np.zeros((2, 1))
    with pytest.raises(ValueError):
        ef.fuse_cds_efficacy(base, ["a", "b"], embeddings=np.ones((3, 4)))  # 3 rows, no index
