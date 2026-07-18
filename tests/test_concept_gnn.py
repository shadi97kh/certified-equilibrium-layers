"""ConceptGNN: the optional RNA-FM channel must preserve the T1 leakage-free property.

The RNA-FM 640-d embedding is added as a single disjoint concept channel. The tests
confirm the feature-dimension manifest and that L_k = 0 (zero cross-channel leakage)
still holds bit-for-bit: intervening on the RNA-FM block changes only the RNA-FM
embedding, and intervening on a base concept leaves the RNA-FM embedding untouched.
The model needs torch; the test skips when torch is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("torch")
import torch  # noqa: E402

from cel.features import CONCEPT_SPECS  # noqa: E402
from cel.models.concept_gnn import RNAFM_CONCEPT, ConceptGNN  # noqa: E402

RNAFM_DIM = 640
BASE_TOTAL = sum(len(feats) for _, feats in CONCEPT_SPECS)


def _base_blocks(batch: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {name: torch.randn(batch, len(feats)) for name, feats in CONCEPT_SPECS}


def test_manifest_without_and_with_rnafm() -> None:
    m0 = ConceptGNN(use_rnafm=False).feature_manifest()
    assert m0["n_concepts"] == 9 and m0["use_rnafm"] is False
    assert m0["rnafm_concept"] is None and m0["rnafm_dim"] == 0
    assert m0["total_feature_dim"] == BASE_TOTAL
    assert RNAFM_CONCEPT not in m0["per_concept_dim"]

    m1 = ConceptGNN(use_rnafm=True).feature_manifest()
    assert m1["n_concepts"] == 10 and m1["use_rnafm"] is True
    assert m1["rnafm_concept"] == RNAFM_CONCEPT and m1["rnafm_dim"] == RNAFM_DIM
    assert m1["per_concept_dim"][RNAFM_CONCEPT] == RNAFM_DIM
    assert m1["total_feature_dim"] == BASE_TOTAL + RNAFM_DIM
    assert m1["leakage_free"] is True


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    model = ConceptGNN(use_rnafm=True)
    written = model.write_manifest(tmp_path / "manifest.json")
    loaded = json.loads((tmp_path / "manifest.json").read_text())
    assert loaded == written and loaded["per_concept_dim"][RNAFM_CONCEPT] == RNAFM_DIM


def test_rnafm_channel_is_leakage_free() -> None:
    model = ConceptGNN(use_rnafm=True, dropout=0.0).eval()
    blocks = _base_blocks()
    rnafm = torch.randn(4, RNAFM_DIM)
    with torch.no_grad():
        embs = model.get_per_concept_embeddings(blocks, rnafm)

        # (1) Intervene on the RNA-FM block: the 9 base concept embeddings must be
        # bit-for-bit identical (the RNA-FM channel never feeds another concept).
        rnafm_perturbed = rnafm.clone()
        rnafm_perturbed[0, 0] += 5.0
        embs_b = model.get_per_concept_embeddings(blocks, rnafm_perturbed)
        for name in model.base_concepts:
            assert torch.equal(embs[name], embs_b[name]), f"RNA-FM leaked into {name}"
        assert not torch.equal(embs[RNAFM_CONCEPT], embs_b[RNAFM_CONCEPT])

        # (2) Intervene on a base concept block: the RNA-FM embedding must be
        # bit-for-bit identical (no concept feeds the RNA-FM channel).
        victim = model.base_concepts[0]
        blocks_c = {k: v.clone() for k, v in blocks.items()}
        blocks_c[victim][0, 0] += 5.0
        embs_c = model.get_per_concept_embeddings(blocks_c, rnafm)
        assert torch.equal(embs[RNAFM_CONCEPT], embs_c[RNAFM_CONCEPT])
        for name in model.base_concepts:
            if name != victim:
                assert torch.equal(embs[name], embs_c[name]), f"{victim} leaked into {name}"


def test_baseline_unchanged_by_rnafm_flag() -> None:
    # With the same seed the 9 base streams are constructed identically whether or not
    # the RNA-FM channel is present, so use_rnafm does not perturb the T1 baseline.
    blocks = _base_blocks()
    torch.manual_seed(7)
    m_off = ConceptGNN(use_rnafm=False, dropout=0.0).eval()
    torch.manual_seed(7)
    m_on = ConceptGNN(use_rnafm=True, dropout=0.0).eval()
    with torch.no_grad():
        e_off = m_off.get_per_concept_embeddings(blocks)
        e_on = m_on.get_per_concept_embeddings(blocks, torch.randn(4, RNAFM_DIM))
    for name in m_off.base_concepts:
        assert torch.equal(e_off[name], e_on[name])


def test_assemble_block_guards() -> None:
    blocks = _base_blocks()
    with pytest.raises(ValueError):
        ConceptGNN(use_rnafm=True).assemble_blocks(blocks, rnafm_emb=None)
    with pytest.raises(ValueError):
        ConceptGNN(use_rnafm=False).assemble_blocks(blocks, rnafm_emb=torch.randn(4, RNAFM_DIM))
    with pytest.raises(ValueError):
        ConceptGNN(use_rnafm=True).assemble_blocks(blocks, rnafm_emb=torch.randn(4, 128))
