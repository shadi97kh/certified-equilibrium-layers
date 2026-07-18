"""Tests for the K=9 concept partition.

The headline test, :func:`test_concept_feature_indices_are_disjoint`, enforces
the non-overlap invariant that the L_k = 0 isolation property depends on.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cel.features import (
    CONCEPT_SPECS,
    CONCEPTS,
    K,
    concept_layout,
    featurize_pair,
    featurize_pairs,
    seed_site_words,
)

GUIDE = "UGAGGUAGUAGGUUGUAUAGU"
# Two contrasting targets: one with a perfect 8mer, one with no seed match.
TARGET_HIT = "CCCAUAUA" + seed_site_words(GUIDE)["8mer"] + "GGGCAUUAACCGGUU"
TARGET_MISS = "AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
PAIRS = [(GUIDE, TARGET_HIT), (GUIDE, TARGET_MISS)]

EXPECTED_NAMES = {
    "seed_complementarity",
    "seed_match_count",
    "utr_site_context",
    "seed_pairing_stability",
    "target_accessibility",
    "on_target_potency",
    "position_bias",
    "gc_content",
    "thermo_asymmetry",
}


def test_nine_named_concepts() -> None:
    assert K == 9
    assert set(CONCEPTS) == EXPECTED_NAMES
    assert len(CONCEPTS) == len(set(CONCEPTS)) == 9


def test_concept_feature_indices_are_disjoint() -> None:
    layout = concept_layout()
    idx = layout.concept_indices()
    # Every concept owns a non-empty index set.
    assert set(idx) == EXPECTED_NAMES
    assert all(len(v) > 0 for v in idx.values())

    # No feature index is shared across any two concepts.
    for a, b in itertools.combinations(CONCEPTS, 2):
        assert set(idx[a]).isdisjoint(idx[b]), f"{a} and {b} share feature indices"

    # The blocks tile [0, total_dim) exactly (a partition, no gaps, no overlap).
    union = sorted(i for v in idx.values() for i in v)
    assert union == list(range(layout.total_dim))

    # Each concept's width matches its declared feature-name list.
    for concept, feats in CONCEPT_SPECS:
        assert len(idx[concept]) == len(feats)

    # Feature names are globally unique.
    assert len(layout.feature_names) == len(set(layout.feature_names))


def test_featurize_shapes_and_blocks() -> None:
    feats = featurize_pairs(PAIRS)
    layout = feats.layout
    assert feats.matrix.shape == (2, layout.total_dim)
    assert np.isfinite(feats.matrix).all()

    # Each block matches its slice of the concatenated matrix.
    for concept, sl in layout.concept_slices.items():
        block = feats.block(concept)
        assert block.shape == (2, sl.stop - sl.start)
        assert np.array_equal(block, feats.matrix[:, sl])

    # Concatenating the blocks in concept order reconstructs the matrix.
    rebuilt = np.hstack([feats.block(c) for c in CONCEPTS])
    assert np.array_equal(rebuilt, feats.matrix)


def test_perfect_site_signals() -> None:
    one = featurize_pair(GUIDE, TARGET_HIT)
    sc = one["seed_complementarity"]
    assert sc[0] == 1.0  # best site grade (8mer)
    assert sc[3] == 1.0  # has canonical
    assert one["seed_match_count"][0] == 1.0  # one 8mer
    assert np.array_equal(one["position_bias"], np.ones(7))


def test_no_site_zeros() -> None:
    one = featurize_pair(GUIDE, TARGET_MISS)
    assert np.array_equal(one["seed_complementarity"], np.zeros(4))
    assert np.array_equal(one["seed_match_count"], np.zeros(4))


def test_empty_batch() -> None:
    feats = featurize_pairs([])
    assert feats.matrix.shape == (0, concept_layout().total_dim)
    for concept, sub in CONCEPT_SPECS:
        assert feats.block(concept).shape == (0, len(sub))


def test_to_torch_blocks() -> None:
    torch = pytest.importorskip("torch")
    feats = featurize_pairs(PAIRS)
    tblocks = feats.to_torch_blocks()
    assert set(tblocks) == EXPECTED_NAMES
    for concept, sl in feats.layout.concept_slices.items():
        t = tblocks[concept]
        assert t.shape == (2, sl.stop - sl.start)
        assert t.dtype == torch.float32
