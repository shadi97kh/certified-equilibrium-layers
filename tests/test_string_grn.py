"""Tests for the STRING v12 GRN loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cel.data import StringGRN, load_string_grn

P1 = "10116.ENSRNOP001"
P2 = "10116.ENSRNOP002"
P3 = "10116.ENSRNOP003"
P4 = "10116.ENSRNOP004"
P5 = "10116.ENSRNOP005"


def test_threshold_filters_nodes_and_edges(fixtures_dir: Path, cache_root: Path) -> None:
    grn = load_string_grn(fixtures_dir / "string_tiny.txt", cache_root=cache_root, threshold=700)
    assert isinstance(grn, StringGRN)
    # P5 only appears in a sub-700 edge, so it is dropped; P1..P4 survive.
    assert grn.nodes == [P1, P2, P3, P4]
    assert grn.num_nodes() == 4
    # Surviving undirected edges: P1-P2, P2-P3, P3-P4, P1-P4.
    assert grn.num_edges() == 4


def test_lower_threshold_adds_nodes(fixtures_dir: Path, cache_root: Path) -> None:
    grn = load_string_grn(fixtures_dir / "string_tiny.txt", cache_root=cache_root, threshold=300)
    assert P5 in grn.nodes
    assert grn.num_nodes() == 5


def test_symmetric_normalized(fixtures_dir: Path, cache_root: Path) -> None:
    grn = load_string_grn(fixtures_dir / "string_tiny.txt", cache_root=cache_root, threshold=700)
    s = grn.S
    assert sp.issparse(s)
    # Symmetry: S == S^T.
    diff = s - s.T
    assert abs(diff).max() < 1e-12
    # Non-negative weights, zero diagonal (no self loops in the fixture).
    assert s.min() >= 0.0
    assert np.allclose(s.diagonal(), 0.0)
    # Symmetric-normalized form: D^(-1/2) A D^(-1/2) has spectral entries <= 1.
    assert abs(s).max() <= 1.0 + 1e-9


def test_cache_is_idempotent(fixtures_dir: Path, cache_root: Path) -> None:
    raw = fixtures_dir / "string_tiny.txt"
    first = load_string_grn(raw, cache_root=cache_root, threshold=700)
    second = load_string_grn(raw, cache_root=cache_root, threshold=700)
    assert first.nodes == second.nodes
    assert (first.S != second.S).nnz == 0
    assert list(cache_root.glob("string_grn/*/S.npz"))


def test_to_edge_index(fixtures_dir: Path, cache_root: Path) -> None:
    torch = pytest.importorskip("torch")
    grn = load_string_grn(fixtures_dir / "string_tiny.txt", cache_root=cache_root, threshold=700)
    edge_index, edge_weight = grn.to_edge_index()
    assert edge_index.dtype == torch.long
    assert edge_index.shape[0] == 2
    # One directed entry per nonzero of the symmetric S (2 per undirected edge).
    assert edge_index.shape[1] == grn.S.nnz
    assert edge_weight.shape[0] == grn.S.nnz
    assert int(edge_index.max()) < grn.num_nodes()


def test_missing_file_raises(cache_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_string_grn("/no/such/links.txt.gz", cache_root=cache_root)
