"""Tests for the signed/directed GRN loaders and the undirected STRING variant.

REAL DATA ONLY: loaders raise with the exact URL when the real file is absent.
The signed/directed tests run on a tiny REAL TRRUST slice (real curated edges,
including the TP53 <-> MDM2 feedback loop); see tests/fixtures/grn/NOTE.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cel.data import (
    SignedDirectedGRN,
    StringGRN,
    load_dorothea,
    load_string_undirected,
    load_trrust,
)
from cel.data.grn import TRRUST_URLS


def _trrust(fixtures_dir: Path) -> Path:
    return fixtures_dir / "grn" / "trrust_rawdata.human.tsv"


def test_load_trrust_signed_directed(fixtures_dir: Path, cache_root: Path) -> None:
    g = load_trrust(_trrust(fixtures_dir), cache_root=cache_root)
    assert isinstance(g, SignedDirectedGRN)
    assert g.n_nodes == 8 and {"TP53", "MDM2", "CDKN1A", "MYC"} <= set(g.nodes)
    assert g.n_edges == 7  # 8 rows minus the dropped Unknown edge
    assert g.n_unknown_dropped == 1
    # |S| is row sub-stochastic, so the absolute row sums are <= 1.
    abs_rowsum = np.asarray(abs(g.S).sum(axis=1)).ravel()
    assert abs_rowsum.max() <= 1.0 + 1e-9
    # Signs are exactly +1 / -1.
    assert set(np.unique(g.signs.data).tolist()) <= {-1.0, 1.0}


def test_sign_pattern_activation_and_repression(fixtures_dir: Path, cache_root: Path) -> None:
    g = load_trrust(_trrust(fixtures_dir), cache_root=cache_root)
    idx = g.index()
    # TP53 activates MDM2 (+1); MDM2 represses TP53 (-1): the real feedback loop.
    assert g.signs[idx["MDM2"], idx["TP53"]] == 1.0
    assert g.signs[idx["TP53"], idx["MDM2"]] == -1.0
    assert g.signs[idx["MYC"], idx["TP53"]] == -1.0  # TP53 -| MYC


def test_spectral_radius_le_one_and_norm_ge_radius(fixtures_dir: Path, cache_root: Path) -> None:
    g = load_trrust(_trrust(fixtures_dir), cache_root=cache_root)
    rho = g.spectral_radius()
    norm = g.spectral_norm()
    assert rho <= 1.0 + 1e-9  # contraction-ready: rho(S) <= 1
    # For a signed, directed (non-symmetric) S the spectral norm can exceed rho.
    assert norm >= rho - 1e-9


def test_propagate_reuses_p_alpha(fixtures_dir: Path, cache_root: Path) -> None:
    g = load_trrust(_trrust(fixtures_dir), cache_root=cache_root)
    idx = g.index()
    x0 = np.zeros(g.n_nodes)
    x0[idx["TP53"]] = 1.0
    y = g.propagate(x0, alpha=0.5, num_steps=64)
    assert y.shape == (g.n_nodes,)
    assert np.isfinite(y).all()
    assert y[idx["TP53"]] != 0.0  # the perturbed source carries signal


def test_include_unknown_keeps_the_edge(fixtures_dir: Path, cache_root: Path) -> None:
    g = load_trrust(_trrust(fixtures_dir), cache_root=cache_root, include_unknown=True)
    assert g.n_unknown_dropped == 0
    assert "EGFR" in g.nodes  # the Unknown-mode TP53 -> EGFR edge is retained


def test_trrust_missing_raises_with_url(cache_root: Path, tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_trrust(tmp_path / "absent.tsv", cache_root=cache_root)
    msg = str(exc.value)
    assert TRRUST_URLS["human"] in msg and "REAL DATA ONLY" in msg


def test_dorothea_missing_raises_with_source(cache_root: Path, tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_dorothea(tmp_path / "absent.csv", cache_root=cache_root)
    assert "DoRothEA" in str(exc.value) and "REAL DATA ONLY" in str(exc.value)


def test_string_undirected_variant_is_symmetric(fixtures_dir: Path, cache_root: Path) -> None:
    grn = load_string_undirected(
        fixtures_dir / "string_tiny.txt", cache_root=cache_root, threshold=700
    )
    assert isinstance(grn, StringGRN)
    diff = grn.S - grn.S.T
    assert abs(diff).max() < 1e-9  # STRING variant is undirected (symmetric S)


def test_cache_is_idempotent(fixtures_dir: Path, cache_root: Path) -> None:
    first = load_trrust(_trrust(fixtures_dir), cache_root=cache_root)
    second = load_trrust(_trrust(fixtures_dir), cache_root=cache_root)
    assert first.nodes == second.nodes
    assert (first.S != second.S).nnz == 0
    assert list(cache_root.glob("trrust_human/*/S.npz"))
