"""Tests for the GSE184929 off-target loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cel.data import GEOOffTargetDataset, load_gse184929


def test_load_detects_contrasts(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_gse184929(fixtures_dir / "gse184929_tiny.tsv", cache_root=cache_root)
    assert isinstance(ds, GEOOffTargetDataset)
    assert len(ds) == 6
    assert ds.contrasts == ["D1_10mg", "D1_30mg", "D4_10mg", "D4_30mg"]
    assert ds.log2fc.index.name == "transcript_id"


def test_labels_vector(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_gse184929(fixtures_dir / "gse184929_tiny.tsv", cache_root=cache_root)
    labels = ds.labels("D1_30mg")
    assert labels.shape == (6,)
    # Strongly down-regulated transcript appears in the table.
    assert labels.min() < -2.0
    with pytest.raises(KeyError):
        ds.labels("does_not_exist")


def test_explicit_contrasts(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_gse184929(
        fixtures_dir / "gse184929_tiny.tsv",
        cache_root=cache_root,
        contrasts=("D1_10mg", "D4_30mg"),
    )
    assert ds.contrasts == ["D1_10mg", "D4_30mg"]


def test_cache_is_idempotent(fixtures_dir: Path, cache_root: Path) -> None:
    raw = fixtures_dir / "gse184929_tiny.tsv"
    first = load_gse184929(raw, cache_root=cache_root)
    second = load_gse184929(raw, cache_root=cache_root)
    assert np.array_equal(first.transcript_ids, second.transcript_ids)
    assert first.contrasts == second.contrasts
    assert list(cache_root.glob("gse184929/*/log2fc.parquet"))


def test_missing_file_raises(cache_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gse184929("/no/such/geo.tsv", cache_root=cache_root)
