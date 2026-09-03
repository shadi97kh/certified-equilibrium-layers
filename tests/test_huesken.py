"""Tests for the Huesken efficacy loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cel.data import HueskenDataset, canonical_split, load_huesken


def test_canonical_split_sizes() -> None:
    split = canonical_split(2431)
    assert (split == "train").sum() == 2182
    assert (split == "test").sum() == 249


def test_canonical_split_clamps_small() -> None:
    split = canonical_split(5, train_size=2182)
    assert list(split) == ["train"] * 5


def test_load_basic(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_huesken(fixtures_dir / "huesken_tiny.tsv", cache_root=cache_root)
    assert isinstance(ds, HueskenDataset)
    assert len(ds) == 8
    assert ds.efficacy.dtype == float
    assert ds.efficacy.min() >= 0.0 and ds.efficacy.max() <= 1.0
    # Sequences are upper-cased and alphabet-validated.
    assert all(set(s) <= set("ACGUTN") for s in ds.sequences)


def test_split_from_file(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_huesken(fixtures_dir / "huesken_tiny.tsv", cache_root=cache_root)
    train_seqs, train_eff = ds.train()
    test_seqs, test_eff = ds.test()
    assert len(train_seqs) == 5
    assert len(test_seqs) == 3
    assert len(train_eff) == 5 and len(test_eff) == 3


def test_cache_is_idempotent(fixtures_dir: Path, cache_root: Path) -> None:
    raw = fixtures_dir / "huesken_tiny.tsv"
    first = load_huesken(raw, cache_root=cache_root)
    # Second call must hit the cache and return identical content.
    second = load_huesken(raw, cache_root=cache_root)
    assert first.sequences == second.sequences
    assert np.array_equal(first.efficacy, second.efficacy)
    assert np.array_equal(first.split, second.split)
    # A parquet artifact exists under the cache root.
    assert list(cache_root.glob("huesken/*/huesken.parquet"))


def test_missing_file_raises(cache_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_huesken("/no/such/huesken.tsv", cache_root=cache_root)
