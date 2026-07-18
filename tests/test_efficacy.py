"""Tests for the efficacy loaders (role: efficacy_backbone). REAL DATA ONLY."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cel.data import (
    EFFICACY_BACKBONE,
    EFFICACY_STUDIES,
    EfficacyDataset,
    load_efficacy_suite,
    load_huesken_efficacy,
    load_katoh,
    load_oligoformer_efficacy,
    load_reynolds,
    load_vickers,
)
from cel.data.efficacy import COLUMNS


def _raw(fixtures_dir: Path) -> Path:
    return fixtures_dir / "oligoformer"


def _eff(fixtures_dir: Path) -> Path:
    return fixtures_dir / "efficacy"


def test_load_unified_suite(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    assert isinstance(ds, EfficacyDataset)
    assert ds.role == EFFICACY_BACKBONE
    assert list(ds.frame.columns) == COLUMNS
    assert len(ds) == 8  # 3 huesken + 2 katoh + 2 mix + 1 shabalina
    # Pooled provenance: Huesken, Katoh, the two mix studies, and Shabalina.
    assert {
        "Huesken 2005",
        "Katoh (Taka)",
        "Reynolds 2004",
        "Ui-Tei 2004",
        "Shabalina 2006",
    } <= set(ds.sources())
    assert "H1299" in ds.cell_lines()


def test_efficacy_is_normalized_to_unit_interval(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    eff = ds.frame["efficacy"].to_numpy()
    assert eff.min() >= 0.0 and eff.max() <= 1.0
    # Katoh fixture is percent-scaled (85.0 -> 0.85).
    katoh = ds.frame[ds.frame["source_study"] == "Katoh (Taka)"]["efficacy"].to_numpy()
    assert np.isclose(katoh.max(), 0.85)


def test_subset_sources(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root, sources=["huesken"])
    assert ds.sources() == ["Huesken 2005"]
    assert len(ds) == 3


def test_filter_study(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    only = ds.filter_study("Reynolds 2004")
    assert len(only) == 1
    assert only.sources() == ["Reynolds 2004"]


def test_cache_is_idempotent(fixtures_dir: Path, cache_root: Path) -> None:
    first = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    second = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    assert first.frame.equals(second.frame)
    assert list(cache_root.glob("efficacy/*/frame.parquet"))


def test_missing_dir_raises(cache_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_oligoformer_efficacy("/no/such/dir", cache_root=cache_root)


# --- REAL per-study loaders (cel.data.efficacy v2) ------------------------
def test_registry_covers_the_studies() -> None:
    assert set(EFFICACY_STUDIES) == {
        "huesken",
        "reynolds",
        "vickers",
        "harborth",
        "uitei",
        "khvorova",
        "katoh",
        "mix",
    }
    # Every study carries a citation and an exact fetch URL (REAL DATA ONLY).
    for meta in EFFICACY_STUDIES.values():
        assert meta["citation"] and isinstance(meta["citation"], str)
        assert meta["url"].startswith("http")


def test_huesken_loader_exposes_standard_split(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_huesken_efficacy(_eff(fixtures_dir), cache_root=cache_root)
    assert isinstance(ds, EfficacyDataset)
    assert ds.role == EFFICACY_BACKBONE
    assert ds.has_split
    assert set(COLUMNS) <= set(ds.frame.columns)
    assert len(ds.train()) == 3 and len(ds.test()) == 2
    eff = ds.frame["efficacy"].to_numpy()
    assert eff.min() >= 0.0 and eff.max() <= 1.0


def test_non_huesken_dataset_has_no_split(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_reynolds(_eff(fixtures_dir), cache_root=cache_root)
    assert not ds.has_split
    with pytest.raises(ValueError, match="no split"):
        ds.train()


def test_per_study_selection_from_master(fixtures_dir: Path, cache_root: Path) -> None:
    # No reynolds.csv present -> selected out of the master compilation by study.
    reynolds = load_reynolds(_eff(fixtures_dir), cache_root=cache_root)
    assert len(reynolds) == 1
    assert reynolds.sources() == ["Reynolds 2004"]
    vickers = load_vickers(_eff(fixtures_dir), cache_root=cache_root)
    assert vickers.sources() == ["Vickers 2003"]


def test_master_percent_efficacy_normalized(fixtures_dir: Path, cache_root: Path) -> None:
    katoh = load_katoh(_eff(fixtures_dir), cache_root=cache_root)
    eff = katoh.frame["efficacy"].to_numpy()
    assert eff.min() >= 0.0 and eff.max() <= 1.0
    assert np.isclose(eff.max(), 0.90)  # 90.0 percent -> 0.90


def test_suite_concatenates_requested_studies(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_efficacy_suite(
        studies=["reynolds", "vickers", "katoh"],
        raw_dir=_eff(fixtures_dir),
        cache_root=cache_root,
    )
    assert list(ds.frame.columns) == COLUMNS
    assert len(ds) == 3
    assert {"Reynolds 2004", "Vickers 2003", "Katoh (Taka)"} == set(ds.sources())


def test_cache_parquet_is_written(fixtures_dir: Path, cache_root: Path) -> None:
    load_reynolds(_eff(fixtures_dir), cache_root=cache_root)
    assert list(cache_root.glob("efficacy_reynolds/*/frame.parquet"))


def test_missing_per_study_file_raises_with_exact_url(cache_root: Path, tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_vickers(tmp_path / "empty", cache_root=cache_root)
    msg = str(exc.value)
    assert EFFICACY_STUDIES["vickers"]["url"] in msg
    assert "REAL DATA ONLY" in msg


def test_missing_suite_lists_missing_urls(cache_root: Path, tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_efficacy_suite(raw_dir=tmp_path / "empty", cache_root=cache_root)
    msg = str(exc.value)
    assert EFFICACY_STUDIES["huesken"]["url"] in msg


# --- OligoFormer schema: 19mer = sense, guide = reverse_complement -----------
def test_reverse_complement_is_involution_and_rna() -> None:
    from cel.data import reverse_complement

    # RNA complement over {A,C,G,U}: A<->U, G<->C, then reverse.
    assert reverse_complement("AUGC") == "GCAU"
    for seq in ("CUAAUAUGUUAAUUGAUUU", "GAUUUAUACAAUUCCUUUC", "AGACCAAAAUUAAAUAAGA"):
        rc = reverse_complement(seq)
        assert len(rc) == len(seq)
        assert set(rc) <= set("ACGU")
        assert reverse_complement(rc) == seq  # involution: rc(rc(x)) == x
        assert rc != seq  # the guide differs from the sense on these real rows


def test_oligoformer_transform_sense_guide_efficacy(
    fixtures_dir: Path, cache_root: Path, tmp_path
) -> None:
    # Place the real OligoFormer-schema slice as huesken.csv and load it.
    import shutil

    from cel.data import load_huesken_efficacy, reverse_complement

    raw = tmp_path / "efficacy"
    raw.mkdir()
    shutil.copy2(fixtures_dir / "efficacy" / "oligoformer_hu_slice.csv", raw / "huesken.csv")
    ds = load_huesken_efficacy(raw, cache_root=cache_root)
    f = ds.frame
    assert len(f) == 6
    # sense is the given 19mer; antisense is its reverse complement (the guide).
    assert f["sense"].iloc[0] == "CUAAUAUGUUAAUUGAUUU"
    assert f["antisense"].iloc[0] == reverse_complement("CUAAUAUGUUAAUUGAUUU")
    assert bool((f["antisense"] != f["sense"]).all())  # guide != sense
    # efficacy = 1 - expression/100 (row 1: 65.548 -> 0.3445).
    assert np.isclose(f["efficacy"].iloc[0], 1.0 - 65.54809843 / 100.0)
    assert f["efficacy"].min() >= 0.0 and f["efficacy"].max() <= 1.0
