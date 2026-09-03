"""Tests for the off-target loaders (role: offtarget_certification)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cel.data import (
    OFFTARGET_CERTIFICATION,
    OffTargetDataset,
    load_birmingham2006,
    load_jackson2003,
    load_jackson2006,
    load_seedseq,
)
from cel.data import offtarget as ot
from cel.data.offtarget import COLUMNS


def _ot(fixtures_dir: Path, name: str) -> Path:
    return fixtures_dir / "offtarget" / f"{name}.csv"


def test_jackson2003_seed_derived_from_antisense(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_jackson2003(_ot(fixtures_dir, "jackson2003"), cache_root=cache_root)
    assert isinstance(ds, OffTargetDataset)
    assert ds.role == OFFTARGET_CERTIFICATION
    assert ds.assay == "microarray"
    assert list(ds.frame.columns) == COLUMNS
    assert len(ds) == 3
    # No seed column => derived from antisense positions 2-8.
    assert ds.frame["seed"].iloc[0] == "GAGGUAG"
    assert "supplement" in ds.provenance
    assert ds.sources() == ["jackson2003"]  # defaulted to the dataset name


def test_jackson2006_uses_explicit_seed(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_jackson2006(_ot(fixtures_dir, "jackson2006"), cache_root=cache_root)
    assert set(ds.frame["seed"]) == {"GAGGUAG"}
    assert ds.transcripts() == ["NM_000010", "NM_000011"]


def test_birmingham2006_source_column(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_birmingham2006(_ot(fixtures_dir, "birmingham2006"), cache_root=cache_root)
    assert ds.sources() == ["birmingham2006"]
    assert len(ds) == 2


def test_seedseq_aggregated_has_source_flags(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_seedseq(_ot(fixtures_dir, "seedseq"), cache_root=cache_root)
    assert ds.sources() == ["birmingham2006", "jackson2003", "jackson2006"]
    assert len(ds) == 3


def test_seedseq_union_fallback(
    fixtures_dir: Path, cache_root: Path, tmp_path, monkeypatch
) -> None:
    # No aggregated seedseq table => build the union from available members,
    # tagging each call with its source.
    def fake_default(name: str) -> Path:
        if name == "seedseq":
            return tmp_path / "absent_seedseq.csv"
        return _ot(fixtures_dir, name)

    monkeypatch.setattr(ot, "_default_path", fake_default)
    ds = load_seedseq(cache_root=cache_root)
    assert set(ds.sources()) == {"jackson2003", "jackson2006", "birmingham2006"}
    assert "built from available members" in ds.provenance


def test_missing_file_raises(cache_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jackson2003("/no/such/jackson.csv", cache_root=cache_root)


# --- SeedMatchR KS metric (off-target metric for Theorem 4) -------------------
def test_ks_d_statistic_matches_scipy() -> None:
    import numpy as np
    from scipy import stats

    from cel.data import seedmatch_ks_test

    seed = np.array([-3.1, -0.95, -0.80, -0.62, -1.2, -0.7])
    bg = np.array([0.05, -0.10, 0.12, -0.05, 0.30, -0.20, 0.0])
    log2fc = np.concatenate([seed, bg])
    mask = np.array([True] * len(seed) + [False] * len(bg))
    for alt in ("greater", "less", "two-sided"):
        r = seedmatch_ks_test(log2fc, mask, alternative=alt)
        s = stats.ks_2samp(seed, bg, alternative=alt, method="asymp")
        assert np.isclose(r.statistic, s.statistic)  # D is unambiguous
        assert r.n_seed_match == len(seed) and r.n_background == len(bg)


def test_ks_one_sided_pvalue_matches_r_formula() -> None:
    import numpy as np

    from cel.data import seedmatch_ks_test

    seed = np.array([-3.1, -0.95, -0.80, -0.62])
    bg = np.array([0.05, -0.10, 0.12, -0.05])
    log2fc = np.concatenate([seed, bg])
    mask = np.array([True] * 4 + [False] * 4)
    r = seedmatch_ks_test(log2fc, mask, alternative="greater")
    n, m = 4, 4
    ne = n * m / (n + m)
    expected = float(np.exp(-2.0 * ne * r.statistic**2))  # R ks.test / SeedMatchR form
    assert np.isclose(r.pvalue, expected)


def test_ks_orientation_seed_matched_more_repressed() -> None:
    import numpy as np

    from cel.data import seedmatch_ks_test

    log2fc = np.array([-2.0, -1.5, -1.0, 0.1, 0.2, 0.3])
    mask = np.array([True, True, True, False, False, False])
    greater = seedmatch_ks_test(log2fc, mask, alternative="greater")
    less = seedmatch_ks_test(log2fc, mask, alternative="less")
    assert greater.pvalue < 0.05  # seed-matched enriched for down-regulation
    assert less.pvalue > 0.5  # the opposite direction is not significant


def test_ks_identical_distributions_gives_unit_pvalue() -> None:
    import numpy as np

    from cel.data import seedmatch_ks_test

    log2fc = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
    mask = np.array([True] * 4 + [False] * 4)
    r = seedmatch_ks_test(log2fc, mask, alternative="greater")
    assert r.statistic == 0.0
    assert r.pvalue == 1.0


def test_ks_requires_both_groups() -> None:
    import numpy as np
    import pytest as _pytest

    from cel.data import seedmatch_ks_test

    with _pytest.raises(ValueError, match="seed-matched"):
        seedmatch_ks_test(np.array([0.1, 0.2]), np.array([True, True]))


# --- Schlegel 2022 / GSE184929 RNA-seq loader (SeedMatchR export) -------------
def _sm(fixtures_dir: Path) -> Path:
    return fixtures_dir / "offtarget" / "seedmatchr"


def test_schlegel_loads_real_slice(fixtures_dir: Path, cache_root: Path) -> None:
    from cel.data import SeedMatchRDataset, load_schlegel2022

    ds = load_schlegel2022(
        "Schlegel_2022_Ttr_D1_30mkg", raw_dir=_sm(fixtures_dir), cache_root=cache_root
    )
    assert isinstance(ds, SeedMatchRDataset)
    assert ds.role == OFFTARGET_CERTIFICATION
    assert ds.sirna_id == "Ttr_D1"
    assert "rat liver" in ds.tissue and "RNA-seq" in ds.assay
    assert list(ds.frame.columns) == [
        "transcript_id",
        "gene_name",
        "log2fc",
        "pvalue",
        "padj",
        "seed_match_count",
        "seed_match",
    ]
    assert len(ds) == 8
    # On-target Ttr is present and strongly down-regulated.
    ttr = ds.frame[ds.frame["gene_name"] == "Ttr"]
    assert float(ttr["log2fc"].iloc[0]) < -2.0
    assert ds.has_seed_annotation


def test_schlegel_ks_evaluation_via_dataset(fixtures_dir: Path, cache_root: Path) -> None:
    from cel.data import load_schlegel2022

    ds = load_schlegel2022(
        "Schlegel_2022_Ttr_D1_30mkg", raw_dir=_sm(fixtures_dir), cache_root=cache_root
    )
    res = ds.ks(alternative="greater")
    # Seed-matched transcripts are more repressed -> significant enrichment.
    assert res.n_seed_match == 4 and res.n_background == 4
    assert res.statistic > 0.9
    assert res.pvalue < 0.05


def test_schlegel_missing_raises_with_source(cache_root: Path, tmp_path) -> None:
    from cel.data import SEEDMATCHR_SOURCE, load_schlegel2022

    with pytest.raises(FileNotFoundError) as exc:
        load_schlegel2022(
            "Schlegel_2022_Ttr_D4_30mkg", raw_dir=tmp_path / "empty", cache_root=cache_root
        )
    msg = str(exc.value)
    assert "SeedMatchR" in msg and "REAL DATA ONLY" in msg
    assert SEEDMATCHR_SOURCE.split(";")[0] in msg


def test_schlegel_unknown_dataset_raises(cache_root: Path, tmp_path) -> None:
    from cel.data import load_schlegel2022

    with pytest.raises(KeyError, match="unknown SeedMatchR dataset"):
        load_schlegel2022("not_a_dataset", raw_dir=tmp_path, cache_root=cache_root)


def test_schlegel_loads_geo_direct_schema(fixtures_dir: Path, cache_root: Path) -> None:
    # The R-free path: the raw GEO GSE184929 DESeq2 table (ensembl_gene_id,
    # log2FoldChange, pvalue, padj, symbol) loads directly via download_offtarget.sh,
    # with no seed annotation yet (computed natively in a later step, not faked).
    from cel.data import load_schlegel2022

    raw = fixtures_dir / "offtarget" / "seedmatchr_geo"
    ds = load_schlegel2022("Schlegel_2022_Ttr_D1_30mkg", raw_dir=raw, cache_root=cache_root)
    assert ds.sirna_id == "Ttr_D1"
    # log2FoldChange parsed; on-target Ttr is strongly down-regulated.
    ttr = ds.frame[ds.frame["gene_name"] == "Ttr"]
    assert float(ttr["log2fc"].iloc[0]) < -5.0
    # The GEO table carries no seed-match column: annotation deferred, not fabricated.
    assert ds.has_seed_annotation is False
    assert ds.frame["seed_match"].isna().all()
