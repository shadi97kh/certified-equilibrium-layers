"""CASCADE-OT miRNA auxiliary corpus: loaders raise on missing, real example build, frozen split.

The gitignored caches (McGeary matrix, seeds, RefSeq map) back the build; tests that need them
skip when absent. The committed corpus_built.json / corpus_split.json invariants are checked
without re-running the full 72743-cell build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cel.data import mirna_aux as ma  # noqa: E402

CACHES_PRESENT = (
    ma.MATRIX_PATH.exists() and ma.SEEDS_PATH.exists() and ma.REFSEQ2SYMBOL_PATH.exists()
)
CASCADE_OT = REPO / "results" / "cascade_ot"


def test_loaders_raise_on_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ma, "MATRIX_PATH", tmp_path / "nope.txt.gz")
    monkeypatch.setattr(ma, "SEEDS_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(ma, "REFSEQ2SYMBOL_PATH", tmp_path / "nope2.json")
    for loader in (ma.load_matrix, ma.load_seeds, ma.load_refseq2symbol):
        with pytest.raises(FileNotFoundError):
            loader()


def test_seed_duplex_dg_is_real() -> None:
    pytest.importorskip("RNA")
    dg = ma.seed_duplex_dg("GAGGUAG")  # let-7 seed
    assert dg < 0.0  # a paired duplex has negative free energy


def test_cell_count_constant() -> None:
    assert ma.MCGEARY_N_CELLS == 4279 * 17 == 72743


def test_seeds_and_build_subset_real() -> None:
    if not CACHES_PRESENT:
        pytest.skip("gitignored McGeary / seed / RefSeq caches not present")
    pytest.importorskip("RNA")

    seeds = ma.load_seeds()
    assert len(seeds) == 17
    assert seeds["let7"]["seed_2_8"] == "GAGGUAG"  # cross-checked vs TargetScan
    assert all(len(v["seed_2_8"]) == 7 for v in seeds.values())

    matrix = ma.load_matrix()
    assert matrix.shape == (4279, 17)

    df = ma.build_examples(limit=40)  # 40 transcripts x 17 miRNAs
    assert len(df) == 40 * 17
    assert set(df.columns) >= {
        "mirna",
        "seed",
        "refseq",
        "symbol",
        "log_tpm_repression",
        "has_3utr",
        "best_site_type",
        "site_count",
        "seed_dg",
    }
    # Real signal direction: site-bearing transcripts are more repressed (lower log-TPM).
    site = df["site_count"] > 0
    if site.any() and (~site & df["has_3utr"]).any():
        assert (
            df.loc[site, "log_tpm_repression"].mean()
            <= df.loc[~site & df["has_3utr"], "log_tpm_repression"].mean() + 1.0
        )


def test_committed_corpus_and_frozen_split() -> None:
    built = CASCADE_OT / "corpus_built.json"
    split = CASCADE_OT / "corpus_split.json"
    if not (built.exists() and split.exists()):
        pytest.skip("corpus artifacts not built yet")

    c = json.loads(built.read_text())
    assert c["n_cells_prefilter"] == 72743 == c["n_cells_prefilter_expected"]
    assert c["n_mirnas"] == 17 and c["n_transcripts"] == 4279
    assert c["n_examples_with_seed_site"] > 0
    assert c["signal_check"]["site_more_repressed"] is True

    s = json.loads(split.read_text())
    assert s["frozen"] is True
    assert s["held_out_transfer"]["never_in_training"] is True
    assert s["held_out_transfer"]["n_conditions"] == 6
    assert s["held_out_transfer"]["n_independent_studies"] == 2
    assert s["train"]["n_cells"] == 72743
