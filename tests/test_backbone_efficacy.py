"""Hard-data-policy tests for the backbone efficacy benchmark.

The experiment trains ONLY on the real OligoFormer suite; with the CSVs absent
it raises rather than falling back to generated data. The positive path is
exercised on the committed real-schema fixtures under tests/fixtures/oligoformer.
torch is imported only inside train_and_eval, so the loader path runs in the
torch-free CI image and the end-to-end run is torch-gated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))

import exp_backbone_efficacy as ex  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "oligoformer"


def test_reported_baselines_are_cited_not_fabricated() -> None:
    # Every baseline carries a source citation; only transcribed values are set,
    # the rest are null (not invented).
    for b in ex.REPORTED_BASELINES:
        assert b["source"] and isinstance(b["source"], str)
        assert b["reported_value"] is None or isinstance(b["reported_value"], int | float)
        assert b["metric"] == "pearson"
    names = {b["name"]: b["reported_value"] for b in ex.REPORTED_BASELINES}
    assert names["BIOPREDsi (Huesken ANN)"] == 0.66  # transcribed reference point


def test_target_defaults_to_perfect_complement() -> None:
    from cel.features.seedmatch import revcomp

    g = "UGAGGUAGUAGGUUGUAUAGU"
    assert ex._target_for(g, "") == revcomp(g)  # on-target site when no sense given
    assert ex._target_for(g, "ACGU") == "ACGU"  # provided sense used as-is


def test_real_datasets_raises_without_csvs(tmp_path) -> None:
    # No OligoFormer CSVs present -> FileNotFoundError, never generated data.
    with pytest.raises(FileNotFoundError):
        ex.real_datasets(tmp_path / "absent", tmp_path / "cache")


def test_real_datasets_on_fixtures(tmp_path) -> None:
    feats, labels = ex.real_datasets(FIXTURES, tmp_path / "cache")
    assert "huesken_train" in feats
    assert len(feats["huesken_train"]) > 0
    width = ex.concept_layout().total_dim
    for key, x in feats.items():
        if len(x):
            assert x.shape[1] == width
        assert len(x) == len(labels[key])


def test_end_to_end_smoke_on_fixtures(tmp_path) -> None:
    pytest.importorskip("torch")
    feats, labels = ex.real_datasets(FIXTURES, tmp_path / "cache")
    res = ex.train_and_eval(feats, labels, seed=0, epochs=3, lr=0.01)
    assert set(res["per_dataset"]) == set(ex.DATASETS)
