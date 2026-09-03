"""Off-target data-feasibility probe: real corpus counts + LOSO verdict.

Runs the assessment on the real off-target tables and committed KS results; skipped when
the gitignored raw off-target data is absent. The miRNA-aux example count is asserted only
when the small McGeary matrix has been cached locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cel.experiments import ot_data_feasibility as feas  # noqa: E402

RAW_PATEL = REPO / "data" / "raw" / "offtarget" / "patel" / "GSE164289_siGGGGGC_vs_siNT1_48h.csv"


def test_feasibility_corpus_and_verdict() -> None:
    if not RAW_PATEL.exists():
        pytest.skip("gitignored raw off-target tables not present")

    numbers = feas.run()
    assert numbers["status"] == "ok"

    rc = numbers["real_offtarget_corpus"]
    assert set(rc["studies"]) == {
        "Schlegel_2022_GSE184929",
        "Patel_2022_GSE164289",
        "Patel_2022_GSE196839",
    }
    assert rc["studies"]["Schlegel_2022_GSE184929"]["n_conditions"] == 3
    assert rc["studies"]["Patel_2022_GSE164289"]["n_conditions"] == 2
    assert rc["studies"]["Patel_2022_GSE196839"]["n_conditions"] == 1
    assert rc["n_geo_series"] == 3
    assert rc["n_independent_papers"] == 2
    assert rc["n_conditions_total"] == 6
    # Every condition carries a real gene count.
    for s in rc["studies"].values():
        for c in s["conditions"].values():
            assert c["n_genes"] and c["n_genes"] > 1000

    # The verdict: too few independent studies to train ON the off-target corpus with LOSO.
    v = numbers["verdict"]
    assert v["learned_predictor_trainable_on_offtarget_corpus_with_loso"] is False
    assert v["n_independent_papers"] == 2

    # miRNA auxiliary corpus: TargetScan files reported by size, never downloaded.
    ts = numbers["mirna_auxiliary_corpus"]["targetscan_context_plus_plus"]
    assert len(ts["files"]) == 4 and all(f["downloaded"] is False for f in ts["files"])
    assert any(f["bytes"] > 5e8 for f in ts["files"])  # the large nonconserved file is flagged

    mcg = numbers["mirna_auxiliary_corpus"]["mcgeary_2019_agorbns"]
    assert mcg["geo"] == "GSE140217" and mcg["repression_matrix_bytes"] == feas.MCGEARY_BYTES
    if mcg.get("downloaded"):  # the small matrix was cached -> real example count
        assert mcg["n_transcripts"] == 4279
        assert mcg["n_transfected_mirnas"] == 17
        assert mcg["n_measured_examples"] == 4279 * 17 == 72743
