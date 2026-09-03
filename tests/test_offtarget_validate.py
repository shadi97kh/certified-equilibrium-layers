"""offtarget_validate: CASCADE off-target ranking + T4 bound vs the SeedMatchR ECDF shift.

Runs on the real Schlegel 2022 Ttr GSE184929 contrasts (native annotation when the
SeedMatchR R CSV is absent) and checks the measured ECDF shift, the CASCADE predicted
KS, the rank concordance, and that the T4 certified bound binds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("scipy")
pytest.importorskip("sklearn")  # not used directly, but keeps the env consistent

from cel.experiments import offtarget_validate as ov  # noqa: E402

D1 = "Schlegel_2022_Ttr_D1_30mkg"


def test_validate_against_seedmatchr_ecdf() -> None:
    if not (REPO / "data" / "raw" / "offtarget" / "seedmatchr" / f"{D1}.csv").exists():
        pytest.skip("real GSE184929 SeedMatchR-export DESeq2 tables not present")

    numbers = ov.run()
    assert numbers["status"] == "ok"
    assert "native" in numbers["reference_source"] or "SeedMatchR" in numbers["reference_source"]

    pc = numbers["per_contrast"]
    assert D1 in pc
    c = pc[D1]
    # The measured SeedMatchR ECDF shift is real and present for the parent guide.
    assert c["seedmatchr_ks_D"] is not None and c["seedmatchr_ks_D"] > 0.05
    assert 0.0 < c["seedmatchr_ks_p"] < 0.05
    # CASCADE produces a predicted-off-target KS and a finite rank concordance.
    assert c["cascade_predicted_ks_D"] is not None
    assert np.isfinite(c["cascade_rank_spearman_pred_vs_measured"])
    # The T4 certified upper bound is positive, has a per-gene rowwise refinement, and binds.
    assert c["certified_U_global"] > 0.0
    assert c["certified_U_rowwise_max"] is not None
    assert c["predicted_vs_measured_deviation"] <= c["certified_U_global"] * (1.0 + 1e-9)
    assert c["bound_binds"] is True

    assert numbers["summary"]["t4_bound_binds_all_contrasts"] is True


def test_pick_column_helper() -> None:
    import pandas as pd

    df = pd.DataFrame({"Gene_Name": ["A"], "log2FoldChange": [0.1], "mer7m8": [2]})
    assert ov._pick(df, ["symbol", "gene_name"]) == "Gene_Name"
    assert ov._pick(df, ["log2FoldChange"]) == "log2FoldChange"
    assert ov._pick(df, ["nope"]) is None
