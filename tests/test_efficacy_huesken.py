"""Huesken 2005 efficacy readout: the without-RNA-FM arm is real; RNA-FM arm gated.

Runs the real CASCADE-feature efficacy readout on the Huesken 2005 OligoFormer subset
and checks the conventional split, the siRNADiscovery anti-leakage cleaning, and that
Spearman / PCC are real and informative. The RNA-FM arm is recorded pending when the
'fm' package and embedding cache are both absent (no synthetic fallback).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from cel.experiments import efficacy_huesken as ex  # noqa: E402

REAL_EFFICACY = REPO / "data" / "raw" / "efficacy" / "huesken.csv"


def test_anti_leakage_removes_shared_guides() -> None:
    guides = ["AAAA", "CCCC", "GGGG", "AAAA", "UUUU"]
    is_train = np.array([True, True, False, False, False])  # AAAA leaks into test (idx 3)
    mask, n_leaked = ex.anti_leakage_test_mask(guides, is_train)
    assert n_leaked == 1
    assert not mask[3]  # the leaked AAAA test row is removed
    assert mask[2] and mask[4]  # GGGG / UUUU kept


def test_efficacy_readout_on_real_huesken() -> None:
    if not REAL_EFFICACY.exists():
        pytest.skip("real Huesken efficacy data not present")

    numbers = ex.run()
    assert numbers["status"] == "ok"
    sp = numbers["split"]
    assert sp["n_train"] == 2182
    assert sp["n_test_after_anti_leakage"] <= sp["n_test_conventional"]
    assert numbers["anti_leakage"]["n_test_guides_removed"] >= 0

    w = numbers["without_rnafm"]
    assert -1.0 <= w["spearman"] <= 1.0 and -1.0 <= w["pcc"] <= 1.0
    assert np.isfinite(w["spearman"]) and np.isfinite(w["pcc"])
    # The CASCADE features carry real on-target signal: both correlations are clearly positive.
    assert w["spearman"] > 0.2 and w["pcc"] > 0.2
    assert w["n_features"] >= 9

    wr = numbers["with_rnafm"]
    if wr.get("status") == "pending_real_data":
        assert "fm" in wr["source"] or "RNA-FM" in wr["source"]
    else:
        # When RNA-FM is available the arm reports the same metric keys plus the 640-d width.
        assert -1.0 <= wr["spearman"] <= 1.0 and wr["rnafm_dim"] == 640
        assert wr["n_features"] == w["n_features"] + 640
