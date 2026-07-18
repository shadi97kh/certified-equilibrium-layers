"""T5 certified-feasible selection: conformal/BH math (unit) + a real small enumeration.

The conformal p-value and Benjamini-Hochberg routines are checked on closed-form cases;
the end-to-end run is checked on a small seed subset against the real Huesken calibration
and mouse TRRUST off-target bound (skipped only if ViennaRNA is absent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cel.design import t5_certified_select as cs  # noqa: E402


def test_conformal_pvalues_closed_form() -> None:
    cal_pred = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    cal_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0])  # nulls = first three (y < 0.5)
    p = cs.conformal_pvalues(cal_pred, cal_y, np.array([0.95, 0.05]), eta=0.5)
    # 0.95 exceeds all 3 nulls -> p = (1+0)/(1+3) = 0.25; 0.05 below all -> (1+3)/4 = 1.0
    assert np.isclose(p[0], 0.25) and np.isclose(p[1], 1.0)
    # No calibration nulls -> fully conservative (p = 1).
    p2 = cs.conformal_pvalues(np.array([0.7, 0.8]), np.array([0.9, 0.9]), np.array([0.95]), eta=0.5)
    assert np.allclose(p2, 1.0)


def test_benjamini_hochberg_closed_form() -> None:
    pvals = np.array([0.01, 0.02, 0.5, 0.9])
    sel, thresh = cs.benjamini_hochberg(pvals, q=0.1)
    # thresholds q*k/m = [0.025, 0.05, 0.075, 0.1]; largest k with p_(k) <= thresh is k=2 -> p<=0.02
    assert np.isclose(thresh, 0.02)
    assert sel.tolist() == [True, True, False, False]
    # Nothing passes -> empty selection.
    sel0, _ = cs.benjamini_hochberg(np.array([0.9, 0.95]), q=0.1)
    assert not sel0.any()


def test_enumerate_seeds() -> None:
    assert len(cs.enumerate_seeds(n_seeds=10)) == 10
    full = cs.enumerate_seeds()
    assert len(full) == 4**cs.SEED_LEN == 16384


def test_run_small_real() -> None:
    pytest.importorskip("RNA")  # ViennaRNA for the certified-U seed duplex
    if not (REPO / "data" / "raw" / "efficacy" / "huesken.csv").exists():
        pytest.skip("real Huesken efficacy data not present")
    if not (REPO / "results" / "aaai" / "cache" / "mouse_trrust_3utr_r104.json").exists():
        pytest.skip("mouse TRRUST 3'UTR cache not present (needs a prior pyensembl mouse build)")

    numbers = cs.run(n_seeds=128, eta=0.6, q=0.2)
    assert numbers["status"] == "ok" and numbers["n_candidates"] == 128
    a = numbers["assumptions"]
    assert a["tau"] > 0 and a["calibration_n"] > 0

    guides = numbers["guides"]
    assert len(guides) == 128
    for g in guides:
        assert g["certified_U"] >= 0.0
        assert 0.0 <= g["predicted_efficacy"] <= 1.0
        assert 0.0 < g["conformal_pvalue"] <= 1.0
        # A finally-selected guide must satisfy the exact off-target certificate U <= tau
        # and be conformally selected.
        if g["selected"]:
            assert g["certified_U"] <= a["tau"] * (1.0 + 1e-9)
            assert g["conformal_selected"]
    assert numbers["n_selected"] == sum(g["selected"] for g in guides)
    assert (
        numbers["n_selected"] <= numbers["n_threshold_selected"] + numbers["n_conformal_selected"]
    )
