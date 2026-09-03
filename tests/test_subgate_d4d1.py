"""Stage 0 sub-gate: monotone modification-aware readout must give E(D4) < E(D1) on GSE184929.

Needs ViennaRNA + the real Schlegel GSE184929 seed annotation; skipped otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cel.experiments import subgate_d4d1 as sg  # noqa: E402


def test_ddg_is_destabilizing() -> None:
    pytest.importorskip("RNA")
    d = sg.ddg_calibration()
    # The (S)-GNA g7 mismatch makes the duplex weaker: dG_D4 less favorable, occupancy lower.
    assert d["dG_D4_kcal_per_mol"] > d["dG_D1_kcal_per_mol"]
    assert d["ddG_kcal_per_mol"] > 0 and d["destabilizing"] is True
    assert d["occupancy_D4"] < d["occupancy_D1"]
    assert d["delta_tm_calibration"]["status"] == "pending_real_data"


def test_subgate_passes_monotone_fails_nonmonotone() -> None:
    pytest.importorskip("RNA")
    if not (REPO / "data" / "raw" / "offtarget" / "seedmatchr" / f"{sg.D1_CONTRAST}.csv").exists():
        pytest.skip("real GSE184929 Schlegel seed annotation not present")

    g = sg.run()
    assert g["status"] == "ok"
    # The monotone readout reproduces the strict inequality (the blocking sub-gate).
    assert g["monotone"]["D4_lt_D1"] is True
    assert g["monotone"]["E_D4"] < g["monotone"]["E_D1"]
    assert g["subgate"] == "PASS"
    # The non-monotone count-only readout cannot distinguish D1 from D4 (motivates the monotone design).
    assert g["non_monotone_control"]["D4_lt_D1"] is False
    assert (
        g["non_monotone_control"]["E_D4_count_only"] == g["non_monotone_control"]["E_D1_count_only"]
    )
