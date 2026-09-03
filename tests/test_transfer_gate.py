"""Stage 4 transfer gate: pre-registered PASS/FAIL on the off-target predictor's transfer.

Runs against the committed real transfer + KS results; skipped when they are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cel.experiments import transfer_gate as tg  # noqa: E402


def test_gate_verdict_and_structure() -> None:
    if not (tg.TRANSFER.exists() and tg.OFFTARGET_T4.exists()):
        pytest.skip("predictor_transfer.json / offtarget_t4.json not present")

    g = tg.run()
    assert g["status"] == "ok"
    assert g["pre_registered"]["margin_spearman"] == 0.05
    assert g["verdict"] in {"PASS", "FAIL"}

    c1 = g["criterion_1_margin"]
    # The margin criterion passes iff the learned mean beats the baseline by the margin.
    assert c1["pass"] == (c1["delta"] >= 0.05)
    c2 = g["criterion_2_sgna_direction"]
    # The (S)-GNA criterion needs BOTH the measured ordering and the predicted ordering to hold.
    assert c2["reproduced"] == (
        c2["measured_parent_more_offtarget"] and c2["predicted_parent_more_offtarget"]
    )
    # The verdict is the conjunction of the two criteria.
    assert (g["verdict"] == "PASS") == (c1["pass"] and c2["reproduced"])


def test_committed_gate_is_fail() -> None:
    # On the current real numbers the gate is a documented FAIL (learned does not beat the
    # mechanistic baseline; the predictor cannot distinguish the (S)-GNA modification).
    if not (tg.TRANSFER.exists() and tg.OFFTARGET_T4.exists()):
        pytest.skip("transfer inputs not present")
    g = tg.run()
    assert g["verdict"] == "FAIL"
    assert g["criterion_1_margin"]["pass"] is False
    assert g["criterion_2_sgna_direction"]["reproduced"] is False
