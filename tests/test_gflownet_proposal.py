"""GFlowNet proposal: the T4-certified reward + diversity + enumeration baseline are real;
the BioSeq-GFN-AL sampling arm is pending when the repo is not cloned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cel.design import gflownet_proposal as gp  # noqa: E402
from cel.design.t5_certified_select import TTR_D1, TTR_SEED, guide_for  # noqa: E402


def test_diversity_closed_form() -> None:
    g = "A" * 19
    assert gp.diversity([g, g])["mean_pairwise_hamming"] == 0.0
    g2 = "C" + "A" * 18  # differs at one position
    d = gp.diversity([g, g2])
    assert d["n_unique"] == 2 and d["mean_pairwise_hamming"] == 1.0


def test_gflownet_propose_pending_without_repo() -> None:
    # BioSeq-GFN-AL is not cloned in this environment: the sampler must raise ImportError.
    with pytest.raises(ImportError):
        gp.gflownet_propose(lambda guides: np.ones(len(guides)), n_samples=4, n_train_steps=1)


def test_reward_gate_and_baseline_real() -> None:
    pytest.importorskip("RNA")
    if not (REPO / "data" / "raw" / "efficacy" / "huesken.csv").exists():
        pytest.skip("real Huesken efficacy data not present")
    if not (REPO / "results" / "aaai" / "cache" / "mouse_trrust_3utr_r104.json").exists():
        pytest.skip("mouse TRRUST 3'UTR cache not present")

    ctx = gp.build_context()
    u_parent = float(gp._seed_U([TTR_SEED], ctx["p_norm"], ctx["s2"])[0])
    reward = gp.make_reward(ctx, tau=u_parent)

    # The parent guide is within its own bound (U == u_parent <= tau), so reward = efficacy > 0.
    parent_guide = guide_for(TTR_D1, TTR_SEED)
    r_parent = reward([parent_guide])[0]
    assert r_parent > 0.0
    # A GC-rich seed binds far more strongly (larger U); it is gated to reward 0 at this tau.
    gc_guide = guide_for(TTR_D1, "GGGGGCG")
    u_gc = float(gp._seed_U(["GGGGGCG"], ctx["p_norm"], ctx["s2"])[0])
    assert u_gc > u_parent  # stronger off-target binding
    assert reward([gc_guide])[0] == 0.0

    # Enumeration baseline over a small seed subset is real and well-formed.
    base = gp.enumeration_baseline(ctx, tau=u_parent, eta=0.6, n_seeds=64)
    assert base["n_candidates"] == 64
    assert 0 <= base["n_within_certified_bound"] <= 64
    assert base["diversity_within_bound"]["mean_pairwise_hamming"] <= gp.SEED_LEN  # scaffold-locked
