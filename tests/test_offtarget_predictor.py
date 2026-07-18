"""CASCADE-OT off-target predictor: leakage-free head (Theorem 1) + transfer-result invariants.

The load-bearing property is machine-zero cross-channel leakage; the committed transfer JSON
is checked for structure (leakage 0.0, all held-out conditions present, baseline comparison).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("torch")
import torch  # noqa: E402

from cel.models.offtarget_predictor import (  # noqa: E402
    CONCEPT_DIMS,
    OffTargetPredictor,
    assert_leakage_free,
    build_feature_blocks,
)


def _blocks(n: int = 32):
    rng = np.random.default_rng(0)
    np_blocks = build_feature_blocks(
        site_count=rng.integers(0, 5, n).astype(float),
        seed_dg=rng.normal(-8, 2, n),
        site_grade=rng.uniform(0, 1, n),
        utr_len=rng.integers(50, 5000, n).astype(float),
    )
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in np_blocks.items()}


def test_build_feature_blocks_shapes() -> None:
    b = build_feature_blocks(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5))
    assert b["site_load"].shape == (5, 2)
    assert b["seed_pairing_stability"].shape == (5, 2)
    assert b["site_grade"].shape == (5, 1) and b["target_accessibility"].shape == (5, 1)


def test_leakage_free_is_machine_zero() -> None:
    model = OffTargetPredictor()
    blocks = _blocks()
    assert model.leakage_offdiag_max(blocks) == 0.0  # each concept head reads only its own block
    assert assert_leakage_free(model, blocks) == 0.0
    # The fail path triggers: with a negative tolerance even zero leakage must raise.
    with pytest.raises(RuntimeError):
        assert_leakage_free(model, blocks, tol=-1.0)


def test_forward_and_train_step_runs() -> None:
    torch.manual_seed(0)
    model = OffTargetPredictor()
    blocks = _blocks(64)
    y = torch.randn(64)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    before = float(torch.nn.functional.huber_loss(model(blocks), y))
    for _ in range(50):
        opt.zero_grad()
        torch.nn.functional.huber_loss(model(blocks), y).backward()
        opt.step()
    after = float(torch.nn.functional.huber_loss(model(blocks), y))
    assert after <= before  # the readout is trainable
    assert model(blocks).shape == (64,)


def test_rnafm_channel_is_frozen() -> None:
    model = OffTargetPredictor(rnafm_dim=8)
    frozen = model.heads["rnafm"]
    assert all(not p.requires_grad for p in frozen.parameters())
    assert "rnafm" in CONCEPT_DIMS or model.rnafm_dim == 8


def test_committed_transfer_invariants() -> None:
    path = REPO / "results" / "cascade_ot" / "predictor_transfer.json"
    if not path.exists():
        pytest.skip("transfer result not built yet")
    j = json.loads(path.read_text())
    assert j["training"]["leakage_offdiag_max"] == 0.0
    assert j["training"]["rnafm_frozen_features"] is False
    conds = j["conditions"]
    assert len(conds) == 6  # 3 Patel + 3 Schlegel held-out conditions
    for c in conds.values():
        assert "spearman_learned" in c and "spearman_baseline_seedcount" in c
        assert "ks_direction_reproduced" in c
    s = j["summary"]
    assert "learned_beats_baseline_mean" in s and "mean_spearman_learned" in s
