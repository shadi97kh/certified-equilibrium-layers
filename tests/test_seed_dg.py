"""The promoted seed-duplex dG module (cel.seed_dg) on REAL Huesken sequences.

Pins that cel.seed_dg.seed_duplex_dg reproduces the EXACT dG inputs Theorem 3
consumes (exp_t3_constraint._seed_dgs), computed on the real Huesken antisense /
target pairs. ViennaRNA and the real Huesken data are both gated: the test skips
when either is absent (the lightweight import path never needs ViennaRNA).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("RNA")  # ViennaRNA bindings; skip the real-dG test without them

REAL_EFFICACY = REPO / "data" / "raw" / "efficacy"


def test_seed_dg_importable_without_calling_vienna() -> None:
    # Importing the module must NOT require ViennaRNA (gated); only calling does.
    from cel import seed_dg

    assert hasattr(seed_dg, "seed_duplex_dg")
    assert seed_dg.SEED_START == 1 and seed_dg.SEED_LEN == 7  # seed = guide nt 2-8


def test_thermo_reexports_the_promoted_module() -> None:
    # cel.features.thermo re-exports the canonical implementation unchanged.
    from cel.features.thermo import seed_duplex_dg as thermo_dg
    from cel.seed_dg import seed_duplex_dg as module_dg

    assert thermo_dg is module_dg


def test_seed_dg_matches_t3_inputs_on_real_huesken(cache_root: Path) -> None:
    from cel.data import load_huesken_efficacy
    from cel.seed_dg import seed_duplex_dg

    if not (REAL_EFFICACY / "huesken.csv").exists():
        pytest.skip("real Huesken efficacy data not present")

    import exp_t3_constraint as t3  # the T3 experiment that owns the existing dG inputs

    ds = load_huesken_efficacy(REAL_EFFICACY, cache_root=cache_root)
    frame = ds.train().frame.head(64)  # a real subset is enough to pin the path

    # The EXACT existing T3 inputs (exp_t3_constraint._seed_dgs).
    t3_dg = t3._seed_dgs(frame)
    # The promoted module on the SAME (antisense, target) pairs T3 builds.
    mine = np.array(
        [
            seed_duplex_dg(a, t3._target_for(a, s))
            for a, s in zip(frame["antisense"], frame["sense"], strict=True)
        ],
        dtype=float,
    )

    assert mine.shape == t3_dg.shape and mine.size >= 32
    assert np.all(np.isfinite(mine))
    assert np.allclose(mine, t3_dg, atol=1e-9, rtol=0.0)  # promoted module == T3 inputs
    # Real seed:target duplexes are favorable: the dG values are negative.
    assert float(np.median(mine)) < 0.0
