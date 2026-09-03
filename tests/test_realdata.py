"""Hard-data-policy tests for the AAAI off-target certification experiment.

The experiment uses ONLY real measured data. With no real data present, every
data path raises rather than fabricating numbers. These tests assert that
contract (no generated graphs, no signal injection, no fabricated confidences).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))

import exp_realdata as ex  # noqa: E402


def test_enumerates_all_offtarget_datasets() -> None:
    from cel.data import OFFTARGET_CERTIFICATION, datasets_with_role

    assert set(ex.OFFTARGET_LOADERS) == set(datasets_with_role(OFFTARGET_CERTIFICATION))


def test_load_measured_calls_raises_without_real_data() -> None:
    # No raw off-target file present: the real loader raises, never returns data.
    with pytest.raises(FileNotFoundError):
        ex.load_measured_calls("jackson2003")


def test_build_real_inputs_raises_without_real_data() -> None:
    # First the measured calls are loaded (absent -> FileNotFoundError); if those
    # were present the unbuilt sequence-to-x0 join raises NotImplementedError.
    with pytest.raises((FileNotFoundError, NotImplementedError)):
        ex.build_real_inputs("gse184929", string_links="/nonexistent/string.links")


def test_threshold_confidence_keeps_high_confidence_edges() -> None:
    rows, cols, data = [0, 1, 0, 2], [1, 0, 2, 0], [0.9, 0.9, 0.3, 0.3]
    C = sp.coo_matrix((data, (rows, cols)), shape=(3, 3)).tocsr()
    kept = ex.threshold_confidence(C, 0.5)
    assert kept.nnz == 2  # only the 0.9 edges survive the STRING cut
    assert float(kept.max()) == 0.9
