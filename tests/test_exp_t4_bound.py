"""Tests for experiments/aaai/exp_t4_bound.py (numpy/scipy; no torch)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))
sys.path.insert(0, str(REPO / "src"))

import exp_t4_bound as ex  # noqa: E402

from cel.verify import stability as st  # noqa: E402


def _grn(n: int) -> sp.csr_matrix:
    a = sp.random(n, n, density=0.1, random_state=0)
    a = (a + a.T) * 0.5
    return st.symmetric_normalized_variant(a.tocsr())


def test_assess_dataset_bound_holds_and_meaningful() -> None:
    rng = np.random.default_rng(0)
    n, alpha = 60, 0.5
    s = _grn(n)
    e = rng.standard_normal(n)
    e_pred = ex.ob.propagate(s, e, alpha)
    e_measured = e_pred + 0.02 * rng.standard_normal(n)  # close prediction
    mask = rng.random(n) < 0.3
    res = ex.assess_dataset(
        "d",
        "rna_seq",
        e_pred,
        e_measured,
        s,
        alpha,
        e,
        e_err_norm=0.05,
        s_err_norm=0.05,
        seed_match_mask=mask,
    )
    assert res["bound_holds"] is True
    assert res["certified_bound"]["bound"] > 0
    assert res["rowwise_refinement"]["max_bound"] <= res["certified_bound"]["bound"] + 1e-9
    assert res["rowwise_refinement"]["per_transcript_holds"] is True
    assert res["ks_measured"]["D"] is not None and res["ks_predicted"]["D"] is not None


def test_acceptance_logic() -> None:
    assert ex.acceptance({})["flag"] == "PENDING"

    def d(holds, meaningful, vacuous, assay="rna_seq"):
        return {"bound_holds": holds, "meaningful": meaningful, "vacuous": vacuous, "assay": assay}

    assert ex.acceptance({"x": d(True, True, False)})["flag"] == "PASS"
    bad = ex.acceptance({"x": d(True, False, True)})
    assert bad["flag"] == "FAIL" and "x" in bad["vacuous_datasets"]
    assert ex.acceptance({"x": d(False, False, False)})["flag"] == "FAIL"


def test_cross_assay_reports_both_assays() -> None:
    measured = {
        "per_contrast": {
            "g1": {"ks_D": 0.8},
            "g2": {"ks_D": 0.4},
        }
    }
    microarray = {
        "status": "ok",
        "per_dataset": {
            "j": {"ks_measured": {"D": 0.5}},
            "b": {"ks_measured": {"D": 0.6}},
        },
    }
    ca = ex.cross_assay(measured, microarray)
    assert abs(ca["rna_seq_mean_ks_D"] - 0.6) < 1e-9
    assert abs(ca["microarray_mean_ks_D"] - 0.55) < 1e-9


def test_cross_assay_microarray_pending_is_null_not_faked() -> None:
    measured = {"per_contrast": {"g": {"ks_D": 0.16}}}
    microarray = {
        "status": "pending_real_data",
        "per_dataset": {},
        "pending": {"jackson2006": "..."},
    }
    ca = ex.cross_assay(measured, microarray)
    assert ca["rna_seq_mean_ks_D"] == 0.16
    assert ca["microarray_mean_ks_D"] is None
    assert ca["microarray_status"] == "pending_real_data"


def test_load_real_dataset_raises_without_real_data() -> None:
    # The microarray loader raises FileNotFoundError when its real table is absent.
    with pytest.raises(FileNotFoundError):
        ex.load_real_dataset("jackson2006", string_links=None, alpha=0.5)
    # GSE184929 via the SeedMatchR export is likewise absent.
    with pytest.raises(FileNotFoundError):
        ex.load_real_dataset("gse184929", string_links=None, alpha=0.5)


def test_multiseed_aggregation_is_conservative() -> None:
    def d(holds, meaningful, vac, bound, dev):
        return {
            "g": {
                "assay": "rna_seq",
                "bound_holds": holds,
                "meaningful": meaningful,
                "vacuous": vac,
                "certified_bound": {"bound": bound},
                "measured_deviation": dev,
                "rowwise_refinement": {"max_bound": bound * 0.5},
                "ks_measured": {"D": 0.8},
                "ks_predicted": {"D": 0.7},
            }
        }

    # Bound holds and is meaningful on every seed -> PASS; worst-case bound kept.
    ps = [d(True, True, False, 0.20, 0.04), d(True, True, False, 0.25, 0.04)]
    merged = ex.aggregate_per_dataset(ps, [0, 1])
    assert merged["g"]["certified_bound"]["bound"] == 0.25  # worst seed
    assert merged["g"]["per_seed_bound"] == [0.20, 0.25]
    acc = ex.aggregate_acceptance(merged, ps, [0, 1])
    assert acc["flag"] == "PASS" and acc["n_seeds"] == 2

    # Bound fails on one seed -> conservative FAIL.
    ps2 = [d(True, True, False, 0.2, 0.04), d(False, False, False, 0.2, 0.5)]
    acc2 = ex.aggregate_acceptance(ex.aggregate_per_dataset(ps2, [0, 1]), ps2, [0, 1])
    assert acc2["flag"] == "FAIL"

    # All seeds pending (empty per_dataset) -> PENDING.
    acc3 = ex.aggregate_acceptance(ex.aggregate_per_dataset([{}, {}], [0, 1]), [{}, {}], [0, 1])
    assert acc3["flag"] == "PENDING"
