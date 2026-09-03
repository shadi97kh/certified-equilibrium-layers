"""Tests for experiments/aaai/exp_t2_stability.py.

Runs on small in-memory signed graphs (no real-data file or torch needed); the
real-GRN path is exercised by the experiment's own run on real TRRUST.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))
sys.path.insert(0, str(REPO / "src"))

import exp_t2_stability as ex  # noqa: E402

from cel.verify import stability as st  # noqa: E402

ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _broadcast(k: int = 6) -> sp.csr_matrix:
    rows = list(range(1, k + 1))
    data = [1.0] * (k - 1) + [-1.0]
    return sp.coo_matrix((data, (rows, [0] * k)), shape=(k + 1, k + 1)).tocsr()


def test_analyze_graph_structure() -> None:
    S = _broadcast(6)
    res = ex.analyze_graph(S, "directed", ALPHAS, samples=40, power_iters=80, probes=4, seed=0)
    assert res["n_nodes"] == 7
    assert np.isclose(res["operator_2norm"], np.sqrt(6), rtol=1e-6)
    assert res["inf_norm"] <= 1.0 + 1e-9
    assert len(res["per_alpha"]) == len(ALPHAS)
    for r in res["per_alpha"]:
        assert r["operator_2norm"] >= r["measured_l2_max_ratio"] - 1e-6  # true >= sampled


def test_acceptance_restricted_on_directed_vs_symmetric() -> None:
    S = _broadcast(6)
    directed = ex.analyze_graph(S, "d", ALPHAS, samples=60, power_iters=120, probes=4, seed=0)
    symmetric = ex.analyze_graph(
        st.symmetric_normalized_variant(S),
        "s",
        ALPHAS,
        samples=60,
        power_iters=120,
        probes=4,
        seed=0,
    )
    acc = ex.acceptance(directed, symmetric)
    # Non-normal directed graph: rho-based L2 bound fails, but the inf-norm
    # certificate holds and the symmetric variant satisfies the L2 rho-bound.
    assert acc["l2_rho_constant_holds_directed"] is False
    assert acc["inf_norm_nonexpansive_directed"] is True
    assert acc["symmetric_l2_rho_constant_holds"] is True
    assert acc["flag"] == "RESTRICTED"
    assert acc["operator_2norm_directed"] > acc["rho_directed"]  # non-normality


def test_acceptance_pass_on_symmetric_as_directed() -> None:
    # A symmetric operator used in both roles: the L2 rho-bound holds directly.
    S = st.symmetric_normalized_variant(_broadcast(5))
    sym = ex.analyze_graph(S, "s", ALPHAS, samples=60, power_iters=160, probes=4, seed=0)
    acc = ex.acceptance(sym, sym)
    assert acc["l2_rho_constant_holds_directed"] is True
    assert acc["flag"] == "PASS"


def test_json_safe_replaces_non_finite() -> None:
    out = ex._json_safe({"a": float("inf"), "b": [1.0, float("nan")], "c": 0.5})
    assert out == {"a": None, "b": [1.0, None], "c": 0.5}


def test_aggregate_analyses_is_conservative() -> None:
    S = _broadcast(6)
    runs = [
        ex.analyze_graph(S, "d", ALPHAS, samples=40, power_iters=100, probes=4, seed=s)
        for s in (0, 1, 2)
    ]
    agg = ex._aggregate_analyses(runs, [0, 1, 2])
    assert agg["seeds"] == [0, 1, 2]
    assert len(agg["per_alpha"]) == len(ALPHAS)
    # Each aggregated quantity is the worst-case (max) across seeds.
    for i, r in enumerate(agg["per_alpha"]):
        across = [run["per_alpha"][i]["operator_2norm"] for run in runs]
        assert r["operator_2norm"] == max(across)
        meas = [run["per_alpha"][i]["measured_l2_max_ratio"] for run in runs]
        assert r["measured_l2_max_ratio"] == max(meas)
    # Graph invariants are seed-independent and preserved.
    assert agg["rho"] == runs[0]["rho"] and agg["operator_2norm"] == runs[0]["operator_2norm"]
