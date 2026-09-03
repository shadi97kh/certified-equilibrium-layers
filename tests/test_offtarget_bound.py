"""Tests for the Theorem 4 certified off-target bound (cel.verify.offtarget_bound)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cel.verify import offtarget_bound as ob
from cel.verify import stability as st


def _grn(n: int, seed: int) -> sp.csr_matrix:
    a = sp.random(n, n, density=0.1, random_state=seed)
    a = (a + a.T) * 0.5
    return st.symmetric_normalized_variant(a.tocsr())


def test_certified_bound_holds() -> None:
    rng = np.random.default_rng(0)
    n, alpha = 50, 0.5
    s = _grn(n, 0)
    ds = (sp.random(n, n, density=0.05, random_state=1) * 0.02).tocsr()
    ds = (ds + ds.T) * 0.5
    s_true = (s + ds).tocsr()
    e = rng.standard_normal(n)
    e_true = e + 0.01 * rng.standard_normal(n)

    e_real = ob.propagate(s, e, alpha)
    e_real_true = ob.propagate(s_true, e_true, alpha)
    deviation = float(np.linalg.norm(e_real - e_real_true))

    cert = ob.certified_bound(
        s,
        alpha,
        float(np.linalg.norm(e_true)),
        float(np.linalg.norm(e - e_true)),
        st.operator_2norm(s - s_true),
        p_prime_norm=ob.p_alpha_norm(s_true, alpha),
    )
    assert deviation <= cert["bound"]  # the certified bound holds
    assert cert["seed_error_term"] > 0 and cert["grn_edge_term"] > 0


def test_second_resolvent_identity_is_exact() -> None:
    n, alpha = 30, 0.4
    s = _grn(n, 2)
    ds = (sp.random(n, n, density=0.1, random_state=3) * 0.03).tocsr()
    s_prime = (s + (ds + ds.T) * 0.5).tocsr()
    p = ob.dense_p_alpha(s, alpha)
    pp = ob.dense_p_alpha(s_prime, alpha)
    lhs = p - pp
    rhs = ((1 - alpha) / alpha) * p @ (s - s_prime).toarray() @ pp
    assert float(np.abs(lhs - rhs).max()) < 1e-10  # identity holds to machine precision


def test_rowwise_refinement_is_tighter_and_holds() -> None:
    rng = np.random.default_rng(1)
    n, alpha = 40, 0.5
    s = _grn(n, 4)
    ds = (sp.random(n, n, density=0.05, random_state=5) * 0.02).tocsr()
    s_true = (s + (ds + ds.T) * 0.5).tocsr()
    e = rng.standard_normal(n)
    e_true = e + 0.01 * rng.standard_normal(n)
    e_real, e_real_true = ob.propagate(s, e, alpha), ob.propagate(s_true, e_true, alpha)

    e_norm = float(np.linalg.norm(e_true))
    e_err = float(np.linalg.norm(e - e_true))
    s_err = st.operator_2norm(s - s_true)
    cert = ob.certified_bound(
        s, alpha, e_norm, e_err, s_err, p_prime_norm=ob.p_alpha_norm(s_true, alpha)
    )
    rw = ob.rowwise_bound(s, alpha, e_norm, e_err, s_err)
    assert rw.max() <= cert["bound"] + 1e-9  # every per-transcript bound is no looser
    assert bool(np.all(np.abs(e_real - e_real_true) <= rw + 1e-9))  # and per-transcript holds


def test_verify_and_vacuous() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    meas = np.array([1.1, 1.9, 3.05])
    v = ob.verify_bound(pred, meas, bound=1.0)
    assert v["holds"] is True and v["deviation"] < 1.0
    assert ob.verify_bound(pred, meas, bound=0.01)["holds"] is False

    # A bound far larger than the measured scale is vacuous.
    vac = ob.is_vacuous(1e6, meas, factor=10.0)
    assert vac["vacuous"] is True and vac["bound_over_measured"] > 10.0
    assert ob.is_vacuous(0.5, meas, factor=10.0)["vacuous"] is False
