"""RC-equivariant linear layer (Mallet and Vert 2021): f(g . x) == g . f(x) on real guides.

Empirically verifies the equivariance on real Huesken 2005 guide sequences, and that a
plain (untied) linear layer breaks it. Needs torch; skipped when torch is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("torch")
import torch  # noqa: E402

from cel.models.rc_equivariant import (  # noqa: E402
    RCEquivariantLinear,
    _real_guides,
    complement_permutation,
    equivariance_check,
    one_hot_guides,
    pairing_involution,
    rc_action,
)


def test_complement_and_pairing_are_involutions() -> None:
    m_in = complement_permutation()
    assert torch.equal(m_in @ m_in, torch.eye(4))  # complement is its own inverse
    m_out = pairing_involution(8)
    assert torch.equal(m_out @ m_out, torch.eye(8))
    with pytest.raises(ValueError):
        pairing_involution(5)  # odd c_out has no pairing involution


def test_one_hot_and_rc_action() -> None:
    x = one_hot_guides(["ACGU", "GGCC"])
    assert x.shape == (2, 4, 4) and torch.equal(x.sum(-1), torch.ones(2, 4))
    # RC of "ACGU" (reverse + complement A<->U,C<->G) is "ACGU" itself.
    m = complement_permutation()
    rc = rc_action(x[:1], m)
    assert torch.equal(rc[0], x[0])  # ACGU is its own reverse complement


def test_layer_is_equivariant_on_synthetic_batch() -> None:
    torch.manual_seed(0)
    layer = RCEquivariantLinear(c_in=4, c_out=8).double()
    x = one_hot_guides(["ACGUACGUACGUACGUACG", "GGGGCCCCAAAAUUUUACG"]).double()
    with torch.no_grad():
        lhs = layer(rc_action(x, layer.M_in))
        rhs = rc_action(layer(x), layer.M_out)
    assert torch.max(torch.abs(lhs - rhs)) < 1e-10
    assert layer.weight_tying_residual() < 1e-10


def test_equivariance_on_real_huesken_guides() -> None:
    if not (REPO / "data" / "huesken2005.csv").exists():
        pytest.skip("real Huesken guides not present")
    guides = _real_guides(n=64)
    assert len(guides) == 64 and all(len(g) == 19 for g in guides)

    result = equivariance_check(guides)
    assert result["n_guides"] == 64 and result["guide_len"] == 19
    # The tied layer is equivariant to machine precision on the real guides.
    assert result["max_equivariance_error"] < result["tolerance"]
    assert result["equivariant"] is True
    assert result["weight_tying_residual"] < result["tolerance"]
    assert result["rc_involution_error"] < result["tolerance"]
    # The non-equivariant control genuinely breaks equivariance (the test is meaningful).
    assert result["control_breaks_equivariance"] is True
    assert result["nonequivariant_control_error"] > result["tolerance"]
