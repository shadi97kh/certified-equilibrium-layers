"""Reverse-complement equivariant linear layer (Mallet and Vert, 2021).

The reverse-complement (RC) group is Z/2 acting on a one-hot nucleotide sequence
X in R^{L x C}: g . X = reverse over the length axis AND complement the channels
(A<->U, C<->G), i.e. g . X = reverse_L(X @ M), where M is the complement permutation
on the channels (an involution). On the OUTPUT channels the group acts through a
chosen involution M_out: g . Y = reverse_L(Y @ M_out).

A position-wise linear map f(X) = X @ W + b (no mixing across positions) is
RC-EQUIVARIANT, f(g . X) = g . f(X), if and only if its channel weights commute with
the two representations and the bias is fixed by the output representation:

    M_in @ W = W @ M_out      and      M_out @ b = b.

This module enforces both by parameter sharing (the Reynolds / group-average projection
used by Mallet and Vert): from free parameters W0, b0 it forms

    W = (W0 + M_in @ W0 @ M_out) / 2,     b = (b0 + M_out @ b0) / 2,

which satisfy M_in W = W M_out and M_out b = b exactly (M_in, M_out are involutions),
so the layer is equivariant by construction. The position-wise map commutes with the
length reversal, so the full RC action is respected. equivariance_check verifies this
empirically on real guide sequences and writes results/rc_equivariance_check.json.

Needs torch; cel.models guards the import so the torch-free path is unaffected.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[3]

# Channel order and Watson-Crick complement for the one-hot encoding.
ALPHABET = "ACGU"
_COMPLEMENT = {"A": "U", "U": "A", "C": "G", "G": "C", "T": "A"}


def complement_permutation() -> Tensor:
    """Complement permutation matrix M on the ACGU channels (anti-diagonal involution)."""
    n = len(ALPHABET)
    m = torch.zeros(n, n)
    for i, base in enumerate(ALPHABET):
        m[i, ALPHABET.index(_COMPLEMENT[base])] = 1.0
    return m


def pairing_involution(c_out: int) -> Tensor:
    """Output-channel representation: swap consecutive pairs (the Z/2 regular rep)."""
    if c_out % 2 != 0:
        raise ValueError(f"c_out must be even for the pairing involution, got {c_out}")
    m = torch.zeros(c_out, c_out)
    for k in range(0, c_out, 2):
        m[k, k + 1] = 1.0
        m[k + 1, k] = 1.0
    return m


def rc_action(x: Tensor, channel_rep: Tensor) -> Tensor:
    """Group action g . X = reverse over the length axis then complement the channels.

    x is [B, L, C]; channel_rep is the [C, C] involution for that channel space.
    """
    return torch.flip(x, dims=[1]) @ channel_rep


class RCEquivariantLinear(nn.Module):
    """Position-wise reverse-complement equivariant linear layer (Mallet and Vert 2021).

    Maps [B, L, c_in] -> [B, L, c_out] with weights tied so that f(g . x) = g . f(x),
    where g acts by reverse-complement on the input (complement permutation on c_in) and
    by the pairing involution on the output (c_out must be even).
    """

    def __init__(self, c_in: int = 4, c_out: int = 8, bias: bool = True) -> None:
        super().__init__()
        if c_in != len(ALPHABET):
            raise ValueError(f"c_in must be {len(ALPHABET)} (one-hot ACGU), got {c_in}")
        self.c_in, self.c_out = c_in, c_out
        self.register_buffer("M_in", complement_permutation())
        self.register_buffer("M_out", pairing_involution(c_out))
        self.W0 = nn.Parameter(torch.randn(c_in, c_out) * 0.1)
        self.b0 = nn.Parameter(torch.zeros(c_out)) if bias else None

    def equivariant_weight(self) -> Tensor:
        return 0.5 * (self.W0 + self.M_in @ self.W0 @ self.M_out)

    def equivariant_bias(self) -> Tensor | None:
        if self.b0 is None:
            return None
        return 0.5 * (self.b0 + self.b0 @ self.M_out)

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.equivariant_weight()
        b = self.equivariant_bias()
        return y if b is None else y + b

    def weight_tying_residual(self) -> float:
        """||M_in W - W M_out||_inf, the equivariance constraint residual (should be 0)."""
        w = self.equivariant_weight()
        return float(torch.max(torch.abs(self.M_in @ w - w @ self.M_out)))


def one_hot_guides(guides: list[str]) -> Tensor:
    """Encode equal-length RNA guides as a [B, L, 4] one-hot tensor over ACGU."""
    seqs = [g.strip().upper().replace("T", "U") for g in guides]
    if not seqs:
        raise ValueError("no guide sequences provided")
    length = len(seqs[0])
    if any(len(s) != length for s in seqs):
        raise ValueError("all guides must share one length for the one-hot RC check")
    idx = {b: i for i, b in enumerate(ALPHABET)}
    x = torch.zeros(len(seqs), length, len(ALPHABET))
    for b, s in enumerate(seqs):
        for j, base in enumerate(s):
            if base not in idx:
                raise ValueError(f"guide {b} has a non-ACGU base {base!r}")
            x[b, j, idx[base]] = 1.0
    return x


def equivariance_check(
    guides: list[str], c_out: int = 8, tol: float = 1e-10, seed: int = 0
) -> dict:
    """Empirically verify f(g . x) = g . f(x) on real guides, with a non-equivariant control."""
    torch.manual_seed(seed)
    x = one_hot_guides(guides).double()
    layer = RCEquivariantLinear(c_in=len(ALPHABET), c_out=c_out).double()

    with torch.no_grad():
        fx = layer(x)
        f_rc = layer(rc_action(x, layer.M_in))  # f(g . x)
        rc_f = rc_action(fx, layer.M_out)  # g . f(x)
        equiv_err = float(torch.max(torch.abs(f_rc - rc_f)))

        # RC is an involution on the inputs (sanity on the group action).
        involution_err = float(
            torch.max(torch.abs(rc_action(rc_action(x, layer.M_in), layer.M_in) - x))
        )

        # Non-equivariant control: a plain (untied) position-wise linear breaks equivariance.
        plain = nn.Linear(len(ALPHABET), c_out, bias=True).double()
        fxp = plain(x)
        ctrl_err = float(
            torch.max(torch.abs(plain(rc_action(x, layer.M_in)) - rc_action(fxp, layer.M_out)))
        )

    return {
        "reference": "Mallet and Vert 2021, reverse-complement equivariant networks for DNA",
        "n_guides": len(guides),
        "guide_len": int(x.shape[1]),
        "c_in": len(ALPHABET),
        "c_out": c_out,
        "dtype": "float64",
        "tolerance": tol,
        "max_equivariance_error": equiv_err,
        "equivariant": bool(equiv_err < tol),
        "weight_tying_residual": layer.weight_tying_residual(),
        "rc_involution_error": involution_err,
        "nonequivariant_control_error": ctrl_err,
        "control_breaks_equivariance": bool(ctrl_err > tol),
    }


def _real_guides(n: int = 64) -> list[str]:
    """First n real Huesken 2005 guide sequences (all 19nt) from data/huesken2005.csv."""
    path = REPO / "data" / "huesken2005.csv"
    if not path.exists():
        raise FileNotFoundError(f"real Huesken guides not present at {path}")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return [r["guide"].strip().upper() for r in rows[:n]]


def main() -> int:
    guides = _real_guides()
    result = equivariance_check(guides)
    out = REPO / "results" / "rc_equivariance_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[rc-equiv] {result['n_guides']} real {result['guide_len']}nt guides | "
        f"max equivariance error {result['max_equivariance_error']:.2e} "
        f"(< {result['tolerance']:g}: {result['equivariant']}) | weight-tying residual "
        f"{result['weight_tying_residual']:.2e} | control error {result['nonequivariant_control_error']:.2e}"
    )
    print(f"[rc-equiv] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
