"""Seed-duplex free energy (ViennaRNA), promoted to a dedicated committed module.

:func:`seed_duplex_dg` hybridizes the guide SEED (nucleotides 2-8, the canonical
siRNA / miRNA seed) to its best site in a 3'UTR with ``RNA.duplexfold`` (Turner
nearest-neighbor parameters) and returns the duplex free energy in kcal/mol. A more
negative dG is a stronger seed:target duplex.

This is the canonical home for the seed-duplex dG that feeds the REAL theorem
inputs: the Theorem 3 mass-action / thermodynamic constraint (dG -> knockdown via a
Langmuir occupancy law) and the Theorem 4 off-target input e. It was previously
inlined in :mod:`cel.features.thermo`; that module now re-exports this function
so existing call sites are unchanged.

Sign convention (confirmed by ``exp_t3_constraint.vienna_sign_convention``):
``RNA.duplexfold`` returns the hybridization MFE directly, NEGATIVE for a favorable
duplex. The value is used as-is, with no sign flip.

ViennaRNA is REQUIRED to call :func:`seed_duplex_dg` (it RAISES with the install
command when absent), but it is imported under a ``try`` / ``except`` so that merely
IMPORTING this module never needs ViennaRNA: the lightweight (numpy / scipy) import
path stays ViennaRNA-free.
"""

from __future__ import annotations

try:  # ViennaRNA is gated: importing this module never requires it.
    import RNA as _RNA

    HAS_VIENNA = True
except ImportError:  # pragma: no cover - exercised only in ViennaRNA-free envs
    _RNA = None
    HAS_VIENNA = False

VIENNA_INSTALL = "pip install ViennaRNA==2.6.4  (or: conda install -c bioconda viennarna)"

# Canonical siRNA / miRNA seed = guide nucleotides 2-8 (1-indexed, inclusive),
# i.e. the 0-indexed slice guide[1:8].
SEED_START = 1
SEED_LEN = 7


def require_vienna() -> None:
    """Raise ImportError with the install command if ViennaRNA is unavailable."""
    if _RNA is None:
        raise ImportError(
            "ViennaRNA is required for the real seed-duplex free energy used by Theorem 3 / "
            f"Theorem 4, but it is not installed. {VIENNA_INSTALL}"
        )


def seed_duplex_dg(
    guide: str, utr: str, seed_start: int = SEED_START, seed_len: int = SEED_LEN
) -> float:
    """Seed (guide nt 2-8) duplex free energy dG (kcal/mol) vs its best 3'UTR site.

    The guide seed (positions ``seed_start+1`` .. ``seed_start+seed_len``, default 2
    to 8) is hybridized to its best site in the 3'UTR with ``RNA.duplexfold``. A more
    negative dG is a stronger seed:target duplex (MFE used directly, no sign flip).
    Raises ImportError if ViennaRNA is absent and ValueError if the seed or UTR is
    shorter than a single base pair.
    """
    from cel.features.seedmatch import to_rna

    require_vienna()
    seed = to_rna(guide)[seed_start : seed_start + seed_len]
    target = to_rna(utr)
    if len(seed) < 2 or len(target) < 2:
        raise ValueError("guide seed and 3'UTR must each be at least 2 nt")
    return float(_RNA.duplexfold(seed, target).energy)
