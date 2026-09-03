"""Seed extraction and seed-match scanning for siRNA guides.

The seed logic follows the canonical miRNA / siRNA seed-site taxonomy used by
TargetScan (Lewis et al. 2005; Agarwal et al. 2015, eLife) and reimplemented in
scanMiR (Soutschek et al. 2022, Bioinformatics) and SeedMatchR (the IDT siRNA
off-target package). For a guide (antisense) strand read 5' to 3', the seed is
positions 2 to 8. A candidate 3'UTR target is scanned for Watson-Crick
complementary sites of decreasing strength:

    8mer        positions 2 to 8 match, plus an A opposite guide position 1
    7mer-m8     positions 2 to 8 match
    7mer-A1     positions 2 to 7 match, plus an A opposite guide position 1
    6mer        positions 2 to 7 match
    offset-6mer positions 3 to 8 match

Orientation: a guide read 5' to 3' pairs the target antiparallel, so the seed
site word in the target (read 5' to 3') is the reverse complement of the seed
region. The m8 extension sits on the 5' side of the target site and the A1
position on the 3' side. We reimplement the scan here (no shelling out) so the
feature layer has no R dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

# Strength grades for the canonical site types (monotone with affinity, in the
# TargetScan / scanMiR ordering). Used by the seed_complementarity concept.
SITE_GRADE: dict[str, float] = {
    "8mer": 1.0,
    "7mer-m8": 0.8,
    "7mer-A1": 0.7,
    "6mer": 0.5,
    "offset-6mer": 0.4,
}
SITE_TYPES: tuple[str, ...] = ("8mer", "7mer-m8", "7mer-A1", "6mer", "offset-6mer")

_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}

# Seed is guide positions 2 to 8 (1-indexed, inclusive).
SEED_START = 2
SEED_END = 8


@dataclass(frozen=True)
class Site:
    """A seed-match site in a target transcript (target coordinates, 0-indexed)."""

    site_type: str
    start: int
    end: int  # exclusive
    grade: float


def to_rna(seq: str) -> str:
    """Upper-case, strip whitespace, and map DNA T to RNA U."""
    return str(seq).strip().upper().replace(" ", "").replace("T", "U")


def complement(base: str) -> str:
    return _COMPLEMENT.get(base, "N")


def revcomp(seq: str) -> str:
    """Reverse complement of a nucleotide string (DNA T is read as RNA U; unknown bases map to N)."""
    return "".join(complement(b) for b in reversed(to_rna(seq)))


def extract_seed(guide: str, start: int = SEED_START, end: int = SEED_END) -> str:
    """Return the seed substring (guide positions ``start``..``end``, 1-indexed)."""
    g = to_rna(guide)
    if len(g) < end:
        raise ValueError(f"guide too short ({len(g)} nt) to extract seed positions {start}-{end}")
    return g[start - 1 : end]


def seed_site_words(guide: str) -> dict[str, str]:
    """Return the target-strand (5' to 3') word for each canonical site type."""
    g = to_rna(guide)
    if len(g) < SEED_END:
        raise ValueError(f"guide too short ({len(g)} nt) for seed scanning")
    core6 = revcomp(g[1:7])  # positions 2 to 7
    core7 = revcomp(g[1:8])  # positions 2 to 8
    offset = revcomp(g[2:8])  # positions 3 to 8
    return {
        "8mer": core7 + "A",
        "7mer-m8": core7,
        "7mer-A1": core6 + "A",
        "6mer": core6,
        "offset-6mer": offset,
    }


def _find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    out: list[int] = []
    i = haystack.find(needle)
    while i != -1:
        out.append(i)
        i = haystack.find(needle, i + 1)
    return out


def find_sites(guide: str, target: str) -> list[Site]:
    """Find all canonical seed-match sites of ``guide`` in ``target``.

    Each 6mer-core occurrence is classified as the strongest type it supports
    (m8 extension on the 5' side, A1 on the 3' side). offset-6mer sites are
    reported only where they do not overlap a 6mer-core site.
    """
    g = to_rna(guide)
    t = to_rna(target)
    if len(g) < SEED_END or len(t) < 6:
        return []

    core6 = revcomp(g[1:7])
    comp_g8 = complement(g[7])  # nucleotide a 7mer-m8 needs on the 5' side
    sites: list[Site] = []
    covered: set[int] = set()

    for i in _find_all(t, core6):
        # core6 spans t[i : i+6]; its 5' end pairs guide position 7.
        has_m8 = i - 1 >= 0 and t[i - 1] == comp_g8
        has_a1 = i + 6 < len(t) and t[i + 6] == "A"
        if has_m8 and has_a1:
            stype, start, end = "8mer", i - 1, i + 7
        elif has_m8:
            stype, start, end = "7mer-m8", i - 1, i + 6
        elif has_a1:
            stype, start, end = "7mer-A1", i, i + 7
        else:
            stype, start, end = "6mer", i, i + 6
        sites.append(Site(stype, start, end, SITE_GRADE[stype]))
        covered.update(range(start, end))

    offset = revcomp(g[2:8])
    for i in _find_all(t, offset):
        span = range(i, i + 6)
        if any(p in covered for p in span):
            continue
        sites.append(Site("offset-6mer", i, i + 6, SITE_GRADE["offset-6mer"]))

    sites.sort(key=lambda s: (-s.grade, s.start))
    return sites


def best_site(sites: list[Site]) -> Site | None:
    """Return the strongest (then 5'-most) site, or None when there are none."""
    return sites[0] if sites else None


def seed_match_score(guide: str, target: str) -> float:
    """Best canonical site grade in [0, 1] (0 when no seed match is present)."""
    b = best_site(find_sites(guide, target))
    return b.grade if b is not None else 0.0
