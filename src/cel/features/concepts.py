"""K=9 per-concept feature blocks for CASCADE.

The model's by-construction zero cross-concept leakage property (L_k = 0)
rests on the input features being partitioned into K = 9 concept channels whose
feature indices never overlap. This module builds exactly that partition: each
concept owns a contiguous, non-overlapping block of columns, and the layout is
exposed so a test can enforce disjointness.

The nine concepts (fixed order, fixed names):

    1. seed_complementarity   strength / quality of the best seed-match site
    2. seed_match_count       counts of canonical sites by type
    3. utr_site_context       sequence context around the best site in the UTR
    4. seed_pairing_stability seed:target duplex thermodynamics (ViennaRNA)
    5. target_accessibility   site unpaired probability / opening energy
    6. on_target_potency      guide intrinsic-efficacy position rules
    7. position_bias          per-seed-position complementarity profile
    8. gc_content             GC composition of guide / seed / site / UTR
    9. thermo_asymmetry       guide duplex end-stability asymmetry

Concepts 1, 2, 3, 7 derive from :mod:`cel.features.seedmatch`; concepts 4,
5, 9 from :mod:`cel.features.thermo`; concepts 6, 8 from guide / target
composition. No raw quantity is shared across two concept blocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import seedmatch, thermo
from .seedmatch import Site, best_site, find_sites, to_rna

# Ordered (concept_name, [feature_names]) spec. The width of each block is the
# length of its feature-name list; blocks are concatenated in this order.
CONCEPT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "seed_complementarity",
        (
            "sc_best_site_grade",
            "sc_site_types_present_frac",
            "sc_best_site_len_norm",
            "sc_has_canonical",
        ),
    ),
    (
        "seed_match_count",
        ("smc_n_8mer", "smc_n_7mer", "smc_n_6mer", "smc_n_total"),
    ),
    (
        "utr_site_context",
        (
            "usc_local_au_frac",
            "usc_dist_to_5p_norm",
            "usc_dist_to_3p_norm",
            "usc_site_density_per100",
        ),
    ),
    (
        "seed_pairing_stability",
        ("sps_seed_duplex_mfe", "sps_seed_duplex_mfe_per_bp", "sps_supp_pairing_mfe"),
    ),
    (
        "target_accessibility",
        ("tacc_site_unpaired_prob", "tacc_opening_energy", "tacc_flank_unpaired_prob"),
    ),
    (
        "on_target_potency",
        (
            "otp_ui_tei_au_5p_antisense",
            "otp_ui_tei_gc_5p_sense",
            "otp_ui_tei_au_rich_seed",
            "otp_ui_tei_no_gc_stretch",
            "otp_reynolds_score",
        ),
    ),
    (
        "position_bias",
        ("pb_pos2", "pb_pos3", "pb_pos4", "pb_pos5", "pb_pos6", "pb_pos7", "pb_pos8"),
    ),
    (
        "gc_content",
        ("gc_guide_frac", "gc_seed_frac", "gc_site_frac", "gc_utr_frac"),
    ),
    (
        "thermo_asymmetry",
        ("tasym_dg_5p_end", "tasym_dg_3p_end", "tasym_asym_score"),
    ),
)

CONCEPTS: tuple[str, ...] = tuple(name for name, _ in CONCEPT_SPECS)
K: int = len(CONCEPTS)
assert K == 9, "CASCADE fixes K = 9 concept channels"


@dataclass(frozen=True)
class ConceptLayout:
    """Index layout of the concept partition (data-independent)."""

    feature_names: list[str]
    concept_slices: dict[str, slice]
    total_dim: int

    def concept_indices(self) -> dict[str, list[int]]:
        return {name: list(range(sl.start, sl.stop)) for name, sl in self.concept_slices.items()}


def concept_layout() -> ConceptLayout:
    """Return the fixed concept partition layout (no feature computation)."""
    names: list[str] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for concept, feats in CONCEPT_SPECS:
        slices[concept] = slice(cursor, cursor + len(feats))
        names.extend(feats)
        cursor += len(feats)
    return ConceptLayout(feature_names=names, concept_slices=slices, total_dim=cursor)


LAYOUT = concept_layout()


@dataclass(frozen=True)
class ConceptFeatures:
    """Assembled per-concept features for a batch of (guide, target) pairs.

    ``matrix`` is the concatenated [N, total_dim] array. ``blocks`` holds the K
    separate, non-overlapping per-concept matrices (this separation is what
    later guarantees L_k = 0).
    """

    matrix: np.ndarray
    blocks: dict[str, np.ndarray]
    layout: ConceptLayout

    def block(self, concept: str) -> np.ndarray:
        return self.blocks[concept]

    def to_torch_blocks(self) -> dict[str, object]:
        """Return the K per-concept matrices as torch tensors (lazy import)."""
        import torch

        return {
            name: torch.from_numpy(np.ascontiguousarray(b)).float()
            for name, b in self.blocks.items()
        }


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(b in ("G", "C") for b in seq) / len(seq)


def _longest_gc_run(seq: str) -> int:
    best = run = 0
    for b in seq:
        run = run + 1 if b in ("G", "C") else 0
        best = max(best, run)
    return best


def _seed_complementarity_block(sites: list[Site]) -> list[float]:
    b = best_site(sites)
    if b is None:
        return [0.0, 0.0, 0.0, 0.0]
    types_present = {s.site_type for s in sites}
    return [
        b.grade,
        len(types_present) / len(seedmatch.SITE_TYPES),
        (b.end - b.start) / 8.0,
        1.0,
    ]


def _seed_match_count_block(sites: list[Site]) -> list[float]:
    n_8 = sum(s.site_type == "8mer" for s in sites)
    n_7 = sum(s.site_type in ("7mer-m8", "7mer-A1") for s in sites)
    n_6 = sum(s.site_type in ("6mer", "offset-6mer") for s in sites)
    return [float(n_8), float(n_7), float(n_6), float(len(sites))]


def _utr_site_context_block(target: str, sites: list[Site]) -> list[float]:
    n = len(target)
    b = best_site(sites)
    if b is None or n == 0:
        return [
            _au_fraction(target),
            0.5,
            0.5,
            (len(sites) / n * 100.0) if n else 0.0,
        ]
    center = (b.start + b.end) // 2
    flank = target[max(0, center - 30) : min(n, center + 30)]
    return [
        _au_fraction(flank),
        b.start / n,
        (n - b.end) / n,
        len(sites) / n * 100.0,
    ]


def _on_target_potency_block(guide: str) -> list[float]:
    g = guide
    seed = g[1:8]
    au_5p = 1.0 if g and g[0] in ("A", "U") else 0.0
    gc_5p_sense = 1.0 if g and g[-1] in ("G", "C") else 0.0
    au_rich_seed = 1.0 if sum(b in ("A", "U") for b in seed) >= 4 else 0.0
    no_gc_stretch = 1.0 if _longest_gc_run(g) <= 9 else 0.0
    # Reynolds-style position criteria (composition / GC handled by gc_content).
    crit = [
        len(g) > 18 and g[18] == "A",
        len(g) > 2 and g[2] == "A",
        len(g) > 9 and g[9] == "U",
        len(g) > 12 and g[12] != "G",
        bool(g) and g[-1] in ("A", "U"),
    ]
    reynolds = sum(crit) / len(crit)
    return [au_5p, gc_5p_sense, au_rich_seed, no_gc_stretch, reynolds]


def _position_bias_block(guide: str, target: str) -> list[float]:
    # Best length-7 window in the target by Watson-Crick complement to the seed.
    seed7 = guide[1:8]  # positions 2 to 8
    ideal = seedmatch.revcomp(seed7)  # the perfect 7mer site (5' to 3')
    if len(target) < 7 or len(ideal) < 7:
        return [0.0] * 7
    best_matches: list[int] | None = None
    best_score = -1
    for i in range(len(target) - 6):
        window = target[i : i + 7]
        matches = [int(window[k] == ideal[k]) for k in range(7)]
        score = sum(matches)
        if score > best_score:
            best_score = score
            best_matches = matches
    # ideal[k] pairs guide position (8 - k); reorder to seed positions 2..8.
    assert best_matches is not None
    return [float(best_matches[8 - p]) for p in range(2, 9)]


def _gc_content_block(guide: str, target: str, sites: list[Site]) -> list[float]:
    b = best_site(sites)
    site_word = target[b.start : b.end] if b is not None else ""
    return [
        _gc_fraction(guide),
        _gc_fraction(guide[1:8]),
        _gc_fraction(site_word),
        _gc_fraction(target),
    ]


def _au_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(b in ("A", "U") for b in seq) / len(seq)


def featurize_pair(guide: str, target: str) -> dict[str, np.ndarray]:
    """Compute the K per-concept feature blocks for one (guide, target) pair."""
    g = to_rna(guide)
    t = to_rna(target)
    sites = find_sites(g, t)
    b = best_site(sites)

    sps = thermo.seed_pairing_stability(g, t, sites)
    tacc = thermo.target_accessibility(t, b)
    tasym = thermo.thermo_asymmetry(g)

    blocks: dict[str, list[float]] = {
        "seed_complementarity": _seed_complementarity_block(sites),
        "seed_match_count": _seed_match_count_block(sites),
        "utr_site_context": _utr_site_context_block(t, sites),
        "seed_pairing_stability": list(sps),
        "target_accessibility": list(tacc),
        "on_target_potency": _on_target_potency_block(g),
        "position_bias": _position_bias_block(g, t),
        "gc_content": _gc_content_block(g, t, sites),
        "thermo_asymmetry": list(tasym),
    }
    out: dict[str, np.ndarray] = {}
    for concept, feats in CONCEPT_SPECS:
        vec = np.asarray(blocks[concept], dtype=np.float64)
        if vec.shape != (len(feats),):
            raise ValueError(f"concept {concept} produced {vec.shape}, expected ({len(feats)},)")
        if not np.isfinite(vec).all():
            raise ValueError(f"concept {concept} produced non-finite features")
        out[concept] = vec
    return out


def featurize_pairs(pairs: Sequence[tuple[str, str]]) -> ConceptFeatures:
    """Featurize a batch of (guide, target) pairs into per-concept blocks.

    Returns a :class:`ConceptFeatures` with the concatenated matrix and the K
    separate per-concept matrices.
    """
    per_concept: dict[str, list[np.ndarray]] = {name: [] for name in CONCEPTS}
    for guide, target in pairs:
        one = featurize_pair(guide, target)
        for name in CONCEPTS:
            per_concept[name].append(one[name])

    n = len(pairs)
    blocks: dict[str, np.ndarray] = {}
    for concept, feats in CONCEPT_SPECS:
        if n:
            blocks[concept] = np.vstack(per_concept[concept])
        else:
            blocks[concept] = np.empty((0, len(feats)), dtype=np.float64)

    matrix = (
        np.hstack([blocks[name] for name in CONCEPTS])
        if n
        else np.empty((0, LAYOUT.total_dim), dtype=np.float64)
    )
    return ConceptFeatures(matrix=matrix, blocks=blocks, layout=LAYOUT)
