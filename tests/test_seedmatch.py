"""Tests for the seed-match scanner."""

from __future__ import annotations

from cel.features import (
    extract_seed,
    find_sites,
    revcomp,
    seed_match_score,
    seed_site_words,
)
from cel.features.seedmatch import to_rna

# let-7-like 21 nt guide (antisense, 5' to 3'); seed (2-8) is GAGGUAG.
GUIDE = "UGAGGUAGUAGGUUGUAUAGU"


def test_extract_seed_is_positions_2_to_8() -> None:
    assert extract_seed(GUIDE) == "GAGGUAG"
    assert extract_seed(GUIDE) == to_rna(GUIDE)[1:8]


def test_revcomp_rna() -> None:
    assert revcomp("GAGGUAG") == "CUACCUC"
    # T is normalized to U on the way in.
    assert revcomp("GAGGTAG") == "CUACCUC"


def test_site_words_nesting() -> None:
    w = seed_site_words(GUIDE)
    assert w["6mer"] == "UACCUC"
    assert w["7mer-m8"] == "CUACCUC"
    assert w["8mer"] == w["7mer-m8"] + "A"
    assert w["7mer-A1"] == w["6mer"] + "A"
    assert w["offset-6mer"] == revcomp(to_rna(GUIDE)[2:8])


def test_find_8mer_site() -> None:
    words = seed_site_words(GUIDE)
    target = "CCCAUAUA" + words["8mer"] + "GGGCAUUAAC"
    sites = find_sites(GUIDE, target)
    best = sites[0]
    assert best.site_type == "8mer"
    assert best.grade == 1.0
    # Site spans the 8mer word we embedded.
    assert target[best.start : best.end] == words["8mer"]
    assert seed_match_score(GUIDE, target) == 1.0


def test_classify_7mer_m8_vs_a1() -> None:
    words = seed_site_words(GUIDE)
    # m8 extension without a 3' A => 7mer-m8 (guard with non-A neighbors).
    m8_target = "CCC" + words["7mer-m8"] + "GGG"
    assert find_sites(GUIDE, m8_target)[0].site_type == "7mer-m8"
    # A1 (6mer core + trailing A) but 5' neighbor breaks m8 => 7mer-A1.
    a1_target = "CCG" + words["7mer-A1"] + "GGG"
    assert find_sites(GUIDE, a1_target)[0].site_type == "7mer-A1"


def test_no_site() -> None:
    assert find_sites(GUIDE, "AAAAAAAAAAAAAAAAAAAA") == []
    assert seed_match_score(GUIDE, "AAAAAAAAAAAAAAAAAAAA") == 0.0


def test_short_guide_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="too short"):
        extract_seed("ACG")
