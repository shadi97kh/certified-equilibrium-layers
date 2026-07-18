"""Tests for the thermodynamic feature module (ViennaRNA + NN fallback)."""

from __future__ import annotations

from cel.features import thermo
from cel.features.seedmatch import find_sites
from cel.features.thermo import (
    nn_duplex_mfe,
    seed_pairing_stability,
    target_accessibility,
    thermo_asymmetry,
)

GUIDE = "UGAGGUAGUAGGUUGUAUAGU"


def test_nn_duplex_mfe_gc_more_stable_than_au() -> None:
    # A GC-rich duplex is more stable (more negative dG) than an AU-rich one.
    assert nn_duplex_mfe("GGGGGG") < nn_duplex_mfe("AAAAAA")
    # Too-short strands contribute no stacking energy.
    assert nn_duplex_mfe("G") == 0.0


def test_thermo_asymmetry_sign() -> None:
    # Weak (AU) 5' end and strong (GC) 3' end => positive asymmetry score.
    guide = "AAUU" + "A" * 13 + "GGCC"
    dg5, dg3, asym = thermo_asymmetry(guide)
    assert dg5 > dg3  # 5' end less stable (less negative)
    assert asym > 0.0
    assert abs(asym - (dg5 - dg3)) < 1e-9


def test_seed_pairing_stability_finite_and_negative() -> None:
    words_target = "CCCAUAUA" + "CUACCUCA" + "GGGCAUUAAC"
    sites = find_sites(GUIDE, words_target)
    mfe, per_bp, supp = seed_pairing_stability(GUIDE, words_target, sites)
    assert mfe < 0.0
    assert abs(per_bp - mfe / 7.0) < 1e-9
    assert supp <= 0.0  # 0 when absent, negative when a supplementary match exists


def test_target_accessibility_in_range() -> None:
    target = "GGGCAUUAACCGGUUAAUGCCCAAAUUUGGGCCCAAAUUU"
    sites = find_sites(GUIDE, target)
    site = sites[0] if sites else None
    up, opening, flank = target_accessibility(target, site)
    assert 0.0 <= up <= 1.0
    assert 0.0 <= flank <= 1.0
    assert opening >= 0.0  # opening a structured site costs energy


def test_fallback_path_without_vienna(monkeypatch) -> None:
    # Force the ViennaRNA-free branch and confirm features stay well-defined.
    monkeypatch.setattr(thermo, "HAS_VIENNA", False)
    monkeypatch.setattr(thermo, "_RNA", None)
    target = "GGGCAUUAACCGGUUAAUGCCCAAAUUUGGG" + "CUACCUCA" + "GGG"
    sites = find_sites(GUIDE, target)
    mfe, per_bp, _supp = seed_pairing_stability(GUIDE, target, sites)
    assert mfe < 0.0  # NN duplex energy of the seed
    up, opening, flank = target_accessibility(target, sites[0] if sites else None)
    assert 0.0 <= up <= 1.0
    assert 0.0 <= flank <= 1.0
    assert opening >= 0.0


# --- REAL ViennaRNA biophysical inputs for T3 / T4 ---------------------------
GUIDE_T = "UGAGGUAGUAGGUUGUAUAGU"  # let-7 guide (seed positions 2-8 = GAGGUAG)
UTR_WITH_SITE = "GGGCAUUAACCGGUUAAUGCCC" + "CUACCUCA" + "GGGUUUAAACCC"  # CUACCUC = seed site


def test_seed_duplex_dg_real_is_negative_and_gc_stronger() -> None:
    import pytest

    pytest.importorskip("RNA")
    from cel.features.thermo import seed_duplex_dg

    dg = seed_duplex_dg(GUIDE_T, UTR_WITH_SITE)
    assert dg < 0.0  # a real seed:site duplex is favorable
    # A GC-rich guide seed against its complement binds more tightly than AU-rich.
    gc = seed_duplex_dg("AGGGCCGGGCCAAAAAAAAAA", "UUUUGGCCCGGCCCUUUUUUU")
    au = seed_duplex_dg("AAAUUUAAAUUUAAAUUUAAA", "UUUAAAUUUAAAUUUAAAUUU")
    assert gc < au


def test_thermo_input_vector_real() -> None:
    import numpy as np
    import pytest

    pytest.importorskip("RNA")
    from cel.features.thermo import thermo_input_vector

    e = thermo_input_vector(GUIDE_T, UTR_WITH_SITE)
    assert e.shape == (5,)
    assert np.isfinite(e).all()
    assert e[0] < 0.0  # seed duplex dG
    assert e[2] >= 0.0  # opening energy is a non-negative cost
    assert 0.0 <= e[3] <= 1.0 and 0.0 <= e[4] <= 1.0  # unpaired probabilities


def test_require_vienna_raises_with_install_command(monkeypatch) -> None:
    import pytest

    from cel import seed_dg
    from cel.features import thermo

    # Force the ViennaRNA-absent branch and confirm the real functions raise.
    monkeypatch.setattr(thermo, "_RNA", None)
    with pytest.raises(ImportError, match="ViennaRNA"):
        thermo.require_vienna()
    # seed_duplex_dg is promoted to cel.seed_dg (re-exported by thermo); its gate
    # lives there, so the ViennaRNA-absent branch is forced on the canonical module.
    monkeypatch.setattr(seed_dg, "_RNA", None)
    with pytest.raises(ImportError) as exc:
        thermo.seed_duplex_dg(GUIDE_T, UTR_WITH_SITE)
    assert "pip install ViennaRNA" in str(exc.value)


def test_crosscheck_targetscan_missing_table_raises(tmp_path) -> None:
    import pytest

    pytest.importorskip("RNA")
    from cel.features.thermo import crosscheck_targetscan

    with pytest.raises(FileNotFoundError, match="REAL DATA ONLY"):
        crosscheck_targetscan([(GUIDE_T, UTR_WITH_SITE)], tmp_path / "absent_targetscan.csv")
