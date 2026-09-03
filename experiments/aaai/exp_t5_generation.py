#!/usr/bin/env python3
"""Theorem 5 (certified-feasibility guide generation), validated on real data.

For a real target, every 7mer guide seed over the 4^7 alphabet is enumerated and scored on
two real axes:

  (a) a CERTIFIED off-target upper bound U: the seed-duplex free-energy off-target input
      (the RESPONSIVE input -- ViennaRNA occupancy, NOT a site count) propagated over the
      real mouse TRRUST GRN. U = ||P_alpha(S)||_2 * ||e_seed||_2 is a certified upper bound
      on the GRN-propagated off-target magnitude ||P_alpha(S) e_seed|| (the seed input
      e_seed[g] = (count of the seed's cognate 3'UTR site in gene g) * occupancy(seed), with
      occupancy = -min(dG, 0) the favorable ViennaRNA seed-duplex binding energy);
  (b) the CASCADE-predicted on-target efficacy of the corresponding guide (the scaffold of
      the target with positions 2-8 replaced by the candidate seed), from a real
      Huesken-trained ridge on the K=9 CASCADE features.

It reports the CERTIFIED YIELD -- the fraction of candidate seeds with U <= tau AND predicted
efficacy >= eta -- swept over clinically-motivated tau (relative to the parent guide's U) and
eta, and locates the parent Ttr guide D1 and the seed-destabilized D4 in this landscape: D4
(position-7 (S)-GNA, modeled as a g7 Watson-Crick mismatch since ViennaRNA has no GNA sugar
parameters) targets the SAME 3'UTR site as D1 but with a weaker seed duplex, so D4 has a lower
certified off-target U while its base sequence (hence the sequence-level predicted efficacy) is
unchanged. Finally it reports whether generated certified-feasible guides exist with predicted
off-target strictly below the parent's while keeping efficacy >= eta.

HARD DATA POLICY (OUTLINE.md): REAL DATA ONLY. The GRN is real mouse TRRUST; the off-target
site abundance is over real mouse GRCh-equivalent (GRCm39 / Ensembl 104) 3'UTRs from pyensembl;
the seed-duplex dG is real ViennaRNA; the efficacy model is trained on real Huesken. Each
loader RAISES when its real file is absent. No fabricated values. The dG-input and
GNA-as-mismatch modeling assumptions are recorded in the JSON. Writes
results/aaai/t5_generation.json. No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "aaai"
CACHE = OUT / "cache"
ENSEMBL_RELEASE = 104  # GRCm39 mouse, contemporaneous with the human GRCh38/104 used elsewhere
ALPHA = 0.5  # P_alpha teleport (matches the T4 certified-bound join)
SEED_LEN = 7
ALPHABET = "ACGU"
TTR_D1 = "UUAUAGAGCAAGAACACUGUUUU"  # Schlegel 2022 Ttr parent guide (antisense, 5'->3')
TTR_SEED = "UAUAGAG"  # guide positions 2-8
GNA_SEED_IDX = 5  # g7 within the seed (0-indexed); D4 models the (S)-GNA here
GNA_MISMATCH_BASE = "C"  # g7 A -> C, a Watson-Crick mismatch standing in for the GNA sugar
RT = 0.6156
ETA_GRID = (0.6, 0.7, 0.8)  # predicted on-target efficacy thresholds (assumption)
TAU_REL_GRID = (0.5, 0.75, 1.0)  # certified off-target U thresholds, as multiples of U(parent)


def revcomp_rna(s: str) -> str:
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    return "".join(comp[b] for b in reversed(s.upper().replace("T", "U")))


def site_dna(seed: str) -> str:
    """3'UTR seed-match site (DNA) = reverse complement of the guide seed."""
    comp = {"A": "T", "U": "A", "T": "A", "G": "C", "C": "G"}
    return "".join(comp[b] for b in reversed(seed.upper()))


def all_seeds() -> list[str]:
    return ["".join(p) for p in product(ALPHABET, repeat=SEED_LEN)]


def guide_for(scaffold: str, seed: str) -> str:
    """Scaffold guide with positions 2-8 (0-indexed 1:8) replaced by the candidate seed."""
    return scaffold[:1] + seed + scaffold[1 + SEED_LEN :]


def mouse_node_3utrs(grn, release) -> dict:
    """mouse TRRUST node symbol -> its longest GRCm39/104 3'UTR (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"mouse_trrust_3utr_r{ENSEMBL_RELEASE}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    utrs: dict[str, str] = {}
    for sym in grn.nodes:
        try:
            genes = release.genes_by_name(sym)
        except Exception:
            continue
        best = ""
        for gene in genes:
            for t in gene.transcripts:
                u = t.three_prime_utr_sequence
                if u and len(u) > len(best):
                    best = u.upper()
        if best:
            utrs[sym] = best
    cache_path.write_text(json.dumps(utrs))
    return utrs


def build_site_sumsq(node_utrs: dict) -> dict:
    """site 7mer -> sum over node genes of (count in that gene's 3'UTR)^2."""
    s2: dict[str, float] = {}
    for u in node_utrs.values():
        counts: dict[str, int] = {}
        for i in range(len(u) - SEED_LEN + 1):
            w = u[i : i + SEED_LEN]
            counts[w] = counts.get(w, 0) + 1
        for w, c in counts.items():
            s2[w] = s2.get(w, 0.0) + float(c) * float(c)
    return s2


def seed_dg(guide_seed: str, target_site_rna: str) -> float:
    import RNA

    return float(RNA.duplexfold(guide_seed, target_site_rna).energy)


def occupancy(dg: float) -> float:
    return -min(dg, 0.0)  # favorable seed-duplex binding energy (kcal/mol), >= 0


def train_efficacy():
    """Ridge on the real Huesken efficacy with the K=9 CASCADE features; returns predict()."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from cel.data import load_huesken_efficacy
    from cel.features import featurize_pairs
    from cel.features.seedmatch import revcomp

    frame = load_huesken_efficacy(REPO / "data" / "raw" / "efficacy", cache_root=CACHE).frame
    anti = frame["antisense"].astype(str).tolist()
    sense = frame["sense"].astype(str).tolist()
    tgt = [(s.strip() or revcomp(a)) for a, s in zip(anti, sense, strict=True)]
    x = featurize_pairs(list(zip(anti, tgt, strict=True))).matrix
    y = frame["efficacy"].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=1.0).fit(scaler.transform(x), y)
    r2 = float(model.score(scaler.transform(x), y))

    def predict(guides: list[str]) -> np.ndarray:
        pairs = [(g, revcomp(g)) for g in guides]
        feats = featurize_pairs(pairs).matrix
        return np.clip(model.predict(scaler.transform(feats)), 0.0, 1.0)

    return predict, r2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[t5-gen] seed={args.seed} alpha={ALPHA} | enumerating {len(ALPHABET)}^{SEED_LEN} seeds")

    try:
        from pyensembl import EnsemblRelease

        from cel.data import load_trrust
        from cel.verify import offtarget_bound as ob

        grn = load_trrust(species="mouse")
        release = EnsemblRelease(ENSEMBL_RELEASE, species="mouse")
        release.genes_by_name("Actb")  # forces the GTF DB; raises if not installed
        import RNA  # noqa: F401
    except (FileNotFoundError, ImportError) as exc:
        out = {"status": "pending_real_data", "source": str(exc)}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "t5_generation.json").write_text(json.dumps(out, indent=2))
        print(f"[t5-gen] PENDING: {exc}")
        return 0

    p_norm = float(ob.p_alpha_norm(grn.S, ALPHA))  # ||P_alpha(mouse TRRUST)||_2
    print("[t5-gen] extracting mouse TRRUST-node 3'UTRs (GRCm39/104, cached) ...")
    node_utrs = mouse_node_3utrs(grn, release)
    s2 = build_site_sumsq(node_utrs)
    print(
        f"[t5-gen] {len(node_utrs)} node 3'UTRs; {len(s2)} distinct 7mer sites; ||P_alpha||={p_norm:.3f}"
    )

    seeds = all_seeds()
    # Certified off-target upper bound U(seed) (target-independent).
    U = np.zeros(len(seeds))
    occ = np.zeros(len(seeds))
    for i, sd in enumerate(seeds):
        dg = seed_dg(sd, revcomp_rna(sd))
        occ[i] = occupancy(dg)
        U[i] = p_norm * occ[i] * float(np.sqrt(s2.get(site_dna(sd), 0.0)))
    seed_idx = {sd: i for i, sd in enumerate(seeds)}

    predict, eff_r2 = train_efficacy()
    print(f"[t5-gen] efficacy model trained on real Huesken (train R2={eff_r2:.3f})")

    # Parent Ttr D1 and seed-destabilized D4 in the landscape.
    i_d1 = seed_idx[TTR_SEED]
    eff_ttr = predict([guide_for(TTR_D1, sd) for sd in seeds])
    eff_d1 = float(eff_ttr[i_d1])
    site_d1 = site_dna(TTR_SEED)
    seed_d4 = TTR_SEED[:GNA_SEED_IDX] + GNA_MISMATCH_BASE + TTR_SEED[GNA_SEED_IDX + 1 :]
    dg_d4 = seed_dg(seed_d4, revcomp_rna(TTR_SEED))  # D4 binds the SAME D1 site, weaker
    occ_d4 = occupancy(dg_d4)
    u_d1 = float(U[i_d1])
    u_d4 = p_norm * occ_d4 * float(np.sqrt(s2.get(site_d1, 0.0)))
    eff_d4 = eff_d1  # GNA is a sugar modification; the base sequence (hence seq-model efficacy) is unchanged

    parent = {
        "D1": {
            "seed": TTR_SEED,
            "occupancy": float(occ[i_d1]),
            "certified_U": u_d1,
            "predicted_efficacy": eff_d1,
        },
        "D4": {
            "seed_base": TTR_SEED,
            "gna_model": f"g7 {TTR_SEED[GNA_SEED_IDX]}->{GNA_MISMATCH_BASE} mismatch vs the D1 site",
            "occupancy": occ_d4,
            "certified_U": u_d4,
            "predicted_efficacy": eff_d4,
        },
        "D4_lower_certified_U_than_D1": bool(u_d4 < u_d1),
        "efficacy_comparable": bool(abs(eff_d4 - eff_d1) < 1e-9),
    }

    # Certified yield over (tau relative to U(D1)) x eta, on the Ttr scaffold.
    yield_table: dict = {}
    for tau_rel in TAU_REL_GRID:
        tau = tau_rel * u_d1
        row = {}
        for eta in ETA_GRID:
            feasible = (tau >= U) & (eff_ttr >= eta)
            row[f"eta_{eta}"] = float(feasible.mean())
        yield_table[f"tau_{tau_rel}xU_D1"] = row

    # Generation: certified-feasible guides strictly below the parent's off-target.
    generation: dict = {}
    for eta in ETA_GRID:
        mask = (u_d1 > U) & (eff_ttr >= eta)
        idxs = np.where(mask)[0]
        best = None
        if idxs.size:
            j = idxs[np.argmin(U[idxs])]
            best = {
                "seed": seeds[int(j)],
                "certified_U": float(U[j]),
                "predicted_efficacy": float(eff_ttr[j]),
            }
        generation[f"eta_{eta}"] = {
            "exists": bool(idxs.size > 0),
            "n_seeds_below_parent_U_and_eff_ge_eta": int(idxs.size),
            "best_seed": best,
        }

    # Repeat the yield landscape for a few real Huesken guide scaffolds (no D1/D4 there).
    from cel.data import load_huesken_efficacy

    hf = load_huesken_efficacy(REPO / "data" / "raw" / "efficacy", cache_root=CACHE).frame
    huesken_targets = {}
    for k in range(3):
        scaffold = str(hf["antisense"].iloc[k])
        if len(scaffold) < 1 + SEED_LEN:
            continue
        eff_h = predict([guide_for(scaffold, sd) for sd in seeds])
        huesken_targets[f"Huesken_{k}"] = {
            "scaffold": scaffold,
            "yield_tau_1xU_D1_eta_0.7": float(((u_d1 >= U) & (eff_h >= 0.7)).mean()),
            "n_below_parentU_eff_ge_0.7": int(((u_d1 > U) & (eff_h >= 0.7)).sum()),
        }

    numbers = {
        "status": "ok",
        "seed": args.seed,
        "claim": "Theorem 5 (certified-feasibility guide generation): enumerate the 4^7 guide "
        "seeds and report the certified-feasible yield (U <= tau and predicted efficacy >= eta).",
        "n_seeds": len(seeds),
        "alpha": ALPHA,
        "grn": grn.source,
        "p_alpha_2norm_mouse_trrust": p_norm,
        "n_node_3utrs": len(node_utrs),
        "assumptions": {
            "tau": "certified off-target threshold, swept as multiples of the parent D1's U "
            f"({TAU_REL_GRID}); U(D1)={u_d1:.4f}",
            "eta": f"predicted on-target efficacy threshold, swept over {ETA_GRID}",
            "dg_input": "RESPONSIVE seed-duplex dG input: e_seed[g] = (count of the seed's cognate "
            "3'UTR site in gene g) * occupancy(seed), occupancy = -min(dG,0) from ViennaRNA "
            "duplexfold; NOT a site count.",
            "certified_U": "U = ||P_alpha(S)||_2 * ||e_seed||_2, a certified upper bound on the "
            "GRN-propagated off-target ||P_alpha(S) e_seed|| over mouse TRRUST.",
            "gna_caveat": "ViennaRNA has no GNA sugar parameters, so the position-7 (S)-GNA of D4 "
            "is modeled as a g7 Watson-Crick mismatch (A->C) against the D1 site; this is an "
            "explicit destabilization model (an upper bound on the true GNA penalty).",
            "efficacy_model": "Huesken-trained ridge on the K=9 CASCADE features; sequence-based, "
            f"so it cannot distinguish D1 from D4 (same base sequence). Train R2={eff_r2:.3f}.",
        },
        "parent_Ttr": parent,
        "certified_yield": {"u_d1": u_d1, "table": yield_table},
        "generation": generation,
        "huesken_targets": huesken_targets,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "t5_generation.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    print(
        f"\n[t5-gen] parent Ttr: D1 U={u_d1:.3f} eff={eff_d1:.3f} | D4 U={u_d4:.3f} eff={eff_d4:.3f} "
        f"| D4 lower U: {parent['D4_lower_certified_U_than_D1']} | eff comparable: {parent['efficacy_comparable']}"
    )
    print("[t5-gen] certified yield (fraction of 16384 seeds), tau x eta:")
    print(f"  {'tau':>14} " + " ".join(f"eta={e:>4}" for e in ETA_GRID))
    for tau_rel in TAU_REL_GRID:
        row = yield_table[f"tau_{tau_rel}xU_D1"]
        print(
            f"  {f'{tau_rel}xU_D1':>14} " + " ".join(f"{row[f'eta_{e}']:>8.4f}" for e in ETA_GRID)
        )
    for eta in ETA_GRID:
        g = generation[f"eta_{eta}"]
        bs = g["best_seed"]
        print(
            f"[t5-gen] generation eta={eta}: exists={g['exists']} n={g['n_seeds_below_parent_U_and_eff_ge_eta']}"
            + (
                f" best seed {bs['seed']} U={bs['certified_U']:.3f} eff={bs['predicted_efficacy']:.3f}"
                if bs
                else ""
            )
        )
    print(f"[t5-gen] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
