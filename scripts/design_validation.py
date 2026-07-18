"""Validate the certified-optimal minimal seed-modification design on real siRNA off-target data.

Question: does exact enumeration over single-position seed modifications recover the
EXPERIMENTALLY optimal position (g7 for Schlegel, position 2 for Jackson)?

Real data only, no synthetic fallback, no em-dashes. Reuses solve_f (coupled single-pool
equilibrium) from scripts/offtarget_phase_probe.py in the mirna_baselines project.

scanMiR status: scanMiR 1.16.0 + scanMiRData ARE installed, but they cannot produce per-site Kd
for these NOVEL siRNA seeds: no scanMiRData KdModel shares the Ttr seed, and getKdModel cannot
build a model from sequence alone (it needs the McGeary-CNN 12-mer Kd table, which is not bundled).
This does NOT compromise the verdict, because the verdict is affinity-INDEPENDENT: it is fixed by
the position-to-site incidence, and canonical seed sites make seed positions 3-7 have IDENTICAL
incidence (every canonical window includes 3-7), so g({3})=...=g({7}) for ANY K and delta. The
site-type K used below only sets the common magnitude; it is stated as such and does not decide the
ranking of the core positions. The incidence is built from REAL seed complementarity (validated
cel.features.seedmatch on real 3'UTRs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, "/home/shadi/mirna_baselines/scripts")
from offtarget_phase_probe import solve_f  # noqa: E402

RESULTS = REPO / "results"
OUT = RESULTS / "design_validation.log"
RAT_UTR = REPO / "data" / "raw" / "reference" / "rat_3utr.fa"
SM_DIR = REPO / "data" / "raw" / "offtarget" / "seedmatchr"

# Seed positions 2..8; canonical site-type pairing windows (guide positions).
WINDOWS = {
    "8mer": {2, 3, 4, 5, 6, 7, 8},
    "7mer-m8": {2, 3, 4, 5, 6, 7, 8},
    "7mer-A1": {2, 3, 4, 5, 6, 7},
    "6mer": {2, 3, 4, 5, 6, 7},
    "offset-6mer": {3, 4, 5, 6, 7, 8},
}
POSITIONS = [2, 3, 4, 5, 6, 7, 8]
# Site-type Kd (magnitude only; the core-position verdict is Kd-independent, see module docstring).
K_BY_SITE = {"8mer": 0.02, "7mer-m8": 0.06, "7mer-A1": 0.10, "6mer": 0.40, "offset-6mer": 0.80}
DELTA = 2.0  # fixed per-position destabilization (K -> K*e^delta); verdict is delta-independent
K_ON = 1e-4  # tight fully-complementary on-target
X_ON = 1.0
M_FACTOR = 0.1  # nominal guide pool (subsaturating); does not affect the incidence verdict


def load_fasta(path: Path) -> dict[str, str]:
    seqs, gid, buf = {}, None, []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if gid:
                    seqs[gid] = "".join(buf)
                gid, buf = line[1:].strip().split()[0], []
            else:
                buf.append(line.strip())
    if gid:
        seqs[gid] = "".join(buf)
    return seqs


def build_sites(seed: str, utr_by_gene: dict[str, str], gene_ids: list[str]):
    """Real off-target sites: (type, window) for each canonical seed site in the 3'UTRs."""
    from cel.features.seedmatch import find_sites

    guide = "A" + seed  # dummy position 1; find_sites reads seed as guide positions 2-8
    types = []
    for gid in gene_ids:
        u = utr_by_gene.get(gid)
        if not u or len(u) < 8:
            continue
        for s in find_sites(guide, u):
            if s.site_type in WINDOWS:
                types.append(s.site_type)
    return types


def incidence(types: list[str]) -> dict[int, list[int]]:
    return {p: [i for i, t in enumerate(types) if p in WINDOWS[t]] for p in POSITIONS}


def phi_and_ontarget(types, S, M):
    """Coupled Phi(S) over off-target sites + on-target occupancy, on-target in the shared pool."""
    K0 = np.array([K_BY_SITE[t] for t in types])
    mult = np.array(
        [np.exp(DELTA * len(S & WINDOWS[t])) for t in types]
    )  # raise K where S pairs the site
    Koff = K0 * mult
    K_on = K_ON * np.exp(
        DELTA * len(S & {2, 3, 4, 5, 6, 7, 8})
    )  # on-target fully paired: any seed pos hits it
    x_all = np.concatenate([[X_ON], np.ones(len(types))])
    K_all = np.concatenate([[K_on], Koff])
    f = solve_f(M, x_all, K_all)
    phi = float(np.sum(f / (Koff + f)))
    theta_on = float(f / (K_on + f))
    return phi, theta_on


def evaluate_guide(name, seed, species, utr, gene_ids, log):
    types = build_sites(seed, utr, gene_ids)
    inc = incidence(types)
    n_sites = len(types)
    from collections import Counter

    tc = Counter(types)
    M = M_FACTOR * max(n_sites, 1)

    log(f"=== {name}: seed {seed} ({species}), {n_sites} canonical off-target sites ===")
    log(f"  site-type counts: {dict(tc)}")
    log("  INCIDENCE (guide seed position -> number of off-target sites it pairs):")
    for p in POSITIONS:
        log(f"    position {p}: {len(inc[p])} sites")
    core = [len(inc[p]) for p in (3, 4, 5, 6, 7)]
    position_blind = len(set(core)) == 1
    log(f"  positions 3-7 identical incidence (position-blind core): {position_blind} ({core})")

    phi0, theta0 = phi_and_ontarget(types, frozenset(), M)
    rows = []
    for p in POSITIONS:
        phi, theta_on = phi_and_ontarget(types, frozenset({p}), M)
        rows.append({"position": p, "g": phi0 - phi, "theta_on": theta_on, "n_sites": len(inc[p])})
    rows.sort(key=lambda r: -r["g"])
    log(
        "  CERTIFIED single-position suppression ranking (g larger = better), with on-target occupancy:"
    )
    for rk, r in enumerate(rows, 1):
        log(
            f"    rank {rk}: position {r['position']}  g={r['g']:.5g}  on-target theta={r['theta_on']:.4f}  ({r['n_sites']} sites)"
        )

    core_g = sorted({round(r["g"], 9) for r in rows if r["position"] in (3, 4, 5, 6, 7)})
    top_g = rows[0]["g"]
    top_positions = [r["position"] for r in rows if abs(r["g"] - top_g) < 1e-9]
    return {
        "name": name,
        "position_blind_core": position_blind,
        "core_g_values": core_g,
        "top_positions": top_positions,
        "ranking": rows,
        "theta_on_baseline": theta0,
    }


def main() -> int:
    lines: list[str] = []

    def log(s: str):
        print(s)
        lines.append(s)

    log("CERTIFIED-OPTIMAL SEED-MODIFICATION DESIGN VALIDATION on real siRNA off-target data")
    log("scanMiR 1.16.0 installed, but no per-site Kd for these NOVEL siRNA seeds (no matching")
    log("scanMiRData model; getKdModel needs the McGeary-CNN Kd table). The verdict below is")
    log("affinity-INDEPENDENT: it is fixed by the real position-to-site incidence.\n")

    # (a) Schlegel: rat Ttr, seed UAUAGAG, experimental optimum = position 7 (the g7 GNA).
    utr = load_fasta(RAT_UTR)
    d1 = pd.read_csv(SM_DIR / "Schlegel_2022_Ttr_D1_30mkg.csv", sep="\t")
    gene_ids = list(d1["ensembl_gene_id"].astype(str))
    schlegel = evaluate_guide("Schlegel (rat Ttr)", "UAUAGAG", "rat Rnor", utr, gene_ids, log)
    log("")

    log("SUMMARY")
    exp_opt = 7
    recovered_uniquely = schlegel["top_positions"] == [exp_opt]
    in_top_tie = exp_opt in schlegel["top_positions"]
    log(f"  Schlegel experimental optimum = position {exp_opt} (g7 GNA).")
    log(
        f"    model top-ranked positions: {schlegel['top_positions']}  (uniquely g7: {recovered_uniquely}; g7 in top tie: {in_top_tie})"
    )
    log(
        f"    position-blind core (3-7 tied): {schlegel['position_blind_core']}  core g values: {schlegel['core_g_values']}"
    )
    log("")
    log(
        "  Jackson (human, position-2 2'-OMe): STRUCTURAL conclusion (seed-independent). Position 2"
    )
    log(
        "  pairs every canonical site EXCEPT offset-6mer (window 3-8), so it always pairs FEWER sites"
    )
    log(
        "  than the core 3-7. In the incidence model its suppression is therefore at or below the core"
    )
    log("  tie, never uniquely on top, so position 2 is NOT recovered as the unique optimum. (The")
    log(
        "  specific Jackson guide 3'UTR sites are not re-run; the ranking of position 2 vs the core is"
    )
    log("  a property of the seed-window incidence, identical for any guide seed.)")
    log("")
    log("STEP 4 (position specificity): the incidence matrix does NOT differ across the core seed")
    log(
        "positions 3-7 (identical site membership), so the model is POSITION-BLIND there. This is a"
    )
    log("structural limitation of a contiguous-seed-window occupancy model, not a pass/fail of the")
    log("affinity theory: canonical seed sites pair the whole seed core, so a single-position seed")
    log(
        "modification is near-equivalent across the core in the model, while the experiments single"
    )
    log(
        "out edge positions (g7, position 2) via chemistry/thermodynamics the model does not encode."
    )
    log(
        "  Cross-check to the measured Schlegel g7 effect: the model assigns g7 the SAME suppression"
    )
    log("  as positions 3-6 (a tie), whereas the experiment isolates g7 (measured off-target drop,")
    log(
        "  paired Wilcoxon p=0.0018 in chemistry_beforeafter.json). The model cannot explain why g7"
    )
    log("  specifically; it only certifies that SOME core-seed modification suppresses off-target.")
    log("")
    verdict_a = "NOT uniquely recovered (g7 sits in a 5-way position-blind tie with positions 3-6)"
    verdict_b = "NOT recovered (position 2 pairs fewer sites than the core, ranked at or below it)"
    log("VERDICT:")
    log(f"  (a) Schlegel g7: {verdict_a}")
    log(f"  (b) Jackson position 2: {verdict_b}")
    log(
        "  Cause: the model is POSITION-BLIND across the seed core (identical incidence for positions"
    )
    log(
        "  3-7), a stated STRUCTURAL limitation, not an affinity-ranking miss. Real per-site Kd would"
    )
    log(
        "  not change this: the core-position tie is exact for any K and delta. The certified design"
    )
    log("  is sound as an OPTIMIZER (exact over the tiny seed space) but the seed-window occupancy")
    log(
        "  MODEL lacks the position resolution to recover the experimentally optimal single position."
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[design-validation] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
