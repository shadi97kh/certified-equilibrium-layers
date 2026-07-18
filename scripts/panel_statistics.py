"""Statistical additions to the certified-direction panel (no new data):
  (1) stratified bootstrap 95% CI on attenuation = 1 - D_mod/D_parent for each GNA row, and whether
      the Ttr and Hao1 CIs overlap (the headline 0.71 vs 0.025 contrast);
  (2) a permutation null for the KS D (transcripts are not independent: gene families / shared 3'UTR
      motifs), reported alongside the analytic KS p;
  (3) the Hao1 isoform fragility disclosed as a number: attenuation under last-isoform vs union.

Reuses the SAME shared universe as certified_panel_filtered (tested in both arms, on-target removed,
union-over-isoforms seed annotation from the long tables). One-sided KS D+ = max(F_matched - F_bg) is
recomputed with a fast numpy routine (identical to scipy alternative='greater') for the resampling
loops. Deterministic: a single fixed-seed numpy Generator drives all resampling. Em-dash free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from cel.features.seedmatch import find_sites  # noqa: E402
from certified_panel_filtered import CONTRASTS, tested_ids  # noqa: E402
from seedmod_loader import _load_fasta  # noqa: E402

SEED = 20240716
N_BOOT = 2000
N_PERM = 2000
OUT_JSON = REPO / "results/panel_statistics.json"
RANK = {"8mer": 0, "7mer-m8": 1, "7mer-A1": 2, "6mer": 3}
CANON = set(RANK)
HAO1_ID = "ENSRNOG00000004601"


def ks_dplus(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided KS D+ = max_x (F_a(x) - F_b(x)); equals scipy ks_2samp(a, b, alternative='greater')."""
    a = np.sort(a)
    b = np.sort(b)
    allx = np.concatenate([a, b])
    allx.sort()
    fa = np.searchsorted(a, allx, side="right") / a.size
    fb = np.searchsorted(b, allx, side="right") / b.size
    return float((fa - fb).max())


def paired_universe(c: dict):
    """Per-transcript (parent_lfc, mod_lfc, seed_matched) on the shared universe (same as the panel)."""
    par = pd.read_csv(REPO / c["parent"], sep=c["sep"])
    mod = pd.read_csv(REPO / c["modified"], sep=c["sep"])
    long = pd.read_csv(REPO / c["long"])
    on = set(par.loc[par["symbol"] == c["on_target"], "ensembl_gene_id"].astype(str))
    universe = (tested_ids(par) & tested_ids(mod)) - on
    sub = long[long["transcript_id"].astype(str).isin(universe)]
    p = sub[sub["arm"] == "parent"][["transcript_id", "log2FoldChange", "seed_matched"]].set_index(
        "transcript_id"
    )
    m = sub[sub["arm"] == "modified"][["transcript_id", "log2FoldChange"]].set_index(
        "transcript_id"
    )
    m = m.loc[p.index]
    return (
        p["log2FoldChange"].to_numpy(),
        m["log2FoldChange"].to_numpy(),
        p["seed_matched"].to_numpy(bool),
    )


def bootstrap_ci(parent_lfc, mod_lfc, matched_mask, rng) -> dict:
    mi0 = np.where(matched_mask)[0]
    bi0 = np.where(~matched_mask)[0]
    dp0 = ks_dplus(parent_lfc[mi0], parent_lfc[bi0])
    dm0 = ks_dplus(mod_lfc[mi0], mod_lfc[bi0])
    attens = np.empty(N_BOOT)
    for it in range(N_BOOT):
        mi = rng.choice(mi0, mi0.size, replace=True)  # stratified: resample matched and background
        bi = rng.choice(bi0, bi0.size, replace=True)
        dp = ks_dplus(parent_lfc[mi], parent_lfc[bi])
        dm = ks_dplus(mod_lfc[mi], mod_lfc[bi])
        attens[it] = 1.0 - dm / dp if dp > 0 else np.nan
    lo, hi = np.nanpercentile(attens, [2.5, 97.5])
    return {
        "D_parent": dp0,
        "D_modified": dm0,
        "attenuation": 1.0 - dm0 / dp0,
        "atten_CI95": [float(lo), float(hi)],
        "n_matched": int(mi0.size),
        "n_background": int(bi0.size),
    }


def permutation_p(matched_lfc, bg_lfc, rng) -> dict:
    obs = ks_dplus(matched_lfc, bg_lfc)
    pool = np.concatenate([matched_lfc, bg_lfc])
    n_m = matched_lfc.size
    ge = 0
    for _ in range(N_PERM):
        perm = rng.permutation(pool)
        if ks_dplus(perm[:n_m], perm[n_m:]) >= obs:
            ge += 1
    p_perm = (ge + 1) / (N_PERM + 1)
    p_ks = float(stats.ks_2samp(matched_lfc, bg_lfc, alternative="greater").pvalue)
    return {
        "D": obs,
        "p_ks_analytic": p_ks,
        "p_permutation": p_perm,
        "n_perm": N_PERM,
        "perm_exceedances": ge,
    }


def hao1_isoform_fragility() -> dict:
    """(3) Hao1 attenuation under last-isoform vs union-over-isoforms 3'UTR annotation."""
    utr = _load_fasta(REPO / "data/raw/reference/rat_3utr.fa")
    guide = "A" + "UGUCGAU"  # Hao1 guide g2-8 seed UGUCGAU (Schlegel 2022 NAR Table S1)

    def best(genes, mode):
        out = {}
        for g in genes:
            isos = utr.get(g) or []
            if mode == "last":
                isos = isos[-1:]
            b = None
            for u in isos:
                if not u or len(u) < 8:
                    continue
                for s in find_sites(guide, u):
                    if s.site_type in CANON and (b is None or RANK[s.site_type] < RANK[b]):
                        b = s.site_type
            out[g] = b
        return out

    def load(fn):
        df = pd.read_csv(REPO / "data/raw/offtarget/geo" / fn, sep="\t")
        df["gid"] = df["ensembl_gene_id"].astype(str)
        return df.set_index("gid")[["baseMean", "log2FoldChange"]]

    par = load("GSE108823_group_siRNA-5.vsMock.DESeq2.txt.gz")
    mod = load("GSE108823_group_siRNA-5-GNA.vsMock.DESeq2.txt.gz")

    def tested(df):
        return set(df[df["log2FoldChange"].notna() & (df["baseMean"] > 0)].index.astype(str))

    universe = sorted((tested(par) & tested(mod)) - {HAO1_ID})
    out = {}
    for mode in ("last", "union"):
        b = best(universe, mode)
        matched = [g for g in universe if b[g] in CANON]
        bg = [g for g in universe if b[g] is None]
        dp = ks_dplus(
            par.loc[matched, "log2FoldChange"].to_numpy(), par.loc[bg, "log2FoldChange"].to_numpy()
        )
        dm = ks_dplus(
            mod.loc[matched, "log2FoldChange"].to_numpy(), mod.loc[bg, "log2FoldChange"].to_numpy()
        )
        out[mode] = {
            "D_parent": dp,
            "D_modified": dm,
            "attenuation": 1.0 - dm / dp,
            "n_matched": len(matched),
        }
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows = {c["series"]: c for c in CONTRASTS}
    result = {"seed": SEED, "n_bootstrap": N_BOOT, "n_permutation": N_PERM, "rows": {}}

    print("Panel statistics (bootstrap CI, permutation null, isoform fragility)")
    print("=" * 92)
    boot = {}
    for series in ("GSE184929", "GSE108823"):
        c = rows[series]
        p_lfc, m_lfc, mask = paired_universe(c)
        bc = bootstrap_ci(p_lfc, m_lfc, mask, rng)
        perm_par = permutation_p(p_lfc[mask], p_lfc[~mask], rng)
        perm_mod = permutation_p(m_lfc[mask], m_lfc[~mask], rng)
        boot[series] = bc
        result["rows"][series] = {
            "bootstrap": bc,
            "permutation_parent": perm_par,
            "permutation_modified": perm_mod,
            "target": c["target"],
        }
        print(
            f"\n[{series} {c['target']}]  n_matched={bc['n_matched']} n_background={bc['n_background']}"
        )
        print(
            f"  (1) attenuation {bc['attenuation']:.3f}  95% bootstrap CI "
            f"[{bc['atten_CI95'][0]:.3f}, {bc['atten_CI95'][1]:.3f}]  (D_parent {bc['D_parent']:.3f}, "
            f"D_mod {bc['D_modified']:.3f})"
        )
        print(
            f"  (2) parent   D={perm_par['D']:.4f}  KS p={perm_par['p_ks_analytic']:.2e}  "
            f"perm p={perm_par['p_permutation']:.2e} ({perm_par['perm_exceedances']}/{N_PERM} exceed)"
        )
        print(
            f"      modified D={perm_mod['D']:.4f}  KS p={perm_mod['p_ks_analytic']:.2e}  "
            f"perm p={perm_mod['p_permutation']:.2e} ({perm_mod['perm_exceedances']}/{N_PERM} exceed)"
        )

    ttr_ci = boot["GSE184929"]["atten_CI95"]
    hao_ci = boot["GSE108823"]["atten_CI95"]
    overlap = not (ttr_ci[1] < hao_ci[0] or hao_ci[1] < ttr_ci[0])
    result["headline_ci_overlap"] = {"Ttr_CI95": ttr_ci, "Hao1_CI95": hao_ci, "overlap": overlap}
    print(
        f"\nHEADLINE: Ttr attenuation CI {[round(v, 3) for v in ttr_ci]} vs Hao1 CI "
        f"{[round(v, 3) for v in hao_ci]}  ->  CIs overlap = {overlap}  "
        f"({'NOT separated' if overlap else 'cleanly separated'})"
    )

    frag = hao1_isoform_fragility()
    result["hao1_isoform_fragility"] = frag
    print("\n(3) Hao1 isoform fragility (a 12x swing under a preprocessing choice):")
    print(
        f"    last-isoform annotation: attenuation {frag['last']['attenuation']:.3f} "
        f"(D {frag['last']['D_parent']:.3f} -> {frag['last']['D_modified']:.3f}, n_matched {frag['last']['n_matched']})"
    )
    print(
        f"    union-over-isoforms    : attenuation {frag['union']['attenuation']:.3f} "
        f"(D {frag['union']['D_parent']:.3f} -> {frag['union']['D_modified']:.3f}, n_matched {frag['union']['n_matched']})"
    )
    print(
        "    CORRECT = union. last-isoform keeps only the last 3'UTR record per gene, mis-binning genes"
    )
    print(
        "    whose seed site sits on another isoform INTO the background; that contaminates the null and"
    )
    print(
        "    inflates the apparent parent->modified drop. union scores a gene as matched if ANY isoform"
    )
    print(
        "    carries a canonical site (SeedMatchR A-only shrank 411->196 under the fix). The honest"
    )
    print(
        "    Hao1 attenuation is ~0.025 (GNA barely reduces this off-target); 0.30 is a last-isoform artifact."
    )
    result["hao1_correct_annotation"] = "union"

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
