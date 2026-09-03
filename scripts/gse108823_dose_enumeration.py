"""GSE108823 dose enumeration + certified-direction test at each available dose for siRNA-5.

Rules out the dose confound for the weaker Hao1 attenuation (0.298 vs Ttr 0.673): enumerate every
sample group in GSE108823 and every siRNA-5 arm, and report which DOSES exist for the siRNA-5 parent
vs siRNA-5-GNA pair.

Finding from GEO metadata (36 GSMs, all cell type = rat hepatocytes BioreclamationIVT lot QDZ; the
paper Methods fix in vitro transfection at 10 nM): GSE108823 has NO in vivo samples and NO dose
variation. The siRNA-5-GNA arm exists in exactly one batch (Fig 6b, GSM2913831-833). So the
parent-vs-GNA pair exists at a SINGLE dose (10 nM in vitro). We therefore do NOT manufacture a dose
series; the dose-series table has one row.

We additionally report, at that same 10 nM dose (clearly labeled NOT a dose series), the seed
enrichment in every siRNA-5 arm as orthogonal validation:
  - siRNA-5 (Fig 5a/6b batch) and siRNA-5_Fig5b (independent batch): parent reproducibility;
  - siRNA-5-iB (RISC-loading blocked): seed signature should VANISH (mechanism control; Janas Fig 5b);
  - siRNA-5-GNA: the modified arm.

Same shared-universe pipeline as the panel: single tested-transcript intersection across the two arms
of the pair, find_sites canonical 6-8mer, on-target Hao1 removed, identical n across arms with the
equality assertion as regression guard. Deterministic, real data only, no em-dashes.
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

from operator_certified_panel import K_BY_SITE, aggregate_occupancy  # noqa: E402
from seedmod_loader import _load_fasta, _seed_match  # noqa: E402

GEO = REPO / "data/raw/offtarget/geo"
SEED = "UGUCGAU"  # Hao1 siRNA-5 guide g2-8 (Schlegel 2022 NAR Table S1, verified)
HAO1_ID = "ENSRNOG00000004601"  # on-target, removed
CANON = {"8mer", "7mer-m8", "7mer-A1", "6mer"}
M_REF, R_REF = 0.10, 2.0
OUT_JSON = REPO / "results/gse108823_dose_enumeration.json"

# All 36 GSMs are rat hepatocytes at 10 nM (GEO metadata + Janas 2018 Methods). Group -> replicate GSMs.
SAMPLE_GROUPS = {
    "mock control (Fig5a/6b)": ["GSM2913798", "GSM2913799", "GSM2913800"],
    "mock control (Fig5b)": ["GSM2913822", "GSM2913823", "GSM2913824"],
    "mock control iB (Fig5b)": ["GSM2913813", "GSM2913814", "GSM2913815"],
    "siRNA-1 (Fig5b)": ["GSM2913825", "GSM2913826", "GSM2913827"],
    "siRNA-1-iB (Fig5b)": ["GSM2913816", "GSM2913817", "GSM2913818"],
    "siRNA-2 (Fig5a)": ["GSM2913801", "GSM2913802", "GSM2913803"],
    "siRNA-4 (Fig5a)": ["GSM2913804", "GSM2913805", "GSM2913806"],
    "siRNA-5 (Fig5a/6b)": ["GSM2913807", "GSM2913808", "GSM2913809"],
    "siRNA-5 (Fig5b)": ["GSM2913828", "GSM2913829", "GSM2913830"],
    "siRNA-5-GNA (Fig6b)": ["GSM2913831", "GSM2913832", "GSM2913833"],
    "siRNA-5-iB (Fig5b)": ["GSM2913819", "GSM2913820", "GSM2913821"],
    "siRNA-6 (Fig5a)": ["GSM2913810", "GSM2913811", "GSM2913812"],
}

ARMS = {  # DESeq2 group-vs-Mock files for siRNA-5 arms (all 10 nM in vitro)
    "siRNA-5": "GSE108823_group_siRNA-5.vsMock.DESeq2.txt.gz",
    "siRNA-5_Fig5b": "GSE108823_group_siRNA-5_Fig5b.vsMock.DESeq2.txt.gz",
    "siRNA-5-GNA": "GSE108823_group_siRNA-5-GNA.vsMock.DESeq2.txt.gz",
    "siRNA-5-iB": "GSE108823_group_siRNA-5-iB.vsMock.DESeq2.txt.gz",
}


def load_deseq(fname: str) -> pd.DataFrame:
    """Handle both formats: (symbol, ensembl_gene_id columns) and (gene id as unnamed index)."""
    df = pd.read_csv(GEO / fname, sep="\t")
    if "ensembl_gene_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ensembl_gene_id"})
    # some files use plain ENSRNOG ids, the Fig5b file uses "Symbol:ENSRNOG..."; extract the ENSRNOG id
    ids = df["ensembl_gene_id"].astype(str)
    df["ensembl_gene_id"] = ids.str.extract(r"(ENSRNOG\d+)", expand=False).fillna(ids)
    return df[["ensembl_gene_id", "baseMean", "log2FoldChange"]]


def tested(df: pd.DataFrame) -> set[str]:
    return set(df.loc[df["log2FoldChange"].notna() & (df["baseMean"] > 0), "ensembl_gene_id"])


def ks_left(matched: np.ndarray, background: np.ndarray, return_arrays: bool = False) -> dict:
    r = stats.ks_2samp(matched, background, alternative="greater")
    out = {
        "ks_D": float(r.statistic),
        "ks_p": float(r.pvalue),
        "n_matched": int(len(matched)),
        "n_background": int(len(background)),
    }
    if return_arrays:  # the SAME arrays the KS D was computed on, for plotting (Figure 2)
        out["matched"] = np.asarray(matched, dtype=float)
        out["background"] = np.asarray(background, dtype=float)
    return out


def annotate(genes: list[str], utr: dict[str, str]) -> set[str]:
    flags = _seed_match(SEED, utr, genes, CANON)
    return {g for g in genes if flags[g][0]}


def arm_enrichment(df: pd.DataFrame, utr: dict[str, str], return_arrays: bool = False) -> dict:
    """Seed enrichment within one arm on its own tested universe (on-target removed)."""
    universe = sorted(tested(df) - {HAO1_ID})
    matched = annotate(universe, utr)
    d = df[df["ensembl_gene_id"].isin(universe)]
    return ks_left(
        d.loc[d["ensembl_gene_id"].isin(matched), "log2FoldChange"].to_numpy(),
        d.loc[~d["ensembl_gene_id"].isin(matched), "log2FoldChange"].to_numpy(),
        return_arrays=return_arrays,
    )


def certified_pair(parent: pd.DataFrame, modified: pd.DataFrame, utr: dict[str, str]) -> dict:
    """Shared-universe certified-direction test for the one real pair (siRNA-5 vs siRNA-5-GNA)."""
    universe = sorted((tested(parent) & tested(modified)) - {HAO1_ID})
    matched = annotate(universe, utr)
    res = {}
    for arm, df in (("parent", parent), ("modified", modified)):
        d = df[df["ensembl_gene_id"].isin(universe)]
        res[arm] = ks_left(
            d.loc[d["ensembl_gene_id"].isin(matched), "log2FoldChange"].to_numpy(),
            d.loc[~d["ensembl_gene_id"].isin(matched), "log2FoldChange"].to_numpy(),
        )
    if (
        res["parent"]["n_matched"] != res["modified"]["n_matched"]
        or res["parent"]["n_background"] != res["modified"]["n_background"]
    ):
        raise AssertionError(f"shared-universe violation: {res['parent']} vs {res['modified']}")
    # operator on the shared universe seed-matched canonical set
    pm = _seed_match(SEED, utr, universe, CANON)
    base = dict(
        zip(
            parent["ensembl_gene_id"],
            pd.to_numeric(parent["baseMean"], errors="coerce"),
            strict=True,
        )
    )
    rows = [
        (g, pm[g][1], pm[g][2])
        for g in universe
        if pm[g][0] and pm[g][1] in K_BY_SITE and pd.notna(base.get(g))
    ]
    x = np.array([base[g] * n for g, _t, n in rows], float)
    x = x / x.sum()
    k = np.array([K_BY_SITE[t] for _g, t, _n in rows], float)
    dO = aggregate_occupancy(x, R_REF * k, M_REF) - aggregate_occupancy(x, k, M_REF)
    return {
        "n_matched": res["parent"]["n_matched"],
        "n_background": res["parent"]["n_background"],
        "parent": res["parent"],
        "modified": res["modified"],
        "attenuation": float(1.0 - res["modified"]["ks_D"] / res["parent"]["ks_D"]),
        "operator_predicted_change": float(dO),
        "sign_match": bool(dO < 0 and res["modified"]["ks_D"] < res["parent"]["ks_D"]),
    }


def main() -> int:
    utr = _load_fasta(REPO / "data/raw/reference/rat_3utr.fa")
    dfs = {name: load_deseq(f) for name, f in ARMS.items()}

    pair = certified_pair(dfs["siRNA-5"], dfs["siRNA-5-GNA"], utr)
    enrich = {name: arm_enrichment(df, utr) for name, df in dfs.items()}

    report = {
        "cell_type": "rat hepatocytes (BioreclamationIVT lot QDZ), in vitro",
        "dose": "10 nM (single dose; Janas 2018 Methods; no in vivo, no dose variation in GEO)",
        "n_samples": 36,
        "sample_groups": {g: len(v) for g, v in SAMPLE_GROUPS.items()},
        "sirna5_arms_present": list(ARMS),
        "doses_for_parent_vs_GNA_pair": 1,
        "dose_series": [{"dose": "10 nM", **pair}],
        "same_dose_arm_enrichments": enrich,
        "seed_source": "Schlegel 2022 NAR Suppl Table S1 (Hao1 D6/D9); verified",
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))

    print("GSE108823 sample-group enumeration")
    print("=" * 92)
    print(
        "36 GSMs, ALL cell type = rat hepatocytes (in vitro), single dose 10 nM. Groups (x3 reps):"
    )
    for g, gsms in SAMPLE_GROUPS.items():
        print(f"  {g:26s} n={len(gsms)}  {gsms[0]}..{gsms[-1]}")
    print("\nProcessed DESeq2 group-vs-Mock files for siRNA-5 arms:", ", ".join(ARMS))
    print("\nDOSES for the siRNA-5 parent vs siRNA-5-GNA pair: ONE (10 nM in vitro).")
    print(
        "  GSE108823 has no in vivo samples and no dose variation; siRNA-5-GNA exists in one batch"
    )
    print("  (Fig 6b) only. A dose series cannot be built here and is not manufactured.")

    print("\nDOSE SERIES TABLE (one row):")
    print(
        f"{'dose':>7s} {'n_match':>8s} {'n_bg':>7s} {'parentD':>8s} {'parent_p':>10s} "
        f"{'modD':>7s} {'mod_p':>10s} {'atten':>7s} {'pred.dOcc':>11s} {'sign':>5s}"
    )
    p = pair
    print(
        f"{'10 nM':>7s} {p['n_matched']:>8d} {p['n_background']:>7d} {p['parent']['ks_D']:>8.4f} "
        f"{p['parent']['ks_p']:>10.2e} {p['modified']['ks_D']:>7.4f} {p['modified']['ks_p']:>10.2e} "
        f"{p['attenuation']:>7.4f} {p['operator_predicted_change']:>+11.3e} {p['sign_match']!s:>5s}"
    )

    print("\nSAME-DOSE (10 nM) arm enrichments [NOT a dose series; batch/chemistry controls]:")
    for name, e in enrich.items():
        tag = {
            "siRNA-5": "parent (Fig5a/6b)",
            "siRNA-5_Fig5b": "parent (Fig5b batch, reproducibility)",
            "siRNA-5-GNA": "GNA-modified",
            "siRNA-5-iB": "RISC-block control (expect null)",
        }[name]
        print(f"  {name:15s} D={e['ks_D']:.4f} p={e['ks_p']:.2e} (n_match={e['n_matched']})  {tag}")
    print(
        "\nInterpretation: the parent seed signature reproduces across batches; the RISC-block (iB)"
    )
    print(
        "arm abolishes it (mechanism control), matching Janas 2018 Fig 5b. The residual GNA signature"
    )
    print(
        "(D=0.111) is therefore NOT a dose artifact: it is at the same 10 nM as the parent. The 0.298"
    )
    print(
        "vs 0.673 gap versus GSE184929 is confounded by BOTH assay (in vitro 10 nM vs in vivo 30 mkg)"
    )
    print(
        "AND guide; it cannot be attributed to dose from GSE108823 alone (no dose variation here)."
    )
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
