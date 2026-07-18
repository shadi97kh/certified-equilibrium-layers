#!/usr/bin/env python3
"""Annotate the GSE184929 Ttr DESeq2 tables with the native seed-match call.

REAL DATA ONLY. The guide is the confirmed Schlegel 2022 Ttr siRNA antisense
(Table 2, duplexes D1 / D4 share the identical base sequence and seed):

    guide (antisense, 5'->3'):  UUAUAGAGCAAGAACACUGUUUU
    seed (guide positions 2-8): UAUAGAG

For each rat gene in the DESeq2 tables, every annotated 3'UTR (Ensembl BioMart,
Rattus norvegicus) is scanned for canonical seed sites with the native
cel.features.seedmatch logic. A gene is "seed-matched" if any of its 3'UTRs
carries an effective canonical site (8mer / 7mer-m8 / 7mer-A1). Genes with no
3'UTR in the reference get an empty annotation (NaN), so they are excluded from
the KS test rather than silently called background. No values are generated.

The GSE184929 rat-liver tables are aligned to Rnor_6.0 (GEO Sample_data_processing),
so the build-matched 3'UTRs come from the Ensembl apr2020 archive via
scripts/download_rat_3utr.sh. Writes the seed_match_count column back into each
data/raw/offtarget/seedmatchr/<dataset>.csv (so cel.data.load_schlegel2022 sees
has_seed_annotation for the Theorem 4 KS) and prints the per-dataset SeedMatchR
one-sided KS under both background definitions (all genes, and 3'UTR-bearing only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cel.data import seedmatch_ks_test  # noqa: E402
from cel.features.seedmatch import find_sites  # noqa: E402

GUIDE = "UUAUAGAGCAAGAACACUGUUUU"  # Schlegel 2022 NAR Table 2, Ttr D1/D4 (shared seed UAUAGAG)
CANONICAL = {"8mer", "7mer-m8", "7mer-A1"}  # effective seed sites (TargetScan canonical)
DATASETS = (
    "Schlegel_2022_Ttr_D1_30mkg",
    "Schlegel_2022_Ttr_D4_30mkg",
    "Schlegel_2022_Ttr_D1_10mkg",
)
SM_DIR = REPO / "data" / "raw" / "offtarget" / "seedmatchr"
# Rnor_6.0 3'UTRs: the GSE184929 rat-liver DESeq2 tables are aligned to Rnor_6.0
# (GEO Sample_data_processing), so the build-matched reference is the Ensembl
# apr2020 archive (release 100, the Rnor_6.0 assembly).
UTR_FASTA = REPO / "data" / "raw" / "reference" / "rat_3utr_rnor6.fa"


def parse_3utr_fasta(path: Path) -> dict[str, list[str]]:
    """Map ensembl_gene_id -> list of 3'UTR strings (drop 'Sequence unavailable')."""
    gene2utr: dict[str, list[str]] = {}
    gid: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if gid and buf:
            seq = "".join(buf)
            if seq and "unavailable" not in seq.lower():
                gene2utr.setdefault(gid, []).append(seq)

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                flush()
                gid = line[1:].split("|")[0].split()[0]
                buf = []
            else:
                buf.append(line)
        flush()
    return gene2utr


def gene_seed_counts(utrs: list[str]) -> dict[str, int]:
    """Canonical-site counts (8mer/7mer-m8/7mer-A1/6mer/offset-6mer) over a gene's 3'UTRs."""
    counts: dict[str, int] = {}
    for utr in utrs:
        for site in find_sites(GUIDE, utr):
            counts[site.site_type] = counts.get(site.site_type, 0) + 1
    return counts


def annotate(gene2utr: dict[str, list[str]]) -> dict:
    summary: dict = {"guide": GUIDE, "seed": "UAUAGAG", "datasets": {}}
    # Per-gene counts computed once over the union, reused across datasets.
    per_gene: dict[str, int | None] = {}
    per_gene_canonical: dict[str, int | None] = {}
    for ds in DATASETS:
        path = SM_DIR / f"{ds}.csv"
        df = pd.read_csv(path, sep=None, engine="python")
        smc: list[float] = []
        for gid in df["ensembl_gene_id"].astype(str):
            if gid not in gene2utr:
                smc.append(float("nan"))  # no 3'UTR in the reference -> excluded
                continue
            if gid not in per_gene:
                c = gene_seed_counts(gene2utr[gid])
                per_gene_canonical[gid] = sum(c.get(t, 0) for t in CANONICAL)
                per_gene[gid] = per_gene_canonical[gid]
            smc.append(float(per_gene_canonical[gid]))
        df["seed_match_count"] = smc
        df.to_csv(path, sep="\t", index=False)

        finite = df["seed_match_count"].notna()
        lfc = pd.to_numeric(df["log2FoldChange"], errors="coerce")
        is_seed = df["seed_match_count"] > 0  # NaN (no 3'UTR) compares False

        # Two background definitions for the SeedMatchR one-sided KS:
        #  - "3'UTR-bearing only": background = genes that HAVE a 3'UTR but no seed site.
        #  - "all genes": background = every other gene with a finite log2FC (no-3'UTR
        #    genes counted as background, since with no 3'UTR they carry no seed site).
        sub_utr = finite & lfc.notna()
        ks_utr = seedmatch_ks_test(
            lfc[sub_utr].to_numpy(), is_seed[sub_utr].to_numpy(), alternative="greater"
        )
        sub_all = lfc.notna()
        ks_all = seedmatch_ks_test(
            lfc[sub_all].to_numpy(), is_seed[sub_all].to_numpy(), alternative="greater"
        )
        summary["datasets"][ds] = {
            "n_genes": int(len(df)),
            "n_with_3utr": int(finite.sum()),
            "n_seed_matched": int((is_seed & sub_utr).sum()),
            "ks_3utr_only": {
                "D": ks_utr.statistic,
                "p": ks_utr.pvalue,
                "n_seed": ks_utr.n_seed_match,
                "n_bg": ks_utr.n_background,
            },
            "ks_all_genes": {
                "D": ks_all.statistic,
                "p": ks_all.pvalue,
                "n_seed": ks_all.n_seed_match,
                "n_bg": ks_all.n_background,
            },
        }
    return summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--utr-fasta", type=Path, default=UTR_FASTA, help="rat 3'UTR FASTA (build-matched)"
    )
    args = ap.parse_args()
    if not args.utr_fasta.exists():
        raise FileNotFoundError(
            f"rat 3'UTR reference absent at {args.utr_fasta}. Fetch the build-matched 3'UTRs "
            "from Ensembl BioMart (rnorvegicus_gene_ensembl, ensembl_gene_id + 3utr, FASTA); the "
            "GSE184929 rat-liver DESeq2 tables are Rnor_6.0, so use the apr2020 archive."
        )
    gene2utr = parse_3utr_fasta(args.utr_fasta)
    print(
        f"[seedmatch] 3'UTR reference {args.utr_fasta.name}: {len(gene2utr)} genes with >=1 3'UTR"
    )
    summary = annotate(gene2utr)
    print(
        f"[seedmatch] guide={summary['guide']}  seed={summary['seed']}  sites={sorted(CANONICAL)}"
    )
    for ds, s in summary["datasets"].items():
        cov = 100.0 * s["n_with_3utr"] / s["n_genes"]
        u, a = s["ks_3utr_only"], s["ks_all_genes"]
        print(
            f"\n{ds}:"
            f"\n  3'UTR coverage: {s['n_with_3utr']}/{s['n_genes']} ({cov:.1f}%)  seed-matched genes: {s['n_seed_matched']}"
            f"\n  KS (3'UTR-bearing only):  D={u['D']:.4f}  p={u['p']:.2e}  (n_seed={u['n_seed']} n_bg={u['n_bg']})"
            f"\n  KS (all genes):           D={a['D']:.4f}  p={a['p']:.2e}  (n_seed={a['n_seed']} n_bg={a['n_bg']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
