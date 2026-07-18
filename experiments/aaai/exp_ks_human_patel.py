#!/usr/bin/env python3
"""Native seed-match KS on the real human Patel 2022 6mer-seed off-target datasets.

Two real GEO datasets (Patel et al. 2022, PMID 35332222), both processed against
genome build hg38 = GRCh38 (CONFIRMED from each GEO record's Sample_data_processing
"Genome_build: hg38"; not assumed):

  GSE164289 (HeyA8 ovarian): authors' DESeq2 differential-expression tables.
    - siGGGGGC vs siNT1 (GEO sample "siMxy"; transfection characteristics "10 nM
      siGGGGGC"); 6mer seed GGGGGC, 3'UTR seed-match site GCCCCC (stated in the
      GEO record / paper).
    - siGGCAGU vs siNT1 (GEO sample "si34a"; characteristics "10 nM siGGCAGU");
      6mer seed GGCAGU, 3'UTR site ACTGCC (reverse complement).
  GSE196839 (PC3 prostate + HeyA8): raw HTSeq/featureCounts counts only (siGGGGGC =
    "siMxy" confirmed; SINGLE replicate per condition; NO authors' DE table). A simple
    library-size (CPM) normalized log2FC vs siNT1 is computed from the real counts and
    clearly labeled count-derived (not the authors' DESeq2).

For each contrast the gene's hg38 3'UTR(s) are taken from pyensembl pinned to Ensembl
release 104 (GRCh38, May 2021, contemporaneous with the GSE), a gene is seed-matched
if any of its 3'UTRs contains the seed-match site word, and the SeedMatchR one-sided
KS (seed-matched vs background measured log2FC) is reported per contrast.

HARD DATA POLICY (OUTLINE.md): REAL DATA ONLY. The DESeq2 tables / counts come from
GEO (scripts/download_patel.sh, raises if absent); the 3'UTRs come from pyensembl
(raises if the release is not installed). If a dataset's build cannot be confirmed it
is recorded pending with the exact source. No fabricated values. Writes
results/aaai/ks_human_patel.json. Accepts --seed and logs it. No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.data.offtarget import seedmatch_ks_test  # noqa: E402

OUT = REPO / "results" / "aaai"
PATEL = REPO / "data" / "raw" / "offtarget" / "patel"
ENSEMBL_RELEASE = 104  # GRCh38 (hg38), May 2021; matches the confirmed GSE build

# Confirmed siRNA -> 6mer seed (from each GEO sample's transfection characteristics).
SEEDS = {"siGGGGGC": "GGGGGC", "siGGCAGU": "GGCAGU"}


def revcomp_dna(s: str) -> str:
    s = s.upper().replace("U", "T")  # accept RNA seeds (GGCAGU); emit DNA site words
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(comp[b] for b in reversed(s))


def utr_site(seed: str) -> str:
    """3'UTR seed-match site word = reverse complement of the guide 6mer seed."""
    return revcomp_dna(seed)


def build_site_cache(release, gene_ids, symbols, sites: list[str]) -> dict:
    """gene -> {site: bool} over its hg38 3'UTRs, for the genes seen in the tables.

    Keyed by both Ensembl gene id and uppercase symbol so either table can look up.
    """
    cache: dict[str, dict[str, bool]] = {}

    def flags_for(gene) -> dict[str, bool]:
        utrs = [t.three_prime_utr_sequence for t in gene.transcripts]
        utrs = [u.upper() for u in utrs if u]
        return {site: any(site in u for u in utrs) for site in sites}

    for gid in gene_ids:
        if gid in cache:
            continue
        try:
            gene = release.gene_by_id(gid)
        except Exception:  # gene id absent from this release -> no 3'UTR
            continue
        cache[gid] = flags_for(gene)
        if gene.gene_name:
            cache.setdefault(gene.gene_name.upper(), cache[gid])
    for sym in symbols:
        key = sym.upper()
        if key in cache:
            continue
        try:
            genes = release.genes_by_name(sym)
        except Exception:
            continue
        if not genes:
            continue
        merged: dict[str, bool] = {site: False for site in sites}
        for gene in genes:
            f = flags_for(gene)
            for site in sites:
                merged[site] = merged[site] or f[site]
        cache[key] = merged
    return cache


def ks_for(lfc: np.ndarray, gene_keys: list[str], site: str, cache: dict) -> dict:
    """One-sided KS of seed-matched vs background log2FC over 3'UTR-bearing genes."""
    has_utr = np.array([k in cache for k in gene_keys])
    is_seed = np.array([cache.get(k, {}).get(site, False) for k in gene_keys])
    finite = np.isfinite(lfc)
    keep = has_utr & finite  # 3'UTR-bearing background (genes with a 3'UTR in GRCh38/104)
    res = seedmatch_ks_test(lfc[keep], is_seed[keep], alternative="greater")
    return {
        "ks_D": res.statistic,
        "ks_p": res.pvalue,
        "n_seed_match": res.n_seed_match,
        "n_background": res.n_background,
        "n_with_3utr": int(keep.sum()),
    }


def gse164289(cache_sites, release) -> dict:
    """Authors' DESeq2 KS for siGGGGGC and siGGCAGU vs siNT1 (HeyA8, hg38)."""
    out: dict = {
        "build": "hg38 = GRCh38 (GEO Genome_build: hg38)",
        "cell_line": "HeyA8 ovarian",
        "effect_table": "authors' DESeq2 differential expression (GEO supplementary)",
        "contrasts": {},
    }
    files = {
        "siGGGGGC": PATEL / "GSE164289_siGGGGGC_vs_siNT1_48h.csv",
        "siGGCAGU": PATEL / "GSE164289_siGGCAGU_vs_siNT1_48h.csv",
    }
    for name, path in files.items():
        if not path.exists():
            out["contrasts"][name] = {"status": "pending", "source": f"missing {path.name}"}
            continue
        df = pd.read_csv(path)
        gid = df["Ensembl ID"].astype(str).str.split(".").str[0].tolist()
        lfc = pd.to_numeric(df["log2FoldChange"], errors="coerce").to_numpy()
        site = utr_site(SEEDS[name])
        ks = ks_for(lfc, gid, site, cache_sites)
        out["contrasts"][name] = {
            "seed": SEEDS[name],
            "utr_seed_match_site": site,
            "n_genes_table": int(len(df)),
            **ks,
        }
    return out


def gse196839(cache_sites, release) -> dict:
    """Count-derived KS for siGGGGGC vs siNT1 (PC3, hg38) from the raw HTSeq counts."""
    path = PATEL / "GSE196839_counts.txt"
    out: dict = {
        "build": "hg38 = GRCh38 (GEO Genome_build: hg38)",
        "cell_line": "PC3 prostate",
        "effect_table": "count-derived: CPM-normalized log2((siGGGGGC+1)/(siNT1+1)), SINGLE "
        "replicate per condition (no authors' DESeq2 table in the GEO deposit)",
        "contrasts": {},
    }
    if not path.exists():
        out["status"] = "pending"
        out["source"] = f"missing {path.name}"
        return out
    counts = pd.read_csv(path, sep="\t", index_col=0)
    # siGGGGGC = "siMxy" (confirmed); siNT1 = "siNT"; PC3, 48h single replicate.
    treat, ctrl = "PC3_siMxy_3AB_48h", "PC3_siNT_3AB_48h"
    if treat not in counts.columns or ctrl not in counts.columns:
        out["status"] = "pending"
        out["source"] = f"expected count columns {treat}/{ctrl} absent"
        return out
    sample_cols = [c for c in counts.columns if c.startswith(("PC3_", "HeyA8_"))]
    cpm = counts[sample_cols].apply(lambda s: s / s.sum() * 1e6, axis=0)
    lfc = np.log2((cpm[treat].to_numpy() + 1.0) / (cpm[ctrl].to_numpy() + 1.0))
    symbols = [str(s) for s in counts.index]
    site = utr_site(SEEDS["siGGGGGC"])
    ks = ks_for(lfc, [s.upper() for s in symbols], site, cache_sites)
    out["contrasts"]["siGGGGGC"] = {
        "seed": SEEDS["siGGGGGC"],
        "utr_seed_match_site": site,
        "n_genes_table": int(len(counts)),
        "note": "siCAG (the other treated arm) is not run: its characteristics say 'siCAG', "
        "not explicitly 'siGGCAGU', so the seed is not confirmed from the GEO record.",
        **ks,
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed (logged; result deterministic)"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[ks-patel] seed={args.seed}  Ensembl release {ENSEMBL_RELEASE} (GRCh38 = hg38)")

    sites = sorted({utr_site(s) for s in SEEDS.values()})
    # Sanity: the GGGGGC seed maps to the paper-stated 3'UTR site GCCCCC.
    assert utr_site("GGGGGC") == "GCCCCC", "seed->site convention broken"

    try:
        from pyensembl import EnsemblRelease

        release = EnsemblRelease(ENSEMBL_RELEASE, species="homo_sapiens")
        release.gene_by_id("ENSG00000001084")  # forces the DB; raises if not installed
    except Exception as exc:
        out = {
            "status": "pending_real_data",
            "source": f"pyensembl GRCh38 release {ENSEMBL_RELEASE} not installed "
            f"(run: pyensembl install --release {ENSEMBL_RELEASE} --species homo_sapiens): {exc}",
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "ks_human_patel.json").write_text(json.dumps(out, indent=2))
        print(f"[ks-patel] PENDING: {exc}")
        return 0

    # Build the 3'UTR seed-match cache over the union of genes in the tables.
    gene_ids: set[str] = set()
    symbols: set[str] = set()
    f164 = PATEL / "GSE164289_siGGGGGC_vs_siNT1_48h.csv"
    if f164.exists():
        gene_ids |= set(pd.read_csv(f164)["Ensembl ID"].astype(str).str.split(".").str[0].tolist())
    f196 = PATEL / "GSE196839_counts.txt"
    if f196.exists():
        symbols |= {str(s) for s in pd.read_csv(f196, sep="\t", index_col=0).index}
    print(f"[ks-patel] resolving 3'UTR sites for {len(gene_ids)} ENSG + {len(symbols)} symbols ...")
    cache_sites = build_site_cache(release, gene_ids, symbols, sites)
    n_with_site = {s: sum(1 for v in cache_sites.values() if v.get(s)) for s in sites}
    print(f"[ks-patel] genes with each 3'UTR site (cache): {n_with_site}")

    numbers = {
        "status": "ok",
        "seed": args.seed,
        "study": "Patel et al. 2022 (PMID 35332222), human 6mer-seed off-target",
        "genome_build": "hg38 = GRCh38, confirmed from each GEO Sample_data_processing "
        "(Genome_build: hg38); not assumed",
        "ensembl_release_pinned": ENSEMBL_RELEASE,
        "ensembl_release_note": "GEO records confirm the hg38 (GRCh38) genome build but do not "
        "state the Ensembl annotation release, so pyensembl is pinned to release 104 (GRCh38, "
        "May 2021), contemporaneous with the GSE processing.",
        "metric": "SeedMatchR one-sided KS (D, p): seed-matched vs background log2FC over "
        "GRCh38/104 3'UTR-bearing genes; seed-match = any 3'UTR contains the site word",
        "GSE164289": gse164289(cache_sites, release),
        "GSE196839": gse196839(cache_sites, release),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "ks_human_patel.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    print(f"\n[ks-patel] {numbers['genome_build']}")
    for gse in ("GSE164289", "GSE196839"):
        block = numbers[gse]
        print(f"[ks-patel] {gse} ({block['cell_line']}; {block['effect_table'][:46]}...):")
        for name, c in block.get("contrasts", {}).items():
            if c.get("status") == "pending":
                print(f"     {name}: PENDING ({c.get('source')})")
                continue
            print(
                f"     {name} (seed {c['seed']} -> site {c['utr_seed_match_site']}): "
                f"KS D={c['ks_D']:.4f} p={c['ks_p']:.2e} (seed={c['n_seed_match']} bg={c['n_background']})"
            )
    print(f"[ks-patel] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
