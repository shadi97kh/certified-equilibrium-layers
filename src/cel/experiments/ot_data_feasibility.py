"""Feasibility probe for a LEARNED off-target predictor (data assessment only, no build).

This answers one question: do we have enough REAL supervised off-target data to train and
validate a learned siRNA off-target predictor with leave-one-study-out (LOSO) validation,
or is the independent-study count too low, so the miRNA auxiliary corpus must carry the
training signal?

It does three things:
  1. Assembles the real supervised off-target corpus (per-gene DESeq2 log2FC) across the
     three GEO series we have (Schlegel 2022 GSE184929 rat; Patel 2022 GSE164289 and
     GSE196839 human) and reports n_genes, n_conditions, n_seed_matched per study, then
     counts how many INDEPENDENT studies and conditions exist.
  2. Assesses the miRNA auxiliary corpus (TargetScan context++; McGeary 2019 / scanMiR
     AGO-RBNS Kd and the HeLa transfection repression matrix), reporting downloadable
     files, sizes (from HTTP HEAD; the multi-100-MB TargetScan files are NOT downloaded),
     licenses, and the number of measured (seed, 3'UTR, repression) examples.
  3. States the verdict on LOSO trainability.

REAL DATA ONLY: every count comes from a real local file or a committed real result; URL
sizes come from HTTP HEAD; nothing is fabricated and the large files are not downloaded.
Writes results/cascade_ot/data_feasibility.json. No LaTeX.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"
OUT = RESULTS / "cascade_ot" / "data_feasibility.json"
SCHLEGEL_DIR = REPO / "data" / "raw" / "offtarget" / "seedmatchr"
PATEL_DIR = REPO / "data" / "raw" / "offtarget" / "patel"
MCGEARY_CACHE = (
    REPO
    / "data"
    / "raw"
    / "offtarget"
    / "mirna_aux"
    / ("GSE140217_HeLa_transfection_logtpm_batchnormalized.txt.gz")
)

# TargetScan v8.0 context++ download files; sizes from HTTP HEAD (bytes), not downloaded.
TS_BASE = "https://www.targetscan.org/vert_80/vert_80_data_download/"
TARGETSCAN_FILES = [
    (
        "Predicted_Targets_Context_Scores.default_predictions.txt.zip",
        17_673_917,
        "predicted context++ scores",
    ),
    ("Conserved_Site_Context_Scores.txt.zip", 18_603_027, "conserved-site context++ scores"),
    (
        "Nonconserved_Site_Context_Scores.txt.zip",
        542_498_322,
        "nonconserved-site scores (LARGE, not fetched)",
    ),
    ("miR_Family_Info.txt.zip", 162_636, "miRNA seed-family table"),
]
MCGEARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE140nnn/GSE140217/suppl/"
    "GSE140217_HeLa_transfection_logtpm_batchnormalized.txt.gz"
)
MCGEARY_BYTES = 640_906  # from HTTP HEAD


def _finite_log2fc(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").notna().sum())


def schlegel_corpus() -> dict:
    """Schlegel 2022 GSE184929 (rat liver Ttr): 3 dose/guide conditions, 1 study."""
    from cel.data import SEEDMATCHR_DATASETS, load_schlegel2022

    t4 = json.loads((RESULTS / "offtarget_t4.json").read_text())["per_contrast"]
    conditions = {}
    for name in SEEDMATCHR_DATASETS:
        ds = load_schlegel2022(name, raw_dir=SCHLEGEL_DIR, cache_root=RESULTS / "aaai" / "cache")
        f = ds.frame
        c = t4.get(name, {})
        conditions[name] = {
            "n_genes": int(len(f)),
            "n_genes_with_log2fc": _finite_log2fc(f["log2fc"]),
            "n_seed_matched": c.get("n_seed_matched_aligned"),
            "n_aligned": c.get("n_aligned"),
            "ks_D": c.get("seedmatchr_ks_D"),
            "ks_p": c.get("seedmatchr_ks_p"),
        }
    return {
        "paper": "Schlegel et al. 2022",
        "geo": "GSE184929",
        "organism": "rat (liver)",
        "effect_type": "DESeq2 RNA-seq log2FC vs Mock",
        "n_conditions": len(conditions),
        "conditions": conditions,
        "seed_matched_note": "n_seed_matched is over the TRRUST-aligned subset (n_aligned); the "
        "full-table Rnor_6.0 seed annotation is pending. Source: results/offtarget_t4.json.",
    }


def patel_corpus() -> dict:
    """Patel 2022 GSE164289 (HeyA8, 2 contrasts) and GSE196839 (PC3, 1 contrast)."""
    ks = json.loads((RESULTS / "aaai" / "ks_human_patel.json").read_text())
    out = {}

    g164 = {}
    for tag, fname in [
        ("siGGGGGC", "GSE164289_siGGGGGC_vs_siNT1_48h.csv"),
        ("siGGCAGU", "GSE164289_siGGCAGU_vs_siNT1_48h.csv"),
    ]:
        df = pd.read_csv(PATEL_DIR / fname)
        lfc_col = "log2FoldChange" if "log2FoldChange" in df.columns else df.columns[2]
        c = ks["GSE164289"]["contrasts"][tag]
        g164[tag] = {
            "n_genes": int(len(df)),
            "n_genes_with_log2fc": _finite_log2fc(df[lfc_col]),
            "n_seed_matched": c["n_seed_match"],
            "n_with_3utr": c["n_with_3utr"],
            "ks_D": c["ks_D"],
            "ks_p": c["ks_p"],
        }
    out["Patel_2022_GSE164289"] = {
        "paper": "Patel et al. 2022",
        "geo": "GSE164289",
        "organism": "human (HeyA8 ovarian)",
        "effect_type": "authors' DESeq2 RNA-seq log2FC vs siNT1",
        "n_conditions": len(g164),
        "conditions": g164,
    }

    counts = PATEL_DIR / "GSE196839_counts.txt"
    n_rows = sum(1 for _ in counts.open()) - 1 if counts.exists() else None
    c = ks["GSE196839"]["contrasts"]["siGGGGGC"]
    out["Patel_2022_GSE196839"] = {
        "paper": "Patel et al. 2022",
        "geo": "GSE196839",
        "organism": "human (PC3 prostate)",
        "effect_type": "count-derived CPM log2((siGGGGGC+1)/(siNT1+1)); single replicate",
        "n_conditions": 1,
        "conditions": {
            "siGGGGGC": {
                "n_genes": n_rows,
                "n_seed_matched": c["n_seed_match"],
                "n_with_3utr": c["n_with_3utr"],
                "ks_D": c["ks_D"],
                "ks_p": c["ks_p"],
            }
        },
    }
    return out


def mcgeary_matrix() -> dict:
    """Real (seed, transcript, repression) example count from the cached McGeary matrix."""
    info = {
        "paper": "McGeary, Lin et al. 2019, Science (biochemical basis of microRNA targeting efficacy)",
        "geo": "GSE140217",
        "repression_matrix_url": MCGEARY_URL,
        "repression_matrix_bytes": MCGEARY_BYTES,
        "kd_models": "AGO-RBNS 12-mer Kd affinities; modeled Kd available via scanMiR / scanMiRData "
        "(Bioconductor) and github.com/kslin/miRNA_models.",
    }
    if MCGEARY_CACHE.exists():
        with gzip.open(MCGEARY_CACHE, "rt") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            n_transcripts = sum(1 for _ in fh)
        mirnas = [c for c in header if c.strip()]
        info.update(
            {
                "downloaded": True,
                "n_transcripts": int(n_transcripts),
                "n_transfected_mirnas": len(mirnas),
                "mirnas": mirnas,
                "n_measured_examples": int(n_transcripts) * len(mirnas),
                "examples_note": "REAL measured (miRNA seed, transcript, log-TPM repression) examples "
                "= n_transcripts x n_transfected_mirnas.",
            }
        )
    else:
        info.update(
            {
                "downloaded": False,
                "n_measured_examples": None,
                "examples_note": f"fetch the {MCGEARY_BYTES} byte matrix to count exact dims (rows x "
                "transfected miRNAs); ~4.3k transcripts x 17 miRNAs in the deposit.",
            }
        )
    return info


def mirna_aux() -> dict:
    return {
        "targetscan_context_plus_plus": {
            "version": "TargetScan v8.0 (vert_80)",
            "base_url": TS_BASE,
            "files": [
                {"name": n, "url": TS_BASE + n, "bytes": b, "downloaded": False, "note": note}
                for (n, b, note) in TARGETSCAN_FILES
            ],
            "license": "publicly downloadable (Bartel lab / Whitehead Institute, MIT); free for "
            "academic / non-commercial use; commercial terms to be confirmed.",
            "examples": "context++ scores are PREDICTIONS (a repression proxy), not measured "
            "repression: millions of (miRNA family, target site) rows across ~150 conserved miRNA "
            "families and human 3'UTRs.",
            "sizes_from": "HTTP HEAD on the listed URLs (no file downloaded)",
        },
        "mcgeary_2019_agorbns": {
            **mcgeary_matrix(),
            "license": "GEO data public; scanMiR / scanMiRData on Bioconductor (open source); "
            "github.com/kslin/miRNA_models for the model code (confirm repo license).",
            "urls": {
                "geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140217",
                "github": "https://github.com/kslin/miRNA_models",
                "scanmir": "https://bioconductor.org/packages/scanMiR",
            },
        },
    }


def run() -> dict:
    schlegel = schlegel_corpus()
    patel = patel_corpus()
    studies = {"Schlegel_2022_GSE184929": schlegel, **patel}

    n_geo_series = len(studies)
    n_independent_papers = 2  # Schlegel 2022; Patel 2022 (GSE164289 + GSE196839 share reagents)
    n_conditions_total = sum(s["n_conditions"] for s in studies.values())

    aux = mirna_aux()
    mcg = aux["mcgeary_2019_agorbns"]
    aux_examples = mcg.get("n_measured_examples")

    loso_ok = n_independent_papers >= 5
    verdict = {
        "n_geo_series": n_geo_series,
        "n_independent_papers": n_independent_papers,
        "n_conditions_total": n_conditions_total,
        "loso_folds_available": n_independent_papers,
        "learned_predictor_trainable_on_offtarget_corpus_with_loso": loso_ok,
        "reasoning": "Leave-one-study-out needs on the order of 5+ independent studies for a "
        "meaningful held-out transfer estimate. Here there are only 2 truly independent studies "
        "(Schlegel 2022 rat; Patel 2022 human) across 3 GEO series and "
        f"{n_conditions_total} conditions, and the conditions within a study share siRNA reagents, "
        "seeds, and cell context, so they are not independent folds. Training a learned off-target "
        "predictor ON the real off-target corpus with LOSO is NOT viable: the independent-study "
        "count is low single digits.",
        "recommendation": "Use the miRNA auxiliary corpus as the TRAINING signal "
        f"(McGeary 2019: {aux_examples} real measured (seed, 3'UTR, repression) examples across "
        f"{mcg.get('n_transfected_mirnas')} seeds, plus TargetScan context++ and AGO-RBNS Kd), and "
        "hold out the 2-3 real siRNA off-target studies (GSE184929 / GSE164289 / GSE196839) as a "
        "small leave-one-study-out TRANSFER-validation set, not as the training corpus.",
    }

    return {
        "status": "ok",
        "purpose": "feasibility probe for a learned off-target predictor; data assessment only, no build",
        "real_offtarget_corpus": {
            "studies": studies,
            "n_geo_series": n_geo_series,
            "n_independent_papers": n_independent_papers,
            "n_conditions_total": n_conditions_total,
            "independence_note": "GSE164289 and GSE196839 are both Patel 2022 (same lab; the "
            "siGGGGGC reagent appears in both, different cell lines), so they are NOT statistically "
            "independent studies; the count of truly independent studies is 2.",
        },
        "mirna_auxiliary_corpus": aux,
        "verdict": verdict,
    }


def main() -> int:
    numbers = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    print("[ot-feasibility] REAL supervised off-target corpus (per study / per condition):")
    print(f"  {'study':<26} {'condition':<26} {'n_genes':>8} {'n_seed_matched':>14} {'KS_D':>8}")
    for sname, s in numbers["real_offtarget_corpus"]["studies"].items():
        for cname, c in s["conditions"].items():
            d = c.get("ks_D")
            print(
                f"  {sname:<26} {cname:<26} {c.get('n_genes')!s:>8} "
                f"{c.get('n_seed_matched')!s:>14} {('n/a' if d is None else f'{d:>8.4f}')}"
            )
    rc = numbers["real_offtarget_corpus"]
    print(
        f"[ot-feasibility] {rc['n_geo_series']} GEO series | {rc['n_independent_papers']} independent "
        f"papers | {rc['n_conditions_total']} conditions"
    )
    mcg = numbers["mirna_auxiliary_corpus"]["mcgeary_2019_agorbns"]
    print(
        f"[ot-feasibility] aux corpus McGeary GSE140217: {mcg.get('n_measured_examples')} measured "
        f"(seed, 3'UTR, repression) examples across {mcg.get('n_transfected_mirnas')} seeds"
    )
    v = numbers["verdict"]
    print(
        f"[ot-feasibility] VERDICT: learned predictor trainable on the off-target corpus with LOSO = "
        f"{v['learned_predictor_trainable_on_offtarget_corpus_with_loso']} "
        f"({v['n_independent_papers']} independent studies, low single digits)"
    )
    print(f"[ot-feasibility] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
