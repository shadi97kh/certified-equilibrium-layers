"""Build the CASCADE-OT training corpus from the miRNA auxiliary signal and freeze the split.

Calls cel.data.mirna_aux.build_examples to emit the real (miRNA seed, transcript 3'UTR,
measured log-TPM repression) examples with seed-match site features, writes the full example
table to a gitignored cache, summarizes it to results/cascade_ot/corpus_built.json, and writes
the FROZEN train / held-out-transfer split to results/cascade_ot/corpus_split.json.

The split policy is fixed: the miRNA auxiliary corpus (McGeary) is the TRAINING set; the real
siRNA off-target studies (GSE184929 Schlegel rat, GSE164289 + GSE196839 Patel human) are the
held-out transfer-validation set and are NEVER in training.

REAL DATA ONLY: the examples come from cel.data.mirna_aux (which raises on missing inputs);
the held-out condition counts are read from the committed results/cascade_ot/data_feasibility.json.
No LaTeX.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results" / "cascade_ot"
EXAMPLES_CACHE = REPO / "data" / "raw" / "offtarget" / "mirna_aux" / "cascade_ot_examples.csv.gz"
FEASIBILITY = RESULTS / "data_feasibility.json"


def held_out_conditions() -> dict:
    """The 6 frozen siRNA off-target conditions and their measured-gene counts (transfer set)."""
    if not FEASIBILITY.exists():
        raise FileNotFoundError(f"{FEASIBILITY} absent; run ot_data_feasibility first")
    feas = json.loads(FEASIBILITY.read_text())["real_offtarget_corpus"]
    conditions = {}
    total = 0
    for study_name, study in feas["studies"].items():
        for cond_name, c in study["conditions"].items():
            n_measured = c.get("n_genes_with_log2fc") or c.get("n_genes")
            conditions[f"{study['geo']}:{cond_name}"] = {
                "study": study_name,
                "geo": study["geo"],
                "organism": study["organism"],
                "n_genes": c.get("n_genes"),
                "n_genes_with_log2fc": c.get("n_genes_with_log2fc"),
                "n_seed_matched": c.get("n_seed_matched"),
            }
            total += int(n_measured or 0)
    return {
        "n_independent_studies": feas["n_independent_papers"],
        "n_conditions": feas["n_conditions_total"],
        "conditions": conditions,
        "n_examples_total_measured_genes": total,
    }


def run() -> dict:
    from cel.data import mirna_aux as ma

    df = ma.build_examples()
    EXAMPLES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXAMPLES_CACHE, index=False, compression="gzip")

    with_3utr = df[df["has_3utr"]]
    with_site = df[df["site_count"] > 0]
    site_cells = df["site_count"] > 0
    # Real signal check: site-bearing cells should be more repressed (more negative log-TPM-FC).
    mean_rep_site = (
        float(df.loc[site_cells, "log_tpm_repression"].mean()) if site_cells.any() else None
    )
    mean_rep_nosite = (
        float(df.loc[~site_cells & df["has_3utr"], "log_tpm_repression"].mean())
        if (~site_cells & df["has_3utr"]).any()
        else None
    )

    seeds = ma.load_seeds()
    per_mirna = {}
    for mirna, grp in df.groupby("mirna"):
        s = grp["site_count"] > 0
        per_mirna[mirna] = {
            "seed": seeds[mirna]["seed_2_8"],
            "mirbase_id": seeds[mirna]["mirbase_id"],
            "seed_dg": float(grp["seed_dg"].iloc[0]),
            "n_with_seed_site": int(s.sum()),
            "mean_repression_site": float(grp.loc[s, "log_tpm_repression"].mean())
            if s.any()
            else None,
        }

    corpus = {
        "status": "ok",
        "corpus": "CASCADE-OT miRNA auxiliary training corpus",
        "source": "McGeary 2019 GSE140217 HeLa transfection log-TPM (4279 transcripts x 17 miRNAs)",
        "build": "RefSeq to symbol (mygene) to GRCh38/104 3'UTR (pyensembl); canonical seed-match "
        "sites (cel.features.seedmatch); ViennaRNA seed-duplex dG per seed",
        "n_cells_prefilter": int(len(df)),
        "n_cells_prefilter_expected": ma.MCGEARY_N_CELLS,
        "n_transcripts": ma.MCGEARY_N_TRANSCRIPTS,
        "n_mirnas": ma.MCGEARY_N_MIRNAS,
        "n_transcripts_with_3utr": int(df.loc[df["has_3utr"], "refseq"].nunique()),
        "n_examples_with_3utr": int(len(with_3utr)),
        "n_examples_with_seed_site": int(len(with_site)),
        "feature_schema": ["best_site_type", "site_count", "seed_dg"],
        "site_type_counts": {k: int(v) for k, v in df["best_site_type"].value_counts().items()},
        "signal_check": {
            "mean_log_tpm_repression_site_bearing": mean_rep_site,
            "mean_log_tpm_repression_no_site": mean_rep_nosite,
            "site_more_repressed": (
                bool(mean_rep_site < mean_rep_nosite)
                if (mean_rep_site is not None and mean_rep_nosite is not None)
                else None
            ),
        },
        "per_mirna": per_mirna,
        "examples_cache": str(EXAMPLES_CACHE.relative_to(REPO)) + " (gitignored)",
    }

    held = held_out_conditions()
    split = {
        "status": "ok",
        "frozen": True,
        "policy": "train ONLY on the miRNA auxiliary corpus; the real siRNA off-target studies are "
        "the held-out transfer-validation set and are NEVER in training",
        "train": {
            "source": "McGeary 2019 GSE140217 miRNA auxiliary corpus",
            "n_cells": int(len(df)),
            "n_examples_with_3utr": int(len(with_3utr)),
            "n_examples_with_seed_site": int(len(with_site)),
            "n_mirnas": ma.MCGEARY_N_MIRNAS,
        },
        "held_out_transfer": {**held, "never_in_training": True},
    }
    return {"corpus": corpus, "split": split}


def main() -> int:
    out = run()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "corpus_built.json").write_text(
        json.dumps(out["corpus"], indent=2), encoding="utf-8"
    )
    (RESULTS / "corpus_split.json").write_text(json.dumps(out["split"], indent=2), encoding="utf-8")

    c, s = out["corpus"], out["split"]
    print(
        f"[ot-corpus] built {c['n_cells_prefilter']} cells "
        f"(expected {c['n_cells_prefilter_expected']}) | "
        f"{c['n_transcripts_with_3utr']}/{c['n_transcripts']} transcripts have a 3'UTR | "
        f"{c['n_examples_with_seed_site']} examples carry a seed-match site"
    )
    sc = c["signal_check"]
    print(
        f"[ot-corpus] signal: mean repression site {sc['mean_log_tpm_repression_site_bearing']:.4f} "
        f"vs no-site {sc['mean_log_tpm_repression_no_site']:.4f} "
        f"(site more repressed: {sc['site_more_repressed']})"
    )
    print(
        f"[ot-corpus] FROZEN split: train McGeary {s['train']['n_cells']} cells "
        f"({s['train']['n_examples_with_seed_site']} with site) | held-out transfer "
        f"{s['held_out_transfer']['n_conditions']} conditions across "
        f"{s['held_out_transfer']['n_independent_studies']} independent studies (never in training)"
    )
    print(f"[ot-corpus] wrote {RESULTS / 'corpus_built.json'} and {RESULTS / 'corpus_split.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
