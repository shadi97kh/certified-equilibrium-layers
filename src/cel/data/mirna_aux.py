"""miRNA auxiliary corpus: real (miRNA seed, transcript 3'UTR, measured repression) examples.

This is the TRAINING signal for the CASCADE-OT off-target predictor. It loads the McGeary
2019 (GSE140217) HeLa transfection log-TPM matrix (4279 RefSeq transcripts by 17 transfected
miRNAs) and, for each transfected miRNA, takes its seed (mature positions 2 to 8), resolves
each transcript's 3'UTR via pyensembl (RefSeq accession to gene symbol to GRCh38/104 3'UTR),
enumerates the canonical 3'UTR seed-match sites, and pairs the measured log-TPM repression of
that transcript with the seed-match site features. These features (site type, site count,
ViennaRNA seed-duplex dG) are exactly the inputs the predictor shares with the siRNA
off-target task.

Provenance (all real, fetched and cached under data/raw/offtarget/mirna_aux/, gitignored):
  - repression matrix: GSE140217_HeLa_transfection_logtpm_batchnormalized.txt.gz (McGeary 2019)
  - seeds: mirna_seeds.json, mature guides from miRBase (canonical guide arm; the 16 vertebrate
    seeds were cross-checked against TargetScan v8 Seed+m8; lsy-6 uses cel-lsy-6-5p, a
    non-mammalian control, arm to be confirmed against the McGeary supplement)
  - RefSeq to gene symbol: refseq2symbol.json, resolved via mygene.info (4279/4279)

REAL DATA ONLY: every loader RAISES when its cached real input is absent; nothing is
fabricated and no values are generated.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
AUX_DIR = REPO / "data" / "raw" / "offtarget" / "mirna_aux"
MATRIX_PATH = AUX_DIR / "GSE140217_HeLa_transfection_logtpm_batchnormalized.txt.gz"
SEEDS_PATH = AUX_DIR / "mirna_seeds.json"
REFSEQ2SYMBOL_PATH = AUX_DIR / "refseq2symbol.json"

ENSEMBL_RELEASE = 104  # GRCh38, matching the Patel human off-target 3'UTR build
MCGEARY_N_TRANSCRIPTS = 4279
MCGEARY_N_MIRNAS = 17
MCGEARY_N_CELLS = MCGEARY_N_TRANSCRIPTS * MCGEARY_N_MIRNAS  # 72743 transcript-by-miRNA cells
SEED_LEN = 7
_RNA_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


@dataclass(frozen=True)
class MirnaExample:
    """One real (miRNA seed, transcript 3'UTR, measured repression) training example."""

    mirna: str
    seed: str
    refseq: str
    symbol: str | None
    log_tpm_repression: float
    has_3utr: bool
    utr_len: int
    best_site_type: str
    site_count: int
    seed_dg: float


def load_matrix() -> pd.DataFrame:
    """McGeary HeLa transfection log-TPM matrix (RefSeq index, 17 miRNA columns). Raises if absent."""
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"McGeary repression matrix not found at {MATRIX_PATH}. Fetch GSE140217 "
            "GSE140217_HeLa_transfection_logtpm_batchnormalized.txt.gz from GEO. REAL DATA ONLY."
        )
    with gzip.open(MATRIX_PATH, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    return df


def load_seeds() -> dict:
    """17 transfected miRNAs: name to {mirbase_id, mature, seed_2_8}. Raises if absent."""
    if not SEEDS_PATH.exists():
        raise FileNotFoundError(
            f"miRNA seed table not found at {SEEDS_PATH}. Build it from miRBase mature guides "
            "(positions 2 to 8). REAL DATA ONLY."
        )
    seeds = json.loads(SEEDS_PATH.read_text())
    for name, entry in seeds.items():
        if len(entry.get("seed_2_8", "")) != SEED_LEN:
            raise ValueError(f"seed for {name} is not a {SEED_LEN}-mer: {entry}")
    return seeds


def load_refseq2symbol() -> dict:
    """RefSeq accession (version-stripped) to gene symbol. Raises if absent."""
    if not REFSEQ2SYMBOL_PATH.exists():
        raise FileNotFoundError(
            f"RefSeq to symbol map not found at {REFSEQ2SYMBOL_PATH}. Resolve it via mygene.info "
            "for the McGeary transcript ids. REAL DATA ONLY."
        )
    return json.loads(REFSEQ2SYMBOL_PATH.read_text())


def _revcomp_rna(seq: str) -> str:
    return "".join(_RNA_COMPLEMENT[b] for b in reversed(seq.upper().replace("T", "U")))


def seed_duplex_dg(seed: str) -> float:
    """ViennaRNA free energy of the seed paired to its perfect complement site. Raises if absent."""
    try:
        import RNA
    except ImportError as exc:
        raise ImportError("the seed-duplex dG needs ViennaRNA (the RNA module)") from exc
    return float(RNA.duplexfold(seed.upper().replace("T", "U"), _revcomp_rna(seed)).energy)


def _longest_3utr(release, symbol: str | None, cache: dict) -> str | None:
    if symbol is None:
        return None
    if symbol in cache:
        return cache[symbol]
    best = ""
    try:
        genes = release.genes_by_name(symbol)
    except Exception:
        genes = []
    for gene in genes:
        for transcript in gene.transcripts:
            utr = transcript.three_prime_utr_sequence
            if utr and len(utr) > len(best):
                best = utr.upper()
    cache[symbol] = best or None
    return cache[symbol]


def build_examples(limit: int | None = None) -> pd.DataFrame:
    """Emit the real (seed, 3'UTR, repression) examples: one row per transcript-by-miRNA cell.

    With limit None, n rows equals MCGEARY_N_CELLS (72743) before any site filtering (asserted).
    Cells whose transcript resolves to a GRCh38/104 3'UTR carry the seed-match site features
    (best site type, site count) and the per-seed ViennaRNA seed-duplex dG; cells without a
    resolvable 3'UTR keep site_count 0 and best_site_type 'no_3utr' (reported, never dropped
    silently). limit restricts to the first N transcripts for fast tests.
    """
    from pyensembl import EnsemblRelease

    from cel.features.seedmatch import best_site, find_sites

    matrix = load_matrix()
    seeds = load_seeds()
    refseq2symbol = load_refseq2symbol()
    mirnas = list(matrix.columns)
    release = EnsemblRelease(ENSEMBL_RELEASE, species="human")

    dgs = {name: seed_duplex_dg(entry["seed_2_8"]) for name, entry in seeds.items()}
    utr_cache: dict = {}
    rows: list[dict] = []
    index = matrix.index if limit is None else matrix.index[:limit]
    for refseq in index:
        symbol = refseq2symbol.get(str(refseq).split(".")[0])
        utr = _longest_3utr(release, symbol, utr_cache)
        for mirna in mirnas:
            entry = seeds[mirna]
            repression = float(matrix.at[refseq, mirna])
            if utr:
                sites = find_sites(entry["mature"], utr)
                best = best_site(sites)
                site_count = len(sites)
                best_type = best.site_type if best else "none"
            else:
                site_count = 0
                best_type = "no_3utr"
            rows.append(
                {
                    "mirna": mirna,
                    "seed": entry["seed_2_8"],
                    "refseq": str(refseq),
                    "symbol": symbol,
                    "log_tpm_repression": repression,
                    "has_3utr": utr is not None,
                    "utr_len": len(utr) if utr else 0,
                    "best_site_type": best_type,
                    "site_count": site_count,
                    "seed_dg": dgs[mirna],
                }
            )
    df = pd.DataFrame(rows)
    if limit is None and len(df) != MCGEARY_N_CELLS:
        raise ValueError(
            f"expected {MCGEARY_N_CELLS} transcript-by-miRNA cells "
            f"({MCGEARY_N_TRANSCRIPTS} x {MCGEARY_N_MIRNAS}), built {len(df)}"
        )
    return df
