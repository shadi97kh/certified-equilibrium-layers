"""Measured off-target signature loaders (role: offtarget_certification).

These are the datasets the off-target containment certificate is evaluated
against (the efficacy datasets are NOT; see docs/DATA_CARD.md and
:mod:`cel.data.roles`).

- GSE184929 (RNA-seq) is wired separately in :mod:`cel.data.geo`.
- Jackson 2003 (Nat Biotechnol) and Jackson 2006 (RNA): microarray off-target
  signatures, down-regulated transcripts with 3'UTR seed complementarity. Some
  tables live in the paper supplements rather than a clean GEO matrix, so this
  module parses supplement tables and records the provenance; where a GEO
  accession exists, point the loader at it.
- Birmingham 2006 (Nat Methods): seed-match off-target associations.
- SeedSeq (Das 2013): the aggregated off-target transcriptome table, a
  convenience union over the above with a ``source`` flag on every call.

Each loader returns an :class:`OffTargetDataset` whose frame has columns
siRNA id, antisense, seed, off-target transcript, effect size, cell line or
tissue, assay, and source. Microarray sets are lower resolution and use older
annotation than the GSE184929 RNA-seq; this is recorded honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import CacheKey, cached_frame, sha256_file
from .paths import cache_dir as default_cache_dir
from .paths import raw_dir as default_raw_dir
from .roles import OFFTARGET_CERTIFICATION

PARSER_VERSION = "offtarget-v1"

COLUMNS = [
    "sirna_id",
    "antisense",
    "seed",
    "off_target_transcript",
    "effect_size",
    "cell_line",
    "assay",
    "source",
]

_SIRNA = ("sirna_id", "sirna", "guide_id", "sirna_name", "oligo", "id")
_ANTISENSE = ("antisense", "guide", "antisense_sequence", "as_seq")
_SEED = ("seed", "seed_region", "seed_match", "seed_seq")
_TRANSCRIPT = (
    "off_target_transcript",
    "transcript",
    "off_target",
    "target_transcript",
    "refseq",
    "probe",
    "gene",
)
_EFFECT = (
    "effect_size",
    "log2fc",
    "log2_fold_change",
    "fold_change",
    "logratio",
    "ratio",
    "zscore",
    "score",
)
_CELL = ("cell_line", "cell", "tissue", "cellline")
_ASSAY = ("assay", "assay_type", "platform")
_SOURCE = ("source", "origin", "dataset", "study")

# Fixed provenance per dataset.
OFFTARGET_SOURCES: dict[str, dict] = {
    "jackson2003": {
        "cell_line": "HeLa",
        "assay": "microarray",
        "provenance": "Jackson 2003 (Nat Biotechnol); supplement tables (no clean GEO matrix)",
    },
    "jackson2006": {
        "cell_line": "HeLa",
        "assay": "microarray",
        "provenance": "Jackson 2006 (RNA 12:1179) + companion (RNA 12:1197); microarray data in GEO "
        "GSE5814 / GSE5769 (Homo sapiens), but the siRNA ANTISENSE SEQUENCES needed to derive the "
        "seed are NOT in the GEO deposit (samples carry only target-gene and seed-mutant labels); "
        "the sequences are only in the paper supplement (PMC1484447, not in the PMC Open Access "
        "subset). Native seed-match KS not reproducible from public data.",
    },
    "birmingham2006": {
        "cell_line": "various",
        "assay": "microarray",
        "provenance": "Birmingham 2006 (Nat Methods 3:199, nmeth854); seed-match off-target "
        "associations; NOT deposited in GEO; supplementary-table only.",
    },
    "seedseq": {
        "cell_line": "various",
        "assay": "microarray (aggregated)",
        "provenance": "SeedSeq / Das 2013; aggregated union over Jackson and Birmingham",
    },
}


@dataclass(frozen=True)
class OffTargetDataset:
    """Measured off-target calls for one source (role: offtarget_certification)."""

    name: str
    frame: pd.DataFrame
    assay: str
    provenance: str
    role: str = field(default=OFFTARGET_CERTIFICATION)

    def __len__(self) -> int:
        return len(self.frame)

    def sirnas(self) -> list[str]:
        return sorted(self.frame["sirna_id"].unique().tolist())

    def sources(self) -> list[str]:
        return sorted(self.frame["source"].unique().tolist())

    def transcripts(self) -> list[str]:
        return sorted(self.frame["off_target_transcript"].unique().tolist())


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:
        for low, orig in lowered.items():
            if cand in low:
                return orig
    return None


def _seed_from_antisense(antisense: pd.Series) -> pd.Series:
    """Seed = guide positions 2-8 (RNA), derived when no seed column is present."""
    rna = antisense.astype(str).str.strip().str.upper().str.replace("T", "U", regex=False)
    return rna.str.slice(1, 8)


def _parse_offtarget(path: Path, name: str) -> pd.DataFrame:
    meta = OFFTARGET_SOURCES[name]
    df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    cols = list(df.columns)
    transcript = _pick(cols, _TRANSCRIPT)
    if transcript is None:
        raise ValueError(
            f"{path}: need an off-target transcript column ({_TRANSCRIPT}); found {cols}"
        )
    n = len(df)
    sirna = _pick(cols, _SIRNA)
    anti = _pick(cols, _ANTISENSE)
    seed = _pick(cols, _SEED)
    effect = _pick(cols, _EFFECT)
    cell = _pick(cols, _CELL)
    assay = _pick(cols, _ASSAY)
    src = _pick(cols, _SOURCE)

    antisense = df[anti].astype(str).str.strip().str.upper() if anti else pd.Series([""] * n)
    if seed:
        seed_series = df[seed].astype(str).str.strip().str.upper()
    elif anti:
        seed_series = _seed_from_antisense(df[anti])
    else:
        seed_series = pd.Series([""] * n)

    return pd.DataFrame(
        {
            "sirna_id": (df[sirna].astype(str) if sirna else [f"{name}_sirna"] * n),
            "antisense": antisense.to_numpy(),
            "seed": seed_series.to_numpy(),
            "off_target_transcript": df[transcript].astype(str).to_numpy(),
            "effect_size": (
                pd.to_numeric(df[effect], errors="coerce").to_numpy()
                if effect
                else [float("nan")] * n
            ),
            "cell_line": (df[cell].astype(str) if cell else [meta["cell_line"]] * n),
            "assay": (df[assay].astype(str) if assay else [meta["assay"]] * n),
            "source": (df[src].astype(str) if src else [name] * n),
        }
    )[COLUMNS]


def _default_path(name: str) -> Path:
    return default_raw_dir() / "offtarget" / f"{name}.csv"


def _load_one(
    name: str, raw_path: str | Path | None, cache_root: str | Path | None, rebuild: bool
) -> OffTargetDataset:
    meta = OFFTARGET_SOURCES[name]
    raw_path = Path(raw_path) if raw_path is not None else _default_path(name)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{name} off-target table not found at {raw_path}. Place the parsed supplement "
            "table there; see docs/DATA_CARD.md."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()
    key = CacheKey(name=name, version=PARSER_VERSION, params={"sha256": sha256_file(raw_path)})
    frame = cached_frame(key, cache_root, lambda: _parse_offtarget(raw_path, name), rebuild=rebuild)
    return OffTargetDataset(
        name=name, frame=frame, assay=meta["assay"], provenance=meta["provenance"]
    )


def load_jackson2003(raw_path=None, cache_root=None, rebuild: bool = False) -> OffTargetDataset:
    """Jackson 2003 microarray off-target signatures (supplement tables)."""
    return _load_one("jackson2003", raw_path, cache_root, rebuild)


def load_jackson2006(raw_path=None, cache_root=None, rebuild: bool = False) -> OffTargetDataset:
    """Jackson 2006 microarray off-target signatures (supplement tables)."""
    return _load_one("jackson2006", raw_path, cache_root, rebuild)


def load_birmingham2006(raw_path=None, cache_root=None, rebuild: bool = False) -> OffTargetDataset:
    """Birmingham 2006 seed-match off-target associations (microarray)."""
    return _load_one("birmingham2006", raw_path, cache_root, rebuild)


def load_seedseq(raw_path=None, cache_root=None, rebuild: bool = False) -> OffTargetDataset:
    """SeedSeq (Das 2013) aggregated off-target union with a per-call source flag.

    Reads the aggregated table at ``data/raw/offtarget/seedseq.csv`` if present.
    Otherwise builds the convenience union from whichever individual sources
    (Jackson 2003/2006, Birmingham 2006) are available, tagging each call with
    its ``source``.
    """
    meta = OFFTARGET_SOURCES["seedseq"]
    raw_path = Path(raw_path) if raw_path is not None else _default_path("seedseq")
    if raw_path.exists():
        return _load_one("seedseq", raw_path, cache_root, rebuild)

    members = []
    for loader in (load_jackson2003, load_jackson2006, load_birmingham2006):
        try:
            members.append(loader(cache_root=cache_root, rebuild=rebuild).frame)
        except FileNotFoundError:
            continue
    if not members:
        raise FileNotFoundError(
            f"no SeedSeq table at {raw_path} and no member off-target sources available to "
            "build the union; see docs/DATA_CARD.md."
        )
    union = pd.concat(members, ignore_index=True)[COLUMNS]
    return OffTargetDataset(
        name="seedseq",
        frame=union,
        assay=meta["assay"],
        provenance=meta["provenance"] + " (built from available members)",
    )


# ===========================================================================
# SeedMatchR evaluation metric (off-target metric for Theorem 4), implemented
# natively in Python: a one-sided two-sample Kolmogorov-Smirnov test comparing
# the log2 fold-change ECDF of seed-matched transcripts against non-seed-matched
# (background) transcripts. Seed-matched transcripts are expected to be more
# repressed (more negative log2FC), so the default tests left-shift of the
# seed-matched distribution.
# ===========================================================================
@dataclass(frozen=True)
class KSResult:
    """One-sided KS comparison of seed-matched vs background log2FC."""

    statistic: float  # the KS D statistic (max signed ECDF gap)
    pvalue: float  # asymptotic one-sided / two-sided p-value
    n_seed_match: int
    n_background: int
    alternative: str


def _ks_pvalue_one_sided(d: float, ne: float) -> float:
    """Asymptotic one-sided two-sample KS p-value (Smirnov): exp(-2 ne D^2)."""
    return float(np.exp(-2.0 * ne * d * d))


def _ks_pvalue_two_sided(d: float, ne: float) -> float:
    """Asymptotic two-sided KS p-value via the Kolmogorov series Q(t)."""
    t = np.sqrt(ne) * d
    if t <= 0.0:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = ((-1.0) ** (k - 1)) * np.exp(-2.0 * k * k * t * t)
        total += term
        if abs(term) < 1e-12:
            break
    return float(min(max(2.0 * total, 0.0), 1.0))


def seedmatch_ks_test(
    log2fc,
    is_seed_match,
    alternative: str = "greater",
) -> KSResult:
    """One-sided KS test of seed-matched vs background log2FC (SeedMatchR metric).

    Args:
        log2fc: per-transcript log2 fold-change (array-like).
        is_seed_match: boolean mask, True where the transcript carries a seed
            match to the siRNA guide.
        alternative: which one-sided (or two-sided) statistic, matching
            ``scipy.stats.ks_2samp`` semantics with the seed-matched set as the
            first sample:
            - ``"greater"`` (default): D+ = sup(F_seed - F_background); large when
              seed-matched transcripts are stochastically MORE repressed (more
              negative log2FC). This is the SeedMatchR enrichment direction.
            - ``"less"``: D- = sup(F_background - F_seed).
            - ``"two-sided"``: D = max(D+, D-).

    Returns:
        A :class:`KSResult` with the D statistic and the asymptotic p-value.

    The D statistic is computed exactly from the empirical CDFs; the p-value is
    the asymptotic Smirnov (one-sided) or Kolmogorov-series (two-sided) form. The
    one-sided p-value uses the plain Smirnov tail exp(-2 ne D^2), matching R's
    ``ks.test`` (the backend SeedMatchR calls); this differs slightly from
    ``scipy.stats.ks_2samp``, which adds the Hodges continuity correction, but
    the D statistic is identical to scipy's.
    """
    values = np.asarray(log2fc, dtype=float)
    mask = np.asarray(is_seed_match, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"log2fc and is_seed_match shape mismatch: {values.shape} vs {mask.shape}")
    finite = np.isfinite(values)
    seed = np.sort(values[mask & finite])
    background = np.sort(values[~mask & finite])
    n, m = seed.size, background.size
    if n == 0 or m == 0:
        raise ValueError(
            "need at least one seed-matched and one background transcript with a finite "
            f"log2FC (got {n} seed-matched, {m} background)"
        )

    pooled = np.concatenate([seed, background])
    cdf_seed = np.searchsorted(seed, pooled, side="right") / n
    cdf_bg = np.searchsorted(background, pooled, side="right") / m
    diff = cdf_seed - cdf_bg
    d_plus = float(diff.max())  # sup(F_seed - F_background)
    d_minus = float((-diff).max())  # sup(F_background - F_seed)
    ne = n * m / (n + m)

    if alternative == "greater":
        d, p = d_plus, _ks_pvalue_one_sided(d_plus, ne)
    elif alternative == "less":
        d, p = d_minus, _ks_pvalue_one_sided(d_minus, ne)
    elif alternative == "two-sided":
        d = max(d_plus, d_minus)
        p = _ks_pvalue_two_sided(d, ne)
    else:
        raise ValueError(
            f"alternative must be 'greater', 'less', or 'two-sided'; got {alternative!r}"
        )
    return KSResult(statistic=d, pvalue=p, n_seed_match=n, n_background=m, alternative=alternative)


# ===========================================================================
# GSE184929 / Schlegel et al. 2022 (rat liver Ttr GalNAc-siRNA) RNA-seq, the
# REAL off-target ground truth. The SeedMatchR R package ships the DESeq2 results
# as example data; export them to CSV with scripts/export_seedmatchr_data.R, then
# load per-transcript log2FoldChange, p-value, and the seed-match annotation.
# ===========================================================================
SEEDMATCHR_SOURCE = (
    "Schlegel et al. 2022, GEO GSE184929 (rat liver Ttr GalNAc-siRNA): the real DESeq2 "
    "result tables are the GEO supplementary files; fetch them with scripts/download_offtarget.sh "
    "(no R needed). Equivalently, the SeedMatchR R package re-serves the same tables via "
    "AnnotationHub (install.packages('SeedMatchR'); scripts/export_seedmatchr_data.R)"
)

# Each dataset maps to its REAL GEO GSE184929 supplementary DESeq2 table (the
# authoritative source; SeedMatchR's AnnotationHub copy derives from the same deposit).
SEEDMATCHR_DATASETS: dict[str, dict] = {
    "Schlegel_2022_Ttr_D1_30mkg": {
        "sirna_id": "Ttr_D1",
        "guide_label": "D1",
        "dose": "30 mg/kg",
        "geo_suppl": "GSE184929_Fig3-TTR-2_groupD1_30mkg.vs.Mock.DESeq2.txt.gz",
    },
    "Schlegel_2022_Ttr_D4_30mkg": {
        "sirna_id": "Ttr_D4",
        "guide_label": "D4",
        "dose": "30 mg/kg",
        "geo_suppl": "GSE184929_Fig3-TTR-2_groupD4_30mkg.vs.Mock.DESeq2.txt.gz",
    },
    "Schlegel_2022_Ttr_D1_10mkg": {
        "sirna_id": "Ttr_D1",
        "guide_label": "D1",
        "dose": "10 mg/kg",
        "geo_suppl": "GSE184929_Fig3-TTR-1_groupD1_10mkg.vs.Mock.DESeq2.txt.gz",
    },
}

SEEDMATCHR_TISSUE = "rat liver (Rattus norvegicus)"
SEEDMATCHR_ASSAY = "RNA-seq (DESeq2 results)"

SEEDMATCHR_COLUMNS = [
    "transcript_id",
    "gene_name",
    "log2fc",
    "pvalue",
    "padj",
    "seed_match_count",
    "seed_match",
]

_SM_ID = ("transcript_id", "gene_id", "ensembl", "ensembl_id", "transcript", "gene", "id")
_SM_NAME = ("gene_name", "symbol", "name", "external_gene_name")
_SM_LFC = ("log2foldchange", "log2fc", "log2_fold_change", "logfc", "lfc")
_SM_P = ("pvalue", "p_value", "pval", "p")
_SM_PADJ = ("padj", "fdr", "qvalue", "adj_pvalue", "adj_p")
_SM_SEED = ("seed_match", "seed_matches", "seed_count", "seed_match_count", "n_seed", "matches")


@dataclass(frozen=True)
class SeedMatchRDataset:
    """Per-transcript RNA-seq off-target table for one Schlegel 2022 contrast.

    ``frame`` carries :data:`SEEDMATCHR_COLUMNS`: transcript id, gene name, the
    DESeq2 ``log2fc`` and ``pvalue`` (and ``padj`` when present), and the seed
    match annotation (``seed_match_count`` and the boolean-coded ``seed_match``,
    1.0 where the transcript has at least one guide seed match, else 0.0, NaN if
    the annotation was not exported).
    """

    name: str
    sirna_id: str
    dose: str
    frame: pd.DataFrame
    tissue: str = SEEDMATCHR_TISSUE
    assay: str = SEEDMATCHR_ASSAY
    role: str = field(default=OFFTARGET_CERTIFICATION)

    def __len__(self) -> int:
        return len(self.frame)

    def log2fc(self) -> np.ndarray:
        return self.frame["log2fc"].to_numpy(dtype=float)

    def pvalues(self) -> np.ndarray:
        return self.frame["pvalue"].to_numpy(dtype=float)

    @property
    def has_seed_annotation(self) -> bool:
        return "seed_match" in self.frame.columns and bool(self.frame["seed_match"].notna().any())

    def seed_match_mask(self) -> np.ndarray:
        if not self.has_seed_annotation:
            raise ValueError(
                f"{self.name} has no seed-match annotation; run the seed-match step in "
                "scripts/export_seedmatchr_data.R (a reference and the Ttr guide are required)"
            )
        return self.frame["seed_match"].to_numpy(dtype=float) == 1.0

    def ks(self, alternative: str = "greater") -> KSResult:
        """SeedMatchR KS test: seed-matched vs background log2FC for this contrast."""
        return seedmatch_ks_test(self.log2fc(), self.seed_match_mask(), alternative=alternative)


def _parse_seedmatchr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    cols = list(df.columns)
    lfc = _pick(cols, _SM_LFC)
    if lfc is None:
        raise ValueError(
            f"{path}: need a log2 fold-change column ({_SM_LFC}); found {cols}. Export with "
            "scripts/export_seedmatchr_data.R."
        )
    idc = _pick(cols, _SM_ID) or cols[0]  # write.csv puts the gene id in the first column
    name = _pick(cols, _SM_NAME)
    pcol = _pick(cols, _SM_P)
    padj = _pick(cols, _SM_PADJ)
    seed = _pick(cols, _SM_SEED)
    n = len(df)

    if seed is not None:
        count = pd.to_numeric(df[seed], errors="coerce").to_numpy(dtype=float)
        seed_match = np.where(np.isfinite(count), (count > 0).astype(float), np.nan)
    else:
        count = np.full(n, np.nan)
        seed_match = np.full(n, np.nan)

    return pd.DataFrame(
        {
            "transcript_id": df[idc].astype(str).to_numpy(),
            "gene_name": (df[name].astype(str).to_numpy() if name else [""] * n),
            "log2fc": pd.to_numeric(df[lfc], errors="coerce").to_numpy(dtype=float),
            "pvalue": (
                pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=float)
                if pcol
                else np.full(n, np.nan)
            ),
            "padj": (
                pd.to_numeric(df[padj], errors="coerce").to_numpy(dtype=float)
                if padj
                else np.full(n, np.nan)
            ),
            "seed_match_count": count,
            "seed_match": seed_match,
        }
    )[SEEDMATCHR_COLUMNS]


def default_seedmatchr_dir() -> Path:
    return default_raw_dir() / "offtarget" / "seedmatchr"


def load_schlegel2022(
    dataset: str,
    raw_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> SeedMatchRDataset:
    """Load one Schlegel 2022 GSE184929 contrast (REAL data; raise if absent).

    Args:
        dataset: one of :data:`SEEDMATCHR_DATASETS` (e.g.
            ``"Schlegel_2022_Ttr_D1_30mkg"``).
        raw_dir: directory holding ``<dataset>.csv``. Default
            ``data/raw/offtarget/seedmatchr``.
        cache_root: parquet cache root. Default ``data/cache``.
        rebuild: reparse and overwrite the cache.
    """
    if dataset not in SEEDMATCHR_DATASETS:
        raise KeyError(
            f"unknown SeedMatchR dataset {dataset!r}; known: {sorted(SEEDMATCHR_DATASETS)}"
        )
    meta = SEEDMATCHR_DATASETS[dataset]
    raw_dir = Path(raw_dir) if raw_dir is not None else default_seedmatchr_dir()
    path = raw_dir / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset} not found at {path}. This is the REAL off-target ground truth. "
            f"Fetch it: {SEEDMATCHR_SOURCE}. REAL DATA ONLY: no values are generated."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()
    key = CacheKey(
        name=f"seedmatchr_{dataset}",
        version=PARSER_VERSION,
        params={"sha256": sha256_file(path)},
    )
    frame = cached_frame(key, cache_root, lambda: _parse_seedmatchr(path), rebuild=rebuild)
    return SeedMatchRDataset(
        name=dataset, sirna_id=meta["sirna_id"], dose=meta["dose"], frame=frame
    )
