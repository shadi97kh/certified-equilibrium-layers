"""Loader for GSE184929 (primary off-target ground truth).

GEO accession GSE184929: GalNAc-siRNA seed-destabilization RNA-seq in rat
liver, Ttr-targeting guides D1 and D4 at 10 and 30 mg/kg. The processed
supplementary table is a per-transcript matrix of log2 fold-changes versus
vehicle control across the four dose / guide contrasts.

See docs/DATA_CARD.md section 1 for the rat-liver specificity caveat. The
loader exposes per-transcript log2FC labels and caches a tidy parquet.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import CacheKey, cache_dir_for, sha256_file
from .paths import cache_dir as default_cache_dir
from .paths import raw_dir as default_raw_dir
from .roles import OFFTARGET_CERTIFICATION

PARSER_VERSION = "gse184929-v1"

# GSE184929 is the RNA-seq off-target ground truth (role: offtarget_certification).
ROLE = OFFTARGET_CERTIFICATION

# The four primary dose / guide contrasts documented in the data card.
DEFAULT_CONTRASTS = ("D1_10mg", "D1_30mg", "D4_10mg", "D4_30mg")

_ID_CANDIDATES = (
    "transcript_id",
    "transcript",
    "gene_id",
    "gene",
    "ensembl",
    "ensembl_id",
    "symbol",
    "feature",
    "id",
)
# Substrings that mark a column as a log2 fold-change contrast.
_LFC_MARKERS = ("log2fc", "log2foldchange", "logfc", "lfc", "log2")
# e.g. D1_10mg, D4_30mg (guide + dose).
_CONTRAST_RE = re.compile(r"^d\d+_\d+\s*mg$", re.IGNORECASE)


@dataclass(frozen=True)
class GEOOffTargetDataset:
    """Per-transcript off-target log2FC labels for GSE184929.

    ``log2fc`` is a DataFrame indexed by transcript id whose columns are the
    detected contrasts. ``transcript_ids`` mirrors the index as an array.
    """

    transcript_ids: np.ndarray
    log2fc: pd.DataFrame
    contrasts: list[str]
    role: str = OFFTARGET_CERTIFICATION

    def __len__(self) -> int:
        return len(self.transcript_ids)

    def labels(self, contrast: str) -> np.ndarray:
        """Return the per-transcript log2FC vector for one contrast."""
        if contrast not in self.contrasts:
            raise KeyError(f"unknown contrast {contrast!r}; have {self.contrasts}")
        return self.log2fc[contrast].to_numpy(dtype=float)


def _pick_id_column(columns: list[str], override: str | None) -> str:
    if override is not None:
        if override not in columns:
            raise ValueError(f"id_column {override!r} not in {columns}")
        return override
    lowered = {c.lower(): c for c in columns}
    for cand in _ID_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    # Fall back to the first column (GEO matrices put the feature id first).
    return columns[0]


def _detect_contrasts(df: pd.DataFrame, id_col: str) -> list[str]:
    contrasts: list[str] = []
    for col in df.columns:
        if col == id_col:
            continue
        low = col.lower().replace(" ", "")
        if any(m in low for m in _LFC_MARKERS) or _CONTRAST_RE.match(col.strip()):
            contrasts.append(col)
    if contrasts:
        return contrasts
    # Last resort: every numeric non-id column is treated as a contrast.
    numeric = df.drop(columns=[id_col]).select_dtypes(include="number")
    return list(numeric.columns)


def _parse(
    path: Path,
    contrasts: tuple[str, ...] | None,
    id_column: str | None,
) -> GEOOffTargetDataset:
    df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    cols = list(df.columns)
    id_col = _pick_id_column(cols, id_column)

    if contrasts is not None:
        missing = [c for c in contrasts if c not in cols]
        if missing:
            raise ValueError(f"requested contrasts not in file: {missing}; have {cols}")
        contrast_cols = list(contrasts)
    else:
        contrast_cols = _detect_contrasts(df, id_col)
    if not contrast_cols:
        raise ValueError(f"no log2FC contrast columns detected in {cols}")

    ids = df[id_col].astype(str).to_numpy(dtype=object)
    lfc = df[contrast_cols].apply(pd.to_numeric, errors="coerce")
    lfc.index = pd.Index(ids, name="transcript_id")

    return GEOOffTargetDataset(
        transcript_ids=ids,
        log2fc=lfc,
        contrasts=contrast_cols,
    )


def _cache_files(cdir: Path) -> tuple[Path, Path]:
    return cdir / "log2fc.parquet", cdir / "meta.json"


def _write_cache(cdir: Path, ds: GEOOffTargetDataset) -> None:
    parquet, meta = _cache_files(cdir)
    ds.log2fc.to_parquet(parquet)
    meta.write_text(json.dumps({"contrasts": ds.contrasts}), encoding="utf-8")


def _read_cache(cdir: Path) -> GEOOffTargetDataset:
    parquet, meta = _cache_files(cdir)
    lfc = pd.read_parquet(parquet)
    contrasts = json.loads(meta.read_text(encoding="utf-8"))["contrasts"]
    return GEOOffTargetDataset(
        transcript_ids=lfc.index.to_numpy(dtype=object),
        log2fc=lfc,
        contrasts=contrasts,
    )


def default_raw_path() -> Path:
    """Default location for the processed GSE184929 supplementary table."""
    return default_raw_dir() / "GSE184929" / "GSE184929_processed_log2fc.tsv"


def load_gse184929(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    contrasts: tuple[str, ...] | None = None,
    id_column: str | None = None,
    rebuild: bool = False,
) -> GEOOffTargetDataset:
    """Load GSE184929 per-transcript log2FC labels, caching the parsed form.

    Args:
        raw_path: Processed counts / fold-change table. Defaults to
            ``data/raw/GSE184929/GSE184929_processed_log2fc.tsv``.
        cache_root: Cache root. Defaults to ``data/cache``.
        contrasts: Explicit contrast column names. If omitted, columns are
            auto-detected (log2FC markers or the ``D<g>_<dose>mg`` pattern).
        id_column: Explicit transcript-id column. If omitted, a common id name
            or the first column is used.
        rebuild: Reparse and overwrite the cache even if present.

    Returns:
        A :class:`GEOOffTargetDataset`.
    """
    raw_path = Path(raw_path) if raw_path is not None else default_raw_path()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"GSE184929 processed table not found at {raw_path}. Run "
            "scripts/download_data.sh and place the processed supplementary file "
            "there; see docs/DATA_CARD.md section 1."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()

    key = CacheKey(
        name="gse184929",
        version=PARSER_VERSION,
        params={
            "sha256": sha256_file(raw_path),
            "contrasts": list(contrasts) if contrasts is not None else None,
            "id_column": id_column,
        },
    )
    cdir = cache_dir_for(cache_root, key)
    parquet, meta = _cache_files(cdir)
    if not rebuild and parquet.exists() and meta.exists():
        return _read_cache(cdir)

    ds = _parse(raw_path, contrasts=contrasts, id_column=id_column)
    _write_cache(cdir, ds)
    return ds
