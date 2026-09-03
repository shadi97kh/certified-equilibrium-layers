"""Loader for the Huesken et al. 2005 siRNA efficacy benchmark.

Huesken D et al., Nature Biotechnology 23, 995-1001 (2005), DOI 10.1038/nbt1118.
2431 siRNAs across 34 mRNA targets, with a normalized knockdown efficacy in
[0, 1]. The conventional benchmark split is 2182 train / 249 test.

The loader is column-name tolerant (the supplementary table has circulated in a
few layouts) and caches a tidy parquet under ``data/cache/``. It is idempotent:
a second call with the same inputs reads the cache instead of reparsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import CacheKey, cache_dir_for, sha256_file
from .paths import cache_dir as default_cache_dir
from .paths import raw_dir as default_raw_dir
from .roles import EFFICACY_BACKBONE

# Bump when the parser output layout changes so stale caches are bypassed.
PARSER_VERSION = "huesken-v1"

# Huesken is one source of the on-target efficacy backbone (see efficacy.py).
ROLE = EFFICACY_BACKBONE

# Canonical benchmark sizes (used when the source file carries no split column).
N_TOTAL = 2431
N_TRAIN = 2182
N_TEST = 249

_SEQ_CANDIDATES = ("sequence", "seq", "sirna", "antisense", "guide", "sense")
_EFF_CANDIDATES = (
    "efficacy",
    "inhibition",
    "activity",
    "knockdown",
    "normalized_efficacy",
    "norm_inhibition",
    "value",
)
_SPLIT_CANDIDATES = ("split", "set", "fold", "partition")
_MRNA_CANDIDATES = ("mrna", "target", "gene", "transcript", "mrna_id")

_VALID_BASES = set("ACGUTN")


@dataclass(frozen=True)
class HueskenDataset:
    """Parsed Huesken efficacy benchmark.

    ``sequences`` are upper-cased siRNA strings, ``efficacy`` is a float array
    in [0, 1], ``split`` is an object array of ``"train"`` / ``"test"``, and
    ``mrna`` (optional) is the per-row mRNA target id.
    """

    sequences: list[str]
    efficacy: np.ndarray
    split: np.ndarray
    mrna: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.sequences)

    def _subset(self, which: str) -> tuple[list[str], np.ndarray]:
        mask = self.split == which
        seqs = [s for s, keep in zip(self.sequences, mask, strict=True) if keep]
        return seqs, self.efficacy[mask]

    def train(self) -> tuple[list[str], np.ndarray]:
        """Return (sequences, efficacy) for the train split."""
        return self._subset("train")

    def test(self) -> tuple[list[str], np.ndarray]:
        """Return (sequences, efficacy) for the test split."""
        return self._subset("test")

    def to_frame(self) -> pd.DataFrame:
        data = {
            "sequence": self.sequences,
            "efficacy": self.efficacy,
            "split": self.split,
        }
        if self.mrna is not None:
            data["mrna"] = self.mrna
        return pd.DataFrame(data)


def canonical_split(n: int, train_size: int = N_TRAIN) -> np.ndarray:
    """Return the canonical leading-train / trailing-test assignment.

    The Huesken benchmark designates the first ``train_size`` rows as training
    and the remainder as test. ``train_size`` is clamped to ``n``.
    """
    split = np.array(["test"] * n, dtype=object)
    split[: min(train_size, n)] = "train"
    return split


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    # Substring fallback (e.g. "norm_efficacy_pct" matches "efficacy").
    for cand in candidates:
        for low, orig in lowered.items():
            if cand in low:
                return orig
    return None


def _normalize_sequence(seq: str) -> str:
    s = str(seq).strip().upper().replace(" ", "")
    bad = set(s) - _VALID_BASES
    if bad:
        raise ValueError(f"sequence has invalid bases {sorted(bad)}: {seq!r}")
    return s


def _normalize_efficacy(values: np.ndarray) -> np.ndarray:
    eff = np.asarray(values, dtype=float)
    if not np.isfinite(eff).all():
        raise ValueError("efficacy column contains non-finite values")
    # Percent-scaled tables (0..100) are rescaled to [0, 1].
    if eff.max() > 1.0:
        if eff.max() > 100.0 + 1e-6:
            raise ValueError("efficacy out of range; expected [0, 1] or [0, 100]")
        eff = eff / 100.0
    if eff.min() < -1e-9 or eff.max() > 1.0 + 1e-9:
        raise ValueError("efficacy out of [0, 1] after normalization")
    return np.clip(eff, 0.0, 1.0)


def _parse(path: Path, train_size: int) -> HueskenDataset:
    # sep=None + python engine sniffs tab vs comma vs whitespace.
    df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    cols = list(df.columns)

    seq_col = _pick_column(cols, _SEQ_CANDIDATES)
    eff_col = _pick_column(cols, _EFF_CANDIDATES)
    if seq_col is None or eff_col is None:
        raise ValueError(
            f"could not find sequence/efficacy columns in {cols}; "
            "expected one of "
            f"{_SEQ_CANDIDATES} and {_EFF_CANDIDATES}"
        )

    sequences = [_normalize_sequence(s) for s in df[seq_col].tolist()]
    efficacy = _normalize_efficacy(df[eff_col].to_numpy())

    split_col = _pick_column(cols, _SPLIT_CANDIDATES)
    if split_col is not None:
        split = (
            df[split_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda v: "train" if v.startswith("tr") else "test")
            .to_numpy(dtype=object)
        )
    else:
        split = canonical_split(len(df), train_size=train_size)

    mrna_col = _pick_column(cols, _MRNA_CANDIDATES)
    mrna = df[mrna_col].astype(str).to_numpy(dtype=object) if mrna_col else None

    return HueskenDataset(sequences=sequences, efficacy=efficacy, split=split, mrna=mrna)


def _cache_files(cdir: Path) -> tuple[Path, Path]:
    return cdir / "huesken.parquet", cdir / "meta.json"


def _write_cache(cdir: Path, ds: HueskenDataset) -> None:
    parquet, meta = _cache_files(cdir)
    ds.to_frame().to_parquet(parquet, index=False)
    meta.write_text(json.dumps({"has_mrna": ds.mrna is not None}), encoding="utf-8")


def _read_cache(cdir: Path) -> HueskenDataset:
    parquet, _ = _cache_files(cdir)
    df = pd.read_parquet(parquet)
    mrna = df["mrna"].to_numpy(dtype=object) if "mrna" in df.columns else None
    return HueskenDataset(
        sequences=df["sequence"].tolist(),
        efficacy=df["efficacy"].to_numpy(dtype=float),
        split=df["split"].to_numpy(dtype=object),
        mrna=mrna,
    )


def default_raw_path() -> Path:
    """Default location the download script writes the Huesken table to."""
    return default_raw_dir() / "Huesken" / "huesken_2005.tsv"


def load_huesken(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    train_size: int = N_TRAIN,
    rebuild: bool = False,
) -> HueskenDataset:
    """Load the Huesken efficacy benchmark, caching the parsed form.

    Args:
        raw_path: Source table. Defaults to ``data/raw/Huesken/huesken_2005.tsv``.
        cache_root: Cache root. Defaults to ``data/cache``.
        train_size: Leading-train size used only when the source has no split
            column (canonical default 2182).
        rebuild: Reparse and overwrite the cache even if present.

    Returns:
        A :class:`HueskenDataset`.
    """
    raw_path = Path(raw_path) if raw_path is not None else default_raw_path()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Huesken raw table not found at {raw_path}. Run scripts/download_data.sh "
            "or set HUESKEN_URL; see docs/DATA_CARD.md section 2."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()

    key = CacheKey(
        name="huesken",
        version=PARSER_VERSION,
        params={"sha256": sha256_file(raw_path), "train_size": train_size},
    )
    cdir = cache_dir_for(cache_root, key)
    parquet, meta = _cache_files(cdir)
    if not rebuild and parquet.exists() and meta.exists():
        return _read_cache(cdir)

    ds = _parse(raw_path, train_size=train_size)
    _write_cache(cdir, ds)
    return ds
