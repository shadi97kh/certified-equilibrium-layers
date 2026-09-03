"""On-target siRNA efficacy loaders (role: efficacy_backbone). REAL DATA ONLY.

Per-study loaders for the OligoFormer efficacy compilation and the siRNAEfficacyDB
release:

    Huesken, Reynolds, Vickers, Harborth, Ui-Tei, Khvorova, Katoh (Takayuki).

Each loader returns a tidy table with the antisense and sense sequences, the
target mRNA id, the measured on-target efficacy in [0, 1], and provenance (source
study, cell line, assay type). The Huesken loader additionally exposes the
standard 2182 train / 249 test split (column ``split`` plus ``.train()`` /
``.test()``).

HARD DATA POLICY (OUTLINE.md). This module loads ONLY real measured data. If a
file is absent the loader RAISES ``FileNotFoundError`` with the exact URL to
fetch. There is no fallback that generates rows, sequences, or efficacy values.
Fetch the real files with ``scripts/download_efficacy.sh`` (writes to
``data/raw/efficacy/`` with a sha256 manifest).

Source files. Two layouts are accepted under ``data/raw/efficacy/``:
1. a per-study file ``<study>.csv`` (the layout the download script produces), or
2. a single ``siRNAEfficacyDB`` master table carrying a per-row source study,
   which the per-study loaders filter by study.

These datasets measure ON-TARGET potency. They do NOT validate the off-target
containment certificate (see docs/DATA_CARD.md). Parsed frames cache to
``data/cache/`` as parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import CacheKey, cached_frame, sha256_file
from .huesken import canonical_split
from .paths import cache_dir as default_cache_dir
from .paths import raw_dir as default_raw_dir
from .roles import EFFICACY_BACKBONE

PARSER_VERSION = "efficacy-v3"

# Canonical public sources for the real efficacy compilation. The OligoFormer
# release stores the parsed sets under Comparison methods/Monopoli-RF/datasets/
# (Hu.csv = Huesken, Taka.csv = Katoh, Mixset.csv = the pooled mix); the 19mer
# column is the SENSE strand and the guide is its reverse complement.
OLIGOFORMER_DATA = "https://raw.githubusercontent.com/lulab/OligoFormer/main/data"
OLIGOFORMER_DATASETS = (
    "https://raw.githubusercontent.com/lulab/OligoFormer/main/"
    "Comparison%20methods/Monopoli-RF/datasets"
)
OLIGOFORMER_REPO = "https://github.com/lulab/OligoFormer (Comparison methods/Monopoli-RF/datasets)"
SIRNA_EFFICACY_DB = (
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11665841/ (siRNAEfficacyDB supplement)"
)

# Per-study registry. Every entry is REAL measured data. ``match`` is the token
# used to select this study's rows out of a master compilation; ``url`` is the
# exact file to fetch (raised when the local file is absent); ``source`` names
# the authoritative page when the direct URL must be confirmed.
EFFICACY_STUDIES: dict[str, dict] = {
    "huesken": {
        "study": "Huesken 2005",
        "match": "Huesken",
        "cell_line": "H1299",
        "assay": "branched-DNA residual mRNA",
        "citation": "Huesken et al., Nat Biotechnol 23:995-1001 (2005), 10.1038/nbt1118",
        "n_sirna": 2361,
        "n_mrna": 34,
        "filename": "huesken.csv",
        "url": f"{OLIGOFORMER_DATASETS}/Hu.csv",
        "source": OLIGOFORMER_REPO,
        "has_split": True,
    },
    "reynolds": {
        "study": "Reynolds 2004",
        "match": "Reynolds",
        "cell_line": "HEK293 / HeLa",
        "assay": "reporter / endogenous mRNA knockdown",
        "citation": "Reynolds et al., Nat Biotechnol 22:326-330 (2004), 10.1038/nbt936",
        "n_sirna": 240,
        "n_mrna": 2,
        "filename": "reynolds.csv",
        "url": f"{OLIGOFORMER_DATA}/Reynolds.csv",
        "source": SIRNA_EFFICACY_DB,
        "has_split": False,
    },
    "vickers": {
        "study": "Vickers 2003",
        "match": "Vickers",
        "cell_line": "T24 / HeLa",
        "assay": "endogenous mRNA knockdown",
        "citation": "Vickers et al., J Biol Chem 278:7108-7118 (2003), 10.1074/jbc.M210326200",
        "n_sirna": 76,
        "n_mrna": 2,
        "filename": "vickers.csv",
        "url": f"{OLIGOFORMER_DATA}/Vickers.csv",
        "source": SIRNA_EFFICACY_DB,
        "has_split": False,
    },
    "harborth": {
        "study": "Harborth 2003",
        "match": "Harborth",
        "cell_line": "HeLa",
        "assay": "endogenous mRNA / protein knockdown",
        "citation": "Harborth et al., Antisense Nucleic Acid Drug Dev 13:83-105 (2003), "
        "10.1089/108729003321629638",
        "n_sirna": 44,
        "n_mrna": 6,
        "filename": "harborth.csv",
        "url": f"{OLIGOFORMER_DATA}/Harborth.csv",
        "source": SIRNA_EFFICACY_DB,
        "has_split": False,
    },
    "uitei": {
        "study": "Ui-Tei 2004",
        "match": "Ui-Tei",
        "cell_line": "various",
        "assay": "reporter knockdown",
        "citation": "Ui-Tei et al., Nucleic Acids Res 32:936-948 (2004), 10.1093/nar/gkh247",
        "n_sirna": 62,
        "n_mrna": 2,
        "filename": "uitei.csv",
        "url": f"{OLIGOFORMER_DATA}/Ui-Tei.csv",
        "source": SIRNA_EFFICACY_DB,
        "has_split": False,
    },
    "khvorova": {
        "study": "Khvorova 2003",
        "match": "Khvorova",
        "cell_line": "various",
        "assay": "reporter knockdown",
        "citation": "Khvorova et al., Cell 115:209-216 (2003), 10.1016/S0092-8674(03)00759-1",
        "n_sirna": 90,
        "n_mrna": 2,
        "filename": "khvorova.csv",
        "url": f"{OLIGOFORMER_DATA}/Khvorova.csv",
        "source": SIRNA_EFFICACY_DB,
        "has_split": False,
    },
    "katoh": {
        "study": "Katoh (Taka)",
        "match": "Katoh",
        "cell_line": "HeLa",
        "assay": "luciferase reporter",
        "citation": "Katoh and Suzuki (Takayuki) siRNA efficacy set, via the OligoFormer release",
        "n_sirna": 702,
        "n_mrna": 1,
        "filename": "katoh.csv",
        "url": f"{OLIGOFORMER_DATASETS}/Taka.csv",
        "source": OLIGOFORMER_REPO,
        "has_split": False,
    },
    "mix": {
        "study": "Mix (OligoFormer)",
        "match": "Mix",
        "cell_line": "various",
        "assay": "various",
        "citation": "OligoFormer Mixset (pooled Reynolds / Vickers / Harborth / Ui-Tei / Khvorova)",
        "n_sirna": 472,
        "n_mrna": None,
        "filename": "mix.csv",
        "url": f"{OLIGOFORMER_DATASETS}/Mixset.csv",
        "source": OLIGOFORMER_REPO,
        "has_split": False,
    },
}

# Master compilation file names accepted in place of per-study files.
MASTER_FILES = ("siRNAEfficacyDB.csv", "siRNAEfficacyDB.tsv", "efficacy_master.csv")

_ANTISENSE = ("antisense", "guide", "as", "antisense_sequence", "sirna_antisense", "as_seq")
_SENSE = ("sense", "passenger", "ss", "sense_sequence", "sirna_sense", "ss_seq")
_MRNA = ("mrna_id", "mrna", "target", "gene", "transcript", "target_id")
_EFF = ("efficacy", "inhibition", "activity", "knockdown", "norm_inhibition", "label", "value")
_SOURCE = ("source_study", "source", "study", "dataset", "publication")
_CELL = ("cell_line", "cell", "cellline")
_ASSAY = ("assay", "assay_type", "readout")
_SPLIT = ("split", "set", "fold", "partition")

COLUMNS = ["antisense", "sense", "mrna_id", "efficacy", "source_study", "cell_line", "assay"]


@dataclass(frozen=True)
class EfficacyDataset:
    """On-target efficacy table (role: efficacy_backbone).

    ``frame`` carries :data:`COLUMNS`; a Huesken dataset additionally carries a
    ``split`` column with ``"train"`` / ``"test"`` (the standard 2182 / 249
    split). ``study`` is the registry key when the dataset is a single study.
    """

    frame: pd.DataFrame
    role: str = field(default=EFFICACY_BACKBONE)
    study: str | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def sources(self) -> list[str]:
        return sorted(self.frame["source_study"].unique().tolist())

    def cell_lines(self) -> list[str]:
        return sorted(self.frame["cell_line"].unique().tolist())

    def assays(self) -> list[str]:
        return sorted(self.frame["assay"].unique().tolist())

    def filter_study(self, study: str) -> EfficacyDataset:
        return EfficacyDataset(
            self.frame[self.frame["source_study"] == study].reset_index(drop=True)
        )

    @property
    def has_split(self) -> bool:
        return "split" in self.frame.columns

    def _split_part(self, which: str) -> EfficacyDataset:
        if not self.has_split:
            raise ValueError(
                "this dataset has no split column; only the Huesken loader exposes the "
                "standard 2182 train / 249 test split"
            )
        sub = self.frame[self.frame["split"] == which].reset_index(drop=True)
        return EfficacyDataset(sub, role=self.role, study=self.study)

    def train(self) -> EfficacyDataset:
        return self._split_part("train")

    def test(self) -> EfficacyDataset:
        return self._split_part("test")


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


def _normalize_efficacy(values: np.ndarray) -> np.ndarray:
    eff = np.asarray(values, dtype=float)
    finite = eff[np.isfinite(eff)]
    if finite.size and finite.max() > 1.0:  # percent-scaled -> [0, 1]
        eff = eff / 100.0
    return np.clip(eff, 0.0, 1.0)


_RNA_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}


def reverse_complement(seq: str) -> str:
    """RNA reverse complement over {A, C, G, U}: A<->U, G<->C, then reverse.

    Used to recover the antisense (guide) strand from the OligoFormer 19mer, which
    is the SENSE / passenger strand given 5' to 3'. An involution: rc(rc(x)) == x.
    """
    rna = str(seq).strip().upper().replace("T", "U")
    return "".join(_RNA_COMPLEMENT.get(b, b) for b in reversed(rna))


# OligoFormer release schema (Comparison methods/Monopoli-RF/datasets/*.csv):
# ``19mer`` is the SENSE (passenger) strand 5'->3'; ``expression`` is residual mRNA %.
_OLIGOFORMER_SEQ = ("19mer", "sequence", "seq", "sirna")
_OLIGOFORMER_EXPR = ("expression", "residual_expression", "norm_expression")


def _is_oligoformer(cols: list[str]) -> bool:
    return (
        _pick(cols, _OLIGOFORMER_SEQ) is not None
        and _pick(cols, _OLIGOFORMER_EXPR) is not None
        and _pick(cols, _ANTISENSE) is None
    )


def _parse_oligoformer(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """OligoFormer schema: sense = 19mer (RNA), antisense = rc(19mer), efficacy = 1 - expr/100."""
    meta = EFFICACY_STUDIES[key]
    cols = list(df.columns)
    seq = _pick(cols, _OLIGOFORMER_SEQ)
    expr = _pick(cols, _OLIGOFORMER_EXPR)
    n = len(df)
    sense = df[seq].astype(str).str.strip().str.upper().str.replace("T", "U", regex=False)
    antisense = sense.map(reverse_complement)
    efficacy = np.clip(1.0 - pd.to_numeric(df[expr], errors="coerce").to_numpy() / 100.0, 0.0, 1.0)
    return pd.DataFrame(
        {
            "antisense": antisense.to_numpy(),
            "sense": sense.to_numpy(),
            "mrna_id": [meta["study"]] * n,
            "efficacy": efficacy,
            "source_study": [meta["study"]] * n,
            "cell_line": [meta["cell_line"]] * n,
            "assay": [meta["assay"]] * n,
        }
    )[COLUMNS]


def _to_schema(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Map a raw frame to :data:`COLUMNS` for study ``key`` (column-name tolerant)."""
    cols = list(df.columns)
    if _is_oligoformer(cols):  # OligoFormer 19mer (sense) + expression (residual %)
        return _parse_oligoformer(df, key)
    meta = EFFICACY_STUDIES[key]
    anti = _pick(cols, _ANTISENSE)
    eff = _pick(cols, _EFF)
    if anti is None or eff is None:
        raise ValueError(
            f"{meta['study']} file needs an antisense column ({_ANTISENSE}) and an efficacy "
            f"column ({_EFF}); found {cols}"
        )
    sense = _pick(cols, _SENSE)
    mrna = _pick(cols, _MRNA)
    src = _pick(cols, _SOURCE)
    cell = _pick(cols, _CELL)
    assay = _pick(cols, _ASSAY)
    n = len(df)
    return pd.DataFrame(
        {
            "antisense": df[anti].astype(str).str.strip().str.upper(),
            "sense": (df[sense].astype(str).str.strip().str.upper() if sense else [""] * n),
            "mrna_id": (df[mrna].astype(str) if mrna else [f"{key}_mrna"] * n),
            "efficacy": _normalize_efficacy(df[eff].to_numpy()),
            "source_study": (df[src].astype(str) if src else [meta["study"]] * n),
            "cell_line": (df[cell].astype(str) if cell else [meta["cell_line"]] * n),
            "assay": (df[assay].astype(str) if assay else [meta["assay"]] * n),
        }
    )[COLUMNS]


def _add_split(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Attach the Huesken train/test split: from a file column, else canonical order."""
    out = df.copy()
    split_col = _pick(list(raw.columns), _SPLIT)
    if split_col is not None:
        values = raw[split_col].astype(str).str.strip().str.lower()
        out["split"] = np.where(values.str.startswith("te"), "test", "train")
    else:
        out["split"] = canonical_split(len(out))
    return out


def default_efficacy_dir() -> Path:
    """Default raw directory for the efficacy compilation."""
    return default_raw_dir() / "efficacy"


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", compression="infer")


def _resolve_study_file(key: str, raw_dir: Path) -> tuple[Path, str]:
    """Find the real file backing study ``key``: a per-study file or the master.

    Returns ``(path, mode)`` with ``mode`` in {"single", "master"}. Raises
    ``FileNotFoundError`` with the exact URL when no real file is present.
    """
    meta = EFFICACY_STUDIES[key]
    candidates = list(dict.fromkeys([meta["filename"], f"{key}.csv", f"{key}.tsv", f"{key}.txt"]))
    for name in candidates:
        p = raw_dir / name
        if p.exists():
            return p, "single"
    for name in MASTER_FILES:
        p = raw_dir / name
        if p.exists():
            return p, "master"
    raise FileNotFoundError(
        f"{meta['study']} efficacy table not found under {raw_dir} "
        f"(looked for {candidates} or a master {list(MASTER_FILES)}). "
        f"Fetch the real file: {meta['url']}  (source: {meta['source']}). "
        f"Run scripts/download_efficacy.sh, or place it at {raw_dir / meta['filename']}. "
        "REAL DATA ONLY: no rows are generated when the file is absent."
    )


def _build_study_frame(key: str, path: Path, mode: str, with_split: bool) -> pd.DataFrame:
    raw = _read_table(path)
    if mode == "master":
        col = _pick(list(raw.columns), _SOURCE)
        if col is None:
            raise ValueError(
                f"master efficacy file {path} has no source-study column ({_SOURCE}); cannot "
                f"select the {EFFICACY_STUDIES[key]['study']} rows"
            )
        token = EFFICACY_STUDIES[key]["match"].lower()
        mask = raw[col].astype(str).str.lower().str.contains(token, na=False)
        raw = raw[mask].reset_index(drop=True)
        if raw.empty:
            raise FileNotFoundError(
                f"{EFFICACY_STUDIES[key]['study']} rows not found in master file {path} "
                f"(no source matching {token!r}). Fetch the per-study file: "
                f"{EFFICACY_STUDIES[key]['url']}."
            )
    frame = _to_schema(raw, key)
    if with_split:
        frame = _add_split(frame, raw)
    return frame


def _load_study(
    key: str,
    raw_dir: str | Path | None,
    cache_root: str | Path | None,
    rebuild: bool,
    with_split: bool | None = None,
) -> EfficacyDataset:
    if key not in EFFICACY_STUDIES:
        raise KeyError(f"unknown efficacy study {key!r}; known: {sorted(EFFICACY_STUDIES)}")
    if with_split is None:
        with_split = EFFICACY_STUDIES[key]["has_split"]
    raw_dir = Path(raw_dir) if raw_dir is not None else default_efficacy_dir()
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()
    path, mode = _resolve_study_file(key, raw_dir)
    cache_key = CacheKey(
        name=f"efficacy_{key}",
        version=PARSER_VERSION,
        params={"sha256": sha256_file(path), "mode": mode, "split": bool(with_split)},
    )
    frame = cached_frame(
        cache_key,
        cache_root,
        lambda: _build_study_frame(key, path, mode, with_split),
        rebuild=rebuild,
    )
    return EfficacyDataset(frame, study=key)


# --- Per-study public loaders (REAL data only; raise with exact URL) ----------
def load_huesken_efficacy(raw_dir=None, cache_root=None, rebuild: bool = False) -> EfficacyDataset:
    """Huesken 2005 (2431 siRNA / 34 mRNA), with the standard 2182/249 split."""
    return _load_study("huesken", raw_dir, cache_root, rebuild, with_split=True)


def load_reynolds(raw_dir=None, cache_root=None, rebuild: bool = False) -> EfficacyDataset:
    """Reynolds et al. 2004 siRNA efficacy set."""
    return _load_study("reynolds", raw_dir, cache_root, rebuild)


def load_vickers(raw_dir=None, cache_root=None, rebuild: bool = False) -> EfficacyDataset:
    """Vickers et al. 2003 siRNA efficacy set."""
    return _load_study("vickers", raw_dir, cache_root, rebuild)


def load_harborth(raw_dir=None, cache_root=None, rebuild: bool = False) -> EfficacyDataset:
    """Harborth et al. 2003 siRNA efficacy set."""
    return _load_study("harborth", raw_dir, cache_root, rebuild)


def load_uitei(raw_dir=None, cache_root=None, rebuild: bool = False) -> EfficacyDataset:
    """Ui-Tei et al. 2004 siRNA efficacy set."""
    return _load_study("uitei", raw_dir, cache_root, rebuild)


def load_khvorova(raw_dir=None, cache_root=None, rebuild: bool = False) -> EfficacyDataset:
    """Khvorova et al. 2003 siRNA efficacy set."""
    return _load_study("khvorova", raw_dir, cache_root, rebuild)


def load_katoh(raw_dir=None, cache_root=None, rebuild: bool = False) -> EfficacyDataset:
    """Katoh (Takayuki) 702-siRNA luciferase efficacy set (OligoFormer Taka.csv)."""
    return _load_study("katoh", raw_dir, cache_root, rebuild)


def load_mix(raw_dir=None, cache_root=None, rebuild: bool = False) -> EfficacyDataset:
    """OligoFormer Mixset: pooled cross-dataset efficacy (Reynolds/Vickers/...)."""
    return _load_study("mix", raw_dir, cache_root, rebuild)


def load_efficacy_suite(
    studies: list[str] | None = None,
    raw_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> EfficacyDataset:
    """Load and concatenate several efficacy studies into one tidy table.

    Raises ``FileNotFoundError`` listing the missing studies and their exact URLs
    if any requested study has no real file (REAL DATA ONLY).
    """
    keys = list(studies) if studies is not None else list(EFFICACY_STUDIES)
    raw_dir = Path(raw_dir) if raw_dir is not None else default_efficacy_dir()
    missing = []
    for key in keys:
        try:
            _resolve_study_file(key, raw_dir)
        except FileNotFoundError:
            missing.append(key)
    if missing:
        lines = "; ".join(f"{k} -> {EFFICACY_STUDIES[k]['url']}" for k in missing)
        raise FileNotFoundError(
            f"missing real efficacy files for {missing} under {raw_dir}. Fetch: {lines}. "
            "Run scripts/download_efficacy.sh. REAL DATA ONLY."
        )
    frames = [
        _load_study(k, raw_dir, cache_root, rebuild, with_split=False).frame[COLUMNS] for k in keys
    ]
    return EfficacyDataset(pd.concat(frames, ignore_index=True)[COLUMNS])


# --- Backward-compatible unified loader (OligoFormer four-file layout) ---------
# Retained for the backbone experiment and the existing tests. Reads the older
# data/raw/oligoformer/<source>.csv layout for source in {huesken, katoh, mix,
# shabalina}. New code should prefer the per-study loaders above.
EFFICACY_SOURCES: dict[str, dict] = {
    "huesken": {
        "study": "Huesken 2005",
        "cell_line": "H1299",
        "assay": "branched-DNA residual mRNA",
        "n_sirna": 2431,
        "n_mrna": 34,
    },
    "katoh": {
        "study": "Katoh (Taka)",
        "cell_line": "HeLa",
        "assay": "luciferase reporter",
        "n_sirna": 702,
        "n_mrna": 1,
    },
    "mix": {
        "study": "Mix",
        "cell_line": "various",
        "assay": "various",
        "n_sirna": None,
        "n_mrna": None,
    },
    "shabalina": {
        "study": "Shabalina 2006",
        "cell_line": "various",
        "assay": "various (thermodynamic-filtered compilation, 13 publications)",
        "n_sirna": 653,
        "n_mrna": None,
    },
}

# Studies pooled inside the OligoFormer "mix" file.
MIX_STUDIES = (
    "Amarzguioui 2003",
    "Harborth 2003",
    "Hsieh 2004",
    "Khvorova 2003",
    "Reynolds 2004",
    "Vickers 2003",
    "Ui-Tei 2004",
)


def _parse_source(path: Path, source_key: str) -> pd.DataFrame:
    meta = EFFICACY_SOURCES[source_key]
    df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    cols = list(df.columns)
    anti = _pick(cols, _ANTISENSE)
    eff = _pick(cols, _EFF)
    if anti is None or eff is None:
        raise ValueError(
            f"{path}: need an antisense column ({_ANTISENSE}) and an efficacy column "
            f"({_EFF}); found {cols}"
        )
    sense = _pick(cols, _SENSE)
    mrna = _pick(cols, _MRNA)
    src = _pick(cols, _SOURCE)
    cell = _pick(cols, _CELL)
    assay = _pick(cols, _ASSAY)
    n = len(df)
    return pd.DataFrame(
        {
            "antisense": df[anti].astype(str).str.strip().str.upper(),
            "sense": (df[sense].astype(str).str.strip().str.upper() if sense else [""] * n),
            "mrna_id": (df[mrna].astype(str) if mrna else [f"{source_key}_mrna"] * n),
            "efficacy": _normalize_efficacy(df[eff].to_numpy()),
            "source_study": (df[src].astype(str) if src else [meta["study"]] * n),
            "cell_line": (df[cell].astype(str) if cell else [meta["cell_line"]] * n),
            "assay": (df[assay].astype(str) if assay else [meta["assay"]] * n),
        }
    )[COLUMNS]


def default_raw_dir_efficacy() -> Path:
    return default_raw_dir() / "oligoformer"


def load_oligoformer_efficacy(
    raw_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    sources: list[str] | None = None,
    rebuild: bool = False,
) -> EfficacyDataset:
    """Load the OligoFormer four-file efficacy layout into one tidy table.

    Args:
        raw_dir: directory holding ``<source>.csv`` files. Default
            ``data/raw/oligoformer``.
        cache_root: parquet cache root. Default ``data/cache``.
        sources: subset of {huesken, katoh, mix, shabalina}. Default: all present.
        rebuild: reparse and overwrite the cache.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else default_raw_dir_efficacy()
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()
    keys = sources if sources is not None else list(EFFICACY_SOURCES)

    present = [(k, raw_dir / f"{k}.csv") for k in keys if (raw_dir / f"{k}.csv").exists()]
    if not present:
        raise FileNotFoundError(
            f"no OligoFormer efficacy CSVs found under {raw_dir} (expected "
            f"{[k + '.csv' for k in keys]}). Fetch the real files with "
            "scripts/download_efficacy.sh; see docs/DATA_CARD.md."
        )

    digest = {k: sha256_file(p) for k, p in present}
    key = CacheKey(name="efficacy", version=PARSER_VERSION, params={"sources": digest})

    def build() -> pd.DataFrame:
        frames = [_parse_source(p, k) for k, p in present]
        return pd.concat(frames, ignore_index=True)[COLUMNS]

    return EfficacyDataset(cached_frame(key, cache_root, build, rebuild=rebuild))
