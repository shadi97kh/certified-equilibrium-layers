"""Data loader for the certified-direction siRNA off-target experiment.

Per parent-vs-modified contrast it produces a standardized LONG-FORMAT table with columns:
  contrast, series, organism, transcript_id, arm (parent/modified), log2FoldChange,
  seed_matched, best_site_type, n_canonical_sites
(the seed_matched flag is the precomputed seed-match membership; alternatively the 3'UTR sequence
is available from utr_fasta).

NO SYNTHETIC DATA: each loader RAISES with a clear message if its processed inputs are not on disk,
and never fabricates expression values or fills missing transcripts. For series where only raw
counts/FASTQ exist, the loader STOPS and lists exactly which processed files the Prompt 2
preprocessing pipeline must produce; it never runs alignment.

Deterministic, no random, no em-dashes. Prints, per contrast, the row count and the number of
seed-matched transcripts once loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_CONFIG = Path(__file__).resolve().parent / "seedmod_loader_config.json"
DEFAULT_OUTDIR = REPO / "results" / "seedmod_loaded"
REQUIRED = (
    "name",
    "series",
    "organism",
    "guide",
    "seed",
    "modification",
    "dose",
    "parent_arm",
    "modified_arm",
    "utr_fasta",
    "gene_id_col",
    "logfc_col",
)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if "contrasts" not in cfg or not cfg["contrasts"]:
        raise ValueError(f"{path}: no 'contrasts'")
    return cfg


def _validate(c: dict) -> None:
    for field in REQUIRED:
        if field not in c:
            raise ValueError(f"contrast '{c.get('name', '?')}' missing required field: {field}")
    for arm in ("parent_arm", "modified_arm"):
        for k in ("label", "sample_ids", "file"):
            if k not in c[arm]:
                raise ValueError(f"contrast '{c['name']}' {arm} missing '{k}'")


def _load_fasta(path: Path) -> dict[str, list[str]]:
    """Return gene -> list of ALL its 3'UTR isoform sequences. The union (not the last isoform) is
    used for seed matching, so genes whose site sits on a non-last isoform are not mis-binned into the
    background."""
    seqs: dict[str, list[str]] = {}
    gid, buf = None, []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if gid:
                    seqs.setdefault(gid, []).append("".join(buf))
                gid, buf = line[1:].strip().split()[0], []
            else:
                buf.append(line.strip())
    if gid:
        seqs.setdefault(gid, []).append("".join(buf))
    return seqs


_SITE_RANK = {"8mer": 0, "7mer-m8": 1, "7mer-A1": 2, "6mer": 3}  # smaller = tighter


def _load_deseq(path: Path, gene_col: str, logfc_col: str, sep: str) -> dict[str, float]:
    df = pd.read_csv(path, sep=sep)  # pandas reads .gz transparently
    for col in (gene_col, logfc_col):
        if col not in df.columns:
            raise ValueError(
                f"{path}: required column '{col}' absent (have {list(df.columns)[:8]})"
            )
    lfc = pd.to_numeric(df[logfc_col], errors="coerce")
    out = {}
    for g, v in zip(df[gene_col].astype(str), lfc, strict=True):
        if pd.notna(v):
            out[g] = float(v)  # real DESeq2 value only; missing transcripts are never filled
    return out


def _seed_match(seed: str, utr_by_gene: dict[str, list[str]], genes, canonical: set[str]):
    """UNION over 3'UTR isoforms: a gene is seed-matched if ANY isoform carries a canonical 6-8mer
    site; n_canonical_sites = MAX over isoforms; best_site_type = tightest site across all isoforms."""
    from cel.features.seedmatch import find_sites

    guide = "A" + seed  # dummy position 1; find_sites reads the seed as guide positions 2-8
    flags = {}
    for g in genes:
        best, nmax = None, 0
        for u in utr_by_gene.get(g) or []:
            if not u or len(u) < 8:
                continue
            n = 0
            for s in find_sites(guide, u):
                if s.site_type in canonical:
                    n += 1
                    if best is None or _SITE_RANK[s.site_type] < _SITE_RANK[best]:
                        best = s.site_type
            nmax = max(nmax, n)
        flags[g] = (best is not None, best, nmax)
    return flags


def load_contrast(c: dict, repo: Path, canonical: set[str]) -> pd.DataFrame:
    """Return the standardized long-format table, or RAISE if processed inputs are absent."""
    _validate(c)
    parent_file = repo / c["parent_arm"]["file"]
    modified_file = repo / c["modified_arm"]["file"]
    utr_fasta = repo / c["utr_fasta"]
    missing = [str(p) for p in (parent_file, modified_file, utr_fasta) if not p.is_file()]

    if c.get("processed_status") == "needs_preprocessing" or missing:
        need = missing if missing else [str(parent_file), str(modified_file), str(utr_fasta)]
        raise FileNotFoundError(
            f"[{c['name']}] STOP: required processed inputs are not on disk. Produce these via the "
            f"Prompt 2 preprocessing pipeline first (this loader does not run alignment):\n"
            + "\n".join(f"    - {m}" for m in need)
            + f"\n  note: {c.get('assignment_note', '')}"
        )

    seed = c["seed"]["value"]
    if not seed:
        raise ValueError(
            f"[{c['name']}] seed sequence is null (VERIFY): {c['seed']['source']}. "
            "Source the guide seed before computing seed-match membership."
        )

    parent = _load_deseq(parent_file, c["gene_id_col"], c["logfc_col"], c.get("sep", "\t"))
    modified = _load_deseq(modified_file, c["gene_id_col"], c["logfc_col"], c.get("sep", "\t"))
    utr_by_gene = _load_fasta(utr_fasta)

    genes = [g for g in parent if g in modified]  # transcripts measured in BOTH arms (real values)
    flags = _seed_match(seed, utr_by_gene, genes, canonical)

    rows = []
    for g in genes:
        matched, best, n = flags[g]
        for arm, table in (("parent", parent), ("modified", modified)):
            rows.append(
                {
                    "contrast": c["name"],
                    "series": c["series"],
                    "organism": c["organism"],
                    "transcript_id": g,
                    "arm": arm,
                    "log2FoldChange": table[g],
                    "seed_matched": matched,
                    "best_site_type": best,
                    "n_canonical_sites": n,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "contrast",
            "series",
            "organism",
            "transcript_id",
            "arm",
            "log2FoldChange",
            "seed_matched",
            "best_site_type",
            "n_canonical_sites",
        ],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="certified-direction siRNA off-target data loader")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    cfg = load_config(args.config)
    canonical = set(cfg.get("canonical_site_types", ["8mer", "7mer-m8", "7mer-A1", "6mer"]))
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Certified-direction siRNA off-target loader")
    loaded, blocked = 0, 0
    for c in cfg["contrasts"]:
        print(
            f"\n[{c['name']}]  series={c['series']}  organism={c['organism']}  status={c.get('processed_status')}"
        )
        try:
            df = load_contrast(c, REPO, canonical)
        except (FileNotFoundError, ValueError) as e:
            blocked += 1
            print(f"  BLOCKED (no synthetic fallback):\n  {e!s}")
            continue
        loaded += 1
        n_sm = int(df.loc[df["arm"] == "parent", "seed_matched"].sum())
        n_tx = df["transcript_id"].nunique()
        out = args.outdir / f"{c['name']}.long.csv.gz"
        df.to_csv(
            out, index=False, compression={"method": "gzip", "mtime": 0}
        )  # mtime=0 -> byte-deterministic
        print(f"  LOADED: rows={len(df)}  transcripts={n_tx}  seed-matched transcripts={n_sm}")
        print(
            f"          arms: {c['parent_arm']['label']} vs {c['modified_arm']['label']}; "
            f"seed={c['seed']['value']} ({c['modification']['chemistry']} at seed pos {c['modification']['seed_position']})"
        )
        print(f"          wrote {out}")
    print(
        f"\nSUMMARY: {loaded} contrast(s) loaded, {blocked} blocked awaiting real processed inputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
