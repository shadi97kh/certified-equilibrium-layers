"""Loader for the STRING v12 protein interaction backbone.

Parses a STRING ``protein.links`` file (space-separated triples
``protein1 protein2 combined_score`` with scores in 0..1000), filters by a
combined-score threshold, and builds a symmetric-normalized sparse adjacency

    S = D^(-1/2) A D^(-1/2)

where A is the undirected, score-weighted adjacency (weights are
``combined_score / 1000``). S is exposed as scipy CSR; an ``edge_index`` for
torch_geometric is produced lazily (torch is imported only inside
:meth:`StringGRN.to_edge_index`) so the data layer stays importable without
torch installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .cache import CacheKey, cache_dir_for, sha256_file
from .paths import cache_dir as default_cache_dir
from .paths import raw_dir as default_raw_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

PARSER_VERSION = "string-v12-v1"

# STRING "high confidence" cutoff (see docs/DATA_CARD.md section 3).
DEFAULT_THRESHOLD = 700
DEFAULT_TAXID = 10116  # Rattus norvegicus, matching GSE184929.


@dataclass(frozen=True)
class StringGRN:
    """Symmetric-normalized STRING interaction graph.

    ``nodes`` is the ordered protein-id list (list index == node index). ``S``
    is the symmetric-normalized CSR adjacency. ``threshold`` records the
    combined-score cutoff used.
    """

    nodes: list[str]
    S: sp.csr_matrix
    threshold: int

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        """Number of undirected edges (S is symmetric; divide nnz by two)."""
        return int(self.S.nnz // 2)

    def node_index(self) -> dict[str, int]:
        return {node: i for i, node in enumerate(self.nodes)}

    def to_edge_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(edge_index, edge_weight)`` for torch_geometric.

        ``edge_index`` is a ``[2, E]`` long tensor and ``edge_weight`` the
        matching ``[E]`` float tensor of normalized weights. torch is imported
        lazily so this module imports without it.
        """
        import torch

        coo = self.S.tocoo()
        edge_index = torch.from_numpy(np.vstack([coo.row, coo.col])).to(torch.long)
        edge_weight = torch.from_numpy(coo.data.astype(np.float32))
        return edge_index, edge_weight


def _symmetric_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(adj.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        dinv_sqrt = 1.0 / np.sqrt(deg)
    dinv_sqrt[~np.isfinite(dinv_sqrt)] = 0.0
    d = sp.diags(dinv_sqrt)
    return (d @ adj @ d).tocsr()


def _confidence_adjacency(path: Path, threshold: int) -> tuple[list[str], sp.csr_matrix]:
    """Parse STRING into the score-weighted confidence adjacency A (weights in [0, 1]).

    A is symmetric with A_ij = combined_score / 1000 (the per-edge confidence) on
    the thresholded, undirected support. This is the mean adjacency A_bar used by
    the inhomogeneous random-subgraph noise model; the symmetric normalization
    that produces S is applied separately.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        compression="infer",
        usecols=["protein1", "protein2", "combined_score"],
    )
    df = df[df["combined_score"] >= threshold]
    if df.empty:
        raise ValueError(f"no STRING edges survive combined_score >= {threshold}")

    nodes = sorted(set(df["protein1"]) | set(df["protein2"]))
    idx = {node: i for i, node in enumerate(nodes)}
    a = df["protein1"].map(idx).to_numpy()
    b = df["protein2"].map(idx).to_numpy()
    score = df["combined_score"].to_numpy(dtype=float) / 1000.0

    # Canonicalize each undirected pair (lo, hi) and keep the max score, which
    # collapses the two directed rows STRING lists per interaction.
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    self_loop = lo == hi
    pairs = pd.DataFrame({"lo": lo[~self_loop], "hi": hi[~self_loop], "w": score[~self_loop]})
    pairs = pairs.groupby(["lo", "hi"], as_index=False)["w"].max()

    n = len(nodes)
    rows = np.concatenate([pairs["lo"].to_numpy(), pairs["hi"].to_numpy()])
    cols = np.concatenate([pairs["hi"].to_numpy(), pairs["lo"].to_numpy()])
    data = np.concatenate([pairs["w"].to_numpy(), pairs["w"].to_numpy()])
    return nodes, sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _parse(path: Path, threshold: int) -> StringGRN:
    nodes, adj = _confidence_adjacency(path, threshold)
    s = _symmetric_normalize(adj)
    return StringGRN(nodes=nodes, S=s, threshold=threshold)


def _cache_files(cdir: Path) -> tuple[Path, Path]:
    return cdir / "S.npz", cdir / "nodes.json"


def _write_cache(cdir: Path, grn: StringGRN) -> None:
    npz, nodes_json = _cache_files(cdir)
    sp.save_npz(npz, grn.S)
    nodes_json.write_text(
        json.dumps({"nodes": grn.nodes, "threshold": grn.threshold}),
        encoding="utf-8",
    )


def _read_cache(cdir: Path) -> StringGRN:
    npz, nodes_json = _cache_files(cdir)
    s = sp.load_npz(npz).tocsr()
    meta = json.loads(nodes_json.read_text(encoding="utf-8"))
    return StringGRN(nodes=meta["nodes"], S=s, threshold=meta["threshold"])


def default_raw_path(taxid: int = DEFAULT_TAXID) -> Path:
    """Default location for the STRING protein.links file."""
    return default_raw_dir() / "STRING" / f"{taxid}.protein.links.v12.0.txt.gz"


def load_string_grn(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    threshold: int = DEFAULT_THRESHOLD,
    taxid: int = DEFAULT_TAXID,
    rebuild: bool = False,
) -> StringGRN:
    """Load a STRING GRN, build symmetric-normalized S, and cache it.

    Args:
        raw_path: STRING ``protein.links`` file. Defaults to the per-taxid path
            written by scripts/download_data.sh.
        cache_root: Cache root. Defaults to ``data/cache``.
        threshold: Minimum combined_score (0..1000) to keep an edge. Default 700.
        taxid: Organism taxonomy id, used only to build the default path.
        rebuild: Reparse and overwrite the cache even if present.

    Returns:
        A :class:`StringGRN`.
    """
    raw_path = Path(raw_path) if raw_path is not None else default_raw_path(taxid)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"STRING links file not found at {raw_path}. Run scripts/download_data.sh "
            "(optionally with STRING_TAXID); see docs/DATA_CARD.md section 3."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()

    key = CacheKey(
        name="string_grn",
        version=PARSER_VERSION,
        params={"sha256": sha256_file(raw_path), "threshold": threshold},
    )
    cdir = cache_dir_for(cache_root, key)
    npz, nodes_json = _cache_files(cdir)
    if not rebuild and npz.exists() and nodes_json.exists():
        return _read_cache(cdir)

    grn = _parse(raw_path, threshold=threshold)
    _write_cache(cdir, grn)
    return grn


def load_string_confidence(
    raw_path: str | Path | None = None,
    threshold: int = DEFAULT_THRESHOLD,
    taxid: int = DEFAULT_TAXID,
) -> tuple[list[str], sp.csr_matrix]:
    """Load the raw STRING confidence adjacency A_bar (combined_score / 1000).

    Returns ``(nodes, A_bar)`` where ``A_bar`` is the symmetric score-weighted
    adjacency with entries in [0, 1] on the thresholded support. This is the mean
    adjacency of the inhomogeneous random-subgraph noise model
    (:mod:`certification.noise_model`); ``load_string_grn`` returns the
    symmetric-normalized S used for propagation, which this does not.
    """
    raw_path = Path(raw_path) if raw_path is not None else default_raw_path(taxid)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"STRING links file not found at {raw_path}. Run scripts/download_data.sh "
            "(optionally with STRING_TAXID); see docs/DATA_CARD.md section 3."
        )
    return _confidence_adjacency(raw_path, threshold=threshold)
