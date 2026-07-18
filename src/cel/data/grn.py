"""Regulatory networks for the propagation operator P_alpha. REAL DATA ONLY.

Two flavors:

- **Signed / directed** (for Theorem 2, contraction / stability on signed,
  directed GRNs). Loaded from curated TF-target resources:
  - TRRUST v2 (``load_trrust``): manually curated, signed, directed TF-target
    interactions with an Activation / Repression mode.
  - DoRothEA (``load_dorothea``): regulons with a confidence level (A-E) and a
    mode of regulation (mor in {+1, -1}); the high-confidence A-C levels are the
    default.
  Both build a signed, directed, row-normalized transition ``S`` whose absolute
  value is sub-stochastic, so the spectral radius rho(S) <= 1. The sparse ``S``,
  the +1 / -1 sign pattern, and rho(S) are exposed.

- **Undirected / unsigned** (``load_string_undirected``). STRING v12
  confidence-weighted associations, reused from :mod:`cel.data.string_grn`.
  STRING edges are undirected and unsigned, so this is the undirected variant,
  NOT the signed/directed T2 network (see docs/DATA_CARD.md).

Propagation reuses the shared P_alpha operator
(:func:`cel.verify.containment.propagate`); this module never re-implements
it. REAL DATA ONLY: every loader RAISES ``FileNotFoundError`` with the exact URL
when its real file is absent. Fetch with ``scripts/download_grn.sh``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .cache import CacheKey, cache_dir_for, sha256_file
from .paths import cache_dir as default_cache_dir
from .paths import raw_dir as default_raw_dir
from .string_grn import StringGRN, load_string_grn

PARSER_VERSION = "grn-v1"

# TRRUST v2 curated, signed, directed TF-target tables.
TRRUST_URLS: dict[str, str] = {
    "human": "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
    "mouse": "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv",
}
DOROTHEA_SOURCE = (
    "DoRothEA regulons (saezlab), via decoupleR / OmniPath. Export a CSV with "
    "columns tf, target, confidence (A-E), mor (+1/-1) to data/raw/grn/dorothea.csv. "
    "See https://saezlab.github.io/dorothea/"
)

# CollecTRI signed, directed TF-target network, served by the OmniPath REST API.
COLLECTRI_ORGANISMS: dict[str, int] = {"human": 9606, "mouse": 10090, "rat": 10116}
COLLECTRI_URL = (
    "https://omnipathdb.org/interactions?datasets=collectri&organisms={organism}"
    "&genesymbols=yes&fields=sources,references"
)

# TF-target regulation mode -> sign.
_MODE_SIGN = {"activation": 1.0, "repression": -1.0}


@dataclass(frozen=True)
class SignedDirectedGRN:
    """A signed, directed, row-normalized transition operator over genes.

    ``S`` (CSR) is the propagation operator: ``S[target, tf]`` carries the signed,
    row-normalized influence of a TF on a target, so a perturbation on TFs flows
    to targets under P_alpha. ``signs`` (CSR) is the matching +1 / -1 pattern.
    The absolute value of ``S`` is row sub-stochastic, hence rho(S) <= 1.
    """

    nodes: list[str]
    S: sp.csr_matrix
    signs: sp.csr_matrix
    source: str
    n_edges: int
    n_unknown_dropped: int = 0
    role: str = field(default="grn_signed_directed")

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def spectral_radius(self) -> float:
        """Spectral radius rho(S) = max |eigenvalue| (<= 1 by construction)."""
        return signed_spectral_radius(self.S)

    def spectral_norm(self) -> float:
        """Spectral norm ||S||_2 = largest singular value.

        For a signed, directed (non-symmetric) S this can exceed rho(S); it is
        computed by SVD, not by the symmetric-only power-iteration estimator in
        :mod:`cel.verify.containment`.
        """
        return signed_spectral_norm(self.S)

    def propagate(self, x0: np.ndarray, alpha: float, num_steps: int = 64) -> np.ndarray:
        """Diffuse ``x0`` with the shared P_alpha operator (no re-implementation)."""
        from cel.verify.containment import propagate

        return propagate(self.S, np.asarray(x0, dtype=float), alpha=alpha, num_steps=num_steps)

    def index(self) -> dict[str, int]:
        return {node: i for i, node in enumerate(self.nodes)}


def signed_spectral_radius(S: sp.spmatrix) -> float:
    """Spectral radius of a (possibly non-symmetric, signed) sparse matrix."""
    n = S.shape[0]
    if n == 0:
        return 0.0
    if n <= 256:
        vals = np.linalg.eigvals(S.toarray())
    else:
        import scipy.sparse.linalg as sla

        try:
            vals = sla.eigs(
                S.tocsc().astype(float), k=1, which="LM", return_eigenvectors=False, maxiter=5000
            )
        except Exception:  # - ARPACK convergence fallback
            if n > 4000:
                raise
            vals = np.linalg.eigvals(S.toarray())
    return float(np.max(np.abs(vals))) if len(vals) else 0.0


def signed_spectral_norm(S: sp.spmatrix) -> float:
    """Spectral norm ||S||_2 (largest singular value) of a sparse matrix."""
    n = S.shape[0]
    if n == 0:
        return 0.0
    if n <= 256:
        return float(np.linalg.norm(S.toarray(), 2))
    import scipy.sparse.linalg as sla

    try:
        return float(sla.svds(S.astype(float), k=1, return_singular_vectors=False)[0])
    except Exception:  # - ARPACK fallback for tiny / awkward inputs
        if n > 4000:
            raise
        return float(np.linalg.norm(S.toarray(), 2))


def _build_signed_transition(
    tfs: list[str], targets: list[str], signs: list[float]
) -> tuple[list[str], sp.csr_matrix, sp.csr_matrix]:
    """Row-normalize a signed directed edge list into a transition + sign pattern."""
    nodes = sorted(set(tfs) | set(targets))
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    rows = [idx[t] for t in targets]  # target receives the edge
    cols = [idx[s] for s in tfs]  # from the TF
    weights = np.asarray(signs, dtype=float)
    W = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    W.sum_duplicates()
    # Row-normalize by the absolute incoming influence so |S| is sub-stochastic.
    absrow = np.asarray(abs(W).sum(axis=1)).ravel()
    dinv = np.divide(1.0, absrow, out=np.zeros_like(absrow), where=absrow > 0)
    S = (sp.diags(dinv) @ W).tocsr()
    S.eliminate_zeros()
    sign_pattern = S.copy()
    sign_pattern.data = np.sign(sign_pattern.data)
    return nodes, S, sign_pattern


def _grn_cache(cdir: Path):
    return cdir / "S.npz", cdir / "signs.npz", cdir / "meta.json"


def _write_grn_cache(cdir: Path, grn: SignedDirectedGRN) -> None:
    spath, signpath, meta = _grn_cache(cdir)
    sp.save_npz(spath, grn.S)
    sp.save_npz(signpath, grn.signs)
    meta.write_text(
        json.dumps(
            {
                "nodes": grn.nodes,
                "source": grn.source,
                "n_edges": grn.n_edges,
                "n_unknown_dropped": grn.n_unknown_dropped,
                "role": grn.role,
            }
        ),
        encoding="utf-8",
    )


def _read_grn_cache(cdir: Path) -> SignedDirectedGRN:
    spath, signpath, meta = _grn_cache(cdir)
    info = json.loads(meta.read_text(encoding="utf-8"))
    return SignedDirectedGRN(
        nodes=info["nodes"],
        S=sp.load_npz(spath).tocsr(),
        signs=sp.load_npz(signpath).tocsr(),
        source=info["source"],
        n_edges=info["n_edges"],
        n_unknown_dropped=info["n_unknown_dropped"],
        role=info.get("role", "grn_signed_directed"),
    )


def _cached_grn(name: str, path: Path, build, cache_root: Path, rebuild: bool) -> SignedDirectedGRN:
    key = CacheKey(name=name, version=PARSER_VERSION, params={"sha256": sha256_file(path)})
    cdir = cache_dir_for(cache_root, key)
    spath, _signpath, meta = _grn_cache(cdir)
    if not rebuild and spath.exists() and meta.exists():
        return _read_grn_cache(cdir)
    grn = build()
    _write_grn_cache(cdir, grn)
    return grn


def default_grn_dir() -> Path:
    return default_raw_dir() / "grn"


def load_trrust(
    raw_path: str | Path | None = None,
    species: str = "human",
    cache_root: str | Path | None = None,
    include_unknown: bool = False,
    rebuild: bool = False,
) -> SignedDirectedGRN:
    """Load TRRUST v2 (curated, signed, directed TF-target) as a P_alpha operator.

    Args:
        raw_path: TRRUST raw TSV. Default ``data/raw/grn/trrust_rawdata.<species>.tsv``.
        species: ``"human"`` or ``"mouse"``.
        cache_root: cache root. Default ``data/cache``.
        include_unknown: keep edges whose mode is ``Unknown`` (default drop them;
            they have no sign and cannot enter the signed network).
        rebuild: reparse and overwrite the cache.
    """
    if species not in TRRUST_URLS:
        raise ValueError(f"species must be one of {sorted(TRRUST_URLS)}; got {species!r}")
    raw_path = (
        Path(raw_path)
        if raw_path is not None
        else default_grn_dir() / f"trrust_rawdata.{species}.tsv"
    )
    if not raw_path.exists():
        raise FileNotFoundError(
            f"TRRUST v2 {species} table not found at {raw_path}. Fetch the real file: "
            f"{TRRUST_URLS[species]} (run scripts/download_grn.sh). REAL DATA ONLY."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()

    def build() -> SignedDirectedGRN:
        df = pd.read_csv(raw_path, sep="\t", header=None, dtype=str, comment="#")
        df = df.iloc[:, :3]
        df.columns = ["tf", "target", "mode"]
        if str(df["tf"].iloc[0]).strip().lower() in ("tf", "transcription factor"):
            df = df.iloc[1:].reset_index(drop=True)
        modes = df["mode"].astype(str).str.strip().str.lower()
        known = modes.isin(_MODE_SIGN)
        n_unknown = int((~known).sum())
        if not include_unknown:
            df, modes = df[known].reset_index(drop=True), modes[known].reset_index(drop=True)
        signs = [_MODE_SIGN.get(m, 1.0) for m in modes]
        tfs = df["tf"].astype(str).str.strip().tolist()
        targets = df["target"].astype(str).str.strip().tolist()
        nodes, S, sign_pattern = _build_signed_transition(tfs, targets, signs)
        return SignedDirectedGRN(
            nodes=nodes,
            S=S,
            signs=sign_pattern,
            source=f"TRRUST v2 ({species}); {TRRUST_URLS[species]}",
            n_edges=int(S.nnz),
            n_unknown_dropped=0 if include_unknown else n_unknown,
        )

    return _cached_grn(f"trrust_{species}", raw_path, build, cache_root, rebuild)


def load_collectri(
    raw_path: str | Path | None = None,
    species: str = "mouse",
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> SignedDirectedGRN:
    """Load CollecTRI (signed, directed TF-target) from an OmniPath REST export.

    CollecTRI (Mueller-Dott 2023) is a curated meta-resource of signed, directed
    TF-target interactions. The OmniPath interactions endpoint serves it as a TSV
    with ``source_genesymbol`` / ``target_genesymbol``, ``is_directed`` and the
    signed flags ``is_stimulation`` (activation, +1) / ``is_inhibition``
    (repression, -1). Only directed edges enter S; the sign is -1 for a pure
    inhibition and +1 otherwise (activation or unsigned-default, matching the
    CollecTRI convention).

    Args:
        raw_path: TSV path. Default ``data/raw/grn/collectri_<species>.tsv``.
        species: ``"human"``, ``"mouse"`` (rat for GSE184929), or ``"rat"``.
        cache_root: cache root. Default ``data/cache``.
        rebuild: reparse and overwrite the cache.
    """
    if species not in COLLECTRI_ORGANISMS:
        raise ValueError(f"species must be one of {sorted(COLLECTRI_ORGANISMS)}; got {species!r}")
    raw_path = (
        Path(raw_path) if raw_path is not None else default_grn_dir() / f"collectri_{species}.tsv"
    )
    url = COLLECTRI_URL.format(organism=COLLECTRI_ORGANISMS[species])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"CollecTRI {species} table not found at {raw_path}. Fetch the real file: "
            f"{url} (run scripts/download_grn.sh). REAL DATA ONLY."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()

    def build() -> SignedDirectedGRN:
        df = pd.read_csv(raw_path, sep="\t", dtype=str)
        need = {
            "source_genesymbol",
            "target_genesymbol",
            "is_directed",
            "is_stimulation",
            "is_inhibition",
        }
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{raw_path}: CollecTRI export missing columns {sorted(missing)}")
        df = df[df["source_genesymbol"].notna() & df["target_genesymbol"].notna()]

        def _flag(col: str) -> np.ndarray:
            return df[col].astype(str).str.strip().isin(["True", "1", "true"]).to_numpy()

        df = df[_flag("is_directed")].reset_index(drop=True)  # directed edges only
        stim, inhib = _flag("is_stimulation"), _flag("is_inhibition")
        signs = np.where(inhib & ~stim, -1.0, 1.0).tolist()  # pure inhibition -> -1, else +1
        tfs = df["source_genesymbol"].astype(str).str.strip().tolist()
        targets = df["target_genesymbol"].astype(str).str.strip().tolist()
        nodes, S, sign_pattern = _build_signed_transition(tfs, targets, signs)
        return SignedDirectedGRN(
            nodes=nodes,
            S=S,
            signs=sign_pattern,
            source=f"CollecTRI ({species}) via OmniPath; {url}",
            n_edges=int(S.nnz),
        )

    return _cached_grn(f"collectri_{species}", raw_path, build, cache_root, rebuild)


_DORO_TF = ("tf", "source", "regulator")
_DORO_TARGET = ("target", "gene")
_DORO_CONF = ("confidence", "conf", "level")
_DORO_MOR = ("mor", "weight", "sign", "mode")


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def load_dorothea(
    raw_path: str | Path | None = None,
    confidence: tuple[str, ...] = ("A", "B", "C"),
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> SignedDirectedGRN:
    """Load DoRothEA regulons (signed, directed) at the given confidence levels.

    Expects a CSV with columns tf / target / confidence / mor (mode of
    regulation, +1 activation, -1 repression). Default keeps confidence A, B, C.
    """
    raw_path = Path(raw_path) if raw_path is not None else default_grn_dir() / "dorothea.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"DoRothEA regulon table not found at {raw_path}. {DOROTHEA_SOURCE}. REAL DATA ONLY."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()
    keep = {c.upper() for c in confidence}

    def build() -> SignedDirectedGRN:
        df = pd.read_csv(raw_path, sep=None, engine="python")
        cols = list(df.columns)
        tf = _pick(cols, _DORO_TF)
        target = _pick(cols, _DORO_TARGET)
        mor = _pick(cols, _DORO_MOR)
        conf = _pick(cols, _DORO_CONF)
        if tf is None or target is None or mor is None:
            raise ValueError(
                f"{raw_path}: need tf ({_DORO_TF}), target ({_DORO_TARGET}), and mor "
                f"({_DORO_MOR}) columns; found {cols}"
            )
        if conf is not None:
            df = df[df[conf].astype(str).str.upper().isin(keep)].reset_index(drop=True)
        mor_values = pd.to_numeric(df[mor], errors="coerce").fillna(0.0)
        nonzero = mor_values != 0.0
        df, mor_values = (
            df[nonzero].reset_index(drop=True),
            mor_values[nonzero].reset_index(drop=True),
        )
        signs = np.sign(mor_values.to_numpy(dtype=float)).tolist()
        tfs = df[tf].astype(str).str.strip().tolist()
        targets = df[target].astype(str).str.strip().tolist()
        nodes, S, sign_pattern = _build_signed_transition(tfs, targets, signs)
        return SignedDirectedGRN(
            nodes=nodes,
            S=S,
            signs=sign_pattern,
            source=f"DoRothEA regulons, confidence {sorted(keep)}",
            n_edges=int(S.nnz),
        )

    return _cached_grn("dorothea", raw_path, build, cache_root, rebuild)


def load_string_undirected(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    threshold: int = 700,
    rebuild: bool = False,
) -> StringGRN:
    """Undirected, unsigned STRING v12 confidence variant (reuses string_grn).

    STRING edges are undirected and unsigned, so the returned operator is the
    symmetric-normalized UNDIRECTED variant: use it for undirected propagation,
    NOT as the signed/directed T2 network. For the signed/directed operator use
    :func:`load_trrust` or :func:`load_dorothea`.
    """
    return load_string_grn(raw_path, cache_root=cache_root, threshold=threshold, rebuild=rebuild)
