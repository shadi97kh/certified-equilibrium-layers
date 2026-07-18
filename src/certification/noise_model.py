"""Random-edge noise models on the inferred GRN.

Primary model (Project 2 C2): the inhomogeneous random-subgraph model. Each
candidate edge (i, j) carries a STRING combined confidence c_ij in [0, 1]; the
edge is included independently with probability c_ij, giving a random binary
adjacency A. The mean adjacency is therefore A_bar with entries c_ij, exposed by
:meth:`EdgeBernoulliNoise.mean_adjacency`. The propagation operator is built
from a normalization of A: row-stochastic D^{-1} A (the clean default for the
go/no-go), the raw adjacency, or the symmetric normalization D^{-1/2} A D^{-1/2}
(the variant used for the spectral contribution C5).

Also kept: the earlier Gaussian per-edge :class:`RandomEdgeNoise` used by the
resolvent-perturbation certificate in :mod:`certification.certify`.

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from cel.verify.containment import spectral_clip

NormalizationScheme = str  # one of "row", "sym", "none"


def normalize_adjacency(A: sp.spmatrix, scheme: NormalizationScheme = "row") -> sp.csr_matrix:
    """Normalize an adjacency into a propagation operator.

    - ``"row"``: row-stochastic D^{-1} A (spectral radius 1; clean PageRank case).
    - ``"sym"``: symmetric D^{-1/2} A D^{-1/2} (eigenvalues in [-1, 1]; C5 variant).
    - ``"none"``: the raw adjacency A (may not be contractive; use with care).

    Zero-degree nodes get an all-zero row (they keep only the teleport term).
    """
    a = A.tocsr().astype(float)
    if scheme == "none":
        return a
    deg = np.asarray(a.sum(axis=1)).ravel()
    if scheme == "row":
        with np.errstate(divide="ignore"):
            dinv = np.where(deg > 0, 1.0 / deg, 0.0)
        return (sp.diags(dinv) @ a).tocsr()
    if scheme == "sym":
        with np.errstate(divide="ignore"):
            dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        d = sp.diags(dinv_sqrt)
        return (d @ a @ d).tocsr()
    raise ValueError(f"unknown normalization scheme {scheme!r}; use 'row', 'sym', or 'none'")


@dataclass(frozen=True)
class EdgeBernoulliNoise:
    """Inhomogeneous random-subgraph noise: include edge (i, j) with prob c_ij.

    ``C`` is the symmetric confidence matrix (the mean adjacency A_bar) with
    entries in [0, 1] on the candidate-edge support. ``nodes`` is optional node
    metadata. A sampled adjacency is binary (an included edge has weight 1), so
    E[A] = C exactly.
    """

    C: sp.csr_matrix
    nodes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.C.shape[0] != self.C.shape[1]:
            raise ValueError("confidence matrix C must be square")
        if self.C.nnz and (self.C.data.min() < -1e-9 or self.C.data.max() > 1.0 + 1e-9):
            raise ValueError("confidence entries c_ij must lie in [0, 1]")

    @property
    def n_nodes(self) -> int:
        return self.C.shape[0]

    def mean_adjacency(self) -> sp.csr_matrix:
        """Return the mean adjacency A_bar = E[A], with entries c_ij."""
        return self.C.tocsr()

    def sample_adjacency(self, rng: np.random.Generator) -> sp.csr_matrix:
        """Draw one random binary symmetric subgraph A (edge (i,j) w.p. c_ij)."""
        triu = sp.triu(self.C, k=1).tocoo()
        keep = rng.random(triu.data.shape[0]) < triu.data
        r = triu.row[keep]
        c = triu.col[keep]
        rows = np.concatenate([r, c])
        cols = np.concatenate([c, r])
        data = np.ones(rows.shape[0], dtype=float)
        return sp.coo_matrix((data, (rows, cols)), shape=self.C.shape).tocsr()

    def sample_operator(
        self, rng: np.random.Generator, scheme: NormalizationScheme = "row"
    ) -> sp.csr_matrix:
        """Sample a subgraph and return its normalized propagation operator S(A)."""
        return normalize_adjacency(self.sample_adjacency(rng), scheme)

    @classmethod
    def from_confidence(cls, C: sp.spmatrix, nodes: list[str] | None = None) -> EdgeBernoulliNoise:
        """Build from a confidence matrix (clipped to [0, 1] on the support)."""
        c = C.tocsr().astype(float)
        c.data = np.clip(c.data, 0.0, 1.0)
        return cls(C=c, nodes=nodes)

    @classmethod
    def from_string(cls, raw_path, threshold: int = 700, taxid: int = 10116) -> EdgeBernoulliNoise:
        """Build the model from a STRING links file (c_ij = combined_score / 1000)."""
        from cel.data import load_string_confidence

        nodes, a_bar = load_string_confidence(raw_path, threshold=threshold, taxid=taxid)
        return cls(C=a_bar.tocsr(), nodes=nodes)


# ---------------------------------------------------------------------------
# Legacy Gaussian per-edge model (used by certification.certify).
# ---------------------------------------------------------------------------
def upper_edges(S: sp.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row, col, weight) of the strict upper triangle of symmetric S."""
    coo = sp.triu(S, k=1).tocoo()
    return coo.row, coo.col, coo.data


def confidence_from_weights(weights: np.ndarray) -> np.ndarray:
    """Map edge weights to a confidence in [0, 1] (relative to the strongest edge)."""
    mx = float(weights.max()) if weights.size else 1.0
    return weights / (mx + 1e-12)


@dataclass(frozen=True)
class RandomEdgeNoise:
    """Per-edge Gaussian perturbation with edge std = base_std * (1 - confidence).

    ``edge_confidence`` (aligned to the strict-upper-triangle edge order of the
    graph passed to :meth:`perturb`) overrides the weight-derived confidence.
    """

    base_std: float = 0.1
    nonneg: bool = True
    edge_confidence: np.ndarray | None = None

    def perturb(self, S: sp.spmatrix, rng: np.random.Generator) -> sp.csr_matrix:
        """Draw a perturbed, symmetric, spectral-clipped adjacency S'."""
        row, col, w = upper_edges(S)
        conf = (
            self.edge_confidence if self.edge_confidence is not None else confidence_from_weights(w)
        )
        conf = np.clip(np.asarray(conf, dtype=float), 0.0, 1.0)
        std = self.base_std * (1.0 - conf)
        wp = w + std * rng.standard_normal(w.shape[0])
        if self.nonneg:
            wp = np.clip(wp, 0.0, None)
        n = S.shape[0]
        data = np.concatenate([wp, wp])
        rows = np.concatenate([row, col])
        cols = np.concatenate([col, row])
        sp_perturbed = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        clipped, _ = spectral_clip(sp_perturbed)
        return clipped
