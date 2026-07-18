"""Reactome graph builder (TRACKED GAP, stubbed).

The certification project uses a regulatory / interaction graph as the
propagation backbone. STRING is wired (see
:func:`cel.data.load_string_grn`); Reactome is the second backbone and is
NOT implemented yet.

The target API mirrors the STRING builder: parse the Reactome functional
interaction (FI) network or pathway graph for the relevant organism, filter by
a confidence / score threshold, and build the same symmetric-normalized sparse
adjacency S = D^(-1/2) A D^(-1/2), returned as a :class:`StringGRN`-shaped
object (so downstream code that consumes ``S`` / ``edge_index`` works unchanged).

TODO: download the Reactome FI table (e.g. FIsInGene_*.txt) or the pathway
participant graph, map identifiers onto the same node space as STRING, build A,
symmetric-normalize, and return. Until then this raises NotImplementedError and
``tests/test_shared_core.py`` carries a visible xfail-strict gap test.
"""

from __future__ import annotations

from pathlib import Path

from .string_grn import StringGRN

# Bump alongside the (future) builder implementation.
BUILDER_VERSION = "reactome-stub-v0"

# Reactome's "high confidence" FI score cutoff (placeholder default for the
# future implementation; FI scores are in [0, 1]).
DEFAULT_FI_THRESHOLD = 0.5


def load_reactome_graph(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    threshold: float = DEFAULT_FI_THRESHOLD,
    rebuild: bool = False,
) -> StringGRN:
    """Stub: build a symmetric-normalized GRN from the Reactome FI / pathway graph.

    Returns a :class:`StringGRN`-shaped object once implemented. See the module
    docstring for the intended parsing pipeline.
    """
    raise NotImplementedError(
        "Reactome graph builder is a tracked gap; the symmetric-normalized S "
        "must match the STRING builder's output shape. See "
        "cel.data.reactome and the gap test in tests/test_shared_core.py."
    )
