"""Tests for the full CASCADE model (torch-gated)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cel.models.cascade import CASCADE  # noqa: E402
from cel.models.ccmp_backbone import CCMPBackbone, default_concept_dims  # noqa: E402
from cel.models.propagation_head import PropagationHead  # noqa: E402


def _model() -> CASCADE:
    torch.manual_seed(0)
    backbone = CCMPBackbone(hidden_dim=16, concept_emb_dim=16, n_layers=2)
    prop = PropagationHead(num_steps=10, alpha_init=0.2)
    return CASCADE(backbone, prop).eval()


def _graph(n: int):
    # Simple ring graph (symmetric), unit weights.
    rows = list(range(n)) + [(i + 1) % n for i in range(n)]
    cols = [(i + 1) % n for i in range(n)] + list(range(n))
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.ones(len(rows), dtype=torch.float32)
    return edge_index, edge_weight


def test_forward_shape_and_finite() -> None:
    model = _model()
    n = 6
    feats = torch.randn(n, default_concept_dims_total())
    ei, ew = _graph(n)
    y = model(feats, ei, ew, num_nodes=n)
    assert y.shape == (n,)
    assert torch.isfinite(y).all()


def default_concept_dims_total() -> int:
    return sum(default_concept_dims().values())


def test_alpha_delegates_to_propagation() -> None:
    model = _model()
    assert torch.equal(model.alpha, model.propagation.alpha)
    assert 0.0 < float(model.alpha.detach()) < 1.0


def test_per_concept_embeddings_delegation() -> None:
    model = _model()
    n = 5
    feats = torch.randn(n, default_concept_dims_total())
    embs = model.get_per_concept_embeddings(feats)
    assert len(embs) == 9
    assert all(z.shape[0] == n for z in embs.values())


def test_propagation_does_not_amplify_direct_repression() -> None:
    # Containment preserved end to end: ||P_alpha x0|| <= ||x0|| (||P_alpha|| <= 1).
    model = _model()
    n = 8
    feats = torch.randn(n, default_concept_dims_total())
    ei, ew = _graph(n)
    x0 = model.direct_repression(feats)
    y = model(feats, ei, ew, num_nodes=n)
    assert float(y.detach().norm()) <= float(x0.detach().norm()) + 1e-3


def test_matrix_input_matches_dict_input() -> None:
    model = _model()
    n = 5
    feats = torch.randn(n, default_concept_dims_total())
    ei, ew = _graph(n)
    blocks = model.backbone._as_blocks(feats)
    y_mat = model(feats, ei, ew, num_nodes=n)
    y_dict = model(blocks, ei, ew, num_nodes=n)
    assert torch.equal(y_mat, y_dict)
