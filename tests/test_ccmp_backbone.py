"""Tests for the CCMP backbone, including the L_k = 0 leakage audit.

The headline test, :func:`test_concept_input_intervention_is_isolated`, perturbs
one concept channel's input and asserts every other concept's embedding is
bit-for-bit unchanged (tolerance exactly 0). A gradient-based audit confirms the
cross-concept Jacobian is identically zero.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cel.models.ccmp_backbone import CCMPBackbone, default_concept_dims  # noqa: E402

DIMS = default_concept_dims()


def _random_features(batch: int = 4):
    torch.manual_seed(0)
    return {name: torch.randn(batch, d) for name, d in DIMS.items()}


def _build_model() -> CCMPBackbone:
    torch.manual_seed(0)
    return CCMPBackbone().eval()


def test_construction_matches_feature_layout() -> None:
    model = _build_model()
    assert model.K == 9
    assert model.concepts == list(DIMS)
    assert model.total_dim == sum(DIMS.values()) == 37


def test_per_concept_embeddings_shapes() -> None:
    model = _build_model()
    embs = model.get_per_concept_embeddings(_random_features(4))
    assert list(embs) == model.concepts
    for z in embs.values():
        assert z.shape == (4, 96)  # 3 node types * hidden_dim (32)


def test_forward_repression_scores() -> None:
    model = _build_model()
    rep = model(_random_features(5))
    assert rep.shape == (5,)
    assert torch.isfinite(rep).all()
    # Concatenated-matrix input must give the same result as the block dict.
    feats = _random_features(5)
    mat = torch.cat([feats[n] for n in model.concepts], dim=-1)
    assert torch.equal(model(mat), model(feats))


def test_concept_input_intervention_is_isolated() -> None:
    # Perturbing concept k's input must not move any other concept's embedding.
    model = _build_model()
    feats = _random_features(4)
    base = model.get_per_concept_embeddings(feats)

    for target in model.concepts:
        perturbed = dict(feats)
        torch.manual_seed(123)
        perturbed[target] = feats[target] + 5.0 * torch.randn_like(feats[target])
        after = model.get_per_concept_embeddings(perturbed)

        for name in model.concepts:
            if name == target:
                # The intervened channel must actually respond (not a constant).
                assert not torch.equal(base[name], after[name])
            else:
                # Every other channel is bit-for-bit identical: tolerance 0.
                assert torch.equal(base[name], after[name]), f"leakage from {target} into {name}"


def test_cross_concept_jacobian_is_exactly_zero() -> None:
    # Stronger audit: d z_j / d x_k == 0 for all j != k (autograd, not finite diff).
    model = _build_model()
    feats = _random_features(3)
    target = "seed_complementarity"
    x_k = feats[target].clone().requires_grad_(True)
    inp = dict(feats)
    inp[target] = x_k

    embs = model.get_per_concept_embeddings(inp)
    others = torch.cat([embs[n] for n in model.concepts if n != target], dim=-1)
    grad = torch.autograd.grad(others.sum(), x_k, allow_unused=True)[0]
    # No path connects x_k to the other concepts: grad is None or all zeros.
    assert grad is None or torch.count_nonzero(grad) == 0


def test_own_concept_has_nonzero_jacobian() -> None:
    # Sanity: the intervened concept's own embedding does depend on its input.
    model = _build_model()
    feats = _random_features(3)
    target = "seed_complementarity"
    x_k = feats[target].clone().requires_grad_(True)
    inp = dict(feats)
    inp[target] = x_k

    z_k = model.get_per_concept_embeddings(inp)[target]
    grad = torch.autograd.grad(z_k.sum(), x_k, allow_unused=True)[0]
    assert grad is not None
    assert torch.count_nonzero(grad) > 0
