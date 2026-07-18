"""Tests for the APPNP propagation head (torch-gated)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cel.data import load_string_grn  # noqa: E402
from cel.models.propagation_head import PropagationHead  # noqa: E402
from cel.verify import containment as C  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "string_tiny.txt"


def _random_graph(n: int, p: float, seed: int):
    """Random symmetric graph as (edge_index, edge_weight) with positive weights."""
    g = torch.Generator().manual_seed(seed)
    rows, cols, wts = [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            if torch.rand((), generator=g) < p:
                w = float(torch.rand((), generator=g)) + 0.1
                rows += [a, b]
                cols += [b, a]
                wts += [w, w]
    if not rows:  # guarantee at least one edge
        rows, cols, wts = [0, 1], [1, 0], [1.0, 1.0]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.tensor(wts, dtype=torch.float32)
    return edge_index, edge_weight


def test_alpha_in_unit_interval() -> None:
    for init in (0.01, 0.1, 0.5, 0.9, 0.99):
        head = PropagationHead(alpha_init=init)
        a = float(head.alpha.detach())
        assert 0.0 < a < 1.0
        assert abs(a - init) < 1e-5
    # A large theta still maps strictly inside (0, 1) (sigmoid is bounded).
    head = PropagationHead(alpha_init=0.5)
    with torch.no_grad():
        head.theta.fill_(10.0)
    a = float(head.alpha.detach())
    assert 0.0 < a < 1.0
    # At float32 saturation (very large theta) alpha never exceeds 1.
    with torch.no_grad():
        head.theta.fill_(60.0)
    assert 0.0 < float(head.alpha.detach()) <= 1.0


def test_alpha_init_out_of_range_raises() -> None:
    for bad in (0.0, 1.0, -0.2, 2.0):
        with pytest.raises(ValueError, match="alpha_init"):
            PropagationHead(alpha_init=bad)


def test_operator_norm_at_most_one_across_random_weights() -> None:
    # Containment by construction: ||P_alpha||_2 <= 1 for arbitrary graphs/alpha.
    for seed in range(6):
        n = 10
        edge_index, edge_weight = _random_graph(n, 0.5, seed=seed)
        torch.manual_seed(seed)
        head = PropagationHead(
            num_steps=15, alpha_init=float(np.clip(0.1 + 0.1 * seed, 0.05, 0.95))
        )
        gen = torch.Generator().manual_seed(100 + seed)
        op = head.operator_norm(edge_index, edge_weight, num_nodes=n, iters=80, generator=gen)
        assert op <= 1.0 + 1e-3, f"||P_alpha||={op} exceeds 1 (seed={seed})"


def test_forward_does_not_amplify() -> None:
    edge_index, edge_weight = _random_graph(12, 0.4, seed=3)
    head = PropagationHead(num_steps=20, alpha_init=0.2)
    torch.manual_seed(0)
    h = torch.randn(12)
    out = head(h, edge_index, edge_weight, num_nodes=12)
    assert out.shape == (12,)
    assert float(out.detach().norm()) <= float(h.norm()) + 1e-3


def test_forward_matrix_signal() -> None:
    edge_index, edge_weight = _random_graph(8, 0.6, seed=4)
    head = PropagationHead(num_steps=10, alpha_init=0.3)
    h = torch.randn(8, 5)
    out = head(h, edge_index, edge_weight, num_nodes=8)
    assert out.shape == (8, 5)
    assert torch.isfinite(out).all()


def test_matches_numpy_envelope_on_fixture(cache_root: Path) -> None:
    # The torch head and the numpy audit implement the same APPNP iteration.
    grn = load_string_grn(FIXTURE, cache_root=cache_root, threshold=700)
    n = grn.num_nodes()
    edge_index, edge_weight = grn.to_edge_index()
    alpha = 0.2
    head = PropagationHead(num_steps=64, alpha_init=alpha)

    h = torch.zeros(n)
    h[0] = 1.0
    h_np = h.detach().numpy()
    torch_out = head(h, edge_index, edge_weight, num_nodes=n).detach().numpy()
    numpy_out = C.propagate(grn.S, h_np, alpha=alpha, num_steps=64)
    assert np.allclose(torch_out, numpy_out, atol=1e-5)

    # And the certified envelope holds for the head's output, hop by hop.
    dist = C.hop_distances(grn.S, np.array([0]))
    hnorm = float(np.linalg.norm(h_np))
    for r in range(int(dist[np.isfinite(dist)].max()) + 1):
        emp = float(np.linalg.norm(torch_out[dist >= r])) / hnorm
        assert emp <= (1.0 - alpha) ** r + 1e-6


def test_alpha_is_learnable() -> None:
    edge_index, edge_weight = _random_graph(10, 0.5, seed=7)
    head = PropagationHead(num_steps=10, alpha_init=0.3, learnable_alpha=True)
    h = torch.randn(10)
    out = head(h, edge_index, edge_weight, num_nodes=10)
    loss = out.sum()
    loss.backward()
    assert head.theta.grad is not None
    assert head.theta.requires_grad

    frozen = PropagationHead(alpha_init=0.3, learnable_alpha=False)
    assert not frozen.theta.requires_grad
