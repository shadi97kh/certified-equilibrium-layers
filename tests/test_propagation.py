"""Theorem 2 non-normality contrast on a real 50-node TRRUST v2 subgraph.

The magnetic SIGNED Laplacian L_q^sigma (MSGNN, He 2022) is Hermitian, so
(1-alpha)(I - L_q^sigma) has spectral radius < 1 AND, being normal, its operator
2-norm equals that spectral radius. The real directed personalized-PageRank
operator P_alpha = alpha (I - (1-alpha) S)^{-1} is NON-normal, so ||P_alpha||_2
strictly exceeds its spectral radius: with a row-stochastic directed transition
(Perron rho(S) = 1) this gives ||P_alpha||_2 > 1. That is the phenomenon the
magnetic Laplacian tames.

This test SEPARATES the directed and signed contributions to the T2 phenomenon.
The ||S||_2 ~ 8.3 blow-up on the full TRRUST is DIRECTEDNESS-driven non-normality:
a row-stochastic directed transition has Perron spectral radius 1, and its
asymmetry makes the P_alpha operator 2-norm exceed 1. The SIGNS do the opposite,
they SUPPRESS rather than inflate -- mixed +/- cycles cancel and REDUCE the
spectral radius (rho ~ 0.77 on this 50-node subgraph) instead of growing the norm.
So on the SAME 50 nodes the test asserts: (1) the magnetic SIGNED Laplacian
contracts; (2) the directed row-stochastic |S| (Perron rho = 1) is non-normal with
||P_alpha||_2 > 1; (3) the SIGNED transition has rho < 0.85 and ||P_alpha||_2 < 1.

Skips when torch_geometric_signed_directed or the real TRRUST data is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("torch")
pytest.importorskip("torch_geometric_signed_directed")
pytest.importorskip("networkx")

Q = 0.25
ALPHA = 0.15


def _trrust_scc_subgraph(n_target: int = 50) -> sp.csr_matrix:
    """A strongly connected ``n_target``-node induced subgraph of TRRUST v2 (signed)."""
    import networkx as nx
    from scipy.sparse.csgraph import connected_components

    from cel.data import load_trrust

    grn = load_trrust(species="human")  # raises FileNotFoundError if absent
    w = grn.signs.tocsr()
    adj = abs(w)
    _, labels = connected_components(adj, directed=True, connection="strong")
    scc = np.where(labels == np.bincount(labels).argmax())[0]
    if len(scc) < n_target:
        pytest.skip(f"largest TRRUST SCC has {len(scc)} < {n_target} nodes")
    # Peel the SCC down to n_target, always keeping it strongly connected.
    g = nx.from_scipy_sparse_array(adj[scc][:, scc], create_using=nx.DiGraph)
    while g.number_of_nodes() > n_target:
        order = sorted(g.nodes(), key=lambda u: g.in_degree(u) + g.out_degree(u))
        for u in order:
            h = g.copy()
            h.remove_node(u)
            if nx.is_strongly_connected(h):
                g = h
                break
        else:  # cannot stay strongly connected; drop the lowest-degree node
            g.remove_node(order[0])
    ix = np.sort(scc[np.array(sorted(g.nodes()))])
    return w[ix][:, ix].tocsr()


def test_magnetic_laplacian_contracts_while_real_p_alpha_is_non_normal() -> None:
    from cel.data import load_trrust  # noqa: F401  (import gates the data skip)

    try:
        w_sub = _trrust_scc_subgraph(50)
    except FileNotFoundError:
        pytest.skip("real TRRUST v2 data not present")
    n = w_sub.shape[0]
    assert n == 50

    # (1) Magnetic SIGNED Laplacian L_q^sigma: (1-alpha)(I - L_q^sigma) contracts.
    import torch
    from torch_geometric_signed_directed.utils import get_magnetic_signed_Laplacian

    coo = w_sub.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float)  # the SIGNED +1/-1 weights
    li, lr, lim = get_magnetic_signed_Laplacian(
        edge_index, edge_weight, normalization="sym", q=Q, num_nodes=n
    )
    laplacian = sp.coo_matrix(
        (lr.numpy() + 1j * lim.numpy(), (li[0].numpy(), li[1].numpy())), shape=(n, n)
    ).toarray()
    assert np.allclose(laplacian, laplacian.conj().T, atol=1e-5)  # Hermitian by construction
    evals = np.linalg.eigvalsh(laplacian)  # real eigenvalues (Hermitian)
    operator = (1.0 - ALPHA) * (np.eye(n) - laplacian)
    rho_mag = (1.0 - ALPHA) * float(np.max(np.abs(1.0 - evals)))
    assert rho_mag < 1.0  # spectral radius strictly less than 1
    # Hermitian -> normal -> operator 2-norm equals the spectral radius (no blow-up).
    assert np.isclose(np.linalg.norm(operator, 2), rho_mag, atol=1e-6)

    # (2) Real-valued directed P_alpha is non-normal: rho(S) = 1 but ||P_alpha||_2 > 1.
    absrow = np.asarray(abs(w_sub).sum(1)).ravel()
    dinv = np.divide(1.0, absrow, out=np.zeros_like(absrow), where=absrow > 0)
    s = (sp.diags(dinv) @ abs(w_sub)).toarray()  # row-stochastic directed transition |S|
    rho_s = float(np.max(np.abs(np.linalg.eigvals(s))))
    p_alpha = ALPHA * np.linalg.inv(np.eye(n) - (1.0 - ALPHA) * s)
    norm2 = float(np.linalg.norm(p_alpha, 2))
    assert np.isclose(rho_s, 1.0, atol=1e-6)  # Perron: strongly connected + row-stochastic
    assert norm2 > 1.0  # non-normality: the 2-norm strictly exceeds the spectral radius

    # (3) Contrast: on the SAME 50 nodes the SIGNED transition SUPPRESSES the spectral
    # radius (mixed +/- cycles cancel) rather than inflating the norm, so its P_alpha
    # does NOT blow up. This records that directedness, not the signs, drives the >1.
    s_signed = (sp.diags(dinv) @ w_sub).toarray()  # signed row-normalized transition
    rho_signed = float(np.max(np.abs(np.linalg.eigvals(s_signed))))
    p_signed = ALPHA * np.linalg.inv(np.eye(n) - (1.0 - ALPHA) * s_signed)
    norm2_signed = float(np.linalg.norm(p_signed, 2))
    assert rho_signed < 0.85  # signs reduce the spectral radius below the directed rho = 1
    assert norm2_signed < 1.0  # so the signed P_alpha 2-norm does not exceed 1 at this alpha
    assert rho_signed < rho_s and norm2_signed < norm2  # signed suppresses; directed inflates
