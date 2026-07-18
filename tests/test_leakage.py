"""Tests for the Theorem 1 leakage-audit primitives (cel.verify.leakage).

Pure numpy / sklearn (no torch): the audit functions are model-agnostic, so a
disjoint embedding map and a shared one are checked with plain closures.
"""

from __future__ import annotations

import numpy as np

from cel.verify import leakage as lk

NAMES = ["c0", "c1", "c2"]
SLICES = {"c0": slice(0, 2), "c1": slice(2, 4), "c2": slice(4, 6)}


def test_intervention_disjoint_is_machine_zero_shared_leaks() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))

    def disjoint(x):  # each concept embedding reads only its own block
        return {k: x[:, SLICES[k]].sum(axis=1, keepdims=True) for k in NAMES}

    def shared(x):  # every concept embedding reads all blocks
        s = x.sum(axis=1, keepdims=True)
        return {k: s + float(i) for i, k in enumerate(NAMES)}

    dj = lk.intervention_leakage(disjoint, X, SLICES, NAMES, seed=0)
    sh = lk.intervention_leakage(shared, X, SLICES, NAMES, seed=0)
    assert dj["max_cross_channel_change"] == 0.0  # other channels bit-for-bit unchanged
    for k in NAMES:
        assert dj["per_concept"][k]["self_change"] > 0.0  # the perturbed channel does move
    assert sh["max_cross_channel_change"] > 1e-6  # shared encoder leaks


def test_leakage_mi_disjoint_below_shared() -> None:
    rng = np.random.default_rng(1)
    n = 300
    X = rng.standard_normal((n, 3))  # 3 concepts, one column each
    slices = {"c0": slice(0, 1), "c1": slice(1, 2), "c2": slice(2, 3)}
    y = X[:, 0] + X[:, 1] + X[:, 2] + 0.1 * rng.standard_normal(n)

    # Disjoint c_hat: each predicted score is a function of its OWN block.
    c_hat_disjoint = X.copy()
    # Leaky c_hat: concept 0's score encodes the OTHER concepts (cross leakage),
    # so it leaks whichever other concept the nuisance selector picks.
    c_hat_leaky = X.copy()
    c_hat_leaky[:, 0] = X[:, 1] + X[:, 2]

    dj = lk.leakage_mi(c_hat_disjoint, X, slices, ["c0", "c1", "c2"], y, seed=0)
    leaky = lk.leakage_mi(c_hat_leaky, X, slices, ["c0", "c1", "c2"], y, seed=0)
    assert dj["mean_leakage_auc_gap"] is not None and leaky["mean_leakage_auc_gap"] is not None
    assert leaky["mean_leakage_auc_gap"] > dj["mean_leakage_auc_gap"]
    assert dj["mean_leakage_auc_gap"] < 0.1  # disjoint c_hat carries ~no nuisance info beyond X_k


def test_completeness_score_full_vs_subset() -> None:
    rng = np.random.default_rng(2)
    n = 200
    X = rng.standard_normal((n, 5))
    w = np.array([1.0, -2.0, 0.5, 1.5, -1.0])
    y = X @ w + 0.05 * rng.standard_normal(n)

    full = lk.completeness_score(X, X, y, seed=0)
    subset = lk.completeness_score(X[:, :2], X, y, seed=0)
    assert 0.0 <= full["completeness"] <= 1.0
    assert full["completeness"] > 0.9  # the full concept set explains the task
    assert subset["completeness"] < full["completeness"]  # a strict subset is less complete


def test_concept_true_values_are_block_means() -> None:
    X = np.arange(12, dtype=float).reshape(2, 6)
    ct = lk.concept_true_values(X, SLICES, NAMES)
    assert np.allclose(ct["c0"], X[:, 0:2].mean(axis=1))
    assert set(ct) == set(NAMES)
