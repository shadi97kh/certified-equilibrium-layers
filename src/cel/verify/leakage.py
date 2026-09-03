"""Theorem 1 leakage-audit primitives: cross-concept leakage on a CBM bottleneck.

Three measurements, all model-agnostic (they take callables / arrays, never a
specific model class), so the disjoint per-concept backbone, a shared-encoder CBM,
and a Concept Embedding Model can be audited identically:

1. ``intervention_leakage`` (the architectural L_k = 0 check): perturb the input
   block of concept k and measure the maximum change induced in EVERY OTHER
   concept channel's embedding. For a disjoint per-concept backbone this is 0 to
   machine precision; for a shared encoder it is positive.

2. ``leakage_mi`` (Havasi-style empirical leakage): for each concept k, the
   predicted concept score c_hat_k should carry no information about a
   label-relevant nuisance (the most label-correlated OTHER concept) beyond what
   concept k's own real input block X_k already carries. A probe predicting the
   nuisance from c_hat_k must not beat the X_k baseline: if it does, c_hat_k leaks
   cross-concept information, i.e. I(c_hat_k ; nuisance | X_k) > 0.

3. ``completeness_score`` (Yeh et al. 2020): how well the concept set explains the
   task, R2(y | concept scores) / R2(y | full features), in [0, 1].

Pure numpy with a lazy scikit-learn import (no torch). The probes use random
splits / perturbations; this is verification-layer code, off any reported-data
path.
"""

from __future__ import annotations

import numpy as np


def concept_true_values(X: np.ndarray, slices: dict, names: list[str]) -> dict[str, np.ndarray]:
    """Scalar ground-truth value per concept: the mean of its real input block."""
    return {k: np.asarray(X[:, slices[k]], dtype=float).mean(axis=1) for k in names}


def intervention_leakage(
    embed_fn, X: np.ndarray, slices: dict, names: list[str], eps: float = 1.0, seed: int = 0
) -> dict:
    """Perturb each concept block and measure the change in other channels.

    Args:
        embed_fn: maps a feature matrix (numpy [N, D]) to a dict
            ``{concept_name: embedding array [N, d]}``.
        X: real feature matrix [N, D].
        slices: ``{concept_name: slice}`` partition of the D columns.
        names: concept names (the channels).
        eps: perturbation scale applied to the targeted block.
        seed: RNG seed for the perturbation.

    Returns the per-concept self / other change and the overall
    ``max_cross_channel_change`` (0 to machine precision for a disjoint backbone).
    """
    rng = np.random.default_rng(seed)
    base = embed_fn(X)
    out = {}
    for k in names:
        xp = np.array(X, dtype=float, copy=True)
        block = xp[:, slices[k]]
        xp[:, slices[k]] = block + eps * rng.standard_normal(block.shape)
        pert = embed_fn(xp)
        others = {
            j: float(np.abs(np.asarray(pert[j]) - np.asarray(base[j])).max())
            for j in names
            if j != k
        }
        out[k] = {
            "self_change": float(np.abs(np.asarray(pert[k]) - np.asarray(base[k])).max()),
            "max_other_change": (max(others.values()) if others else 0.0),
            "per_other_change": others,
        }
    return {
        "per_concept": out,
        "max_cross_channel_change": max(v["max_other_change"] for v in out.values()),
    }


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    half = max(1, n // 2)
    return perm[:half], perm[half:]


def _probe_auc(
    features: np.ndarray, target: np.ndarray, tr: np.ndarray, te: np.ndarray
) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    f = np.asarray(features, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    ytr = target[tr]
    yte = target[te]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None  # degenerate split, no AUC defined
    try:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(f[tr], ytr)
        prob = clf.predict_proba(f[te])[:, 1]
        return float(roc_auc_score(yte, prob))
    except Exception:  # - tiny / ill-conditioned probe
        return None


def _most_label_relevant_other(c_true: dict, y: np.ndarray, k: str, names: list[str]) -> str | None:
    best, best_abs = None, -1.0
    for j in names:
        if j == k:
            continue
        cj = c_true[j]
        if np.std(cj) == 0.0 or np.std(y) == 0.0:
            continue
        r = abs(float(np.corrcoef(cj, y)[0, 1]))
        if np.isfinite(r) and r > best_abs:
            best, best_abs = j, r
    return best


def leakage_mi(
    concept_scores: np.ndarray,
    X: np.ndarray,
    slices: dict,
    names: list[str],
    y: np.ndarray,
    seed: int = 0,
) -> dict:
    """Empirical cross-concept leakage: probe AUC from c_hat_k vs the X_k baseline.

    For each concept k, the nuisance is the median-binarized true value of the
    most label-correlated OTHER concept. ``baseline_auc_from_Xk`` is the best a
    probe can do from concept k's real block; ``probe_auc_from_chat_k`` is from the
    predicted concept-k score. ``leakage_auc_gap = max(0, probe - baseline)`` is 0
    when c_hat_k carries no cross-concept information beyond X_k.
    """
    c_true = concept_true_values(X, slices, names)
    tr, te = _split(X.shape[0], seed)
    out = {}
    for ki, k in enumerate(names):
        jstar = _most_label_relevant_other(c_true, y, k, names)
        if jstar is None:
            out[k] = {"nuisance_concept": None, "leakage_auc_gap": None}
            continue
        nz = (c_true[jstar] > np.median(c_true[jstar])).astype(int)
        base_auc = _probe_auc(X[:, slices[k]], nz, tr, te)
        probe_auc = _probe_auc(concept_scores[:, ki], nz, tr, te)
        gap = None if (base_auc is None or probe_auc is None) else max(0.0, probe_auc - base_auc)
        out[k] = {
            "nuisance_concept": jstar,
            "baseline_auc_from_Xk": base_auc,
            "probe_auc_from_chat_k": probe_auc,
            "leakage_auc_gap": gap,
        }
    gaps = [v["leakage_auc_gap"] for v in out.values() if v["leakage_auc_gap"] is not None]
    return {"per_concept": out, "mean_leakage_auc_gap": (float(np.mean(gaps)) if gaps else None)}


def _ridge_r2(features: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> float | None:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    f = np.asarray(features, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    if len(te) < 2 or np.std(y[te]) == 0.0:
        return None
    try:
        model = Ridge(alpha=1.0)
        model.fit(f[tr], y[tr])
        return float(r2_score(y[te], model.predict(f[te])))
    except Exception:  # - tiny / ill-conditioned probe
        return None


def completeness_score(
    concept_scores: np.ndarray, X_full: np.ndarray, y: np.ndarray, seed: int = 0
) -> dict:
    """Yeh-style completeness: R2(y | concept scores) / R2(y | full features) in [0,1]."""
    tr, te = _split(X_full.shape[0], seed)
    r2_c = _ridge_r2(concept_scores, y, tr, te)
    r2_f = _ridge_r2(X_full, y, tr, te)
    if r2_c is None or r2_f is None or r2_f <= 1e-8:
        comp = None
    else:
        comp = max(0.0, min(1.0, r2_c / r2_f))
    return {"r2_from_concepts": r2_c, "r2_from_full_features": r2_f, "completeness": comp}
