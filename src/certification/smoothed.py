"""Smoothed predictor over the random-subgraph edge ensemble.

For the inhomogeneous random-subgraph noise model (:mod:`noise_model`), the
diffusion-propagated prediction y(A) = P_alpha(S(A)) x0 is a random function of
the sampled adjacency A. This module estimates the smoothed predictor by Monte
Carlo over the ensemble and reports two centers:

- y_bar: the Monte-Carlo mean E_A[ P_alpha(S(A)) x0 ] (and the elementwise
  median y_median, which is more robust to heavy-tailed edge dropouts);
- y_plugin: the plug-in mean-graph prediction P_alpha(S(A_bar)) x0, where
  A_bar = E[A] is the confidence matrix.

IMPORTANT: E_A[ P_alpha(S(A)) ] is NOT P_alpha(S(A_bar)), because P_alpha
(and the degree normalization S) are nonlinear in A. So y_bar != y_plugin in
general; ``plugin_gap`` reports the discrepancy. The C2 certificate is centered
at the plug-in mean-graph prediction y_plugin because it is exactly computable
from A_bar without sampling; y_bar (and y_median) are the empirical ensemble
centers. Both are returned so callers can choose and audit the gap.

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from cel.verify.containment import dense_propagation_operator, propagate

from .noise_model import EdgeBernoulliNoise, NormalizationScheme, normalize_adjacency


def apply_p_alpha(
    operator: sp.spmatrix,
    x0: np.ndarray,
    alpha: float,
    method: str = "dense",
    num_steps: int = 64,
) -> np.ndarray:
    """Apply P_alpha built from ``operator`` to x0 (reuses the CASCADE operator).

    ``method="dense"`` uses the exact closed form (good for small graphs and the
    computable plug-in center); ``method="iterative"`` uses power iteration.
    """
    if method == "dense":
        return dense_propagation_operator(operator, alpha) @ np.asarray(x0, dtype=float)
    if method == "iterative":
        return propagate(operator, x0, alpha, num_steps)
    raise ValueError(f"unknown method {method!r}; use 'dense' or 'iterative'")


@dataclass(frozen=True)
class SmoothedPrediction:
    """Smoothed predictor outputs over the edge ensemble."""

    y_bar: np.ndarray  # Monte-Carlo mean E_A[P_alpha(S(A)) x0]
    y_median: np.ndarray  # elementwise ensemble median
    y_plugin: np.ndarray  # plug-in mean-graph prediction P_alpha(S(A_bar)) x0
    y_std: np.ndarray  # per-node ensemble standard deviation
    plugin_gap: float  # ||y_bar - y_plugin||_2 (nonzero: P_alpha nonlinear in A)
    n_samples: int
    scheme: str


def smoothed_prediction(
    x0: np.ndarray,
    noise: EdgeBernoulliNoise,
    alpha: float,
    scheme: NormalizationScheme = "row",
    n_samples: int = 512,
    method: str = "dense",
    num_steps: int = 64,
    rng: np.random.Generator | None = None,
) -> SmoothedPrediction:
    """Estimate the smoothed predictor over the random-subgraph ensemble.

    Returns both the Monte-Carlo center (y_bar, y_median) and the computable
    plug-in mean-graph center y_plugin (see the module docstring on why they
    differ). ``scheme`` defaults to row-stochastic (the clean go/no-go case);
    pass ``"sym"`` for the C5 symmetric-normalized variant.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x0 = np.asarray(x0, dtype=float)

    # Plug-in mean-graph prediction (exact, no sampling).
    m_bar = normalize_adjacency(noise.mean_adjacency(), scheme)
    y_plugin = apply_p_alpha(m_bar, x0, alpha, method, num_steps)

    # Monte-Carlo over the edge ensemble.
    samples = np.empty((n_samples, x0.shape[0]))
    for k in range(n_samples):
        operator = noise.sample_operator(rng, scheme)
        samples[k] = apply_p_alpha(operator, x0, alpha, method, num_steps)

    y_bar = samples.mean(axis=0)
    y_median = np.median(samples, axis=0)
    y_std = samples.std(axis=0)

    return SmoothedPrediction(
        y_bar=y_bar,
        y_median=y_median,
        y_plugin=y_plugin,
        y_std=y_std,
        plugin_gap=float(np.linalg.norm(y_bar - y_plugin)),
        n_samples=n_samples,
        scheme=scheme,
    )
