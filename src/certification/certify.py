"""Probabilistic certified L2 region for the diffusion-propagated prediction.

The prediction is y = P_alpha(S) x0 (the CASCADE propagation, reused). Under the
confidence-calibrated random-edge noise model on S, S' is random and so is
y' = P_alpha(S') x0. This module certifies a radius R such that

    P_{S'}( || y' - y0 ||_2 <= R )  >=  1 - delta,

where y0 is the nominal prediction, with finite-sample confidence 1 - gamma.

Two certificates are produced:

- a Monte-Carlo radius: the (1-delta) quantile of the sampled deviation, taken
  as a nonparametric (1-gamma) upper confidence bound via the binomial order
  statistic (distribution-free);
- an analytic radius: the resolvent perturbation bound
  || y' - y0 || <= ((1-alpha)/alpha) || S' - S ||_2 || x0 ||, evaluated at a
  high-probability perturbation budget. It upper-bounds the Monte-Carlo radius.

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import binom

from cel.verify.containment import operator_norm, propagate, spectral_clip

from .noise_model import RandomEdgeNoise


@dataclass(frozen=True)
class CertResult:
    """A certified L2 region with both Monte-Carlo and analytic radii."""

    radius_mc: float
    radius_analytic: float
    delta: float
    gamma: float
    n_samples: int
    coverage_in_sample: float  # fraction of drawn samples within radius_mc
    perturbation_budget: float  # high-prob ||S' - S||_2 used by the analytic bound
    alpha: float
    insufficient_samples: bool


def _order_statistic_radius(sorted_r: np.ndarray, p: float, gamma: float) -> tuple[float, bool]:
    """Distribution-free (1-gamma) upper confidence bound on the p-quantile.

    Returns the smallest order statistic r_(k) with
    P(Binomial(m, p) <= k-1) >= 1 - gamma, and a flag if no such k exists.
    """
    m = len(sorted_r)
    for k in range(1, m + 1):
        if binom.cdf(k - 1, m, p) >= 1.0 - gamma:
            return float(sorted_r[k - 1]), False
    return float(sorted_r[-1]), True  # not enough samples for this (delta, gamma)


def certify_l2_region(
    x0: np.ndarray,
    S: sp.spmatrix,
    noise: RandomEdgeNoise,
    alpha: float,
    num_steps: int = 64,
    n_samples: int = 1000,
    delta: float = 0.05,
    gamma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CertResult:
    """Certify an L2 region for y = P_alpha(S) x0 under the random-edge noise model."""
    rng = rng if rng is not None else np.random.default_rng(0)
    x0 = np.asarray(x0, dtype=float)
    s_clipped, _ = spectral_clip(S)
    y0 = propagate(s_clipped, x0, alpha, num_steps)

    deviations = np.empty(n_samples)
    op_perturb = np.empty(n_samples)
    for k in range(n_samples):
        s_prime = noise.perturb(S, rng)
        y_prime = propagate(s_prime, x0, alpha, num_steps)
        deviations[k] = float(np.linalg.norm(y_prime - y0))
        op_perturb[k] = operator_norm((s_prime - s_clipped).tocsr())

    sorted_dev = np.sort(deviations)
    radius_mc, insufficient = _order_statistic_radius(sorted_dev, p=1.0 - delta, gamma=gamma)

    # Analytic resolvent-perturbation bound at a high-probability edge budget.
    budget = float(np.quantile(op_perturb, 1.0 - gamma))
    radius_analytic = (1.0 - alpha) / alpha * budget * float(np.linalg.norm(x0))

    return CertResult(
        radius_mc=radius_mc,
        radius_analytic=radius_analytic,
        delta=delta,
        gamma=gamma,
        n_samples=n_samples,
        coverage_in_sample=float(np.mean(deviations <= radius_mc)),
        perturbation_budget=budget,
        alpha=alpha,
        insufficient_samples=insufficient,
    )
