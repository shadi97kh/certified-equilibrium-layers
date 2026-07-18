"""Calibration utilities for the certified region.

The certificate's confidence is only meaningful if the realized out-of-sample
coverage matches the nominal level. :func:`empirical_coverage` checks this on
fresh perturbations, and :func:`calibrate_base_std` tunes the noise scale to a
target certified radius (bisection).

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cel.verify.containment import propagate, spectral_clip

from .certify import certify_l2_region
from .noise_model import RandomEdgeNoise


def empirical_coverage(
    x0: np.ndarray,
    S: sp.spmatrix,
    noise: RandomEdgeNoise,
    alpha: float,
    radius: float,
    num_steps: int = 64,
    n_test: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Fraction of fresh perturbed predictions that fall within ``radius`` of y0."""
    rng = rng if rng is not None else np.random.default_rng(1)
    s_clipped, _ = spectral_clip(S)
    y0 = propagate(s_clipped, x0, alpha, num_steps)
    inside = 0
    for _ in range(n_test):
        y_prime = propagate(noise.perturb(S, rng), x0, alpha, num_steps)
        inside += int(np.linalg.norm(y_prime - y0) <= radius)
    return inside / n_test


def calibrate_base_std(
    x0: np.ndarray,
    S: sp.spmatrix,
    alpha: float,
    target_radius: float,
    delta: float = 0.05,
    gamma: float = 0.05,
    num_steps: int = 64,
    n_samples: int = 400,
    lo: float = 1e-3,
    hi: float = 2.0,
    iters: int = 18,
    seed: int = 0,
) -> float:
    """Bisection on ``base_std`` so the certified radius matches ``target_radius``.

    The certified radius is monotone increasing in the noise scale, so bisection
    converges.
    """

    def radius_at(std: float) -> float:
        noise = RandomEdgeNoise(base_std=std)
        rng = np.random.default_rng(seed)
        return certify_l2_region(
            x0, S, noise, alpha, num_steps, n_samples, delta, gamma, rng
        ).radius_mc

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if radius_at(mid) < target_radius:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
