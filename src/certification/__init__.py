"""Project 2 (AAAI): probabilistic certified L2 region for off-target prediction.

The primary (C2) noise model is the inhomogeneous random-subgraph model: each
candidate edge is included with probability equal to its STRING confidence
(:mod:`noise_model`). The smoothed predictor over this ensemble exposes the
Monte-Carlo center y_bar and the computable plug-in mean-graph center y_plugin
(:mod:`smoothed`). The earlier resolvent-perturbation certificate
(:mod:`certify`, :mod:`calibration`) uses a Gaussian per-edge model and is kept
for comparison.

The package reuses the CASCADE P_alpha operator and the spectral clip from
``cel.verify.containment`` and never re-implements them (OUTLINE.md rule 8).
Pure numpy / scipy, so it runs in the lightweight CI image.
"""

from .calibration import calibrate_base_std, empirical_coverage
from .certificate import (
    BernsteinCertificate,
    MonteCarloCertificate,
    bernstein_b,
    bernstein_certificate,
    bernstein_radius,
    edge_variance_contributions,
    monte_carlo_certificate,
    p_alpha_eigenvalue_range,
    uniform_edge_bound,
    variance_proxy,
)
from .certify import CertResult, certify_l2_region
from .noise_model import (
    EdgeBernoulliNoise,
    RandomEdgeNoise,
    confidence_from_weights,
    normalize_adjacency,
    upper_edges,
)
from .normalized import (
    NormalizedCertificate,
    expected_degrees,
    min_degree_diagnostic,
    normalized_certificate,
)
from .smoothed import SmoothedPrediction, apply_p_alpha, smoothed_prediction

__all__ = [
    # inhomogeneous random-subgraph model (C2)
    "EdgeBernoulliNoise",
    "normalize_adjacency",
    "SmoothedPrediction",
    "smoothed_prediction",
    "apply_p_alpha",
    # matrix-Bernstein certificate (C2)
    "BernsteinCertificate",
    "bernstein_certificate",
    "bernstein_b",
    "bernstein_radius",
    "variance_proxy",
    "uniform_edge_bound",
    "edge_variance_contributions",
    "p_alpha_eigenvalue_range",
    "MonteCarloCertificate",
    "monte_carlo_certificate",
    # symmetric-normalized certificate (C5)
    "NormalizedCertificate",
    "normalized_certificate",
    "min_degree_diagnostic",
    "expected_degrees",
    # legacy Gaussian perturbation model
    "RandomEdgeNoise",
    "confidence_from_weights",
    "upper_edges",
    # resolvent-perturbation certificate
    "CertResult",
    "certify_l2_region",
    "empirical_coverage",
    "calibrate_base_std",
]
