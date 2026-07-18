#!/usr/bin/env python3
"""Adapt Pinet's operator-splitting projection to the CASCADE mass-action polytope.

The CASCADE mass-action / thermodynamic constraint (Theorem 3) pins the ordinary
-least-squares trend of the predicted-knockdown logits y on the design matrix
Z = [1, dG] to the calibrated mass-action line (a, b), where dG is the REAL
ViennaRNA seed-duplex free energy. As an affine equality this is

    Z^T y = Z^T Z (a, b)^T          (the normal equations of the OLS trend),

so the polytope is P = { y : A y = b } with A = Z^T (2 x N) and b = Z^T Z (a, b).
A Pinet ``EqualityConstraint`` projects any prediction onto P with the closed-form
orthogonal projection x - A^+ (A x - b); Pinet backpropagates through it with the
implicit function theorem.

REAL DATA ONLY. dG comes from ViennaRNA on the real Huesken antisense / target
pairs (raises if the data or ViennaRNA is absent). This is a standalone script
(not a cascade submodule) so its JAX / Pinet dependency stays off the core path;
it adds src/ and external/pinet/src to sys.path itself.

Run:  python code/cascade/pinet_massaction.py
Verifies: (1) ||A y - b||_inf < 1e-12 in float64 on Huesken 2431; (2) the
implicit-function-theorem gradient agrees with central finite differences to 1e-6
on a 100-example minibatch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PINET_DISABLE_IMPORT_HOOK", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external" / "pinet" / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # float64 BEFORE any jax array is built
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from pinet import EqualityConstraint, ProjectionInstance  # noqa: E402


class MassActionPolytopeProjection:
    """Pinet projection onto the CASCADE mass-action polytope P = {y : A y = b}.

    Args:
        dG: real ViennaRNA seed-duplex free energies, shape (N,).
        a, b: the calibrated mass-action line logit(knockdown) = a + b * dG.
    """

    def __init__(self, dG: np.ndarray, a: float, b: float) -> None:
        dG = np.asarray(dG, dtype=np.float64)
        z = np.stack([np.ones_like(dG), dG], axis=1)  # Z = [1, dG], (N, 2)
        # Coefficient form: pin the OLS coefficients of y on Z directly. A = Z^+ =
        # (Z^T Z)^{-1} Z^T maps y to its (intercept, slope); the rhs is the O(1)
        # mass-action line (a, b), so the residual is exact to float64 machine
        # precision rather than scaled by the (large) dG moments.
        self.A = np.linalg.pinv(z)  # (2, N) OLS-coefficient map
        self.rhs = np.array([a, b], dtype=np.float64)  # (2,), O(1)
        self.A_j = jnp.asarray(self.A)[None]  # (1, 2, N) Pinet batched a_mat
        self.b_j = jnp.asarray(self.rhs).reshape(1, 2, 1)  # (1, 2, 1)
        self.eq = EqualityConstraint(self.A_j, self.b_j, method="pinv")

    def project(self, x: np.ndarray) -> jnp.ndarray:
        """Project a prediction vector x (shape (N,)) onto P; returns (N,)."""
        xj = jnp.asarray(x, dtype=jnp.float64).reshape(1, -1, 1)
        return self.eq.project(ProjectionInstance(x=xj)).x.reshape(-1)

    def residual_inf(self, y: jnp.ndarray) -> float:
        """||A y - b||_inf, the max absolute constraint violation."""
        yj = jnp.asarray(y, dtype=jnp.float64).reshape(1, -1, 1)
        return float(jnp.max(jnp.abs(self.A_j @ yj - self.b_j)))


def huesken_seed_dg(raw_dir: Path, cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Real Huesken knockdown + ViennaRNA seed-duplex dG (raises if absent)."""
    from cel.data import load_huesken_efficacy
    from cel.features.seedmatch import revcomp
    from cel.features.thermo import require_vienna, seed_duplex_dg

    require_vienna()
    frame = load_huesken_efficacy(raw_dir, cache_root=cache_root).frame
    anti = frame["antisense"].astype(str).tolist()
    sense = frame["sense"].astype(str).tolist()
    targets = [(s.strip() or revcomp(a)) for a, s in zip(anti, sense, strict=True)]
    dG = np.array(
        [seed_duplex_dg(a, t) for a, t in zip(anti, targets, strict=True)], dtype=np.float64
    )
    knockdown = frame["efficacy"].to_numpy(dtype=np.float64)
    return knockdown, dG


def _mass_action_line(dG: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> tuple[float, float]:
    """OLS of logit(y) on [1, dG] (the calibrated mass-action line)."""
    z = np.stack([np.ones_like(dG), dG], axis=1)
    yc = np.clip(y, eps, 1.0 - eps)
    logit = np.log(yc / (1.0 - yc))
    coeffs, *_ = np.linalg.lstsq(z, logit, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def verify() -> int:
    raw_dir = REPO / "data" / "raw" / "efficacy"
    cache_root = REPO / "results" / "aaai" / "cache"
    knockdown, dG = huesken_seed_dg(raw_dir, cache_root)
    n = len(dG)
    a, b = _mass_action_line(dG, knockdown)
    print(
        f"[pinet] Huesken N={n} | mass-action line a={a:.4f} b={b:.4f} | dG in [{dG.min():.2f},{dG.max():.2f}]"
    )

    # (1) Residual on all Huesken: project a real-scale prediction onto the polytope.
    proj = MassActionPolytopeProjection(dG, a, b)
    rng = np.random.default_rng(0)
    logit_kd = np.log(np.clip(knockdown, 1e-3, 1 - 1e-3) / (1 - np.clip(knockdown, 1e-3, 1 - 1e-3)))
    x_pred = logit_kd + rng.standard_normal(n)  # a perturbed real prediction to project
    y = proj.project(x_pred)
    resid = proj.residual_inf(y)
    print(
        f"[pinet] (1) float64 constraint residual ||A y - b||_inf = {resid:.3e}  (< 1e-12: {resid < 1e-12})"
    )

    # (2) IFT gradient vs central finite differences on a 100-example minibatch.
    m = 100
    sub = MassActionPolytopeProjection(dG[:m], a, b)
    x0 = jnp.asarray(x_pred[:m], dtype=jnp.float64)

    def loss(xv: jnp.ndarray) -> jnp.ndarray:
        # a generic differentiable readout of the projected point
        return jnp.sum(sub.project(xv) ** 2)

    g_ift = np.asarray(jax.grad(loss)(x0))  # Pinet IFT backward
    eps = 1e-6
    g_fd = np.zeros(m)
    for i in range(m):
        g_fd[i] = float((loss(x0.at[i].add(eps)) - loss(x0.at[i].add(-eps))) / (2 * eps))
    max_diff = float(np.max(np.abs(g_ift - g_fd)))
    print(
        f"[pinet] (2) IFT grad vs finite-diff max|diff| = {max_diff:.3e}  (< 1e-6: {max_diff < 1e-6})"
    )

    ok = resid < 1e-12 and max_diff < 1e-6
    print(f"[pinet] VERDICT: {'PASS' if ok else 'FAIL'} (residual<1e-12 and FD-agreement<1e-6)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(verify())
