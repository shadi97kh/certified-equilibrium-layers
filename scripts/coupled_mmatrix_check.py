"""Coupled (n >= 2) conservation-law M-matrix check on a REAL multi-guide competition dataset.

The single-pool experiments in this repo validate the DIAGONAL (n = 1) case: one guide against a
target background, where the reduced residual Jacobian is the scalar D = 1 + sum_s x_s sigma'(f).
This script is the coupled counterpart. It requires a dataset in which two or more distinct guides
demonstrably share one Argonaute/RISC pool AND whose readout resolves per-guide (or per-target,
per-guide) occupancy finely enough to populate the OFF-DIAGONAL loading terms.

Model (competitive multi-guide Langmuir on a shared site set):
  guide i has total abundance M_i and free abundance f_i; site s has abundance x_s and
  dissociation constant K_{i,s} for guide i. Site occupancy by guide i is
      theta_{i,s}(f) = (f_i / K_{i,s}) / (1 + sum_j f_j / K_{j,s}),
  and conservation for each guide is
      R_i(f) = f_i + sum_s x_s theta_{i,s}(f) - M_i = 0.
  The COUPLED residual Jacobian is
      J_ij = dR_i/df_j = delta_ij + sum_s x_s dtheta_{i,s}/df_j,
  whose off-diagonal entries (i != j) are strictly negative whenever guides i and j share any site
  with x_s > 0. Those terms are exactly what a single-pool (diagonal) treatment discards.

DATA POLICY. There is NO synthetic fallback and NO diagonal approximation. If the real multi-guide
dataset is absent, or a loader cannot read genuine per-guide occupancy from it, this script RAISES.
Nothing is tuned or fitted here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "results" / "coupled_mmatrix.json"

# A qualifying dataset must provide, from real measurement:
#   guides   : >= 2 distinct guides demonstrably sharing one RISC/AGO pool
#   M        : per-guide total abundance (loaded guide), length n
#   x        : shared site abundances, length m
#   K        : per-guide per-site dissociation constants, shape (n, m)
#   measured_cross_effect : the measured response of guide 0's occupancy to a perturbation of
#                           guide 1's abundance (sign at minimum), for the (c) comparison
#   B0       : measured background loading coefficient at guide 0, for the (d) phase boundary
CANDIDATE_PATHS = (
    REPO / "data" / "raw" / "offtarget" / "multiguide" / "coupled_multiguide.json",
    REPO / "data" / "raw" / "offtarget" / "coupled_multiguide.json",
)
REQUIRED_KEYS = ("guides", "M", "x", "K", "measured_cross_effect", "B0")

TOL = 1e-13
MAX_ITER = 500


def load_coupled_multiguide() -> dict:
    """Load a REAL multi-guide shared-pool dataset, or raise. No substitution, no approximation."""
    found = next((p for p in CANDIDATE_PATHS if p.is_file()), None)
    if found is None:
        searched = "\n  ".join(str(p.relative_to(REPO)) for p in CANDIDATE_PATHS)
        raise FileNotFoundError(
            "No real multi-guide (n >= 2) shared-pool competition dataset is available.\n"
            "The repository's guide datasets (GSE140217/GSE140218, GSE184929, GSE108823, "
            "GSE164289/GSE196839, GSE285764, GSE5814/GSE5769, and the Denzler/Bosson "
            "literature constants) each resolve ONE guide against a background, so every "
            "off-diagonal loading term is unmeasured and the coupled Jacobian cannot be built.\n"
            "Refusing to substitute, approximate, or diagonally collapse anything.\n"
            "Searched:\n  " + searched
        )
    cfg = json.loads(found.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{found} lacks required measured fields: {missing}")
    m_tot = np.asarray(cfg["M"], dtype=float)
    x = np.asarray(cfg["x"], dtype=float)
    k = np.asarray(cfg["K"], dtype=float)
    if m_tot.ndim != 1 or m_tot.size < 2:
        raise ValueError(f"{found}: need >= 2 guides for a coupled system; got {m_tot.size}")
    if k.shape != (m_tot.size, x.size):
        raise ValueError(f"{found}: K must have shape (n_guides, n_sites) = {(m_tot.size, x.size)}")
    if np.any(k <= 0) or np.any(x < 0) or np.any(m_tot <= 0):
        raise ValueError(f"{found}: K must be > 0, x >= 0, M > 0 (real measured quantities)")
    return {
        "source": str(found.relative_to(REPO)),
        "guides": list(cfg["guides"]),
        "M": m_tot,
        "x": x,
        "K": k,
        "measured_cross_effect": float(cfg["measured_cross_effect"]),
        "B0": float(cfg["B0"]),
    }


def occupancy(f: np.ndarray, x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """theta[i, s] = (f_i/K_is) / (1 + sum_j f_j/K_js) for the shared competitive site set."""
    ratio = f[:, None] / k               # (n, m)
    return ratio / (1.0 + ratio.sum(axis=0))[None, :]


def residual(f: np.ndarray, m_tot: np.ndarray, x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """R_i(f) = f_i + sum_s x_s theta_{i,s}(f) - M_i."""
    return f + (occupancy(f, x, k) * x[None, :]).sum(axis=1) - m_tot


def jacobian(f: np.ndarray, x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Coupled residual Jacobian J_ij = delta_ij + sum_s x_s dtheta_{i,s}/df_j (full off-diagonals).

    With ratio r_is = f_i/K_is and denom_s = 1 + sum_j r_js:
        dtheta_{i,s}/df_i = (1/K_is)(denom_s - r_is) / denom_s^2
        dtheta_{i,s}/df_j = -(r_is)(1/K_js) / denom_s^2      (j != i)  -> strictly negative
    """
    n = f.size
    ratio = f[:, None] / k
    denom = 1.0 + ratio.sum(axis=0)          # (m,)
    d2 = denom**2
    j_mat = np.eye(n, dtype=float)
    for i in range(n):
        for jj in range(n):
            if i == jj:
                dth = (1.0 / k[i]) * (denom - ratio[i]) / d2
            else:
                dth = -(ratio[i]) * (1.0 / k[jj]) / d2
            j_mat[i, jj] += float(np.sum(x * dth))
    return j_mat


def solve_equilibrium(m_tot: np.ndarray, x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Damped root-finding on R(f) = 0 with f > 0, using the exact coupled Jacobian."""
    f = 0.5 * m_tot.copy()
    for _ in range(MAX_ITER):
        r = residual(f, m_tot, x, k)
        if float(np.max(np.abs(r))) <= TOL * (1.0 + float(np.max(np.abs(m_tot)))):
            return f
        step = np.linalg.solve(jacobian(f, x, k), r)
        alpha = 1.0
        while alpha > 1e-12 and np.any(f - alpha * step <= 0.0):
            alpha *= 0.5
        f = f - alpha * step
    raise RuntimeError("coupled equilibrium did not converge; refusing to report a partial solve")


def m_matrix_report(j_mat: np.ndarray) -> dict:
    """Nonsingular-M-matrix test: Z-sign pattern, positive leading principal minors, eigenvalues."""
    n = j_mat.shape[0]
    offdiag = j_mat[~np.eye(n, dtype=bool)]
    minors = [float(np.linalg.det(j_mat[: i + 1, : i + 1])) for i in range(n)]
    eig = np.linalg.eigvals(j_mat)
    det = float(np.linalg.det(j_mat))
    z_pattern = bool(np.all(offdiag <= 0.0))
    positive_minors = bool(all(mn > 0.0 for mn in minors))
    return {
        "determinant": det,
        "leading_principal_minors": minors,
        "eigenvalue_real_parts": [float(v) for v in np.real(eig)],
        "eigenvalue_imag_parts": [float(v) for v in np.imag(eig)],
        "max_offdiagonal": float(np.max(offdiag)),
        "z_sign_pattern_offdiag_nonpositive": z_pattern,
        "all_leading_minors_positive": positive_minors,
        "all_eigenvalue_real_parts_positive": bool(np.all(np.real(eig) > 0.0)),
        "nonsingular_M_matrix": bool(z_pattern and positive_minors and abs(det) > 0.0),
    }


def cross_static(f: np.ndarray, x: np.ndarray, k: np.ndarray, j_mat: np.ndarray) -> dict:
    """Certified cross-guide comparative static d(theta_0 total)/dM_1 via the implicit function thm.

    dR/dM = -I, so df/dM = J^{-1}. The occupancy response of guide 0 to guide 1's abundance is
        d(sum_s x_s theta_{0,s}) / dM_1 = sum_s x_s sum_j (dtheta_{0,s}/df_j) (J^{-1})_{j,1}.
    """
    n = f.size
    jinv = np.linalg.inv(j_mat)
    ratio = f[:, None] / k
    denom = 1.0 + ratio.sum(axis=0)
    d2 = denom**2
    dtheta0 = np.empty((n, x.size), dtype=float)
    for jj in range(n):
        if jj == 0:
            dtheta0[jj] = (1.0 / k[0]) * (denom - ratio[0]) / d2
        else:
            dtheta0[jj] = -(ratio[0]) * (1.0 / k[jj]) / d2
    d_occ0_dM1 = float(np.sum(x * (dtheta0 * jinv[:, 1][:, None]).sum(axis=0)))
    return {
        "d_occupancy_guide0_d_M_guide1": d_occ0_dM1,
        "certified_sign": int(np.sign(d_occ0_dM1)),
        "certified_sign_label": "negative (cross-guide de-repression)" if d_occ0_dM1 < 0
        else ("positive" if d_occ0_dM1 > 0 else "zero"),
    }


def phase_boundary(j_mat: np.ndarray, b0: float) -> dict:
    """Locate B_0 (J^{-1})_{00} = 1 relative to the physical operating point."""
    jinv00 = float(np.linalg.inv(j_mat)[0, 0])
    q = b0 * jinv00
    return {
        "J_inv_00": jinv00,
        "B0": b0,
        "B0_times_J_inv_00": q,
        "boundary_value": 1.0,
        "margin_to_boundary": 1.0 - q,
        "side": "sub-critical (B0 (J^-1)_00 < 1)" if q < 1.0
        else ("super-critical (B0 (J^-1)_00 > 1)" if q > 1.0 else "exactly on the boundary"),
    }


def main() -> int:
    data = load_coupled_multiguide()  # raises if the real dataset is absent
    m_tot, x, k = data["M"], data["x"], data["K"]
    f = solve_equilibrium(m_tot, x, k)
    j_mat = jacobian(f, x, k)

    mm = m_matrix_report(j_mat)
    cs = cross_static(f, x, k, j_mat)
    pb = phase_boundary(j_mat, data["B0"])
    measured = data["measured_cross_effect"]
    agree = bool(np.sign(measured) == cs["certified_sign"]) if measured != 0.0 else False

    out = {
        "check": "coupled (n >= 2) conservation-law residual Jacobian: M-matrix, cross-guide "
                 "certified sign, and phase boundary, on real multi-guide data",
        "source": data["source"],
        "guides": data["guides"],
        "n_guides": int(m_tot.size),
        "n_sites": int(x.size),
        "equilibrium_free_guide": [float(v) for v in f],
        "max_abs_residual": float(np.max(np.abs(residual(f, m_tot, x, k)))),
        "coupled_jacobian": [[float(v) for v in row] for row in j_mat],
        "m_matrix": mm,
        "cross_guide_comparative_static": cs,
        "measured_cross_effect": measured,
        "certified_sign_matches_measurement": agree,
        "phase_boundary": pb,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md5 = hashlib.md5(OUT_JSON.read_bytes()).hexdigest()
    print(f"[coupled] wrote {OUT_JSON.relative_to(REPO)}  md5 {md5}")
    print(f"[coupled] nonsingular M-matrix : {mm['nonsingular_M_matrix']}")
    print(f"[coupled] certified sign       : {cs['certified_sign_label']}")
    print(f"[coupled] matches measurement  : {agree}")
    print(f"[coupled] phase boundary       : {pb['side']}, margin {pb['margin_to_boundary']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
