"""Stage D verification: the monotone equilibrium layer's fixed point, IFT gradient, APPNP limit.

Builds MonotoneEquilibriumPropagation over the real human TRRUST operator and checks, in float64:
  1. the forward fixed-point residual is at machine precision (unique fixed point reached);
  2. the equilibrium equals the closed-form resolvent, and with w=1 reproduces the existing
     cel.verify.offtarget_bound.propagate (the APPNP limit);
  3. the implicit-function-theorem gradient matches central finite differences to < 1e-5
     (for alpha, the edge scale w, and sample input entries);
  4. the monotone-operator contraction factor (1-alpha) w rho(S) < 1 (well-posed / unique).
Emits results/cascade_ot/ift_check.json naming the implicit-layer well-posedness + IFT theorem.

REAL DATA ONLY: the operator is the real TRRUST GRN; the probe vector is a fixed deterministic
pattern used only to verify the layer math (no reported biological number). No LaTeX.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "cascade_ot" / "ift_check.json"
GRAD_TOL = 1e-5


def _fd(fn, x, k, h=1e-6):
    import torch

    with torch.no_grad():
        x[k] += h
        lp = float(fn())
        x[k] -= 2 * h
        lm = float(fn())
        x[k] += h
    return (lp - lm) / (2 * h)


def run() -> dict:
    import torch

    from cel.data import load_trrust
    from cel.models.prop_layer import MonotoneEquilibriumPropagation, analytic_propagate
    from cel.verify import offtarget_bound as ob

    grn = load_trrust(species="human")
    s_sparse = grn.S
    n = grn.n_nodes
    s = torch.tensor(s_sparse.toarray(), dtype=torch.float64)
    # Deterministic probe vector (fixed pattern; only used to verify the layer math).
    e = torch.tensor([math.cos(0.1 * i) for i in range(n)], dtype=torch.float64, requires_grad=True)

    layer = MonotoneEquilibriumPropagation(s, alpha_init=0.3)
    alpha = float(layer.alpha())
    w = float(layer.w())

    eq = layer(e)
    beta = (1.0 - alpha) * w
    residual = float(torch.linalg.norm(eq - (alpha * e + beta * (s @ eq))))

    # APPNP limit: layer equilibrium vs closed-form resolvent (same w), and w=1 vs the numpy propagate.
    closed = analytic_propagate(s, e.detach(), alpha, w)
    appnp_layer_vs_closed = float(torch.linalg.norm(eq.detach() - closed))
    closed_w1 = analytic_propagate(s, e.detach(), alpha, 1.0)
    np_prop = torch.tensor(ob.propagate(s_sparse, e.detach().numpy(), alpha), dtype=torch.float64)
    appnp_closed_vs_numpy = float(torch.linalg.norm(closed_w1 - np_prop))

    # IFT gradient vs finite differences on a smooth scalar L = 0.5 ||E||^2.
    loss = 0.5 * (eq * eq).sum()
    loss.backward()

    def _L():
        return 0.5 * (layer(e).detach() ** 2).sum()

    g_alpha_auto = float(layer.alpha_raw.grad)
    g_w_auto = float(layer.w_raw.grad)
    g_alpha_fd = _fd(_L, layer.alpha_raw.data.view(-1), 0)
    g_w_fd = _fd(_L, layer.w_raw.data.view(-1), 0)
    e_idx = [0, n // 3, 2 * n // 3]
    e_auto = [float(e.grad[k]) for k in e_idx]
    e_fd = [_fd(_L, e.data, k) for k in e_idx]

    def rel(a, b):
        return abs(a - b) / (abs(b) + 1e-12)

    grad_errors = {
        "alpha": rel(g_alpha_auto, g_alpha_fd),
        "w": rel(g_w_auto, g_w_fd),
        **{f"e[{k}]": rel(a, b) for k, a, b in zip(e_idx, e_auto, e_fd, strict=True)},
    }
    max_grad_err = max(grad_errors.values())

    # Spectral radius of S -> contraction factor for uniqueness.
    from scipy.sparse.linalg import eigs

    rho = float(abs(eigs(s_sparse.astype(float), k=1, which="LM", return_eigenvectors=False)[0]))
    contraction = beta * rho

    checks = {
        "fixed_point_residual": residual,
        "fixed_point_machine_precision": bool(residual < 1e-9),
        "appnp_layer_vs_closed_form": appnp_layer_vs_closed,
        "appnp_closed_form_vs_numpy_propagate": appnp_closed_vs_numpy,
        "appnp_agreement": bool(appnp_layer_vs_closed < 1e-8 and appnp_closed_vs_numpy < 1e-8),
        "ift_grad_vs_finite_diff_max_rel_error": max_grad_err,
        "ift_grad_pass": bool(max_grad_err < GRAD_TOL),
        "grad_rel_errors": grad_errors,
        "spectral_radius_rho_S": rho,
        "contraction_factor": contraction,
        "unique_fixed_point": bool(contraction < 1.0),
    }
    passed = bool(
        checks["fixed_point_machine_precision"]
        and checks["appnp_agreement"]
        and checks["ift_grad_pass"]
        and checks["unique_fixed_point"]
    )
    return {
        "status": "ok",
        "stage": "Stage D: monotone equilibrium propagation layer (monDEQ/IGNN) with IFT backward",
        "supports": "implicit-layer well-posedness (unique fixed point of the non-normal P_alpha) "
        "and IFT-exact gradient; the differentiable-propagation headline contribution",
        "operator": grn.source,
        "n_nodes": n,
        "alpha": alpha,
        "edge_scale_w": w,
        "dtype": "float64",
        "checks": checks,
        "verdict": "PASS" if passed else "FAIL",
    }


def main() -> int:
    numbers = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    c = numbers["checks"]
    print(
        f"[ift-check] {numbers['stage']} | {numbers['n_nodes']} nodes | alpha={numbers['alpha']:.3f} w={numbers['edge_scale_w']:.3f}"
    )
    print(
        f"  fixed-point residual {c['fixed_point_residual']:.2e} (machine precision: {c['fixed_point_machine_precision']})"
    )
    print(
        f"  APPNP: layer-vs-closed {c['appnp_layer_vs_closed_form']:.2e}, closed-vs-numpy {c['appnp_closed_form_vs_numpy_propagate']:.2e} (agree: {c['appnp_agreement']})"
    )
    print(
        f"  IFT grad vs finite-diff max rel error {c['ift_grad_vs_finite_diff_max_rel_error']:.2e} (< {GRAD_TOL}: {c['ift_grad_pass']})"
    )
    print(
        f"  spectral radius rho(S)={c['spectral_radius_rho_S']:.4f}, contraction (1-a)w*rho={c['contraction_factor']:.4f} (unique: {c['unique_fixed_point']})"
    )
    print(f"  VERDICT: {numbers['verdict']}")
    print(f"[ift-check] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
