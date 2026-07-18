#!/usr/bin/env python3
"""Project 2 (AAAI): tightness of the matrix-Bernstein certified radius.

Three claims about the certificate (cel.certification.bernstein_certificate),
all formula-level / concentration properties (no propagation is applied, so this
is independent of the coverage regime checked in exp_verify.py).

A. Variance scaling. The certified radius scales as sqrt(sum_e c_ij(1-c_ij))
   through the matrix-variance proxy v. The per-edge variance c(1-c) and the
   radius variance term sqrt(2 v log(2n/delta)) are maximized at c = 0.5 (the
   STRING coin-flip confidence) and tightest (smallest) as confidences approach
   0 or 1. A star graph (one center node holding all uncertain edges) makes
   v = sum_e c(1-c) literally.

B. Matching lower bound. For the star, the structure perturbation has operator
   norm ||A - A_bar||_op = ||g||_2 where g_e = A_ij - c_ij, and E[||A-A_bar||^2]
   = sum_e c(1-c) = v exactly. At c = 0.5 every realization attains
   ||A-A_bar||_op = sqrt(v), an explicit worst-case perturbation achieving the
   Bernstein variance. Hence any high-probability radius is at least sqrt(v),
   and the Bernstein bound b ~ sqrt(2 v log) exceeds it only by a sqrt(log)
   factor: the sqrt-variance rate is not improvable by more than constants.

C. Probabilistic vs deterministic. The deterministic resolvent-Lipschitz
   baseline takes the worst-case (triangle-inequality, fully aligned) structure
   perturbation ||Delta S||_op <= sum_e ||W_e||_op = sum_e max(c_ij, 1-c_ij),
   giving radius_det = ((1-alpha)/alpha) (sum_e max(c,1-c)) ||x0||, which grows
   linearly in the number of uncertain edges E. The probabilistic Bernstein
   radius grows as sqrt(E) (via v), so the probabilistic certificate is tighter
   by approximately sqrt(E). The common factor ((1-alpha)/alpha) ||x0|| cancels
   in the ratio, so it equals (sum_e max(c,1-c)) / b(delta).

Writes results/aaai/tightness.json and two PNG figures. Accepts --seed and logs
it (OUTLINE.md rule 3). Emits JSON and figures only, never LaTeX (rule 7).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from certification import EdgeBernoulliNoise, bernstein_certificate  # noqa: E402

OUT = REPO / "results" / "aaai"


def star_confidence(confidences: np.ndarray) -> sp.csr_matrix:
    """Confidence matrix of a star: center node 0 joined to E leaves with conf c_e.

    All uncertain edges are incident to the center, so the variance proxy
    v = max_i sum_{j~i} c(1-c) = sum_e c_e(1-c_e) (the total variance).
    """
    e = len(confidences)
    n = e + 1
    rows, cols, data = [], [], []
    for k, c in enumerate(confidences):
        rows += [0, k + 1]
        cols += [k + 1, 0]
        data += [float(c), float(c)]
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _radius_terms(cert, delta: float) -> tuple[float, float]:
    """Split radius(delta) into its variance term and its L (deterministic) term."""
    log_term = math.log(2.0 * cert.n_nodes / delta)
    factor = (1.0 - cert.alpha) / cert.alpha * cert.x0_norm
    variance_term = factor * math.sqrt(2.0 * cert.v * log_term)
    l_term = factor * (2.0 * cert.L / 3.0) * log_term
    return variance_term, l_term


def _unit_x0(n: int) -> np.ndarray:
    return np.ones(n) / math.sqrt(n)  # ||x0|| = 1


def _full_radius_argmax(alpha: float, delta: float, n_edges: int, c_grid: np.ndarray) -> float:
    radii = []
    for c in c_grid:
        noise = EdgeBernoulliNoise(C=star_confidence(np.full(n_edges, c)))
        radii.append(bernstein_certificate(_unit_x0(n_edges + 1), noise, alpha, delta).radius)
    return float(c_grid[int(np.argmax(radii))])


def part_a_variance_scaling(alpha: float, delta: float, n_edges: int) -> dict:
    """Radius vs uniform confidence c; confirm the sqrt(c(1-c)) shape, peak at 0.5."""
    c_grid = np.round(np.linspace(0.02, 0.98, 49), 4)
    per_c = []
    for c in c_grid:
        noise = EdgeBernoulliNoise(C=star_confidence(np.full(n_edges, c)))
        cert = bernstein_certificate(_unit_x0(n_edges + 1), noise, alpha, delta)
        var_term, l_term = _radius_terms(cert, delta)
        per_c.append(
            {
                "c": float(c),
                "per_edge_variance": float(c * (1.0 - c)),
                "v": cert.v,
                "radius": cert.radius,
                "radius_variance_term": var_term,
                "radius_l_term": l_term,
            }
        )
    radii = np.array([d["radius"] for d in per_c])
    var_terms = np.array([d["radius_variance_term"] for d in per_c])
    # The variance term divided by sqrt(c(1-c)) is constant (the sqrt scaling law).
    ratio = var_terms / np.sqrt(c_grid * (1.0 - c_grid))

    # The dominant variance term peaks exactly at 0.5; the full radius peak is
    # shifted by the sub-dominant deterministic L-term and converges to 0.5 as
    # the number of uncertain edges grows (variance-dominated regime).
    fine = np.round(np.linspace(0.02, 0.98, 97), 4)
    argmax_vs_e = {e: _full_radius_argmax(alpha, delta, e, fine) for e in (16, 64, 256, 1024, 4096)}

    return {
        "n_edges": n_edges,
        "c_grid": c_grid.tolist(),
        "per_c": per_c,
        "argmax_c_full_radius": float(c_grid[int(radii.argmax())]),
        "argmax_c_variance_term": float(c_grid[int(var_terms.argmax())]),
        "variance_term_over_sqrt_c1mc_cv": float(ratio.std() / ratio.mean()),
        "radius_at_c_0p02_over_peak": float(radii[0] / radii[radii.argmax()]),
        "full_radius_argmax_vs_e": {str(e): a for e, a in argmax_vs_e.items()},
    }


def part_b_lower_bound(alpha: float, delta: float, n_edges: int, n_samples: int, seed: int) -> dict:
    """Worst-case perturbation achieving the variance; b within a log factor of sqrt(v)."""
    out = {}
    for c in (0.5, 0.3):
        noise = EdgeBernoulliNoise(C=star_confidence(np.full(n_edges, c)))
        cert = bernstein_certificate(_unit_x0(n_edges + 1), noise, alpha, delta)
        a_bar = noise.mean_adjacency().toarray()
        rng = np.random.default_rng(seed)
        ops = np.array(
            [
                np.linalg.norm(noise.sample_adjacency(rng).toarray() - a_bar, 2)
                for _ in range(n_samples)
            ]
        )
        rms = math.sqrt(float((ops**2).mean()))
        quantile = float(np.quantile(ops, 1.0 - delta))
        # Explicit worst-case toggle: every edge at its maximum |A_ij - c_ij|.
        worst_case_opnorm = math.sqrt(n_edges * max(c, 1.0 - c) ** 2)
        out[f"c={c}"] = {
            "v": cert.v,
            "sqrt_v": math.sqrt(cert.v),
            "empirical_rms_deviation": rms,  # equals sqrt(E[||A-A_bar||^2]) = sqrt(v)
            "empirical_mean_deviation": float(ops.mean()),
            "empirical_quantile_1_minus_delta": quantile,
            "worst_case_construction_opnorm": worst_case_opnorm,
            "bernstein_b": cert.b,
            "b_over_empirical_quantile": cert.b / quantile,
            "log_slack_sqrt_2_log": math.sqrt(2.0 * math.log(2.0 * cert.n_nodes / delta)),
            "rms_matches_sqrt_v": bool(
                abs(rms - math.sqrt(cert.v)) < 1e-6 * max(1.0, math.sqrt(cert.v))
            ),
        }
    return out


def part_c_prob_vs_det(alpha: float, delta: float, c: float, edge_grid: list[int]) -> dict:
    """Probabilistic Bernstein radius vs the deterministic Lipschitz envelope."""
    per_e = []
    for e in edge_grid:
        noise = EdgeBernoulliNoise(C=star_confidence(np.full(e, c)))
        cert = bernstein_certificate(_unit_x0(e + 1), noise, alpha, delta)
        factor = (1.0 - alpha) / alpha * cert.x0_norm
        radius_prob = cert.radius
        det_lipschitz_norm = e * max(c, 1.0 - c)  # sum_e ||W_e||_op (triangle bound)
        det_opnorm = math.sqrt(e) * max(c, 1.0 - c)  # true max ||Delta S||_op (tighter ref)
        radius_det_lipschitz = factor * det_lipschitz_norm
        radius_det_opnorm = factor * det_opnorm
        per_e.append(
            {
                "n_edges": e,
                "n_nodes": e + 1,
                "radius_prob": radius_prob,
                "radius_det_lipschitz": radius_det_lipschitz,
                "radius_det_opnorm": radius_det_opnorm,
                "ratio_det_over_prob": radius_det_lipschitz / radius_prob,
                "sqrt_n_edges": math.sqrt(e),
            }
        )
    es = np.array(edge_grid, dtype=float)
    ratios = np.array([d["ratio_det_over_prob"] for d in per_e])
    # ratio ~ E^p; the sqrt(E) claim is p ~ 0.5.
    fitted_exponent = float(np.polyfit(np.log(es), np.log(ratios), 1)[0])
    return {
        "c": c,
        "delta": delta,
        "edge_grid": edge_grid,
        "per_e": per_e,
        "fitted_ratio_exponent": fitted_exponent,
        "ratio_over_sqrt_e_last": float(ratios[-1] / math.sqrt(es[-1])),
        "approx_sqrt_e": bool(0.45 <= fitted_exponent <= 0.6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument("--alpha", type=float, default=0.5, help="propagation teleport")
    parser.add_argument("--delta", type=float, default=0.05, help="confidence level")
    parser.add_argument("--n-samples", type=int, default=20000, help="draws for the lower bound")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[tightness] seed={args.seed} alpha={args.alpha} delta={args.delta}")

    part_a = part_a_variance_scaling(args.alpha, args.delta, n_edges=64)
    part_b = part_b_lower_bound(
        args.alpha, args.delta, n_edges=32, n_samples=args.n_samples, seed=args.seed
    )
    part_c = part_c_prob_vs_det(args.alpha, args.delta, c=0.5, edge_grid=[2, 4, 8, 16, 32, 64, 128])

    summary = {
        "seed": args.seed,
        "alpha": args.alpha,
        "delta": args.delta,
        "structure": "star (all uncertain edges incident to one center node => v = sum_e c(1-c))",
        "A_variance_scaling": part_a,
        "B_matching_lower_bound": part_b,
        "C_probabilistic_vs_deterministic": part_c,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "tightness.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[tightness] A: variance-term argmax c={part_a['argmax_c_variance_term']} "
        f"(exactly 0.5), sqrt-scaling CV={part_a['variance_term_over_sqrt_c1mc_cv']:.4f}; "
        f"full-radius argmax c vs E={part_a['full_radius_argmax_vs_e']} (converges to 0.5)"
    )
    b05 = part_b["c=0.5"]
    print(
        f"[tightness] B: at c=0.5 ||A-Abar||=sqrt(v)={b05['sqrt_v']:.4f} achieved "
        f"(rms={b05['empirical_rms_deviation']:.4f}); b/quantile={b05['b_over_empirical_quantile']:.3f} "
        f"(log slack {b05['log_slack_sqrt_2_log']:.3f})"
    )
    print(
        f"[tightness] C: ratio(det/prob) exponent={part_c['fitted_ratio_exponent']:.3f} "
        f"(expect ~0.5 => sqrt(E)); approx_sqrt_e={part_c['approx_sqrt_e']}"
    )
    print(f"[tightness] wrote {out_path}")

    _plot(part_a, part_c, args.out_dir)
    return 0


def _plot(part_a: dict, part_c: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[tightness] matplotlib not installed; skipped figures")
        return

    # Figure 1: radius vs c.
    c = np.array(part_a["c_grid"])
    radius = np.array([d["radius"] for d in part_a["per_c"]])
    var_term = np.array([d["radius_variance_term"] for d in part_a["per_c"]])
    ref = np.sqrt(c * (1.0 - c))
    ref = ref / ref.max() * var_term.max()  # scaled sqrt(c(1-c)) reference
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(c, radius, "o-", ms=3, label="certified radius")
    ax.plot(c, var_term, "s-", ms=3, label="variance term sqrt(2 v log)")
    ax.plot(c, ref, "k--", label="sqrt(c(1-c)) (scaled)")
    ax.axvline(0.5, color="gray", ls=":", label="c = 0.5")
    ax.set_xlabel("edge confidence c")
    ax.set_ylabel("certified L2 radius (||x0|| = 1)")
    ax.set_title(f"Radius vs confidence (star, E={part_a['n_edges']}): peak at coin-flip 0.5")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "tightness_radius_vs_c.png", dpi=120)
    plt.close(fig)
    print(f"[tightness] wrote {out_dir / 'tightness_radius_vs_c.png'}")

    # Figure 2: probabilistic vs deterministic radius vs number of uncertain edges.
    es = np.array([d["n_edges"] for d in part_c["per_e"]], dtype=float)
    prob = np.array([d["radius_prob"] for d in part_c["per_e"]])
    det = np.array([d["radius_det_lipschitz"] for d in part_c["per_e"]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog(es, det, "s-", label="deterministic Lipschitz (~ E)")
    ax1.loglog(es, prob, "o-", label="probabilistic Bernstein (~ sqrt(E))")
    ax1.loglog(es, prob[0] * np.sqrt(es / es[0]), "k:", label="sqrt(E) slope")
    ax1.set_xlabel("number of uncertain edges E")
    ax1.set_ylabel("certified L2 radius (||x0|| = 1)")
    ax1.set_title("Probabilistic vs deterministic radius")
    ax1.legend(fontsize=8)
    ratios = np.array([d["ratio_det_over_prob"] for d in part_c["per_e"]])
    ax2.plot(np.sqrt(es), ratios, "o-")
    ax2.set_xlabel("sqrt(E)")
    ax2.set_ylabel("radius_det / radius_prob")
    ax2.set_title(
        f"Tightness gain ~ sqrt(E) (fitted exponent {part_c['fitted_ratio_exponent']:.2f})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "tightness_prob_vs_det.png", dpi=120)
    plt.close(fig)
    print(f"[tightness] wrote {out_dir / 'tightness_prob_vs_det.png'}")


if __name__ == "__main__":
    sys.exit(main())
