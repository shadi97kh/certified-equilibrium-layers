#!/usr/bin/env python3
"""Project 2 (AAAI): certified L2 region for the diffusion-propagated prediction.

Certifies a radius R such that the perturbed prediction y' = P_alpha(S') x0 stays
within R of the nominal y0 with probability 1 - delta (finite-sample confidence
1 - gamma), under the confidence-calibrated random-edge noise model on a real
STRING subgraph. Reports the Monte-Carlo radius, the analytic resolvent bound,
and the out-of-sample coverage across several delta levels.

Writes results/aaai/certified_region.json and a PNG figure. Accepts --seed and
logs it (OUTLINE.md rule 3). Emits JSON and a figure only, never LaTeX or prose
(rule 7).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.data import load_string_grn  # noqa: E402
from cel.verify.containment import spectral_gap  # noqa: E402
from certification import RandomEdgeNoise, certify_l2_region, empirical_coverage  # noqa: E402

OUT = REPO / "results" / "aaai"
DEFAULT_GRAPH = REPO / "tests" / "fixtures" / "string_tiny.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.2, help="propagation teleport")
    parser.add_argument("--base-std", type=float, default=0.15, help="edge-noise scale")
    parser.add_argument("--gamma", type=float, default=0.05, help="finite-sample confidence slack")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--threshold", type=int, default=700, help="STRING combined_score cutoff")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--cache", type=Path, default=OUT / "cache")
    args = parser.parse_args()
    print(
        f"[certify] seed={args.seed} alpha={args.alpha} base_std={args.base_std} graph={args.graph}"
    )

    grn = load_string_grn(args.graph, cache_root=args.cache, threshold=args.threshold)
    n = grn.num_nodes()
    rng = np.random.default_rng(args.seed)
    x0 = rng.standard_normal(n)
    noise = RandomEdgeNoise(base_std=args.base_std)
    print(
        f"[certify] graph: {n} nodes, {grn.num_edges()} edges, spectral_gap={spectral_gap(grn.S):.4f}"
    )

    deltas = [0.2, 0.1, 0.05, 0.01]
    per_delta = []
    for delta in deltas:
        res = certify_l2_region(
            x0,
            grn.S,
            noise,
            alpha=args.alpha,
            n_samples=args.n_samples,
            delta=delta,
            gamma=args.gamma,
            rng=np.random.default_rng(args.seed + 1),
        )
        cov = empirical_coverage(
            x0,
            grn.S,
            noise,
            alpha=args.alpha,
            radius=res.radius_mc,
            n_test=args.n_samples,
            rng=np.random.default_rng(args.seed + 2),
        )
        per_delta.append(
            {
                "delta": delta,
                "radius_mc": res.radius_mc,
                "radius_analytic": res.radius_analytic,
                "coverage_out_of_sample": cov,
                "covers_nominal": bool(cov >= 1.0 - delta - 0.02),
                "analytic_dominates": bool(res.radius_analytic >= res.radius_mc),
                "insufficient_samples": res.insufficient_samples,
            }
        )
        print(
            f"[certify] delta={delta:.2f} R_mc={res.radius_mc:.4f} "
            f"R_analytic={res.radius_analytic:.4f} cov_oos={cov:.3f}"
        )

    result = {
        "seed": args.seed,
        "alpha": args.alpha,
        "base_std": args.base_std,
        "gamma": args.gamma,
        "n_samples": args.n_samples,
        "num_nodes": n,
        "spectral_gap": float(spectral_gap(grn.S)),
        "noise_model": "confidence-calibrated random-edge (edge std = base_std * (1 - confidence))",
        "per_delta": per_delta,
        "all_calibrated": all(d["covers_nominal"] for d in per_delta),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "certified_region.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[certify] wrote {out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ds = [d["delta"] for d in per_delta]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(ds, [d["radius_mc"] for d in per_delta], "o-", label="Monte-Carlo")
        ax1.plot(ds, [d["radius_analytic"] for d in per_delta], "s--", label="analytic bound")
        ax1.set_xlabel("delta")
        ax1.set_ylabel("certified L2 radius")
        ax1.legend()
        ax1.set_title("Certified radius vs delta")
        ax2.plot(ds, [d["coverage_out_of_sample"] for d in per_delta], "o-", label="realized")
        ax2.plot(ds, [1 - d for d in ds], "k--", label="nominal 1-delta")
        ax2.set_xlabel("delta")
        ax2.set_ylabel("coverage")
        ax2.legend()
        ax2.set_title("Out-of-sample coverage")
        fig.tight_layout()
        fig.savefig(OUT / "certified_region.png", dpi=120)
        plt.close(fig)
        print(f"[certify] wrote {OUT / 'certified_region.png'}")
    except ImportError:
        print("[certify] matplotlib not installed; skipped figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
