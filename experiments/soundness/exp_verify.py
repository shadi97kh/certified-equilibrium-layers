#!/usr/bin/env python3
"""Project 2 (AAAI): empirical verification of the matrix-Bernstein certificate.

On synthetic graphs with known edge confidences, draw many random adjacencies
A ~ inhomogeneous random subgraph, compute the true deviation
|| P_alpha(A) x0 - y_plugin ||_2 from the plug-in mean-graph prediction
y_plugin = P_alpha(A_bar) x0, and check empirically that it stays within the
certified radius(delta) (cel.certification.bernstein_certificate) at least a
1 - delta fraction of the time. Sweep delta and report empirical coverage versus
the nominal 1 - delta.

The certified radius MUST NOT be violated more than a delta fraction of the
time. Any violation (empirical violation rate above delta) is recorded
prominently in the JSON and printed, and the script exits non-zero.

Regime. The certificate is the un-normalized case S(A) = A and uses
|| P_alpha ||_op = 1, which holds when the operator is spectrally bounded by 1.
For binary subgraphs that means matchings (disjoint candidate edges, every node
degree <= 1), so the synthetic graphs here are random matchings: every drawn A
then satisfies || A ||_op <= 1 and the certificate's assumption holds. This is
verified at run time (max observed spectral norm is recorded).

Writes results/aaai/coverage.json and results/aaai/coverage.png. Accepts --seed
and logs it (OUTLINE.md rule 3). Emits JSON and a figure only, never LaTeX
(rule 7).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.verify.containment import dense_propagation_operator  # noqa: E402
from certification import EdgeBernoulliNoise, bernstein_certificate  # noqa: E402

OUT = REPO / "results" / "aaai"
DELTA_GRID = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]


def matching_confidence(confidences: np.ndarray) -> sp.csr_matrix:
    """Confidence matrix of a matching: edge e joins nodes 2e, 2e+1 with conf c_e.

    Disjoint edges keep the spectral norm of every binary subgraph at most 1, so
    the un-normalized P_alpha certificate's ||P_alpha||=1 assumption holds.
    """
    k = len(confidences)
    n = 2 * k
    rows, cols, data = [], [], []
    for e, c in enumerate(confidences):
        i, j = 2 * e, 2 * e + 1
        rows += [i, j]
        cols += [j, i]
        data += [float(c), float(c)]
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _configs(seed: int) -> list[dict]:
    """A sweep of random matching ensembles (varying size, confidences, alpha)."""
    rng = np.random.default_rng(seed)
    specs = [
        {"n_edges": 3, "alpha": 0.5, "lo": 0.4, "hi": 0.95},
        {"n_edges": 5, "alpha": 0.5, "lo": 0.3, "hi": 0.9},
        {"n_edges": 8, "alpha": 0.3, "lo": 0.5, "hi": 0.95},
        {"n_edges": 8, "alpha": 0.7, "lo": 0.2, "hi": 0.8},
        {"n_edges": 12, "alpha": 0.5, "lo": 0.3, "hi": 0.7},
    ]
    configs = []
    for idx, s in enumerate(specs):
        confs = rng.uniform(s["lo"], s["hi"], size=s["n_edges"])
        x0 = rng.standard_normal(2 * s["n_edges"])
        configs.append({"id": idx, "alpha": s["alpha"], "confidences": confs, "x0": x0})
    return configs


def _evaluate_config(cfg: dict, n_samples: int, seed: int) -> dict:
    confidences = np.asarray(cfg["confidences"])
    alpha = cfg["alpha"]
    x0 = np.asarray(cfg["x0"])
    c = matching_confidence(confidences)
    noise = EdgeBernoulliNoise(C=c)
    cert = bernstein_certificate(x0, noise, alpha, delta=0.05)

    y_plugin = dense_propagation_operator(noise.mean_adjacency(), alpha) @ x0
    rng = np.random.default_rng(seed)
    devs = np.empty(n_samples)
    max_spec = 0.0
    for t in range(n_samples):
        a = noise.sample_adjacency(rng)
        max_spec = max(max_spec, float(np.linalg.norm(a.toarray(), 2)))
        devs[t] = float(np.linalg.norm(dense_propagation_operator(a, alpha) @ x0 - y_plugin))

    per_delta = []
    for delta in DELTA_GRID:
        radius = cert.radius_at(delta)
        coverage = float(np.mean(devs <= radius))
        violation_rate = 1.0 - coverage
        per_delta.append(
            {
                "delta": delta,
                "nominal_coverage": 1.0 - delta,
                "radius": radius,
                "empirical_coverage": coverage,
                "violation_rate": violation_rate,
                "violated": bool(violation_rate > delta),
            }
        )

    return {
        "id": cfg["id"],
        "n_nodes": noise.n_nodes,
        "n_edges": len(confidences),
        "alpha": alpha,
        "confidences": [round(float(x), 4) for x in confidences],
        "x0_norm": float(np.linalg.norm(x0)),
        "v": cert.v,
        "L": cert.L,
        "max_spectral_norm_observed": max_spec,
        "max_deviation_observed": float(devs.max()),
        "per_delta": per_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument(
        "--n-samples", type=int, default=20000, help="random adjacencies per config"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[verify] seed={args.seed} n_samples={args.n_samples}")

    configs = _configs(args.seed)
    results = [_evaluate_config(c, args.n_samples, args.seed + 1 + c["id"]) for c in configs]

    # The certificate is violated if any (config, delta) has violation_rate > delta.
    excesses = [d["violation_rate"] - d["delta"] for r in results for d in r["per_delta"]]
    max_excess = max(excesses)
    violations = [
        {"config": r["id"], "delta": d["delta"], "violation_rate": d["violation_rate"]}
        for r in results
        for d in r["per_delta"]
        if d["violated"]
    ]
    any_violation = len(violations) > 0

    summary = {
        "seed": args.seed,
        "n_samples": args.n_samples,
        "delta_grid": DELTA_GRID,
        "regime": (
            "spectrally-bounded random matchings (||A||_op <= 1), the regime where "
            "the un-normalized P_alpha certificate assumption ||P_alpha||=1 holds"
        ),
        "max_observed_spectral_norm": max(r["max_spectral_norm_observed"] for r in results),
        "configs": results,
        "n_violations": len(violations),
        "violations": violations,
        "max_excess_violation": max_excess,  # violation_rate - delta; must be <= 0
        "any_violation": any_violation,
        "verdict": "FAIL" if any_violation else "PASS",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "coverage.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for r in results:
        print(
            f"[verify] config {r['id']}: n={r['n_nodes']} alpha={r['alpha']} max||A||={r['max_spectral_norm_observed']:.3f}"
        )
        for d in r["per_delta"]:
            tag = "  <<< VIOLATION" if d["violated"] else ""
            print(
                f"[verify]   delta={d['delta']:.2f} nominal={d['nominal_coverage']:.2f} "
                f"empirical={d['empirical_coverage']:.4f} radius={d['radius']:.3f}{tag}"
            )
    if any_violation:
        print(
            f"[verify] !!! CERTIFICATE VIOLATED in {len(violations)} cases: {violations}",
            file=sys.stderr,
        )
    else:
        print(f"[verify] PASS: no violations (max violation_rate - delta = {max_excess:.4f} <= 0)")
    print(f"[verify] wrote {out_path}")

    _plot(results, args.out_dir / "coverage.png")
    return 1 if any_violation else 0


def _plot(results: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[verify] matplotlib not installed; skipped figure")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    deltas = DELTA_GRID
    nominal = [1.0 - d for d in deltas]
    ax1.plot(deltas, nominal, "k--", label="nominal 1 - delta")
    for r in results:
        cov = [d["empirical_coverage"] for d in r["per_delta"]]
        ax1.plot(deltas, cov, "o-", label=f"cfg {r['id']} (n={r['n_nodes']}, a={r['alpha']})")
    ax1.set_xlabel("delta")
    ax1.set_ylabel("coverage")
    ax1.set_title("Empirical coverage vs nominal (empirical >= nominal = sound)")
    ax1.set_ylim(0.6, 1.02)
    ax1.legend(fontsize=7)

    ax2.plot(deltas, deltas, "k--", label="delta budget")
    for r in results:
        vio = [d["violation_rate"] for d in r["per_delta"]]
        ax2.plot(deltas, vio, "o-", label=f"cfg {r['id']}")
    ax2.set_xlabel("delta")
    ax2.set_ylabel("violation rate")
    ax2.set_title("Violation rate vs delta budget (must stay below the line)")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[verify] wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
