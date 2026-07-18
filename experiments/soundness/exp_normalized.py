#!/usr/bin/env python3
"""Project 2 (AAAI): symmetric-normalized certificate (C5) and its breakdown.

Two parts.

1. Coverage. On a dense synthetic graph where the minimum-expected-degree
   assumption holds (relative degree fluctuation rho < 1), draw random
   adjacencies, compute the symmetric-normalized deviation
   ||S(A) - S(A_bar)||_op and the prediction deviation
   ||P_alpha(S(A)) x0 - P_alpha(S(A_bar)) x0||_2, and confirm they stay within
   the degree-concentration structure bound B_sym and the certified radius_sym
   (cel.certification.normalized_certificate) at least a 1 - delta fraction
   of the time.

2. Breakdown. Build a STRING-like confidence graph and sweep the STRING
   combined-confidence cutoff up toward 0.9. As the graph is thresholded
   sparser, the minimum expected degree falls; the certificate is valid only
   while rho < 1 (the finite-sample Le-Levina-Vershynin minimum-expected-degree
   condition). Report the minimum-degree diagnostic versus cutoff and the
   breakdown cutoff where rho crosses 1 and the certificate stops being valid.

Writes results/aaai/normalized.json and a breakdown figure. Accepts --seed and
logs it (OUTLINE.md rule 3). Emits JSON and figures only, never LaTeX (rule 7).
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
from certification import (  # noqa: E402
    EdgeBernoulliNoise,
    min_degree_diagnostic,
    normalized_certificate,
)
from certification.noise_model import normalize_adjacency  # noqa: E402

OUT = REPO / "results" / "aaai"


def synthetic_confidence(n: int, edge_prob: float, lo: float, hi: float, rng) -> sp.csr_matrix:
    """Random symmetric confidence matrix: each candidate edge has conf ~ U(lo, hi)."""
    iu = np.triu_indices(n, 1)
    present = rng.random(iu[0].shape[0]) < edge_prob
    conf = rng.uniform(lo, hi, iu[0].shape[0]) * present
    rows = np.concatenate([iu[0][present], iu[1][present]])
    cols = np.concatenate([iu[1][present], iu[0][present]])
    data = np.concatenate([conf[present], conf[present]])
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def threshold_confidence(C: sp.csr_matrix, cutoff: float) -> sp.csr_matrix:
    """Keep only edges with confidence >= cutoff (the STRING combined-score cut)."""
    coo = C.tocoo()
    keep = coo.data >= cutoff
    return sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=C.shape).tocsr()


def part_coverage(alpha: float, delta: float, n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 50
    c = synthetic_confidence(n, edge_prob=0.85, lo=0.75, hi=0.97, rng=rng)
    noise = EdgeBernoulliNoise(C=c)
    x0 = rng.standard_normal(n)
    cert = normalized_certificate(x0, noise, alpha, delta)

    s_bar = normalize_adjacency(noise.mean_adjacency(), "sym")
    y_plugin = dense_propagation_operator(s_bar, alpha) @ x0
    struct_dev = np.empty(n_samples)
    pred_dev = np.empty(n_samples)
    for k in range(n_samples):
        s_a = normalize_adjacency(noise.sample_adjacency(rng), "sym")
        struct_dev[k] = float(np.linalg.norm((s_a - s_bar).toarray(), 2))
        pred_dev[k] = float(np.linalg.norm(dense_propagation_operator(s_a, alpha) @ x0 - y_plugin))

    struct_cov = float(np.mean(struct_dev <= cert.structure_bound))
    pred_cov = float(np.mean(pred_dev <= cert.radius_sym))
    return {
        "n_nodes": n,
        "mean_expected_degree": cert.expected_min_degree,  # reported below; keep min too
        "expected_min_degree": cert.expected_min_degree,
        "degree_rel_fluctuation": cert.degree_rel_fluctuation,
        "assumption_holds": cert.assumption_holds,
        "structure_bound": cert.structure_bound,
        "radius_sym": cert.radius_sym,
        "n_samples": n_samples,
        "empirical_structure_quantile": float(np.quantile(struct_dev, 1.0 - delta)),
        "empirical_max_structure_deviation": float(struct_dev.max()),
        "structure_coverage": struct_cov,
        "prediction_coverage": pred_cov,
        "structure_no_violation": bool(struct_cov >= 1.0 - delta),
        "prediction_no_violation": bool(pred_cov >= 1.0 - delta),
    }


def part_breakdown(alpha: float, delta: float, seed: int, llv_const: float) -> dict:
    rng = np.random.default_rng(seed + 1)
    n = 250
    edge_prob = 0.6
    c_full = synthetic_confidence(n, edge_prob=edge_prob, lo=0.4, hi=1.0, rng=rng)
    x0 = rng.standard_normal(n)
    cutoffs = [round(x, 3) for x in np.arange(0.70, 0.951, 0.025)]

    per_cutoff = []
    breakdown_cutoff = None
    llv_breakdown_cutoff = None
    for cutoff in cutoffs:
        c_cut = threshold_confidence(c_full, cutoff)
        noise = EdgeBernoulliNoise(C=c_cut)
        diag = min_degree_diagnostic(noise, delta, llv_const)
        cert = normalized_certificate(x0, noise, alpha, delta, llv_const)
        per_cutoff.append(
            {
                "cutoff": cutoff,
                "n_active": diag["n_active"],
                "mean_expected_degree": diag["mean_expected_degree"],
                "min_expected_degree": diag["expected_min_degree"],
                "degree_rel_fluctuation": diag["degree_rel_fluctuation"],
                "eta": cert.eta if np.isfinite(cert.eta) else None,
                "llv_min_degree_threshold": diag["llv_min_degree_threshold"],
                "llv_condition_holds": diag["llv_condition_holds"],
                "assumption_holds": diag["assumption_holds"],
                "radius_sym": cert.radius_sym if np.isfinite(cert.radius_sym) else None,
            }
        )
        if breakdown_cutoff is None and not diag["assumption_holds"]:
            breakdown_cutoff = cutoff
        if llv_breakdown_cutoff is None and not diag["llv_condition_holds"]:
            llv_breakdown_cutoff = cutoff

    return {
        "graph": {"n_nodes": n, "candidate_edge_prob": edge_prob, "confidence_range": [0.4, 1.0]},
        "llv_const": llv_const,
        "cutoff_grid": cutoffs,
        "per_cutoff": per_cutoff,
        "breakdown_cutoff": breakdown_cutoff,
        "llv_breakdown_cutoff": llv_breakdown_cutoff,
        "criterion": "rho = max_i t_i/dbar_i < 1 (degrees concentrate); breakdown at first cutoff with rho >= 1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument("--alpha", type=float, default=0.5, help="propagation teleport")
    parser.add_argument("--delta", type=float, default=0.1, help="confidence level")
    parser.add_argument("--n-samples", type=int, default=2000, help="coverage draws")
    parser.add_argument("--llv-const", type=float, default=1.0, help="LLV min-degree constant C")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[normalized] seed={args.seed} alpha={args.alpha} delta={args.delta}")

    coverage = part_coverage(args.alpha, args.delta, args.n_samples, args.seed)
    breakdown = part_breakdown(args.alpha, args.delta, args.seed, args.llv_const)

    summary = {
        "seed": args.seed,
        "alpha": args.alpha,
        "delta": args.delta,
        "route": "concentration of the normalized adjacency (Le-Levina-Vershynin style)",
        "assumption": "minimum expected degree: rho = max_i t_i/dbar_i < 1 (degrees concentrate)",
        "coverage": coverage,
        "breakdown": breakdown,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "normalized.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[normalized] coverage: rho={coverage['degree_rel_fluctuation']:.3f} "
        f"B_sym={coverage['structure_bound']:.3f} struct_cov={coverage['structure_coverage']:.4f} "
        f"pred_cov={coverage['prediction_coverage']:.4f} "
        f"(no_violation={coverage['structure_no_violation'] and coverage['prediction_no_violation']})"
    )
    for d in breakdown["per_cutoff"]:
        flag = "" if d["assumption_holds"] else "  <<< BREAKDOWN (rho >= 1)"
        print(
            f"[normalized] cutoff={d['cutoff']:.3f} min_deg={d['min_expected_degree']:.2f} "
            f"rho={d['degree_rel_fluctuation']:.3f} valid={d['assumption_holds']}{flag}"
        )
    print(f"[normalized] breakdown cutoff (rho>=1): {breakdown['breakdown_cutoff']}")
    print(
        f"[normalized] LLV breakdown cutoff (dbar_min < C log n): {breakdown['llv_breakdown_cutoff']}"
    )
    print(f"[normalized] wrote {out_path}")

    _plot(breakdown, args.out_dir / "normalized_breakdown.png")
    return 0


def _plot(breakdown: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[normalized] matplotlib not installed; skipped figure")
        return

    pc = breakdown["per_cutoff"]
    cutoffs = [d["cutoff"] for d in pc]
    min_deg = [d["min_expected_degree"] for d in pc]
    rho = [d["degree_rel_fluctuation"] for d in pc]
    llv = pc[0]["llv_min_degree_threshold"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(cutoffs, min_deg, "o-", label="min expected degree")
    ax1.axhline(llv, color="k", ls="--", label=f"LLV threshold C log n = {llv:.1f}")
    ax1.set_xlabel("STRING combined-confidence cutoff")
    ax1.set_ylabel("minimum expected degree")
    ax1.set_title("Minimum expected degree vs threshold")
    ax1.legend(fontsize=8)

    ax2.plot(cutoffs, rho, "o-", color="C3", label="rho = max_i t_i/dbar_i")
    ax2.axhline(1.0, color="k", ls="--", label="rho = 1 (breakdown)")
    if breakdown["breakdown_cutoff"] is not None:
        ax2.axvline(breakdown["breakdown_cutoff"], color="gray", ls=":", label="breakdown cutoff")
    ax2.set_xlabel("STRING combined-confidence cutoff")
    ax2.set_ylabel("degree relative fluctuation rho")
    ax2.set_title("Certificate validity (rho < 1) vs threshold")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[normalized] wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
