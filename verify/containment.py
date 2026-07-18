#!/usr/bin/env python3
"""Runnable containment verification (Contribution 2).

Loads a real STRING subgraph, picks a seed support set, and checks that the
off-target signal at hop distance r stays under the certified (1-alpha)^r
envelope. Prints a summary, reports the spectral gap and the effective-resistance
distance from the seed support (the spectral upgrade-bound ingredients), and
writes a decay-vs-envelope plot plus a CSV to ``verify/_out/``.

Exits non-zero if the certified envelope is violated, so it can gate ``make
repro``. Per OUTLINE.md rule 3 it accepts ``--seed`` and logs the seed it ran with.

Example:
    python verify/containment.py --seed 0 --alpha 0.15 --graph tests/fixtures/string_tiny.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cel.data import load_string_grn  # noqa: E402
from cel.verify.containment import plot_containment, verify_containment  # noqa: E402

OUT = REPO / "verify" / "_out"
DEFAULT_GRAPH = REPO / "tests" / "fixtures" / "string_tiny.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for seed-set choice")
    parser.add_argument("--alpha", type=float, default=0.15, help="teleport probability in (0, 1)")
    parser.add_argument("--steps", type=int, default=64, help="power-iteration steps")
    parser.add_argument("--threshold", type=int, default=700, help="STRING combined_score cutoff")
    parser.add_argument("--n-seed", type=int, default=1, help="number of seed support nodes")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="STRING links file")
    parser.add_argument("--cache", type=Path, default=None, help="cache root for the parsed graph")
    args = parser.parse_args()

    print(
        f"[containment] seed={args.seed} alpha={args.alpha} steps={args.steps} graph={args.graph}"
    )
    cache_root = args.cache if args.cache is not None else (OUT / "cache")
    grn = load_string_grn(args.graph, cache_root=cache_root, threshold=args.threshold)
    n = grn.num_nodes()
    print(f"[containment] graph: {n} nodes, {grn.num_edges()} edges")

    rng = np.random.default_rng(args.seed)
    seed_idx = rng.choice(n, size=min(args.n_seed, n), replace=False)
    print(f"[containment] seed support nodes: {seed_idx.tolist()}")

    report = verify_containment(grn.S, seed_idx, alpha=args.alpha, num_steps=args.steps)

    print(f"[containment] ||S||_2 (pre-clip) = {report.spectral_norm:.6f}")
    print(f"[containment] spectral gap        = {report.spectral_gap:.6f}")
    print(f"[containment] max eff-resistance  = {float(np.max(report.eff_resistance)):.6f}")
    print("[containment]  r   empirical    envelope")
    for r, emp, env in zip(report.hops, report.empirical, report.envelope, strict=True):
        print(f"[containment] {r:2d}   {emp:.6e}  {env:.6e}")
    print(
        f"[containment] envelope holds: {report.holds} (max violation {report.max_violation:.3e})"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "containment_decay.csv"
    np.savetxt(
        csv_path,
        np.column_stack([report.hops, report.empirical, report.envelope]),
        delimiter=",",
        header="r,empirical,envelope",
        comments="",
    )
    print(f"[containment] wrote {csv_path}")
    try:
        png = plot_containment(report, str(OUT / "containment_decay.png"))
        print(f"[containment] wrote {png}")
    except ImportError:
        print("[containment] matplotlib not installed; skipped plot")

    if not report.holds:
        print("[containment] FAIL: certified envelope violated", file=sys.stderr)
        return 1
    print("[containment] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
