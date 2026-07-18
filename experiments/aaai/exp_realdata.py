#!/usr/bin/env python3
"""Probabilistic off-target containment certificate on REAL off-target data.

Runs the Theorem 4 probabilistic containment certificate
(radius = ((1-alpha)/alpha) * b(delta) * ||x0||) on every dataset tagged
offtarget_certification (cel.data.datasets_with_role): GSE184929 (RNA-seq),
Jackson 2003, Jackson 2006, Birmingham 2006, and the SeedSeq union, with S from
STRING.

HARD DATA POLICY (OUTLINE.md): this experiment uses ONLY real measured data. It
contains no generated data, no signal injection, and no fabricated confidences.
Each loader RAISES FileNotFoundError when its real file is absent, and the parts
of the real join that are not yet wired RAISE NotImplementedError. No numbers are
produced without the real data.

Intended real pipeline per dataset:
1. load the measured off-target calls via the role-tagged loader (the transcripts
   measured to be down-regulated, with effect sizes);
2. load the STRING confidence GRN (load_string_confidence) over those transcripts;
3. compute the per-transcript direct seed-repression x0 from the real 3'UTR
   sequences with the SeedMatchR seed logic;
4. propagate y = P_alpha(S) x0, compute the Theorem 4 certified radius, and report
   the certified-containment fraction, how it tracks the measured calls (contained
   fraction by class and the ROC-AUC of the predicted effect vs the measured
   label), certified radius vs STRING threshold {0.4, 0.7, 0.9}, and the cross-assay
   RNA-seq-vs-microarray comparison.

Step 3 (aligning measured transcripts to STRING nodes and computing x0 from real
sequences) is a tracked data gap and currently RAISES NotImplementedError.

Output: results/aaai/realdata.json and figures under results/aaai/figures/.
Accepts --seed and logs it (OUTLINE.md rule 3). No LaTeX.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.data import (  # noqa: E402
    OFFTARGET_CERTIFICATION,
    datasets_with_role,
    load_birmingham2006,
    load_gse184929,
    load_jackson2003,
    load_jackson2006,
    load_seedseq,
)

OUT = REPO / "results" / "aaai"
THRESHOLDS = (0.4, 0.7, 0.9)

OFFTARGET_LOADERS = {
    "gse184929": load_gse184929,
    "jackson2003": load_jackson2003,
    "jackson2006": load_jackson2006,
    "birmingham2006": load_birmingham2006,
    "seedseq": load_seedseq,
}


def threshold_confidence(C: sp.csr_matrix, cutoff: float) -> sp.csr_matrix:
    """Keep only edges with confidence >= cutoff (the STRING combined-score cut)."""
    coo = C.tocoo()
    keep = coo.data >= cutoff
    return sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=C.shape).tocsr()


def load_measured_calls(name: str):
    """Load the real measured off-target calls for a dataset (raises if absent)."""
    return OFFTARGET_LOADERS[name]()


def build_real_inputs(name: str, string_links: str):
    """Assemble the real (GRN, measured calls, x0) for one off-target dataset.

    Raises FileNotFoundError if the measured calls or the STRING file are absent,
    and NotImplementedError for the transcript-to-node and sequence-to-x0 join
    that is not yet wired. No generated or fabricated values are ever returned.
    """
    from cel.data import load_string_confidence

    calls = load_measured_calls(name)  # real loader; FileNotFoundError if absent
    nodes, confidence = load_string_confidence(string_links, threshold=400)
    raise NotImplementedError(
        f"{name}: aligning the measured off-target transcripts ({len(calls)} calls) to the "
        f"STRING nodes ({confidence.shape[0]}) and computing the per-transcript seed-repression "
        "x0 from real 3'UTR sequences is a tracked data gap. The Theorem 4 certificate runs the "
        "moment that real join is wired; no values are produced from non-real inputs."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument(
        "--string-links",
        type=str,
        required=True,
        help="path to a real STRING protein.links file (required; no fabricated GRN)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[realdata] seed={args.seed} alpha={args.alpha} delta={args.delta}")

    names = datasets_with_role(OFFTARGET_CERTIFICATION)
    print(f"[realdata] off-target datasets (role offtarget_certification): {names}")
    for name in names:
        print(f"[realdata] {name}: loading real measured calls and STRING GRN")
        build_real_inputs(name, args.string_links)  # raises until the real join is wired
    return 0


if __name__ == "__main__":
    sys.exit(main())
