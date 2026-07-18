"""Validate CASCADE's off-target ranking and the T4 certified bound against SeedMatchR.

The SeedMatchR ECDF shift (the one-sided KS D-statistic of seed-matched vs background
measured log2FC) is the canonical off-target signal. This experiment checks, on the real
Schlegel 2022 Ttr GSE184929 contrasts, that:

  1. CASCADE's PREDICTED off-target ranking recovers that shift. The seed input
     e (negative seed-match load) is propagated over the real TRRUST P_alpha to a
     per-gene predicted off-target E = P_alpha(S) e; we report the KS D-statistic of E
     (seed-matched vs background) next to the measured SeedMatchR KS D, and the Spearman
     concordance between the predicted E and the measured log2FC.
  2. The T4 CERTIFIED upper bound U(g) covers the predicted-vs-measured deviation. We
     report the global certified bound and the per-gene rowwise U(g), and whether the
     bound binds (and is not vacuous).

The SeedMatchR reference is read from results/seedmatch_gse184929.csv when present (the
R-package annotation written by external/seedmatchr_gse184929.R); otherwise the native
Python annotation (cel.features.seedmatch via cel.data.load_schlegel2022) is used.
Either way the source is recorded.

REAL DATA ONLY: every input comes from a real loader that RAISES when absent; no value is
fabricated. Writes results/offtarget_t4.json. No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"
SM_DIR = REPO / "data" / "raw" / "offtarget" / "seedmatchr"
CACHE = REPO / "results" / "aaai" / "cache"
ALPHA = 0.5
E_REL_ERR = 0.1
S_REL_ERR = 0.1
VACUOUS_FACTOR = 10.0


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_reference() -> tuple[list[tuple[str, pd.DataFrame]], str]:
    """Per-contrast (symbol, log2fc, seed_match) tables, from SeedMatchR or native."""
    smr = RESULTS / "seedmatch_gse184929.csv"
    if smr.exists():
        df = pd.read_csv(smr)
        sym = _pick(df, ["symbol", "gene_name", "gene", "gene_id"])
        lfc = _pick(df, ["log2FoldChange", "log2fc", "log2_fold_change"])
        sm = _pick(df, ["total_canonical", "seed_match_count", "mer7m8", "is_seed_matched"])
        if sym is None or lfc is None or sm is None:
            raise ValueError(f"{smr} missing a symbol / log2FoldChange / seed-match column")
        out = pd.DataFrame(
            {
                "symbol": df[sym].astype(str),
                "log2fc": pd.to_numeric(df[lfc], errors="coerce"),
                "seed_match": pd.to_numeric(df[sm], errors="coerce").fillna(0.0),
            }
        )
        return [
            ("SeedMatchR_GSE184929", out)
        ], "SeedMatchR R package (results/seedmatch_gse184929.csv)"

    from cel.data import SEEDMATCHR_DATASETS, load_schlegel2022

    refs: list[tuple[str, pd.DataFrame]] = []
    for name in SEEDMATCHR_DATASETS:
        ds = load_schlegel2022(name, raw_dir=SM_DIR, cache_root=CACHE)
        if not ds.has_seed_annotation:
            continue
        f = ds.frame
        refs.append(
            (
                name,
                pd.DataFrame(
                    {
                        "symbol": f["gene_name"].astype(str),
                        "log2fc": pd.to_numeric(f["log2fc"], errors="coerce"),
                        "seed_match": pd.to_numeric(f["seed_match_count"], errors="coerce").fillna(
                            0.0
                        ),
                    }
                ),
            )
        )
    if not refs:
        raise FileNotFoundError(
            "no SeedMatchR reference: results/seedmatch_gse184929.csv absent and the native "
            "annotation is missing; run scripts/annotate_seedmatch.py or the R script first"
        )
    return refs, "native cel.features.seedmatch annotation (build-matched Rnor_6.0 3'UTRs)"


def _align(grn_index: dict, df: pd.DataFrame, n: int) -> dict:
    e = np.zeros(n)
    measured = np.zeros(n)
    seed_mask = np.zeros(n, dtype=bool)
    aligned = np.zeros(n, dtype=bool)
    for sym, lfc, sm in zip(df["symbol"], df["log2fc"], df["seed_match"], strict=True):
        node = grn_index.get(str(sym).strip().upper())
        if node is None or not np.isfinite(lfc):
            continue
        aligned[node] = True
        measured[node] = float(lfc)
        if np.isfinite(sm) and sm > 0:
            e[node] = -float(sm)  # negative seed-match load is the off-target input
            seed_mask[node] = True
    return {"e": e, "measured": measured, "seed_mask": seed_mask, "aligned": aligned}


def run() -> dict:
    from scipy.stats import spearmanr

    from cel.data import load_trrust
    from cel.data.offtarget import seedmatch_ks_test
    from cel.verify import offtarget_bound as ob
    from cel.verify.stability import operator_2norm

    refs, source = load_reference()
    grn = load_trrust(species="human")
    s = grn.S
    n = grn.n_nodes
    grn_index = {sym.upper(): i for i, sym in enumerate(grn.nodes)}
    s2 = operator_2norm(s)
    p_norm = ob.p_alpha_norm(s, ALPHA)
    # Per-gene certified envelope: row norms of the dense resolvent (computed once).
    p_dense = ob.dense_p_alpha(s, ALPHA)
    row_norms = np.linalg.norm(p_dense, axis=1)
    p_global = float(np.linalg.norm(p_dense, 2))

    per_contrast: dict = {}
    for name, df in refs:
        al = _align(grn_index, df, n)
        a = al["aligned"]
        e_pred_full = ob.propagate(s, al["e"], ALPHA)
        e_pred = e_pred_full[a]
        measured = al["measured"][a]
        mask = al["seed_mask"][a]
        e_norm = float(np.linalg.norm(al["e"]))

        # SeedMatchR ECDF shift (measured) and CASCADE predicted off-target KS.
        has_split = bool(mask.any() and (~mask).any())
        ks_meas = seedmatch_ks_test(measured, mask, alternative="greater") if has_split else None
        ks_pred = seedmatch_ks_test(e_pred, mask, alternative="greater") if has_split else None
        # Rank concordance: predicted off-target E vs measured log2FC over aligned genes.
        rho = float(spearmanr(e_pred, measured).statistic) if e_pred.size > 2 else None

        # T4 certified upper bound U(g): global + per-gene rowwise.
        e_err, s_err = E_REL_ERR * e_norm, S_REL_ERR * s2
        cert = ob.certified_bound(
            s, ALPHA, e_norm, e_err, s_err, p_norm=p_norm, p_prime_norm=p_norm
        )
        u_global = cert["bound"]
        u_rowwise = (
            row_norms[a] * e_err
            + ((1.0 - ALPHA) / ALPHA) * row_norms[a] * p_global * s_err * e_norm
        )
        deviation = float(np.linalg.norm(e_pred - measured))
        meas_scale = float(np.linalg.norm(measured))
        vacuous = bool(meas_scale > 0 and u_global > VACUOUS_FACTOR * meas_scale)
        binds = bool(deviation <= u_global * (1.0 + 1e-9) and not vacuous)

        per_contrast[name] = {
            "n_aligned": int(a.sum()),
            "n_seed_matched_aligned": int(mask.sum()),
            "seedmatchr_ks_D": (ks_meas.statistic if ks_meas else None),
            "seedmatchr_ks_p": (ks_meas.pvalue if ks_meas else None),
            "cascade_predicted_ks_D": (ks_pred.statistic if ks_pred else None),
            "cascade_rank_spearman_pred_vs_measured": rho,
            "certified_U_global": u_global,
            "certified_U_rowwise_max": float(u_rowwise.max()) if u_rowwise.size else None,
            "certified_U_rowwise_median": float(np.median(u_rowwise)) if u_rowwise.size else None,
            "predicted_vs_measured_deviation": deviation,
            "bound_binds": binds,
            "bound_vacuous": vacuous,
        }

    # Summary verdict over contrasts with a seed split.
    valid = [c for c in per_contrast.values() if c["seedmatchr_ks_D"] is not None]
    binds_all = bool(valid) and all(c["bound_binds"] for c in valid)
    concordant = bool(valid) and all(
        (c["cascade_rank_spearman_pred_vs_measured"] or 0.0) > 0.0 for c in valid
    )
    return {
        "status": "ok",
        "reference_source": source,
        "grn": grn.source,
        "alpha": ALPHA,
        "p_alpha_2norm": p_norm,
        "uncertainty_budget": {"e_rel_err": E_REL_ERR, "s_rel_err": S_REL_ERR},
        "metric_note": "seedmatchr_ks_D is the measured ECDF shift (seed-matched vs background "
        "log2FC); cascade_predicted_ks_D is the same KS on the propagated off-target E; "
        "certified_U is the T4 upper bound (global and per-gene rowwise).",
        "per_contrast": per_contrast,
        "summary": {
            "cascade_ranking_concordant_with_seedmatchr": concordant,
            "t4_bound_binds_all_contrasts": binds_all,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "offtarget_t4.json")
    args = parser.parse_args()

    numbers = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    print(f"[offtarget-validate] reference: {numbers['reference_source']}")
    print(
        f"  {'contrast':<28} {'SeedMatchR_D':>12} {'CASCADE_D':>10} {'rank_rho':>9} "
        f"{'U_global':>10} {'deviation':>10} {'binds':>6}"
    )
    for name, c in numbers["per_contrast"].items():
        d = c["seedmatchr_ks_D"]
        cd = c["cascade_predicted_ks_D"]
        print(
            f"  {name:<28} {('n/a' if d is None else f'{d:>12.4f}')} "
            f"{('n/a' if cd is None else f'{cd:>10.4f}')} "
            f"{(c['cascade_rank_spearman_pred_vs_measured'] or float('nan')):>9.3f} "
            f"{c['certified_U_global']:>10.3f} {c['predicted_vs_measured_deviation']:>10.3f} "
            f"{c['bound_binds']!s:>6}"
        )
    sm = numbers["summary"]
    print(
        f"[offtarget-validate] concordant with SeedMatchR: "
        f"{sm['cascade_ranking_concordant_with_seedmatchr']} | T4 bound binds all: "
        f"{sm['t4_bound_binds_all_contrasts']}"
    )
    print(f"[offtarget-validate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
