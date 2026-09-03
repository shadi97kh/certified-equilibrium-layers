#!/usr/bin/env python3
"""T5 cross-species transfer: train the off-target predictor on rat, test zero-shot on human.

The predictor is trained on RAT GSE184929 (Schlegel Ttr siRNA, seed UAUAGAG) and
applied ZERO-SHOT to the human Patel 2022 datasets (GSE164289, GSE196839; seed GGGGGC).
Per gene the features are (i) the cognate seed-match abundance in the SPECIES-MATCHED
3'UTR (rat Rnor_6.0 for rat, human GRCh38/104 for human) and (ii) the magnetic-operator
propagated seed signal over the human TRRUST GRN at charge q. A logistic regression is
fit on rat (target: log2FC < 0, the off-target / downregulation direction) and the
rat-learned weights are applied to the human features.

For each q in {0, 0.1, 0.2, 0.25, 0.5} the cross-species AUROC for predicting the
direction of the human log2FC (seed-matched vs unmatched cohort) is reported. The
1-Wasserstein distance between the rat and human 3'UTR seed-match abundance
distributions is also computed.

STOP CONDITION: if the cross-species AUROC stays below 0.6, T5 is DEMOTED from a
theorem to a NUMERICAL OBSERVATION; the output records it as such and does not frame
it as a theorem. The script reports the AUROC and whether T5 survives as a theorem.

HARD DATA POLICY (OUTLINE.md): REAL DATA ONLY. Rat seed counts / log2FC come from the
annotated GSE184929 tables; human log2FC from the Patel GEO tables; human 3'UTRs from
pyensembl GRCh38/104; the GRN from real human TRRUST. Each loader RAISES when its real
file is absent. No fabricated values. Writes results/aaai/t5_crossspecies.json. No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "aaai"
PATEL = REPO / "data" / "raw" / "offtarget" / "patel"
SM_DIR = REPO / "data" / "raw" / "offtarget" / "seedmatchr"
CACHE = OUT / "cache"
ENSEMBL_RELEASE = 104  # GRCh38 = hg38, matches the confirmed human GSE build
ALPHA = 0.15  # magnetic-PPR teleport
Q_GRID = (0.0, 0.1, 0.2, 0.25, 0.5)
HUMAN_SITE = "GCCCCC"  # 3'UTR seed-match site of the human siGGGGGC (Patel)
AUROC_THRESHOLD = 0.6


def magnetic_laplacian_dense(grn, q: float):
    """Dense complex normalized magnetic signed Laplacian L_q^sigma of the GRN."""
    import torch
    from torch_geometric_signed_directed.utils import get_magnetic_signed_Laplacian

    coo = grn.signs.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float)
    li, lr, lim = get_magnetic_signed_Laplacian(
        edge_index, edge_weight, normalization="sym", q=q, num_nodes=grn.n_nodes
    )
    return sp.coo_matrix(
        (lr.numpy() + 1j * lim.numpy(), (li[0].numpy(), li[1].numpy())),
        shape=(grn.n_nodes, grn.n_nodes),
    ).toarray()


def p_mag_magnitude(L, e: np.ndarray, alpha: float) -> np.ndarray:
    """|P_mag e| with P_mag = alpha (alpha I + (1-alpha) L)^{-1} (magnetic PPR)."""
    n = L.shape[0]
    a_mat = alpha * np.eye(n) + (1.0 - alpha) * L  # Hermitian PD (L spectrum in [0,2])
    x = np.linalg.solve(a_mat, alpha * e.astype(complex))
    return np.abs(x)


def human_site_counts(gene_ids, symbols, site: str) -> tuple[dict, dict]:
    """Return (counts, ensg2sym): `site` count per gene over its best GRCh38/104 3'UTR,
    keyed by Ensembl id AND uppercase symbol, plus an Ensembl-id -> uppercase-symbol map
    (so symbol-keyed GRN nodes can be reached from ENSG tables). Cached to JSON."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"human_{site}_counts_r{ENSEMBL_RELEASE}.json"
    if cache_path.exists():
        blob = json.loads(cache_path.read_text())
        return blob["counts"], blob["ensg2sym"]
    from pyensembl import EnsemblRelease

    release = EnsemblRelease(ENSEMBL_RELEASE, species="homo_sapiens")
    counts: dict[str, int] = {}
    ensg2sym: dict[str, str] = {}

    def count_for(gene) -> int:
        best = 0
        for t in gene.transcripts:
            u = t.three_prime_utr_sequence
            if u:
                best = max(best, u.upper().count(site))
        return best

    for gid in gene_ids:
        if gid in counts:
            continue
        try:
            gene = release.gene_by_id(gid)
        except Exception:
            continue
        counts[gid] = count_for(gene)
        if gene.gene_name:
            ensg2sym[gid] = gene.gene_name.upper()
            counts.setdefault(gene.gene_name.upper(), counts[gid])
    for sym in symbols:
        key = sym.upper()
        if key in counts:
            continue
        try:
            genes = release.genes_by_name(sym)
        except Exception:
            continue
        counts[key] = max((count_for(g) for g in genes), default=0) if genes else 0
    cache_path.write_text(json.dumps({"counts": counts, "ensg2sym": ensg2sym}))
    return counts, ensg2sym


def mag_feature(
    grn_index: dict, keys: list[str], seed_counts: np.ndarray, L, alpha: float
) -> np.ndarray:
    """Per-gene magnetic-propagated score: |P_mag e| at the gene's GRN node (0 if absent)."""
    n = len(grn_index)
    e = np.zeros(n)
    for k, c in zip(keys, seed_counts, strict=True):
        node = grn_index.get(k)
        if node is not None:
            e[node] += float(c)
    prop = p_mag_magnitude(L, e, alpha)
    return np.array([prop[grn_index[k]] if k in grn_index else 0.0 for k in keys])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[t5] seed={args.seed} q_grid={Q_GRID} alpha={ALPHA}")

    rat_csv = SM_DIR / "Schlegel_2022_Ttr_D1_30mkg.csv"
    h164 = PATEL / "GSE164289_siGGGGGC_vs_siNT1_48h.csv"
    h196 = PATEL / "GSE196839_counts.txt"
    for p in (rat_csv, h164, h196):
        if not p.exists():
            out = {"status": "pending_real_data", "source": f"missing {p}"}
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "t5_crossspecies.json").write_text(json.dumps(out, indent=2))
            print(f"[t5] PENDING: missing {p}")
            return 0
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        from cel.data import load_trrust

        grn = load_trrust(species="human")
    except (FileNotFoundError, ImportError) as exc:
        out = {"status": "pending_real_data", "source": str(exc)}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "t5_crossspecies.json").write_text(json.dumps(out, indent=2))
        print(f"[t5] PENDING: {exc}")
        return 0

    grn_index = {sym.upper(): i for i, sym in enumerate(grn.nodes)}

    # Rat training data (GSE184929 D1_30mkg): seed_match_count + log2FC, by uppercase symbol.
    rat = pd.read_csv(rat_csv, sep="\t")
    rat = rat[rat["log2FoldChange"].notna() & rat["seed_match_count"].notna()].copy()
    rat_keys = rat["symbol"].astype(str).str.upper().tolist()
    rat_seed = rat["seed_match_count"].to_numpy(dtype=float)
    rat_down = (rat["log2FoldChange"].to_numpy() < 0).astype(int)

    # Human seed-match abundance (GCCCCC counts) over GRCh38/104 3'UTRs.
    g164 = pd.read_csv(h164)
    g164_ens = g164["Ensembl ID"].astype(str).str.split(".").str[0].tolist()
    g196 = pd.read_csv(h196, sep="\t", index_col=0)
    print("[t5] resolving human GCCCCC 3'UTR counts (cached) ...")
    hsite, ensg2sym = human_site_counts(set(g164_ens), {str(s) for s in g196.index}, HUMAN_SITE)

    def human_block(name, count_keys, grn_keys, lfc):
        """count_keys: gene id used to look up the seed count; grn_keys: uppercase symbol
        used for the GRN node; aligned, one per gene."""
        seed = np.array([hsite.get(k, 0) for k in count_keys], dtype=float)
        has_utr = np.array([k in hsite for k in count_keys])
        finite = np.isfinite(lfc)
        keep = has_utr & finite
        return {
            "name": name,
            "keys": [gk for gk, m in zip(grn_keys, keep, strict=True) if m],
            "seed": seed[keep],
            "down": (lfc[keep] < 0).astype(int),
        }

    humans = [
        human_block(
            "GSE164289_siGGGGGC",
            g164_ens,  # count lookup by Ensembl id
            [ensg2sym.get(e, e) for e in g164_ens],  # GRN lookup by symbol (fallback ENSG)
            pd.to_numeric(g164["log2FoldChange"], errors="coerce").to_numpy(),
        ),
    ]
    # GSE196839 count-derived log2FC (PC3 siGGGGGC vs siNT1, single replicate), by symbol.
    sample_cols = [c for c in g196.columns if c.startswith(("PC3_", "HeyA8_"))]
    cpm = g196[sample_cols].apply(lambda s: s / s.sum() * 1e6, axis=0)
    lfc196 = np.log2(
        (cpm["PC3_siMxy_3AB_48h"].to_numpy() + 1.0) / (cpm["PC3_siNT_3AB_48h"].to_numpy() + 1.0)
    )
    sym196 = [str(s).upper() for s in g196.index]
    humans.append(human_block("GSE196839_siGGGGGC", sym196, sym196, lfc196))
    auroc = {}
    for q in Q_GRID:
        L = magnetic_laplacian_dense(grn, q)
        rat_mag = mag_feature(grn_index, rat_keys, rat_seed, L, ALPHA)
        x_rat = np.column_stack([rat_seed, rat_mag])
        scaler = StandardScaler().fit(x_rat)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(scaler.transform(x_rat), rat_down)
        per_ds = {}
        for hb in humans:
            h_mag = mag_feature(grn_index, hb["keys"], hb["seed"], L, ALPHA)
            x_h = np.column_stack([hb["seed"], h_mag])
            p_down = clf.predict_proba(scaler.transform(x_h))[:, 1]
            y = hb["down"]
            au = float(roc_auc_score(y, p_down)) if 0 < y.sum() < len(y) else None
            per_ds[hb["name"]] = {
                "auroc": au,
                "n_genes": int(len(y)),
                "n_down": int(y.sum()),
                "n_seed_matched": int((hb["seed"] > 0).sum()),
            }
        auroc[f"{q}"] = per_ds

    # 1-Wasserstein between rat and human seed-match ABUNDANCE distributions.
    from scipy.stats import wasserstein_distance

    human_abundance = np.concatenate([hb["seed"] for hb in humans])
    w1 = float(wasserstein_distance(rat_seed, human_abundance))

    best_auroc = max(
        (d["auroc"] for q in auroc.values() for d in q.values() if d["auroc"] is not None),
        default=None,
    )
    survives = bool(best_auroc is not None and best_auroc >= AUROC_THRESHOLD)
    status_label = "theorem" if survives else "numerical_observation"

    numbers = {
        "status": "ok",
        "seed": args.seed,
        "design": "train off-target predictor on rat GSE184929 (seed UAUAGAG), zero-shot test on "
        "human Patel GSE164289 / GSE196839 (seed GGGGGC); features = [species-matched seed-match "
        "abundance, magnetic-PPR propagated seed signal over human TRRUST at charge q]; logistic "
        "regression, target log2FC < 0; AUROC for human log2FC direction.",
        "magnetic_operator": f"P_mag(q) = alpha (alpha I + (1-alpha) L_q^sigma)^(-1) on human TRRUST "
        f"({grn.n_nodes} nodes), alpha={ALPHA}; feature = |P_mag(q) e| at the gene's node",
        "species_matched_3utr": "rat seed counts vs rat Rnor_6.0 3'UTRs; human seed counts vs "
        f"GRCh38/{ENSEMBL_RELEASE} 3'UTRs (pyensembl)",
        "auroc_by_q": auroc,
        "wasserstein1_rat_vs_human_seedmatch_abundance": w1,
        "wasserstein_note": "rat abundance = UAUAGAG (7mer) site counts; human = GCCCCC (6mer) "
        "site counts; the gap includes the seed-length difference, not only the species 3'UTRs.",
        "stop_condition": {
            "auroc_threshold": AUROC_THRESHOLD,
            "best_cross_species_auroc": best_auroc,
            "t5_survives_as_theorem": survives,
            "t5_status": status_label,
            "ruling": (
                "Cross-species AUROC stays below 0.6: T5 is DEMOTED from a theorem to a NUMERICAL "
                "OBSERVATION; it must NOT be framed as a theorem."
                if not survives
                else "Cross-species AUROC reaches 0.6 or above: T5 survives as a theorem."
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "t5_crossspecies.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    print("\n[t5] cross-species AUROC (rat -> human zero-shot), by q:")
    names = [hb["name"] for hb in humans]
    print(f"  {'q':>5} " + " ".join(f"{n:>22}" for n in names))
    for q in Q_GRID:
        cells = " ".join(
            f"{(auroc[f'{q}'][n]['auroc'] if auroc[f'{q}'][n]['auroc'] is not None else float('nan')):>22.4f}"
            for n in names
        )
        print(f"  {q:>5} {cells}")
    print(f"[t5] 1-Wasserstein(rat UAUAGAG counts, human GCCCCC counts) = {w1:.4f}")
    sc = numbers["stop_condition"]
    print(
        f"[t5] best cross-species AUROC = {best_auroc:.4f} (threshold {AUROC_THRESHOLD}) -> "
        f"T5 survives as theorem: {sc['t5_survives_as_theorem']}"
    )
    print(f"[t5] T5 STATUS: {sc['t5_status'].upper()}")
    print(f"[t5] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
