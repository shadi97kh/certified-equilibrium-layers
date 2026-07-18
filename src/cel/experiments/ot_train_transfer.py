"""Train the CASCADE-OT predictor on the McGeary corpus; validate by transfer to siRNA studies.

Trains the leakage-free per-concept off-target head (cel.models.offtarget_predictor) ONLY
on the McGeary (seed, 3'UTR, repression) corpus, asserts machine-zero cross-channel leakage,
then predicts on each held-out real siRNA off-target condition (never seen in training) and
reports per-gene Spearman against the measured DESeq2 log2FC, the predicted-vs-measured
seed-vs-background KS direction, and a comparison against the mechanistic seed-match-count
(SeedMatchR / scanMiR) baseline. The analytical P_alpha GRN propagation is reused (not trained)
as a secondary graph-aware metric.

Honest scope: RNA-FM frozen features are off (the 'fm' package is absent); rat 3'UTRs are not
installed, so the Schlegel (rat) condition uses the loader's real seed_match_count with the
sequence-derived concept blocks zeroed (recorded). Within a single condition the per-gene
ranking is driven by site count, so the learned head and the count baseline are expected to be
close per condition; this is reported faithfully.

REAL DATA ONLY: McGeary corpus + real Patel CSVs + real Schlegel loader; nothing fabricated.
Writes results/cascade_ot/predictor_transfer.json. No LaTeX.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results" / "cascade_ot"
CORPUS_CACHE = REPO / "data" / "raw" / "offtarget" / "mirna_aux" / "cascade_ot_examples.csv.gz"
PATEL_DIR = REPO / "data" / "raw" / "offtarget" / "patel"
ALPHA = 0.5
ENSEMBL_RELEASE = 104
SITE_GRADE_6MER = 0.5

# Held-out siRNA conditions: seed (RNA) and the 3'UTR DNA site word scanned (Patel reagents).
PATEL_CONDITIONS = {
    "GSE164289:siGGGGGC": {
        "seed": "GGGGGC",
        "site_word": "GCCCCC",
        "file": "GSE164289_siGGGGGC_vs_siNT1_48h.csv",
        "kind": "deseq",
    },
    "GSE164289:siGGCAGU": {
        "seed": "GGCAGU",
        "site_word": "ACTGCC",
        "file": "GSE164289_siGGCAGU_vs_siNT1_48h.csv",
        "kind": "deseq",
    },
    "GSE196839:siGGGGGC": {
        "seed": "GGGGGC",
        "site_word": "GCCCCC",
        "file": "GSE196839_counts.txt",
        "kind": "counts",
    },
}
SCHLEGEL_SEED = "UAUAGAG"  # Ttr D1 guide seed (positions 2-8)


def _seed_dg(seed_rna: str) -> float:
    import RNA

    comp = {"A": "U", "U": "A", "G": "C", "C": "G", "T": "A"}
    rc = "".join(comp[b] for b in reversed(seed_rna.upper().replace("T", "U")))
    return float(RNA.duplexfold(seed_rna.upper().replace("T", "U"), rc).energy)


# --------------------------------------------------------------------------- training
def load_corpus() -> pd.DataFrame:
    if not CORPUS_CACHE.exists():
        raise FileNotFoundError(f"corpus cache absent at {CORPUS_CACHE}; run ot_build_corpus first")
    with gzip.open(CORPUS_CACHE, "rt") as fh:
        return pd.read_csv(fh)


def _grade(best_site_type: str) -> float:
    from cel.features.seedmatch import SITE_GRADE

    return float(SITE_GRADE.get(best_site_type, 0.0))


def _standardize(blocks: dict, scaler: dict | None) -> tuple[dict, dict]:
    import torch

    out, fitted = {}, {}
    for name, arr in blocks.items():
        if scaler is None:
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            fitted[name] = {"mu": mu.tolist(), "sd": sd.tolist()}
        else:
            mu = np.array(scaler[name]["mu"])
            sd = np.array(scaler[name]["sd"])
        out[name] = torch.tensor((arr - mu) / sd, dtype=torch.float32)
    return out, (fitted if scaler is None else scaler)


def train_predictor(corpus: pd.DataFrame, n_steps: int = 600, seed: int = 0) -> tuple:
    import torch

    from cel.models.offtarget_predictor import (
        OffTargetPredictor,
        assert_leakage_free,
        build_feature_blocks,
    )

    torch.manual_seed(seed)
    df = corpus[corpus["has_3utr"]].copy()
    # Repression target: remove each gene's mean log-TPM across miRNAs; orient so + = repressed.
    gene_mean = df.groupby("refseq")["log_tpm_repression"].transform("mean")
    target = -(df["log_tpm_repression"].to_numpy() - gene_mean.to_numpy())
    grade = df["best_site_type"].map(_grade).to_numpy()
    blocks_np = build_feature_blocks(
        site_count=df["site_count"].to_numpy(),
        seed_dg=df["seed_dg"].to_numpy(),
        site_grade=grade,
        utr_len=df["utr_len"].to_numpy(),
    )
    blocks, scaler = _standardize(blocks_np, None)
    y = torch.tensor((target - target.mean()) / (target.std() + 1e-8), dtype=torch.float32)
    is_site = torch.tensor((df["site_count"].to_numpy() > 0), dtype=torch.bool)

    model = OffTargetPredictor()
    leak = assert_leakage_free(model, blocks)  # fails the run if leakage is detected
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.05)
    for _ in range(n_steps):
        opt.zero_grad()
        pred = model(blocks)
        huber = torch.nn.functional.huber_loss(pred, y, delta=1.0)
        # Differentiable one-sided ECDF-shift (energy) surrogate: push seed scores above background.
        ecdf = (
            -(pred[is_site].mean() - pred[~is_site].mean())
            if is_site.any() and (~is_site).any()
            else pred.new_zeros(())
        )
        (huber + 0.1 * ecdf).backward()
        opt.step()
    from scipy.stats import spearmanr

    with torch.no_grad():
        train_rho = float(spearmanr(model(blocks).numpy(), y.numpy()).statistic)
    return (
        model,
        scaler,
        {"leakage_offdiag_max": leak, "train_spearman": train_rho, "n_examples": int(len(df))},
    )


def predict(model, scaler, site_count, seed_dg, site_grade, utr_len) -> np.ndarray:
    import torch

    from cel.models.offtarget_predictor import build_feature_blocks

    blocks_np = build_feature_blocks(
        site_count, np.full(len(site_count), seed_dg), site_grade, utr_len
    )
    blocks, _ = _standardize(blocks_np, scaler)
    with torch.no_grad():
        return model(blocks).numpy()


# --------------------------------------------------------------------------- held-out conditions
def _human_utr_cache(release, gene_keys: list[str], by_symbol: bool) -> dict:
    cache: dict = {}
    for key in gene_keys:
        if key in cache:
            continue
        best = ""
        try:
            genes = release.genes_by_name(key) if by_symbol else [release.gene_by_id(key)]
        except Exception:
            genes = []
        for gene in genes:
            for t in gene.transcripts:
                u = t.three_prime_utr_sequence
                if u and len(u) > len(best):
                    best = u.upper()
        cache[key] = best
    return cache


def patel_condition(name: str, spec: dict, release) -> pd.DataFrame:
    path = PATEL_DIR / spec["file"]
    if spec["kind"] == "deseq":
        df = pd.read_csv(path)
        gid = df.columns[0]
        out = pd.DataFrame(
            {
                "gene": df[gid].astype(str).str.split(".").str[0],
                "log2fc": pd.to_numeric(df["log2FoldChange"], errors="coerce"),
            }
        )
        by_symbol = False
    else:
        df = pd.read_csv(path, sep="\t", index_col=0)
        cpm = df[["PC3_siMxy_3AB_48h", "PC3_siNT_3AB_48h"]].apply(lambda c: c / c.sum() * 1e6)
        out = pd.DataFrame(
            {
                "gene": df.index.astype(str),
                "log2fc": np.log2(
                    (cpm["PC3_siMxy_3AB_48h"] + 1.0) / (cpm["PC3_siNT_3AB_48h"] + 1.0)
                ).to_numpy(),
            }
        )
        by_symbol = True
    out = out[np.isfinite(out["log2fc"])].reset_index(drop=True)
    utr = _human_utr_cache(release, out["gene"].tolist(), by_symbol)
    word = spec["site_word"]
    out["site_count"] = [_count_word(utr.get(g, ""), word) for g in out["gene"]]
    out["utr_len"] = [len(utr.get(g, "")) for g in out["gene"]]
    out["symbol"] = [(g if by_symbol else _id_symbol(release, g)) for g in out["gene"]]
    return out


def _count_word(seq: str, word: str) -> int:
    if not seq:
        return 0
    n, i = 0, seq.find(word)
    while i != -1:
        n += 1
        i = seq.find(word, i + 1)
    return n


def _id_symbol(release, gid: str) -> str | None:
    try:
        return release.gene_by_id(gid).gene_name
    except Exception:
        return None


def schlegel_condition(name: str):
    from cel.data import load_schlegel2022

    ds = load_schlegel2022(name, raw_dir=REPO / "data" / "raw" / "offtarget" / "seedmatchr")
    f = ds.frame
    out = pd.DataFrame(
        {
            "gene": f["gene_name"].astype(str),
            "symbol": f["gene_name"].astype(str),
            "log2fc": pd.to_numeric(f["log2fc"], errors="coerce"),
            "site_count": pd.to_numeric(f["seed_match_count"], errors="coerce").fillna(0.0),
        }
    )
    out["utr_len"] = 0.0  # rat 3'UTR sequences not installed: accessibility block zeroed
    return out[np.isfinite(out["log2fc"])].reset_index(drop=True)


# --------------------------------------------------------------------------- evaluation
def _ks_dir(values, is_site):
    from cel.data.offtarget import seedmatch_ks_test

    if not (is_site.any() and (~is_site).any()):
        return None, None
    ks = seedmatch_ks_test(np.asarray(values), np.asarray(is_site), alternative="greater")
    gap = float(np.asarray(values)[is_site].mean() - np.asarray(values)[~is_site].mean())
    return float(ks.statistic), gap


def evaluate(
    name: str,
    df: pd.DataFrame,
    model,
    scaler,
    seed_rna: str,
    has_grade: bool,
    has_utr: bool,
    grn=None,
):
    from scipy.stats import spearmanr

    dg = _seed_dg(seed_rna)
    site_count = df["site_count"].to_numpy(dtype=float)
    is_site = site_count > 0
    grade = np.where(is_site, SITE_GRADE_6MER, 0.0) if has_grade else np.zeros(len(df))
    utr_len = df["utr_len"].to_numpy(dtype=float) if has_utr else np.zeros(len(df))
    score = predict(model, scaler, site_count, dg, grade, utr_len)
    log2fc = df["log2fc"].to_numpy(dtype=float)

    # (a) per-gene Spearman: predicted off-target score vs measured repression (-log2FC).
    # All-gene Spearman is dominated by the majority of zero-site genes (tied low scores), so we
    # also report the more discriminating Spearman among the seed-matched genes only.
    rho_learned = float(spearmanr(score, -log2fc).statistic)
    rho_baseline = float(spearmanr(site_count, -log2fc).statistic)  # mechanistic SeedMatchR/scanMiR
    rho_learned_seedonly = (
        float(spearmanr(score[is_site], -log2fc[is_site]).statistic) if is_site.sum() > 2 else None
    )

    # (b) seed-vs-background KS direction: is the seed-matched cohort more repressed in the
    # measurement, and does the predictor put the same cohort at higher predicted off-target?
    meas_D, _ = _ks_dir(log2fc, is_site)
    pred_D, _ = _ks_dir(-score, is_site)
    meas_seed_more_repressed = (
        bool(log2fc[is_site].mean() < log2fc[~is_site].mean())
        if is_site.any() and (~is_site).any()
        else None
    )
    pred_seed_more_offtarget = (
        bool(score[is_site].mean() > score[~is_site].mean())
        if is_site.any() and (~is_site).any()
        else None
    )
    direction_match = (
        meas_seed_more_repressed is not None
        and meas_seed_more_repressed == pred_seed_more_offtarget
    )

    result = {
        "n_genes": int(len(df)),
        "n_seed_matched": int(is_site.sum()),
        "spearman_learned": rho_learned,
        "spearman_learned_seed_matched_only": rho_learned_seedonly,
        "spearman_baseline_seedcount": rho_baseline,
        "measured_ks_D": meas_D,
        "predicted_ks_D": pred_D,
        "predicted_ks_note": "trivially high: the predictor keys on seed-site presence",
        "measured_seed_cohort_more_repressed": meas_seed_more_repressed,
        "predicted_seed_cohort_more_offtarget": pred_seed_more_offtarget,
        "ks_direction_reproduced": bool(direction_match),
    }

    # P_alpha graph-aware secondary metric on the GRN-aligned subset.
    if grn is not None and is_site.any():
        from cel.verify import offtarget_bound as ob

        idx = {s.upper(): i for i, s in enumerate(grn.nodes)}
        e = np.zeros(grn.n_nodes)
        aligned = np.zeros(len(df), dtype=bool)
        node_of = []
        for k, sym in enumerate(df["symbol"]):
            node = idx.get(str(sym).strip().upper()) if sym else None
            node_of.append(node)
            if node is not None:
                aligned[k] = True
                e[node] = max(e[node], score[k])
        propagated = ob.propagate(grn.S, e, ALPHA)
        prop_scores = np.array([propagated[n] if n is not None else 0.0 for n in node_of])
        if aligned.sum() > 2:
            result["spearman_propagated_aligned"] = float(
                spearmanr(prop_scores[aligned], -log2fc[aligned]).statistic
            )
            result["n_aligned_grn"] = int(aligned.sum())
    return result


def run() -> dict:
    from pyensembl import EnsemblRelease

    from cel.data import load_trrust

    corpus = load_corpus()
    model, scaler, train_info = train_predictor(corpus)

    release = EnsemblRelease(ENSEMBL_RELEASE, species="human")
    human_grn = load_trrust(species="human")
    mouse_grn = load_trrust(species="mouse")

    conditions: dict = {}
    for name, spec in PATEL_CONDITIONS.items():
        df = patel_condition(name, spec, release)
        conditions[name] = {
            "study": "Patel_2022",
            "organism": "human",
            **evaluate(
                name, df, model, scaler, spec["seed"], has_grade=True, has_utr=True, grn=human_grn
            ),
        }
    for sname in (
        "Schlegel_2022_Ttr_D1_30mkg",
        "Schlegel_2022_Ttr_D4_30mkg",
        "Schlegel_2022_Ttr_D1_10mkg",
    ):
        df = schlegel_condition(sname)
        conditions[f"GSE184929:{sname}"] = {
            "study": "Schlegel_2022",
            "organism": "rat",
            "note": "rat 3'UTRs not installed: site_grade and accessibility blocks zeroed; "
            "uses the loader real seed_match_count",
            **evaluate(
                sname,
                df,
                model,
                scaler,
                SCHLEGEL_SEED,
                has_grade=False,
                has_utr=False,
                grn=mouse_grn,
            ),
        }

    learned = [c["spearman_learned"] for c in conditions.values()]
    base = [c["spearman_baseline_seedcount"] for c in conditions.values()]
    return {
        "status": "ok",
        "training": {
            "corpus": "McGeary 2019 GSE140217 miRNA auxiliary (train only)",
            "rnafm_frozen_features": False,
            "rnafm_note": "the 'fm' package is absent; RNA-FM channel disabled (no fabrication)",
            **train_info,
        },
        "transfer_policy": "predictor trained ONLY on McGeary; every condition below is held-out",
        "conditions": conditions,
        "summary": {
            "mean_spearman_learned": float(np.mean(learned)),
            "mean_spearman_baseline_seedcount": float(np.mean(base)),
            "learned_beats_baseline_mean": bool(np.mean(learned) > np.mean(base)),
            "ks_direction_reproduced_all": bool(
                all(c["ks_direction_reproduced"] for c in conditions.values())
            ),
        },
    }


def main() -> int:
    numbers = run()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "predictor_transfer.json").write_text(
        json.dumps(numbers, indent=2), encoding="utf-8"
    )

    t = numbers["training"]
    print(
        f"[ot-transfer] trained on {t['n_examples']} McGeary examples | leakage off-diag "
        f"{t['leakage_offdiag_max']:.1e} | train Spearman {t['train_spearman']:.3f}"
    )
    print(
        f"  {'condition':<34}{'n':>7}{'learned_rho':>12}{'base_rho':>10}{'meas_KS':>9}{'pred_KS':>9}{'dir':>5}"
    )
    for name, c in numbers["conditions"].items():
        print(
            f"  {name:<34}{c['n_genes']:>7}{c['spearman_learned']:>12.3f}{c['spearman_baseline_seedcount']:>10.3f}"
            f"{(c['measured_ks_D'] or float('nan')):>9.3f}{(c['predicted_ks_D'] or float('nan')):>9.3f}"
            f"{c['ks_direction_reproduced']!s:>5}"
        )
    s = numbers["summary"]
    print(
        f"[ot-transfer] mean Spearman learned {s['mean_spearman_learned']:.3f} vs baseline "
        f"{s['mean_spearman_baseline_seedcount']:.3f} | KS direction reproduced all: {s['ks_direction_reproduced_all']}"
    )
    print(f"[ot-transfer] wrote {RESULTS / 'predictor_transfer.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
