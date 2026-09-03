"""Stage E/F: retrain the off-target predictor with the modification-aware occupancy feature
and the MonotoneEquilibriumPropagation layer, then re-run the pre-registered criterion-1 gate.

The predictor: per gene, the modification-aware occupancy feature occ = -min(seed-duplex dG, 0)
(so it distinguishes the (S)-GNA D4 from the parent D1) and the seed-match count feed a small
learned binding head -> a non-negative per-gene load; the load is propagated through the
differentiable MonotoneEquilibriumPropagation layer (learned alpha and edge scale, IFT-exact) to
a predicted transcriptome-wide off-target E. The head + propagation are PRE-TRAINED on the
data-rich McGeary GSE140217 AGO-RBNS / transfection corpus (per-gene repression), then FROZEN and
evaluated zero-shot on the held-out real siRNA off-target studies.

The gate is PRE-REGISTERED in the JSON BEFORE evaluation (the thresholds dict is fixed):
  (a) held-out per-gene Spearman beats the SeedMatchR + seed-duplex-Tm baseline (occupancy * count)
      by >= 0.05 mean (pooled across all held-out conditions), AND
  (b) the margin is positive on BOTH the Schlegel and the Patel held-out studies separately, AND
  (c) the D4 < D1 discrimination still holds in the retrained model (E(D4) < E(D1)).

REAL DATA ONLY: real McGeary corpus + real ViennaRNA dG + real TRRUST + the real siRNA studies;
nothing fabricated. Writes results/cascade_ot/criterion1_retrain.json. No LaTeX.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "cascade_ot" / "criterion1_retrain.json"
ENSEMBL_RELEASE = 104
N_STEPS = 600
SEED = 0

# Pre-registered gate thresholds (fixed before any evaluation).
PRE_REGISTERED = {
    "criterion_a_min_mean_margin": 0.05,
    "criterion_b": "the learned-minus-baseline margin is positive on BOTH the Schlegel and Patel "
    "held-out studies separately (not just pooled)",
    "criterion_c": "the D4 < D1 discrimination still holds in the retrained model (E(D4) < E(D1))",
    "baseline": "SeedMatchR + seed-duplex-Tm = occupancy(seed dG) * seed-match count (no learning, "
    "no propagation)",
    "decision_rule": "PASS only if (a) AND (b) AND (c); otherwise FAIL",
}

# D4 (S)-GNA model (matches subgate_d4d1): g7 A->C mismatch standing in for the GNA sugar.
GNA_SEED_IDX = 5
GNA_MISMATCH_BASE = "C"


def _occ(dg: float) -> float:
    return max(-dg, 0.0)


def _features(occ: np.ndarray, count: np.ndarray) -> np.ndarray:
    occ = np.asarray(occ, dtype=float)
    count = np.asarray(count, dtype=float)
    return np.stack([occ, np.log1p(count), occ * count], axis=1)


class _BindingHead:
    """Small learned binding/occupancy head: features -> non-negative per-gene load."""

    def __init__(self, torch):
        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1), torch.nn.Softplus()
        )

    def __call__(self, x):
        return self.net(x).squeeze(-1)


def _align(symbols, grn):
    idx = {s.upper(): i for i, s in enumerate(grn.nodes)}
    nodes, keep = [], []
    for k, s in enumerate(symbols):
        node = idx.get(str(s).strip().upper()) if s else None
        if node is not None:
            nodes.append(node)
            keep.append(k)
    return np.asarray(nodes, dtype=int), np.asarray(keep, dtype=int)


def _grn_tensor(grn, torch, device):
    return torch.tensor(grn.S.toarray(), dtype=torch.float32, device=device)


def train(device: str = "cpu") -> dict:
    import torch

    from cel.data import load_trrust
    from cel.experiments.ot_train_transfer import load_corpus
    from cel.models.prop_layer import MonotoneEquilibriumPropagation

    torch.manual_seed(SEED)
    corpus = load_corpus()
    corpus = corpus[corpus["has_3utr"]].copy()
    gene_mean = corpus.groupby("refseq")["log_tpm_repression"].transform("mean")
    target = -(corpus["log_tpm_repression"].to_numpy() - gene_mean.to_numpy())  # + = repressed
    occ = corpus["seed_dg"].map(_occ).to_numpy()
    feats = _features(occ, corpus["site_count"].to_numpy())
    mu, sd = feats.mean(0), feats.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    scaler = {"mu": mu.tolist(), "sd": sd.tolist()}

    grn = load_trrust(species="human")
    nodes, keep = _align(corpus["symbol"].tolist(), grn)
    mirna_codes = corpus["mirna"].astype("category").cat.codes.to_numpy()[keep]
    n_mir = int(mirna_codes.max()) + 1
    nodes_t = torch.tensor(nodes, device=device, dtype=torch.long)
    mir_t = torch.tensor(mirna_codes, device=device, dtype=torch.long)
    x = torch.tensor(((feats - mu) / sd)[keep], dtype=torch.float32, device=device)
    y = torch.tensor(target[keep], dtype=torch.float32, device=device)
    s = _grn_tensor(grn, torch, device)
    n = grn.n_nodes

    head = _BindingHead(torch)
    head.net.to(device)
    prop = MonotoneEquilibriumPropagation(s.double(), alpha_init=0.5).to(device)
    prop.S = prop.S.float()
    opt = torch.optim.Adam(list(head.net.parameters()) + list(prop.parameters()), lr=0.02)
    huber = torch.nn.functional.huber_loss

    for _ in range(N_STEPS):
        opt.zero_grad()
        load = torch.zeros(n, n_mir, device=device)
        load.index_put_((nodes_t, mir_t), head(x), accumulate=True)
        e_pred = prop(load)  # (n, n_mir)
        pred = e_pred[nodes_t, mir_t]
        loss = huber(pred, y)
        loss.backward()
        opt.step()

    return {
        "head": head,
        "prop": prop,
        "scaler": scaler,
        "alpha": float(prop.alpha()),
        "w": float(prop.w()),
        "n_train_cells": int(len(keep)),
        "n_mirnas": n_mir,
    }


def _predict_E(head, prop, scaler, occ_value, counts, symbols, grn, device):
    import torch

    occ = np.full(len(counts), occ_value, dtype=float)
    feats = (_features(occ, counts) - np.array(scaler["mu"])) / np.array(scaler["sd"])
    nodes, keep = _align(symbols, grn)
    s = _grn_tensor(grn, torch, device)
    prop_local = type(prop)(s.double(), alpha_init=0.5).to(device)
    prop_local.alpha_raw.data = prop.alpha_raw.data.clone()
    prop_local.w_raw.data = prop.w_raw.data.clone()
    prop_local.S = prop_local.S.float()
    with torch.no_grad():
        x = torch.tensor(feats[keep], dtype=torch.float32, device=device)
        load = torch.zeros(grn.n_nodes, device=device)
        load.index_put_((torch.tensor(nodes, device=device),), head(x), accumulate=True)
        e_full = prop_local(load).cpu().numpy()
    pred = np.zeros(len(counts))
    pred[keep] = e_full[nodes]
    return pred, keep


def run(device: str = "cpu") -> dict:
    from scipy.stats import spearmanr

    from cel.data import load_trrust
    from cel.experiments.ot_train_transfer import (
        PATEL_CONDITIONS,
        SCHLEGEL_SEED,
        _seed_dg,
        patel_condition,
        schlegel_condition,
    )

    trained = train(device)
    head, prop, scaler = trained["head"], trained["prop"], trained["scaler"]

    from pyensembl import EnsemblRelease

    release = EnsemblRelease(ENSEMBL_RELEASE, species="human")
    human_grn = load_trrust(species="human")
    mouse_grn = load_trrust(species="mouse")

    # (S)-GNA modified-seed occupancy for D1 vs D4 (real ViennaRNA mismatch model).
    site = "".join({"A": "U", "U": "A", "G": "C", "C": "G"}[b] for b in reversed(SCHLEGEL_SEED))
    import RNA

    occ_d1 = _occ(float(RNA.duplexfold(SCHLEGEL_SEED, site).energy))
    d4_seed = SCHLEGEL_SEED[:GNA_SEED_IDX] + GNA_MISMATCH_BASE + SCHLEGEL_SEED[GNA_SEED_IDX + 1 :]
    occ_d4 = _occ(float(RNA.duplexfold(d4_seed, site).energy))

    conditions: dict = {}

    def _eval(name, df, occ_value, grn, study):
        symbols = df["symbol"].tolist() if "symbol" in df else df["gene"].tolist()
        counts = df["site_count"].to_numpy(dtype=float)
        log2fc = df["log2fc"].to_numpy(dtype=float)
        pred, _keep = _predict_E(head, prop, scaler, occ_value, counts, symbols, grn, device)
        baseline = occ_value * counts  # SeedMatchR + seed-Tm: occupancy * count
        rho_learned = float(spearmanr(pred, -log2fc).statistic)
        rho_baseline = float(spearmanr(baseline, -log2fc).statistic)
        conditions[name] = {
            "study": study,
            "n_genes": int(len(df)),
            "spearman_learned": rho_learned,
            "spearman_baseline": rho_baseline,
            "margin": rho_learned - rho_baseline,
            "mean_pred": float(np.mean(pred)),
        }

    for cname, spec in PATEL_CONDITIONS.items():
        df = patel_condition(cname, spec, release)
        occ_value = _occ(_seed_dg(spec["seed"]))
        _eval(cname, df, occ_value, human_grn, "Patel")
    schlegel_occ = {
        "Schlegel_2022_Ttr_D1_30mkg": occ_d1,
        "Schlegel_2022_Ttr_D4_30mkg": occ_d4,
        "Schlegel_2022_Ttr_D1_10mkg": occ_d1,
    }
    for sname, occ_value in schlegel_occ.items():
        df = schlegel_condition(sname)
        _eval(f"GSE184929:{sname}", df, occ_value, mouse_grn, "Schlegel")

    # Aggregate margins per study + pooled.
    by_study: dict[str, list] = {"Schlegel": [], "Patel": []}
    for c in conditions.values():
        by_study[c["study"]].append(c["margin"])
    schlegel_margin = float(np.mean(by_study["Schlegel"]))
    patel_margin = float(np.mean(by_study["Patel"]))
    pooled_margin = float(np.mean([c["margin"] for c in conditions.values()]))

    # D4 < D1 discrimination (mean predicted off-target on the shared D1 gene set).
    e_d1 = conditions["GSE184929:Schlegel_2022_Ttr_D1_30mkg"]["mean_pred"]
    e_d4 = conditions["GSE184929:Schlegel_2022_Ttr_D4_30mkg"]["mean_pred"]
    d4_lt_d1 = bool(e_d4 < e_d1)

    crit_a = bool(pooled_margin >= PRE_REGISTERED["criterion_a_min_mean_margin"])
    crit_b = bool(schlegel_margin > 0 and patel_margin > 0)
    crit_c = d4_lt_d1
    passed = bool(crit_a and crit_b and crit_c)

    return {
        "status": "ok",
        "stage": "Stage E/F: modification-aware occupancy + MonotoneEquilibriumPropagation, "
        "pre-trained on McGeary, criterion-1 transfer gate",
        "pre_registered": PRE_REGISTERED,
        "training": {
            "corpus": "McGeary GSE140217 (data-rich AGO-RBNS / transfection repression)",
            "n_train_cells": trained["n_train_cells"],
            "n_mirnas": trained["n_mirnas"],
            "learned_alpha": trained["alpha"],
            "learned_edge_scale_w": trained["w"],
            "propagation": "MonotoneEquilibriumPropagation (IFT-exact, unique fixed point)",
        },
        "conditions": conditions,
        "margins": {
            "schlegel_mean": schlegel_margin,
            "patel_mean": patel_margin,
            "pooled_mean": pooled_margin,
        },
        "d4_lt_d1": {"E_D1": e_d1, "E_D4": e_d4, "holds": d4_lt_d1},
        "criteria": {
            "a_pooled_margin": crit_a,
            "b_both_studies_positive": crit_b,
            "c_d4_lt_d1": crit_c,
        },
        "verdict": "PASS" if passed else "FAIL",
        "decision": (
            "PASS: the retrained modification-aware propagation predictor beats the mechanistic "
            "baseline on both studies and preserves D4<D1; the generator (Stage G) is unlocked."
            if passed
            else "FAIL: the retrained predictor does not clear the pre-registered margin on both "
            "studies. The mechanistic SeedMatchR + seed-Tm baseline is not beaten by >= 0.05 mean; "
            "do not build the generator. Pivot to the certified-discrimination + differentiable"
            "-propagation methods result (the D4<D1 sub-gate and IFT layer stand on their own)."
        ),
    }


def main() -> int:
    import torch

    device = (
        "cuda:1"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    numbers = run(device)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    t = numbers["training"]
    print(
        f"[criterion1] trained on {t['n_train_cells']} McGeary cells | alpha={t['learned_alpha']:.3f} w={t['learned_edge_scale_w']:.3f} | device {device}"
    )
    print(f"  {'condition':<36}{'learned':>9}{'baseline':>10}{'margin':>9}")
    for name, c in numbers["conditions"].items():
        print(
            f"  {name:<36}{c['spearman_learned']:>9.3f}{c['spearman_baseline']:>10.3f}{c['margin']:>+9.3f}"
        )
    m = numbers["margins"]
    print(
        f"  margins: Schlegel {m['schlegel_mean']:+.3f}  Patel {m['patel_mean']:+.3f}  pooled {m['pooled_mean']:+.3f}"
    )
    d = numbers["d4_lt_d1"]
    print(f"  D4<D1: E_D1={d['E_D1']:.4f} E_D4={d['E_D4']:.4f} holds={d['holds']}")
    cr = numbers["criteria"]
    print(
        f"  criteria: a(margin>=0.05)={cr['a_pooled_margin']} b(both>0)={cr['b_both_studies_positive']} c(D4<D1)={cr['c_d4_lt_d1']}"
    )
    print(f"  VERDICT: {numbers['verdict']}")
    print(f"[criterion1] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
