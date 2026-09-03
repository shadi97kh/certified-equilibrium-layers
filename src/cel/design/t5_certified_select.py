"""Certified-feasible guide selection with conformal efficacy-FDR control (Theorem 5).

Pipeline:
  1. Enumerate candidate guide seeds over the 4^7 (ACGU, 7mer) alphabet.
  2. For each candidate compute (a) the T4 certified off-target upper bound U(g) and
     (b) the CASCADE predicted on-target efficacy.
  3. Threshold-select the guides with U(g) <= tau and predicted efficacy >= eta.
  4. Wrap the efficacy criterion in a Jin and Candes split-conformal selection
     (conformal p-values + Benjamini-Hochberg) calibrated on a held-out real
     Huesken split, to control the efficacy false discovery rate at level q.

Soundness argument (NO LaTeX; stated here as a docstring only):

  Off-target side (exact, deterministic). For a candidate seed the off-target input
  e_seed has, per gene, the seed-duplex occupancy times the cognate 3'UTR site count.
  The propagated off-target over the GRN is E = P_alpha(S) e_seed, and
  ||E|| = ||P_alpha(S) e_seed|| <= ||P_alpha(S)||_2 ||e_seed||_2 =: U(g). So U(g) is a
  certified upper bound on the propagated off-target L2 magnitude, and U(g) <= tau
  certifies that the propagated off-target is at most tau. This holds for every
  candidate with no distributional assumption.

  Efficacy side (conformal, assumption-stated). We want to select guides whose TRUE
  efficacy is at least eta while controlling the fraction of selected guides whose true
  efficacy is below eta (the efficacy FDR). Following Jin and Candes (2023), we fit an
  efficacy predictor mu_hat on a training split and hold out a CALIBRATION split of real
  Huesken guides with known efficacy. For each candidate j with score V_j = mu_hat(x_j)
  the conformal p-value uses only the calibration NULLS (guides with true efficacy < eta):

      p_j = (1 + #{i in calib : y_i < eta and V_i >= V_j}) / (1 + #{i in calib : y_i < eta}).

  Applying Benjamini-Hochberg to {p_j} at level q selects a subset whose efficacy FDR is
  controlled at q PROVIDED the candidate efficacy residuals are exchangeable with the
  calibration distribution. The candidates here are ENUMERATED (a fixed scaffold with all
  4^7 seeds), not sampled from the Huesken population, so that exchangeability is a
  modeling assumption and the efficacy FDR control is therefore APPROXIMATE. The
  off-target certificate U(g) <= tau is exact regardless; this is stated honestly and not
  overclaimed.

REAL DATA ONLY: the GRN is real mouse TRRUST, the off-target site abundance is over real
GRCm39 / Ensembl-104 3'UTRs, the seed-duplex dG is real ViennaRNA, and the efficacy model
and calibration split are real Huesken. Each loader RAISES when its real input is absent.
Writes results/t5_certified_guides.json. ruff-clean, no em-dashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"
CACHE = REPO / "results" / "aaai" / "cache"
RAW_EFFICACY = REPO / "data" / "raw" / "efficacy"
MOUSE_UTR_CACHE = CACHE / "mouse_trrust_3utr_r104.json"
ENSEMBL_RELEASE = 104
ALPHA = 0.5
SEED_LEN = 7
ALPHABET = "ACGU"
TTR_D1 = "UUAUAGAGCAAGAACACUGUUUU"
TTR_SEED = "UAUAGAG"
DEFAULT_ETA = 0.7
DEFAULT_Q = 0.1


def enumerate_seeds(n_seeds: int | None = None) -> list[str]:
    """All 4^7 seeds (deterministic order); or the first n_seeds for a fast run."""
    seeds = ("".join(p) for p in product(ALPHABET, repeat=SEED_LEN))
    if n_seeds is None:
        return list(seeds)
    out = []
    for i, s in enumerate(seeds):
        if i >= n_seeds:
            break
        out.append(s)
    return out


def _revcomp_rna(s: str) -> str:
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    return "".join(comp[b] for b in reversed(s.upper().replace("T", "U")))


def _site_dna(seed: str) -> str:
    comp = {"A": "T", "U": "A", "T": "A", "G": "C", "C": "G"}
    return "".join(comp[b] for b in reversed(seed.upper()))


def guide_for(scaffold: str, seed: str) -> str:
    return scaffold[:1] + seed + scaffold[1 + SEED_LEN :]


def _mouse_node_utrs(grn) -> dict:
    if MOUSE_UTR_CACHE.exists():
        return json.loads(MOUSE_UTR_CACHE.read_text())
    from pyensembl import EnsemblRelease

    release = EnsemblRelease(ENSEMBL_RELEASE, species="mouse")
    utrs: dict[str, str] = {}
    for sym in grn.nodes:
        try:
            genes = release.genes_by_name(sym)
        except Exception:
            continue
        best = ""
        for gene in genes:
            for t in gene.transcripts:
                u = t.three_prime_utr_sequence
                if u and len(u) > len(best):
                    best = u.upper()
        if best:
            utrs[sym] = best
    CACHE.mkdir(parents=True, exist_ok=True)
    MOUSE_UTR_CACHE.write_text(json.dumps(utrs))
    return utrs


def _site_sumsq(node_utrs: dict) -> dict:
    s2: dict[str, float] = {}
    for u in node_utrs.values():
        counts: dict[str, int] = {}
        for i in range(len(u) - SEED_LEN + 1):
            w = u[i : i + SEED_LEN]
            counts[w] = counts.get(w, 0) + 1
        for w, c in counts.items():
            s2[w] = s2.get(w, 0.0) + float(c) * float(c)
    return s2


def certified_U(seeds: list[str]) -> np.ndarray:
    """Certified off-target upper bound U(g) per seed over mouse TRRUST (real data)."""
    import RNA

    from cel.data import load_trrust
    from cel.verify import offtarget_bound as ob

    grn = load_trrust(species="mouse")
    p_norm = float(ob.p_alpha_norm(grn.S, ALPHA))
    s2 = _site_sumsq(_mouse_node_utrs(grn))
    u = np.zeros(len(seeds))
    for i, sd in enumerate(seeds):
        dg = float(RNA.duplexfold(sd, _revcomp_rna(sd)).energy)
        occ = -min(dg, 0.0)
        u[i] = p_norm * occ * float(np.sqrt(s2.get(_site_dna(sd), 0.0)))
    return u


def efficacy_predictor():
    """Ridge efficacy model on the Huesken TRAIN split; returns (predict, cal_pred, cal_y)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from cel.data import load_huesken_efficacy
    from cel.features import featurize_pairs
    from cel.features.seedmatch import revcomp

    ds = load_huesken_efficacy(RAW_EFFICACY, cache_root=CACHE)
    f = ds.frame
    anti = f["antisense"].astype(str).tolist()
    sense = f["sense"].astype(str).tolist()
    tgt = [(str(s).strip() or revcomp(a)) for a, s in zip(anti, sense, strict=True)]
    x = featurize_pairs(list(zip(anti, tgt, strict=True))).matrix
    y = f["efficacy"].to_numpy(dtype=float)
    is_train = (f["split"].astype(str) == "train").to_numpy()
    scaler = StandardScaler().fit(x[is_train])
    model = Ridge(alpha=1.0).fit(scaler.transform(x[is_train]), y[is_train])

    def predict(guides: list[str]) -> np.ndarray:
        feats = featurize_pairs([(g, revcomp(g)) for g in guides]).matrix
        return np.clip(model.predict(scaler.transform(feats)), 0.0, 1.0)

    # Calibration = the held-out real Huesken split (true efficacy labels).
    cal_pred = np.clip(model.predict(scaler.transform(x[~is_train])), 0.0, 1.0)
    cal_y = y[~is_train]
    return predict, cal_pred, cal_y


def conformal_pvalues(
    cal_pred: np.ndarray, cal_y: np.ndarray, test_pred: np.ndarray, eta: float
) -> np.ndarray:
    """Jin and Candes split-conformal p-values for H0: efficacy < eta (larger-is-better)."""
    null_scores = cal_pred[cal_y < eta]
    n0 = null_scores.size
    if n0 == 0:  # no calibration nulls: cannot reject, fully conservative
        return np.ones_like(test_pred)
    ge = (null_scores[None, :] >= test_pred[:, None]).sum(axis=1)
    return (1.0 + ge) / (1.0 + n0)


def benjamini_hochberg(pvals: np.ndarray, q: float) -> tuple[np.ndarray, float]:
    """BH at level q; returns (selected mask, p-value threshold)."""
    m = pvals.size
    if m == 0:
        return np.zeros(0, dtype=bool), 0.0
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool), 0.0
    k = np.max(np.where(below)[0])
    p_thresh = float(ranked[k])
    return pvals <= p_thresh, p_thresh


def run(
    scaffold: str = TTR_D1,
    tau: float | None = None,
    eta: float = DEFAULT_ETA,
    q: float = DEFAULT_Q,
    n_seeds: int | None = None,
) -> dict:
    seeds = enumerate_seeds(n_seeds)
    predict, cal_pred, cal_y = efficacy_predictor()
    u = certified_U(seeds)
    eff = predict([guide_for(scaffold, s) for s in seeds])

    # Default tau: the parent guide's certified off-target U (off-target no worse than parent).
    u_parent = float(certified_U([TTR_SEED])[0])
    tau = float(u_parent if tau is None else tau)

    pvals = conformal_pvalues(cal_pred, cal_y, eff, eta)
    conformal_selected, p_thresh = benjamini_hochberg(pvals, q)
    threshold_selected = (u <= tau) & (eff >= eta)
    final_selected = (u <= tau) & conformal_selected

    guides = [
        {
            "seed": seeds[i],
            "certified_U": round(float(u[i]), 4),
            "predicted_efficacy": round(float(eff[i]), 5),
            "conformal_pvalue": round(float(pvals[i]), 6),
            "threshold_selected": bool(threshold_selected[i]),
            "conformal_selected": bool(conformal_selected[i]),
            "selected": bool(final_selected[i]),
        }
        for i in range(len(seeds))
    ]
    return {
        "status": "ok",
        "scaffold": scaffold,
        "n_candidates": len(seeds),
        "alpha": ALPHA,
        "assumptions": {
            "tau": tau,
            "tau_note": "default tau is the parent Ttr guide's certified U (off-target no worse "
            "than the parent)"
            if n_seeds is None
            else "default tau as above",
            "eta": eta,
            "q_efficacy_fdr": q,
            "U_definition": "U(g) = ||P_alpha(mouse TRRUST)||_2 * ||e_seed||_2, a certified upper "
            "bound on the propagated off-target; U <= tau is exact.",
            "conformal": "Jin and Candes split-conformal p-values + Benjamini-Hochberg, calibrated "
            "on the held-out real Huesken split; the efficacy FDR control is approximate because "
            "the enumerated candidates are not exchangeable with the Huesken population.",
            "calibration_n": int(cal_y.size),
            "calibration_n_nulls": int((cal_y < eta).sum()),
        },
        "bh_pvalue_threshold": p_thresh,
        "n_threshold_selected": int(threshold_selected.sum()),
        "n_conformal_selected": int(conformal_selected.sum()),
        "n_selected": int(final_selected.sum()),
        "parent_Ttr_D1": {
            "seed": TTR_SEED,
            "certified_U": u_parent,
            "predicted_efficacy": float(predict([scaffold])[0]),
        },
        "guides": guides,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--q", type=float, default=DEFAULT_Q)
    parser.add_argument("--n-seeds", type=int, default=None, help="subset for a fast run")
    parser.add_argument("--out", type=Path, default=RESULTS / "t5_certified_guides.json")
    args = parser.parse_args()

    numbers = run(tau=args.tau, eta=args.eta, q=args.q, n_seeds=args.n_seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    a = numbers["assumptions"]
    print(
        f"[t5-select] candidates {numbers['n_candidates']} | tau={a['tau']:.3f} eta={a['eta']} "
        f"q={a['q_efficacy_fdr']} | calib {a['calibration_n']} ({a['calibration_n_nulls']} nulls)"
    )
    print(
        f"[t5-select] threshold-selected {numbers['n_threshold_selected']} | conformal-selected "
        f"{numbers['n_conformal_selected']} (BH p<= {numbers['bh_pvalue_threshold']:.4f}) | "
        f"FINAL (U<=tau and conformal) {numbers['n_selected']}"
    )
    print(f"[t5-select] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
