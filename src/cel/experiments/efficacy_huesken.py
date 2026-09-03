"""Train the CASCADE efficacy readout with and without RNA-FM on Huesken 2005.

Trains a ridge efficacy readout on the K=9 CASCADE features (the concept-feature
readout) and, as the second arm, on those features concatenated with the 640-d RNA-FM
mean-pooled guide embedding, on the real Huesken 2005 dataset. It reports Spearman and
Pearson (PCC) on the held-out test split to results/efficacy_huesken.json.

Dataset and split (REAL DATA ONLY): the guides, targets, efficacies, and the
conventional train / test split come from the repository's Huesken loader
(cel.data.load_huesken_efficacy), which carries the OligoFormer Hu subset of
Huesken 2005 with a 2182 train / 179 test split. The canonical full Huesken 2005 set is
2431 siRNAs across 34 transcripts with a 2182 / 249 split; the OligoFormer Hu CSV used
here does not carry the per-transcript labels (its mrna_id is a single placeholder), so
the transcript-grouped split cannot be reproduced from this source. The siRNADiscovery
anti-leakage methodology is therefore applied at the guide-SEQUENCE level: any test
siRNA whose guide also appears in train is removed, so no guide sequence leaks across
the split. The exact number kept is reported.

The RNA-FM arm needs the rna-fm ('fm') package (and a precomputed embedding cache or a
GPU to extract on the fly); when neither is available the arm is recorded pending with
the exact source, never a synthetic fallback. Writes results/efficacy_huesken.json.
No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"
RAW_EFFICACY = REPO / "data" / "raw" / "efficacy"
CACHE = REPO / "results" / "aaai" / "cache"
RNAFM_CACHE = CACHE / "huesken_rnafm_pooled.npy"
RIDGE_ALPHA = 1.0


def load_split() -> dict:
    """Real Huesken guides / targets / efficacy and the conventional train-test mask."""
    from cel.data import load_huesken_efficacy
    from cel.features.seedmatch import revcomp

    ds = load_huesken_efficacy(RAW_EFFICACY, cache_root=CACHE)
    f = ds.frame.reset_index(drop=True)
    guides = f["antisense"].astype(str).str.upper().tolist()
    targets = [
        (str(s).strip() or revcomp(a)) for a, s in zip(f["antisense"], f["sense"], strict=True)
    ]
    eff = f["efficacy"].to_numpy(dtype=float)
    is_train = (f["split"].astype(str) == "train").to_numpy()
    return {"guides": guides, "targets": targets, "eff": eff, "is_train": is_train, "n": len(f)}


def anti_leakage_test_mask(guides: list[str], is_train: np.ndarray) -> tuple[np.ndarray, int]:
    """Test mask with any guide also seen in train removed (siRNADiscovery anti-leakage)."""
    train_guides = {g for g, t in zip(guides, is_train, strict=True) if t}
    test_mask = ~is_train
    n_leaked = 0
    for i, (g, in_test) in enumerate(zip(guides, test_mask, strict=True)):
        if in_test and g in train_guides:
            test_mask[i] = False
            n_leaked += 1
    return test_mask, n_leaked


def cascade_features(guides: list[str], targets: list[str]) -> np.ndarray:
    from cel.features import featurize_pairs

    return featurize_pairs(list(zip(guides, targets, strict=True))).matrix


def rnafm_pooled(guides: list[str]) -> np.ndarray:
    """640-d RNA-FM mean-pooled embedding per guide (cached). Raises if RNA-FM is absent."""
    if RNAFM_CACHE.exists():
        cached = np.load(RNAFM_CACHE)
        if cached.shape[0] == len(guides):
            return cached
    from cel.embeddings.rnafm_extract import extract_embeddings

    _, pooled = extract_embeddings(guides)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(RNAFM_CACHE, pooled)
    return pooled


def fit_eval(x_tr, y_tr, x_te, y_te) -> dict:
    """Ridge efficacy readout; returns Spearman and Pearson (PCC) on the test split."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_tr)
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(x_tr), y_tr)
    pred = model.predict(scaler.transform(x_te))
    return {
        "spearman": float(spearmanr(pred, y_te)[0]),
        "pcc": float(pearsonr(pred, y_te)[0]),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_features": int(x_tr.shape[1]),
    }


def run() -> dict:
    data = load_split()
    guides, targets, eff, is_train = (
        data["guides"],
        data["targets"],
        data["eff"],
        data["is_train"],
    )
    test_mask, n_leaked = anti_leakage_test_mask(guides, is_train)
    x = cascade_features(guides, targets)
    x_tr, y_tr = x[is_train], eff[is_train]
    x_te, y_te = x[test_mask], eff[test_mask]

    without = fit_eval(x_tr, y_tr, x_te, y_te)

    # The with-RNA-FM arm is pending ONLY when the rna-fm package is absent (not on the
    # RNA-FM environment). A present-but-missing / unreadable checkpoint is a HARD FAILURE
    # (FileNotFoundError propagates), never a silent fallback to the without-RNA-FM row.
    try:
        pooled = rnafm_pooled(guides)
        xr = np.hstack([x, pooled.astype(float)])
        with_rnafm = fit_eval(xr[is_train], y_tr, xr[test_mask], y_te)
        with_rnafm["rnafm_dim"] = int(pooled.shape[1])
    except ImportError as exc:
        with_rnafm = {
            "status": "pending_real_data",
            "source": f"RNA-FM package not installed (run on the cascade_fm env): {exc}",
        }

    return {
        "status": "ok",
        "dataset": "Huesken 2005 (OligoFormer Hu subset)",
        "n_sirnas": int(data["n"]),
        "readout": f"ridge (alpha={RIDGE_ALPHA:g}) on the K=9 CASCADE features; the RNA-FM arm "
        "concatenates the 640-d RNA-FM mean-pooled guide embedding",
        "split": {
            "method": "conventional train/test split from cel.data.load_huesken_efficacy",
            "n_train": int(is_train.sum()),
            "n_test_conventional": int((~is_train).sum()),
            "n_test_after_anti_leakage": int(test_mask.sum()),
        },
        "anti_leakage": {
            "methodology": "siRNADiscovery anti-leakage at the guide-sequence level: any test "
            "siRNA whose guide also appears in train is removed (no shared guide across the split)",
            "n_test_guides_removed": n_leaked,
            "transcript_grouped_note": "the canonical Huesken 2005 set is 2431 siRNAs across 34 "
            "transcripts with a 2182/249 split; the OligoFormer Hu CSV used here carries 2361 "
            "siRNAs with a 2182/179 split and a placeholder mrna_id, so the transcript-grouped "
            "split is not reproducible from this source.",
        },
        "without_rnafm": without,
        "with_rnafm": with_rnafm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "efficacy_huesken.json")
    args = parser.parse_args()

    numbers = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    w = numbers["without_rnafm"]
    print(
        f"[efficacy-huesken] {numbers['n_sirnas']} siRNAs | train {numbers['split']['n_train']} "
        f"test {numbers['split']['n_test_after_anti_leakage']} "
        f"(anti-leakage removed {numbers['anti_leakage']['n_test_guides_removed']})"
    )
    print(f"[efficacy-huesken] without RNA-FM: Spearman {w['spearman']:.4f}  PCC {w['pcc']:.4f}")
    wr = numbers["with_rnafm"]
    if wr.get("status") == "pending_real_data":
        print(f"[efficacy-huesken] with RNA-FM: PENDING ({wr['source'][:80]}...)")
    else:
        print(
            f"[efficacy-huesken] with RNA-FM:    Spearman {wr['spearman']:.4f}  PCC {wr['pcc']:.4f}"
        )
    print(f"[efficacy-huesken] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
