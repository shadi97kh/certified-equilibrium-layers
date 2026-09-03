#!/usr/bin/env python3
"""Backbone on-target efficacy benchmark on the REAL OligoFormer suite.

Trains the CCMP backbone (cel.models.CCMPBackbone, the per-concept direct
seed-repression predictor, no propagation) as an on-target siRNA efficacy
regressor on the real OligoFormer efficacy suite, under the standard protocol:

  train on the Huesken training split; report Pearson and Spearman on the
  Huesken test split; and report cross-dataset generalization on Taka (Katoh),
  Mix, and Shabalina as independent test sets.

The cross-dataset drop from Huesken-test to the independent sets is the expected
distribution shift; it is reported, not hidden.

HARD DATA POLICY (OUTLINE.md): this experiment uses ONLY the real OligoFormer
CSVs. There is no fallback to generated data. The loader RAISES FileNotFoundError
when the real CSVs are absent (place them under --raw-dir, default
data/raw/oligoformer).

Baselines. Published numbers (BIOPREDsi-style, siRNADiscovery, OligoFormer) are
cited as "reported" with their source; this script does NOT fabricate baseline
numbers and only computes our own model's Pearson / Spearman. Values that are
not transcribed here are left null with a note to fill them from the source
table.

ACCEPTANCE: the backbone must be at least competitive with the reported state of
the art on the Huesken test split; the JSON flags PASS / FAIL.

Writes results/aaai/backbone_efficacy.json and a cross-dataset bar figure.
Accepts --seed and logs it (OUTLINE.md rule 3). No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.features import concept_layout, featurize_pairs  # noqa: E402
from cel.features.seedmatch import revcomp  # noqa: E402

OUT = REPO / "results" / "aaai"
DATASETS = ["huesken_test", "taka", "mix", "shabalina"]

# Published baselines on the Huesken test split. Cite the source; do NOT
# fabricate. reported_value is null where it is not transcribed here.
REPORTED_BASELINES = [
    {
        "name": "BIOPREDsi (Huesken ANN)",
        "source": "Huesken et al., Nat Biotechnol 23:995-1001 (2005), 10.1038/nbt1118",
        "dataset": "Huesken test (249)",
        "metric": "pearson",
        "reported_value": 0.66,
        "note": "Pearson R commonly cited for the BIOPREDsi predictor on the 249-siRNA "
        "Huesken test split; verify against the source table before use.",
        "verify": True,
    },
    {
        "name": "OligoFormer",
        "source": "OligoFormer (public release)",
        "dataset": "Huesken test",
        "metric": "pearson",
        "reported_value": None,
        "note": "transcribe from the OligoFormer paper/repo table; not asserted here.",
        "verify": True,
    },
    {
        "name": "siRNADiscovery",
        "source": "siRNADiscovery (public release)",
        "dataset": "Huesken test",
        "metric": "pearson",
        "reported_value": None,
        "note": "transcribe from the siRNADiscovery table; not asserted here.",
        "verify": True,
    },
]

ACCEPTANCE_TOLERANCE = 0.05  # competitive if our Pearson >= best reported - tolerance


def _featurize(antisense: list[str], targets: list[str]) -> np.ndarray:
    pairs = list(zip(antisense, targets, strict=True))
    return featurize_pairs(pairs).matrix


def _target_for(antisense: str, sense: str) -> str:
    sense = (sense or "").strip()
    return sense if sense else revcomp(antisense)


def real_datasets(raw_dir: Path, cache_root: Path) -> tuple[dict, dict]:
    """Featurize the real OligoFormer suite (raises FileNotFoundError if absent)."""
    from cel.data import canonical_split, load_oligoformer_efficacy

    ds = load_oligoformer_efficacy(raw_dir, cache_root=cache_root)
    frame = ds.frame

    def featurize_frame(sub) -> np.ndarray:
        anti = sub["antisense"].tolist()
        tgt = [_target_for(a, s) for a, s in zip(sub["antisense"], sub["sense"], strict=True)]
        return _featurize(anti, tgt)

    def study_mask(substr: str):
        return frame["source_study"].str.contains(substr, case=False, na=False)

    hue = frame[study_mask("Huesken")].reset_index(drop=True)
    split = canonical_split(len(hue))
    feats, labels = {}, {}
    feats["huesken_train"] = featurize_frame(hue[split == "train"])
    labels["huesken_train"] = hue[split == "train"]["efficacy"].to_numpy()
    feats["huesken_test"] = featurize_frame(hue[split == "test"])
    labels["huesken_test"] = hue[split == "test"]["efficacy"].to_numpy()
    for key, substr in (("taka", "Katoh"), ("shabalina", "Shabalina")):
        sub = frame[study_mask(substr)].reset_index(drop=True)
        feats[key] = featurize_frame(sub)
        labels[key] = sub["efficacy"].to_numpy()
    mix = frame[
        ~(study_mask("Huesken") | study_mask("Katoh") | study_mask("Shabalina"))
    ].reset_index(drop=True)
    feats["mix"] = featurize_frame(mix)
    labels["mix"] = mix["efficacy"].to_numpy()
    return feats, labels


def train_and_eval(feats: dict, labels: dict, seed: int, epochs: int, lr: float) -> dict:
    import torch

    from cel.models import CCMPBackbone

    torch.manual_seed(seed)
    x_tr, y_tr = feats["huesken_train"], labels["huesken_train"]
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8

    def tens(x):
        return torch.tensor((x - mu) / sd, dtype=torch.float32)

    model = CCMPBackbone(hidden_dim=32, concept_emb_dim=32, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    xt, yt = tens(x_tr), torch.tensor(y_tr, dtype=torch.float32)
    final_mse = float("nan")
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = ((model(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
        final_mse = float(loss.detach())

    model.eval()
    from scipy.stats import pearsonr, spearmanr

    per = {}
    with torch.no_grad():
        for key in DATASETS:
            x, y = feats[key], labels[key]
            if len(y) < 2:
                per[key] = {"n": int(len(y)), "pearson": None, "spearman": None}
                continue
            p = model(tens(x)).numpy()
            pear = float(pearsonr(p, y)[0]) if np.std(p) > 0 else None
            spear = float(spearmanr(p, y)[0]) if np.std(p) > 0 else None
            per[key] = {"n": int(len(y)), "pearson": pear, "spearman": spear}
    return {"huesken_train_final_mse": final_mse, "n_train": int(len(y_tr)), "per_dataset": per}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(REPO / "data" / "raw" / "oligoformer"),
        help="real OligoFormer CSV directory (FileNotFoundError if absent)",
    )
    parser.add_argument("--cache", type=Path, default=OUT / "cache")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[backbone] seed={args.seed} epochs={args.epochs} raw_dir={args.raw_dir}")

    feats, labels = real_datasets(Path(args.raw_dir), args.cache)
    res = train_and_eval(feats, labels, args.seed, args.epochs, args.lr)
    per = res["per_dataset"]
    our_huesken = per["huesken_test"]["pearson"]

    reported = [b["reported_value"] for b in REPORTED_BASELINES if b["reported_value"] is not None]
    best_reported = max(reported) if reported else None
    best_name = (
        max(
            (b for b in REPORTED_BASELINES if b["reported_value"] is not None),
            key=lambda b: b["reported_value"],
        )["name"]
        if reported
        else None
    )
    if best_reported is None or our_huesken is None:
        competitive, flag = None, "PENDING"
    else:
        competitive = bool(our_huesken >= best_reported - ACCEPTANCE_TOLERANCE)
        flag = "PASS" if competitive else "FAIL"

    summary = {
        "seed": args.seed,
        "provenance": "real_oligoformer",
        "model": "CCMP backbone (x0 direct-repression predictor), trained as efficacy regressor",
        "n_features": concept_layout().total_dim,
        "protocol": "train on Huesken train; Pearson/Spearman on Huesken test and cross-dataset "
        "Taka/Mix/Shabalina (independent test sets)",
        "train": {
            "huesken_train_final_mse": res["huesken_train_final_mse"],
            "n_train": res["n_train"],
        },
        "our_model": per,
        "reported_baselines": REPORTED_BASELINES,
        "acceptance": {
            "our_huesken_test_pearson": our_huesken,
            "best_reported_pearson": best_reported,
            "best_reported_name": best_name,
            "tolerance": ACCEPTANCE_TOLERANCE,
            "margin": (our_huesken - best_reported) if (best_reported and our_huesken) else None,
            "competitive": competitive,
            "flag": flag,
        },
        "cross_dataset_note": "the drop from Huesken-test to Taka/Mix/Shabalina is the expected "
        "distribution shift; it is reported here, not hidden.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "backbone_efficacy.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key in DATASETS:
        d = per[key]
        print(f"[backbone] {key:14s} n={d['n']:4d} pearson={d['pearson']} spearman={d['spearman']}")
    print(f"[backbone] acceptance {flag} (our={our_huesken} vs best reported={best_reported})")
    print(f"[backbone] wrote {out_path}")
    _plot(summary, args.out_dir / "figures" / "backbone_efficacy.png")
    return 0


def _plot(summary: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[backbone] matplotlib not installed; skipped figure")
        return

    per = summary["our_model"]
    names = DATASETS
    pear = [per[k]["pearson"] if per[k]["pearson"] is not None else 0.0 for k in names]
    spear = [per[k]["spearman"] if per[k]["spearman"] is not None else 0.0 for k in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - 0.2, pear, 0.4, label="Pearson")
    ax.bar(x + 0.2, spear, 0.4, label="Spearman")
    best = summary["acceptance"]["best_reported_pearson"]
    if best is not None:
        ax.axhline(
            best,
            color="k",
            ls="--",
            label=f"reported {summary['acceptance']['best_reported_name']} ({best})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("correlation")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"CCMP backbone efficacy (real OligoFormer); acceptance {summary['acceptance']['flag']}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[backbone] wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
