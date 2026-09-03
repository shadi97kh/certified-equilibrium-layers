"""Ablation of the trainable certified-monotone equilibrium layer: width, M-fraction, features.

REUSES the training path of scripts/learned_layer.py unchanged. Only the swept hyperparameter is
rebound (module attribute; no file is edited):

  * hidden width   -> learned_layer.HIDDEN, read by make_mlp() when the model is constructed
  * M-fraction     -> learned_layer.M_FRACTION, read by assemble() when pools are built
  * feature drop   -> the (N, 7) feature matrix from build_pools() is column-sliced and
                      learned_layer.D_FEAT is set to the reduced width so make_mlp() matches

The seed is FIXED at 20240716 for every cell (initialization, optimizer stream, and the
train/held-out split), so only the ablated axis varies. The four held-out microRNAs are therefore
identical in every cell. Corpus loading raises if the real cache is absent; nothing is synthetic.

Materiality is judged against the measured seed-to-seed noise floor of this same setup
(sample sd 0.012905 of held-out mean Spearman over 10 seeds, results/learned_layer_seeds.json):
a cell whose shift from the reference is smaller than that sd is not distinguishable from seed noise.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import learned_layer as ll  # noqa: E402  (REUSED training path, imported unmodified)

OUT_JSON = REPO / "results" / "learned_layer_ablation.json"

SEED = 20240716
HIDDEN_GRID = [32, 64, 128]
MFRAC_GRID = [0.1, 0.2, 0.5]
REF_HIDDEN = 64
REF_MFRAC = 0.2
# Seed-to-seed sample sd of held-out mean Spearman for this exact setup (10 seeds, same split).
SEED_NOISE_SD = 0.012905

_ORIG = {"HIDDEN": ll.HIDDEN, "M_FRACTION": ll.M_FRACTION, "D_FEAT": ll.D_FEAT, "SEED": ll.SEED}


def _restore() -> None:
    ll.HIDDEN = _ORIG["HIDDEN"]
    ll.M_FRACTION = _ORIG["M_FRACTION"]
    ll.D_FEAT = _ORIG["D_FEAT"]
    ll.SEED = SEED


def run_cell(data: dict, train: list, test: list, hidden: int, mfrac: float,
             drop_idx: int | None) -> dict:
    """Train the equilibrium layer for one ablation cell and report held-out Spearman + certificate."""
    _restore()
    keep = None if drop_idx is None else [i for i in range(_ORIG["D_FEAT"]) if i != drop_idx]
    ll.HIDDEN = hidden
    ll.M_FRACTION = mfrac
    ll.D_FEAT = _ORIG["D_FEAT"] if keep is None else len(keep)

    sub = data if keep is None else {"df": data["df"], "feat": data["feat"][:, keep]}
    pools = ll.assemble(sub, train)["pools"]

    ll.SEED = SEED
    ll.set_determinism()
    eq = ll.EquilibriumLayer()
    n_par = ll.n_params(eq)
    run = ll.train_model(eq, pools, train, test, cert=True)
    traj = run["min_d_traj"]
    _restore()
    return {
        "hidden": hidden,
        "M_fraction": mfrac,
        "dropped_feature": None if drop_idx is None else ll.FEATURES[drop_idx],
        "d_features": _ORIG["D_FEAT"] if keep is None else len(keep),
        "n_params": n_par,
        "heldout_mean_spearman": run["curve"][-1]["heldout_mean_spearman"],
        "certificate_n_steps": int(len(traj)),
        "certificate_min_over_steps": float(min(traj)),
        "certificate_positive_every_step": bool(all(d > 0.0 for d in traj)),
    }


def main() -> int:
    ll.SEED = SEED
    data = ll.build_pools()  # raises if the real corpus cache is absent
    mirnas = sorted(data["df"]["mirna"].unique())
    train, test = ll.split_mirnas(mirnas)

    reference = run_cell(data, train, test, REF_HIDDEN, REF_MFRAC, None)
    ref = reference["heldout_mean_spearman"]

    width = [run_cell(data, train, test, h, REF_MFRAC, None) for h in HIDDEN_GRID]
    mfrac = [run_cell(data, train, test, REF_HIDDEN, m, None) for m in MFRAC_GRID]
    feats = [run_cell(data, train, test, REF_HIDDEN, REF_MFRAC, i)
             for i in range(_ORIG["D_FEAT"])]

    def annotate(cells: list) -> list:
        for c in cells:
            d = c["heldout_mean_spearman"] - ref
            c["delta_vs_reference"] = d
            c["abs_delta_over_seed_sd"] = abs(d) / SEED_NOISE_SD
            c["material"] = bool(abs(d) > SEED_NOISE_SD)
        return cells

    width, mfrac, feats = annotate(width), annotate(mfrac), annotate(feats)
    every = width + mfrac + feats
    cert_ok = all(c["certificate_positive_every_step"] for c in every)
    cert_min = min(c["certificate_min_over_steps"] for c in every)
    cert_steps = sum(c["certificate_n_steps"] for c in every)

    material = [c for c in every if c["material"]]
    immaterial = [c for c in every if not c["material"]]

    def label(c: dict) -> str:
        if c["dropped_feature"] is not None:
            return f"drop {c['dropped_feature']}"
        if c["M_fraction"] != REF_MFRAC:
            return f"M_fraction={c['M_fraction']}"
        return f"hidden={c['hidden']}"

    out = {
        "experiment": "ablation of the trainable equilibrium layer (width, M-fraction, features)",
        "reused_path": "scripts/learned_layer.py (build_pools, split_mirnas, assemble, "
                       "set_determinism, EquilibriumLayer, train_model), unmodified",
        "seed": SEED,
        "heldout_mirnas": test,
        "train_mirnas": train,
        "reference_cell": reference,
        "reference_heldout_mean_spearman": ref,
        "materiality_rule": {
            "seed_noise_sample_sd": SEED_NOISE_SD,
            "source": "results/learned_layer_seeds.json (10 seeds, identical split and setup)",
            "rule": "a cell is material iff |delta vs reference| exceeds the seed-to-seed sd",
        },
        "hidden_width_ablation": width,
        "m_fraction_ablation": mfrac,
        "feature_ablation": feats,
        "material_changes": [
            {"cell": label(c), "heldout": c["heldout_mean_spearman"],
             "delta": c["delta_vs_reference"]} for c in material
        ],
        "immaterial_changes": [
            {"cell": label(c), "heldout": c["heldout_mean_spearman"],
             "delta": c["delta_vs_reference"]} for c in immaterial
        ],
        "certificate": {
            "positive_every_step_every_cell": bool(cert_ok),
            "min_over_all_steps_all_cells": cert_min,
            "total_steps_checked": cert_steps,
            "n_cells": len(every),
            "note": "D = 1 + sum_s x_s sigma'(f) >= 1 by construction (softplus K > 0, A1)",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md5 = hashlib.md5(OUT_JSON.read_bytes()).hexdigest()

    print(f"\nheld-out microRNAs (fixed): {test}")
    print(f"reference (hidden={REF_HIDDEN}, M_fraction={REF_MFRAC}, all features): {ref:.6f}")
    for title, cells in (("hidden width", width), ("M-fraction", mfrac), ("feature drop", feats)):
        print(f"\n  -- {title} --")
        for c in cells:
            flag = "MATERIAL " if c["material"] else "within-noise"
            print(f"    {label(c):<28s} {c['heldout_mean_spearman']:.6f}  "
                  f"delta {c['delta_vs_reference']:+.6f}  ({flag}, "
                  f"{c['abs_delta_over_seed_sd']:.2f}x seed sd)")
    print(f"\ncertificate positive on every step of every cell: {cert_ok} "
          f"(min {cert_min:.6f} over {cert_steps} steps, {len(every)} cells)")
    print(f"wrote {OUT_JSON.relative_to(REPO)}  md5 {md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
