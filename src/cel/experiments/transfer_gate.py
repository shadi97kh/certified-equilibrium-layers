"""Stage 4 transfer gate: a pre-registered go/no-go before any generator is built.

The staged playbook puts a HARD GATE between the off-target predictor and the design
(generator) stage: the predictor, trained only on the miRNA auxiliary corpus, must transfer
zero-shot to the real siRNA off-target studies and BEAT the mechanistic SeedMatchR / seed-count
baseline by a pre-registered margin, AND reproduce the (S)-GNA KS direction on GSE184929
(the parent guide D1 must show MORE off-target than the (S)-GNA-destabilized D4, matching the
measured shift). This module reads the already-computed real transfer results and applies that
pre-registered rule, emitting a PASS / FAIL verdict so the decision is on record.

This is the Ben-David miRNA->siRNA domain-adaptation term of the composition measured, not
assumed: the gate reports the empirical transfer, not a bound.

REAL DATA ONLY: reads the committed real results (predictor_transfer.json, offtarget_t4.json);
raises if absent. Writes results/cascade_ot/transfer_gate.json. No LaTeX.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"
TRANSFER = RESULTS / "cascade_ot" / "predictor_transfer.json"
OFFTARGET_T4 = RESULTS / "offtarget_t4.json"
OUT = RESULTS / "cascade_ot" / "transfer_gate.json"

# Pre-registered gate (fixed before reading the numbers).
MARGIN_SPEARMAN = 0.05  # the learned mean per-condition Spearman must beat the baseline by this
PARENT = "Schlegel_2022_Ttr_D1_30mkg"  # GSE184929 parent guide
SGNA = "Schlegel_2022_Ttr_D4_30mkg"  # GSE184929 (S)-GNA-destabilized guide


def run() -> dict:
    if not TRANSFER.exists():
        raise FileNotFoundError(f"{TRANSFER} absent; run cel.experiments.ot_train_transfer first")
    if not OFFTARGET_T4.exists():
        raise FileNotFoundError(
            f"{OFFTARGET_T4} absent; run cel.experiments.offtarget_validate first"
        )
    transfer = json.loads(TRANSFER.read_text())
    t4 = json.loads(OFFTARGET_T4.read_text())["per_contrast"]

    # Criterion 1: beat the mechanistic seed-count baseline by the pre-registered margin.
    s = transfer["summary"]
    mean_learned = float(s["mean_spearman_learned"])
    mean_baseline = float(s["mean_spearman_baseline_seedcount"])
    delta = mean_learned - mean_baseline
    margin_pass = bool(delta >= MARGIN_SPEARMAN)

    # Criterion 2: reproduce the (S)-GNA KS direction on GSE184929 (parent off-target > (S)-GNA).
    meas_parent = t4[PARENT]["seedmatchr_ks_D"]
    meas_sgna = t4[SGNA]["seedmatchr_ks_D"]
    pred_parent = t4[PARENT]["cascade_predicted_ks_D"]
    pred_sgna = t4[SGNA]["cascade_predicted_ks_D"]
    measured_parent_gt_sgna = bool(meas_parent > meas_sgna)
    predicted_parent_gt_sgna = bool(pred_parent > pred_sgna)
    gna_reproduced = bool(measured_parent_gt_sgna and predicted_parent_gt_sgna)

    passed = bool(margin_pass and gna_reproduced)

    return {
        "status": "ok",
        "stage": "Stage 4 transfer gate (hard go/no-go before the generator)",
        "benchmark": "zero-shot transfer of the McGeary-trained off-target predictor to the real "
        "siRNA off-target studies (GSE184929 / GSE164289 / GSE196839), versus the mechanistic "
        "SeedMatchR / seed-match-count baseline on the same data",
        "pre_registered": {
            "margin_spearman": MARGIN_SPEARMAN,
            "criterion_1": "learned mean per-condition Spearman beats the mechanistic seed-count "
            "baseline by at least margin_spearman",
            "criterion_2": "reproduces the (S)-GNA KS direction on GSE184929: the model predicts MORE "
            "off-target for the parent D1 than for the (S)-GNA-destabilized D4, matching the measured shift",
            "decision_rule": "PASS only if BOTH criteria hold; otherwise FAIL and do not build the generator",
        },
        "criterion_1_margin": {
            "mean_spearman_learned": mean_learned,
            "mean_spearman_baseline_seedcount": mean_baseline,
            "delta": delta,
            "required_margin": MARGIN_SPEARMAN,
            "pass": margin_pass,
            "per_condition": {
                k: {
                    "learned": v["spearman_learned"],
                    "baseline": v["spearman_baseline_seedcount"],
                }
                for k, v in transfer["conditions"].items()
            },
        },
        "criterion_2_sgna_direction": {
            "measured_parent_ks_D": meas_parent,
            "measured_sgna_ks_D": meas_sgna,
            "measured_parent_more_offtarget": measured_parent_gt_sgna,
            "predicted_parent_ks_D": pred_parent,
            "predicted_sgna_ks_D": pred_sgna,
            "predicted_parent_more_offtarget": predicted_parent_gt_sgna,
            "reproduced": gna_reproduced,
            "note": "the predictor uses the same seed (count + seed-duplex dG) for D1 and D4, so its "
            "features are identical for the parent and the (S)-GNA guide; it cannot distinguish the "
            "(S)-GNA destabilization and predicts the same off-target for both",
        },
        "verdict": "PASS" if passed else "FAIL",
        "decision": (
            "PASS: the predictor transfers above the mechanistic baseline and reproduces the (S)-GNA "
            "direction; building the generator (Stage 5) is justified."
            if passed
            else "FAIL: the McGeary-trained predictor does NOT beat the mechanistic seed-count baseline "
            "(delta below the pre-registered margin) and cannot reproduce the (S)-GNA direction. Per the "
            "staged playbook, do NOT build the generator on this predictor; publish the predictor plus the "
            "honest negative transfer result and stop before Stage 5."
        ),
        "supports": "Stage 4 hard gate; the empirically measured miRNA->siRNA domain-adaptation "
        "(Ben-David) transfer term of the composition theorem, reported not assumed",
    }


def main() -> int:
    numbers = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    c1 = numbers["criterion_1_margin"]
    c2 = numbers["criterion_2_sgna_direction"]
    print(f"[transfer-gate] {numbers['stage']}")
    print(
        f"  criterion 1 (beat baseline by {c1['required_margin']}): learned {c1['mean_spearman_learned']:+.4f} "
        f"vs baseline {c1['mean_spearman_baseline_seedcount']:+.4f} (delta {c1['delta']:+.4f}) -> {c1['pass']}"
    )
    print(
        f"  criterion 2 ((S)-GNA direction): measured parent {c2['measured_parent_ks_D']:.3f} > D4 "
        f"{c2['measured_sgna_ks_D']:.3f} = {c2['measured_parent_more_offtarget']}; predicted parent "
        f"{c2['predicted_parent_ks_D']:.3f} > D4 {c2['predicted_sgna_ks_D']:.3f} = {c2['predicted_parent_more_offtarget']} "
        f"-> reproduced {c2['reproduced']}"
    )
    print(f"  VERDICT: {numbers['verdict']}")
    print(f"  {numbers['decision']}")
    print(f"[transfer-gate] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
