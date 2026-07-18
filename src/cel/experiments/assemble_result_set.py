"""Assemble the certified-discrimination + differentiable-propagation result set for the paper.

Collects the four committed gate results into one manifest, pulling every number from the real
source JSONs (never re-deriving or hardcoding) so the manifest stays in sync:

  criterion 2 (modification discrimination), PASS with certificate:
    - subgate_d4d1.json     the monotone modification-aware readout gives E(D4) < E(D1)
    - ift_check.json        the differentiable propagation layer is IFT-exact and well-posed
  criterion 1 (miRNA -> siRNA transfer), FAIL twice pre-registered:
    - transfer_gate.json    count-based predictor, pre-registered margin gate
    - criterion1_retrain.json  learned occupancy + MonotoneEquilibriumPropagation, pre-registered gate

It states the demarcation plainly: criterion 2 PASS with certificate; criterion 1 FAIL twice
pre-registered; the binding constraint is the independent-study count, not model capacity. No
generator code is written; Stage G stays gated.

REAL DATA ONLY: every value is read from a committed real result file. Writes
results/cascade_ot/paper_result_set.json. No LaTeX.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

CO = REPO / "results" / "cascade_ot"
OUT = CO / "paper_result_set.json"

SOURCES = {
    "subgate_d4d1": CO / "subgate_d4d1.json",
    "ift_check": CO / "ift_check.json",
    "transfer_gate": CO / "transfer_gate.json",
    "criterion1_retrain": CO / "criterion1_retrain.json",
}


def _load(name: str) -> dict:
    p = SOURCES[name]
    if not p.exists():
        raise FileNotFoundError(f"required source result missing: {p}")
    return json.loads(p.read_text())


def build() -> dict:
    sub = _load("subgate_d4d1")
    ift = _load("ift_check")
    tg = _load("transfer_gate")
    cr = _load("criterion1_retrain")

    sub_pass = sub["subgate"] == "PASS"
    ift_pass = ift["verdict"] == "PASS"
    tg_fail = tg["verdict"] == "FAIL"
    cr_fail = cr["verdict"] == "FAIL"

    criterion_2 = {
        "claim": "modification discrimination: the certified off-target readout distinguishes a "
        "chemically modified guide from its parent in the correct direction",
        "verdict": "PASS" if (sub_pass and ift_pass) else "INCOMPLETE",
        "with_certificate": True,
        "evidence": {
            "subgate_d4d1": {
                "source": "results/cascade_ot/subgate_d4d1.json",
                "subgate": sub["subgate"],
                "monotone_E_D1": sub["monotone"]["E_D1"],
                "monotone_E_D4": sub["monotone"]["E_D4"],
                "D4_lt_D1": sub["monotone"]["D4_lt_D1"],
                "certificate": sub["monotone"]["certificate"],
                "non_monotone_control_D4_lt_D1": sub["non_monotone_control"]["D4_lt_D1"],
                "ddG_kcal_per_mol": sub["ddg_calibration"]["ddG_kcal_per_mol"],
                "ddG_model": sub["ddg_calibration"]["model"],
                "ddG_destabilizing": sub["ddg_calibration"]["destabilizing"],
            },
            "ift_propagation": {
                "source": "results/cascade_ot/ift_check.json",
                "verdict": ift["verdict"],
                "n_nodes": ift["n_nodes"],
                "operator": ift["operator"],
                "fixed_point_residual": ift["checks"]["fixed_point_residual"],
                "appnp_layer_vs_closed_form": ift["checks"]["appnp_layer_vs_closed_form"],
                "ift_grad_vs_finite_diff_max_rel_error": ift["checks"][
                    "ift_grad_vs_finite_diff_max_rel_error"
                ],
                "spectral_radius_rho_S": ift["checks"]["spectral_radius_rho_S"],
                "contraction_factor": ift["checks"]["contraction_factor"],
                "unique_fixed_point": ift["checks"]["unique_fixed_point"],
            },
        },
    }

    criterion_1 = {
        "claim": "miRNA -> siRNA transfer: a predictor trained on the data-rich miRNA corpus beats "
        "the mechanistic seed baseline on the held-out siRNA off-target studies",
        "verdict": "FAIL" if (tg_fail and cr_fail) else "NOT_FAILED_TWICE",
        "failed_twice_pre_registered": bool(tg_fail and cr_fail),
        "attempts": {
            "transfer_gate_count_based": {
                "source": "results/cascade_ot/transfer_gate.json",
                "pre_registered": tg["pre_registered"],
                "criterion_1_margin": {
                    "mean_spearman_learned": tg["criterion_1_margin"]["mean_spearman_learned"],
                    "mean_spearman_baseline": tg["criterion_1_margin"][
                        "mean_spearman_baseline_seedcount"
                    ],
                    "delta": tg["criterion_1_margin"]["delta"],
                    "required_margin": tg["criterion_1_margin"]["required_margin"],
                    "pass": tg["criterion_1_margin"]["pass"],
                },
                "criterion_2_sgna_direction_reproduced": tg["criterion_2_sgna_direction"][
                    "reproduced"
                ],
                "verdict": tg["verdict"],
            },
            "criterion1_retrain_learned": {
                "source": "results/cascade_ot/criterion1_retrain.json",
                "pre_registered": cr["pre_registered"],
                "model": "modification-aware occupancy + learned binding head + "
                "MonotoneEquilibriumPropagation (learned alpha="
                f"{cr['training']['learned_alpha']:.3f}, w={cr['training']['learned_edge_scale_w']:.3f}"
                "), pre-trained on McGeary",
                "margins": cr["margins"],
                "criteria": cr["criteria"],
                "d4_lt_d1_holds": cr["d4_lt_d1"]["holds"],
                "verdict": cr["verdict"],
            },
        },
    }

    demarcation = {
        "criterion_2_modification_discrimination": "PASS with certificate. The monotone "
        "modification-aware readout reproduces E(D4) < E(D1) by construction (E linear in occupancy, "
        "same sites), a count-only readout cannot, and the differentiable propagation that carries "
        f"the occupancy is IFT-exact on the real {ift['n_nodes']}-node TRRUST operator.",
        "criterion_1_transfer": "FAIL twice, both pre-registered. The count-based predictor missed "
        f"the >= {tg['criterion_1_margin']['required_margin']:.2f} Spearman margin "
        f"(delta {tg['criterion_1_margin']['delta']:.4f}) and could not reproduce the (S)-GNA "
        "direction; the learned occupancy + MonotoneEquilibriumPropagation predictor also missed "
        f"the margin (pooled {cr['margins']['pooled_mean']:.4f}) and was not positive on both "
        f"studies (Schlegel {cr['margins']['schlegel_mean']:.4f}, Patel {cr['margins']['patel_mean']:.4f}).",
        "binding_constraint": "The binding constraint is the independent-study count, not model "
        "capacity. Both a simple count baseline and a learned propagation predictor land at the same "
        "near-zero margin against the mechanistic seed baseline, and the margin oscillates in sign "
        "across the handful of held-out conditions; with only two independent off-target studies "
        "(Schlegel rat, Patel human) there is no statistical room for a model to separate from the "
        "baseline, regardless of capacity. More independent studies, not a bigger model, is the "
        "unlock.",
    }

    return {
        "status": "ok",
        "title": "CASCADE certified-discrimination + differentiable-propagation result set",
        "real_data_only": True,
        "sources": {k: str(v.relative_to(REPO)) for k, v in SOURCES.items()},
        "results": {
            "criterion_2_modification_discrimination": criterion_2,
            "criterion_1_transfer": criterion_1,
        },
        "demarcation": demarcation,
        "stage_g_generator": {
            "status": "GATED",
            "reason": "Stage G (the off-target-aware generator) requires criterion 1 to PASS. It has "
            "FAILED twice pre-registered, so no generator code is written and Stage G stays gated.",
        },
        "headline": "Criterion 2 (modification discrimination) PASSES with a certificate; criterion 1 "
        "(miRNA -> siRNA transfer) FAILS twice pre-registered. The defensible contribution is the "
        "certified modification-aware discrimination and the IFT-exact differentiable propagation "
        "layer, not a generator.",
    }


def main() -> int:
    m = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(m, indent=2), encoding="utf-8")

    c2 = m["results"]["criterion_2_modification_discrimination"]
    c1 = m["results"]["criterion_1_transfer"]
    print(f"[result-set] {m['title']}")
    print(
        f"  criterion 2 (modification discrimination): {c2['verdict']} (with certificate={c2['with_certificate']})"
    )
    sd = c2["evidence"]["subgate_d4d1"]
    print(
        f"    sub-gate D4<D1: E_D1={sd['monotone_E_D1']:.2f} E_D4={sd['monotone_E_D4']:.2f} -> {sd['subgate']}"
    )
    ip = c2["evidence"]["ift_propagation"]
    print(
        f"    IFT propagation: residual={ip['fixed_point_residual']:.1e} grad_rel_err={ip['ift_grad_vs_finite_diff_max_rel_error']:.1e} -> {ip['verdict']}"
    )
    print(
        f"  criterion 1 (transfer): {c1['verdict']} (failed twice pre-registered={c1['failed_twice_pre_registered']})"
    )
    tgm = c1["attempts"]["transfer_gate_count_based"]["criterion_1_margin"]
    print(
        f"    count-based: delta={tgm['delta']:+.4f} (req {tgm['required_margin']}) -> {c1['attempts']['transfer_gate_count_based']['verdict']}"
    )
    crm = c1["attempts"]["criterion1_retrain_learned"]["margins"]
    print(
        f"    learned:     pooled={crm['pooled_mean']:+.4f} Schlegel={crm['schlegel_mean']:+.4f} Patel={crm['patel_mean']:+.4f} -> {c1['attempts']['criterion1_retrain_learned']['verdict']}"
    )
    print(f"  Stage G generator: {m['stage_g_generator']['status']}")
    print(f"[result-set] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
