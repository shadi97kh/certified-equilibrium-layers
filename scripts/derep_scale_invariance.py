"""Scale invariance of the derepression-threshold ordering under the unknown absolute affinity scale.

The titration contrast (scripts/coupling_contrast.py) fixes the RELATIVE site-type affinity ordering
(8mer < 7mer < 6mer, real McGeary/scanMiR) but its ABSOLUTE Kd scale in copies/cell is nominal and
cannot be pinned down from the data. This script sweeps that one nominal scale over more than two
orders of magnitude and asks whether the operator's qualitative ordering of derepression thresholds
(miR-122 above miR-92/25) survives, even as the absolute thresholds move.

REUSE, not re-derivation: the two real study configurations (Denzler miR-122, Bosson miR-92/25), the
shared-pool equilibrium solve, and the 1.5-fold derepression threshold definition are taken from
scripts/coupling_contrast.py unchanged. Only the absolute affinity scale (K8, K7, K6, and the focal
K) is multiplied by the swept factor; the relative ordering among site types is preserved.

No synthetic data: the equilibrium solver is the real bisection solver imported through
coupling_contrast; if it is not importable the import raises. Nothing is tuned or fitted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import coupling_contrast as cc  # noqa: E402  (REUSED real configs + solver; raises if solver absent)

OUT_JSON = REPO / "results" / "derep_scale_invariance.json"

# Nominal absolute affinities from the titration contrast (copies/cell); relative ordering preserved.
K8_0, K7_0, K6_0 = cc.K8, cc.K7, cc.K6
DEREP_FOLD = 1.5  # threshold definition, unchanged from coupling_contrast.run_scenario
# Sweep the one un-pin-downable knob over >2 orders of magnitude (0.1x .. 100x nominal = 3 orders).
SCALES = [float(s) for s in np.geomspace(0.1, 100.0, 10)]
BISECT_TOL = 1e-9


def _set_scale(scale: float) -> None:
    """Rebind the module's absolute affinities to `scale` x nominal (relative ordering unchanged)."""
    cc.K8 = K8_0 * scale
    cc.K7 = K7_0 * scale
    cc.K6 = K6_0 * scale
    cc.K_FOCAL = K8_0 * scale  # focal is the 8mer site


def _restore() -> None:
    cc.K8, cc.K7, cc.K6, cc.K_FOCAL = K8_0, K7_0, K6_0, K8_0


def derep_fold(sc: dict, m_tot: float, x_comp: float, theta0: float) -> float:
    """Operator fold-derepression theta0/theta(x_comp) at the current absolute scale."""
    f = cc.equilibrium_free_mirna(m_tot, x_comp, sc)
    theta = f / (cc.K_FOCAL + f)
    return theta0 / theta


def operator_threshold(sc: dict) -> float | None:
    """Smallest added-competitor abundance at which fold-derepression reaches DEREP_FOLD.

    Located by bracketing then bisection on the monotone-increasing derepression curve. Returns None
    if the threshold is not reached within a very large competitor budget (10^6 x the guide pool).
    """
    m_tot = sc["miRNA_total"]
    f0 = cc.equilibrium_free_mirna(m_tot, 0.0, sc)
    theta0 = f0 / (cc.K_FOCAL + f0)
    cap = 1.0e6 * m_tot
    hi = max(m_tot, 1.0)
    while derep_fold(sc, m_tot, hi, theta0) < DEREP_FOLD:
        hi *= 2.0
        if hi > cap:
            return None
    lo = 0.0
    while (hi - lo) > BISECT_TOL * (1.0 + hi):
        mid = 0.5 * (lo + hi)
        if derep_fold(sc, m_tot, mid, theta0) >= DEREP_FOLD:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main() -> int:
    rows = []
    for scale in SCALES:
        _set_scale(scale)
        thr = {name: operator_threshold(sc) for name, sc in cc.SCENARIOS.items()}
        d, b = thr["Denzler_miR122"], thr["Bosson_miR92_25"]
        both_finite = d is not None and b is not None
        rows.append({
            "affinity_scale_x_nominal": scale,
            "K8_copies_per_cell": cc.K8,
            "denzler_mir122_operator_threshold_added_sites": d,
            "bosson_mir92_25_operator_threshold_added_sites": b,
            "both_finite": both_finite,
            "ordering_mir122_above_mir92_25": bool(both_finite and d > b),
            "threshold_ratio_mir122_over_mir92_25": (d / b) if both_finite and b > 0 else None,
        })
    _restore()

    finite = [r for r in rows if r["both_finite"]]
    ordering_all = all(r["ordering_mir122_above_mir92_25"] for r in finite) and len(finite) > 0
    ratios = [r["threshold_ratio_mir122_over_mir92_25"] for r in finite]
    d_thr = [r["denzler_mir122_operator_threshold_added_sites"] for r in finite]
    b_thr = [r["bosson_mir92_25_operator_threshold_added_sites"] for r in finite]

    out = {
        "experiment": "derepression-threshold ordering vs the unknown absolute affinity scale",
        "reused": "scripts/coupling_contrast.py real configs (Denzler miR-122, Bosson miR-92/25), "
                  "shared-pool solver, and 1.5-fold threshold definition, unmodified",
        "swept_axis": "absolute site-type affinity scale (K8,K7,K6 and focal K) x nominal; "
                      "relative site-type ordering preserved",
        "nominal_affinities_copies_per_cell": {"K8": K8_0, "K7": K7_0, "K6": K6_0},
        "derepression_fold_threshold": DEREP_FOLD,
        "scales": SCALES,
        "scale_span_orders_of_magnitude": float(np.log10(max(SCALES) / min(SCALES))),
        "per_scale": rows,
        "measured_ordering_reference": {
            "denzler_mir122_measured_threshold_added_sites":
                cc.SCENARIOS["Denzler_miR122"]["measured_threshold_added_sites"],
            "bosson_mir92_25_measured_threshold_added_sites":
                cc.SCENARIOS["Bosson_miR92_25"]["measured_threshold_added_sites"],
            "measured_ordering_mir122_above_mir92_25": True,
        },
        "summary": {
            "n_scales": len(rows),
            "n_scales_both_finite": len(finite),
            "ordering_preserved_at_every_finite_scale": bool(ordering_all),
            "threshold_ratio_min": float(min(ratios)) if ratios else None,
            "threshold_ratio_max": float(max(ratios)) if ratios else None,
            "denzler_threshold_range": [float(min(d_thr)), float(max(d_thr))] if d_thr else None,
            "bosson_threshold_range": [float(min(b_thr)), float(max(b_thr))] if b_thr else None,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md5 = hashlib.md5(OUT_JSON.read_bytes()).hexdigest()

    print(f"swept absolute affinity scale over {out['scale_span_orders_of_magnitude']:.1f} "
          f"orders of magnitude ({len(rows)} scales)")
    print(f"{'scale x nom':>12} {'miR-122 thr':>14} {'miR-92/25 thr':>14} {'122>92/25':>10} "
          f"{'ratio':>8}")
    def fmt(v: float | None, spec: str) -> str:
        return format(v, spec) if v is not None else "None"

    for r in rows:
        d = r["denzler_mir122_operator_threshold_added_sites"]
        b = r["bosson_mir92_25_operator_threshold_added_sites"]
        rat = r["threshold_ratio_mir122_over_mir92_25"]
        order = "True" if r["ordering_mir122_above_mir92_25"] else "False"
        print(f"{r['affinity_scale_x_nominal']:>12.4g} {fmt(d, '.4g'):>14} "
              f"{fmt(b, '.4g'):>14} {order:>10} {fmt(rat, '.3g'):>8}")
    s = out["summary"]
    print(f"\nordering (miR-122 threshold > miR-92/25) preserved at every finite scale: "
          f"{s['ordering_preserved_at_every_finite_scale']} "
          f"({s['n_scales_both_finite']}/{s['n_scales']} scales finite); "
          f"ratio in [{s['threshold_ratio_min']:.3g}, {s['threshold_ratio_max']:.3g}]")
    print(f"wrote {OUT_JSON.relative_to(REPO)}  md5 {md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
