"""Robustness verification of the certified comparative-statics theory on the FULL real McGeary
miRNA corpus, at scale. Load-bearing verification for the paper's central claim.

Real data only, no synthetic, no em-dashes. Reuses solve_f (single-pool equilibrium) from the
mirna_baselines phase probe, and the real McGeary GSE140217 corpus (17 miRNAs, real 3'UTR seed
complementarity + measured log-TPM abundance). GSE140218 HEK293 is NOT on disk and is logged as
unavailable, not substituted.

Each McGeary miRNA is transfected ALONE, so the system is a SINGLE POOL. For a single pool the
ceRNA comparative static is affinity-form-independent and UNCONDITIONALLY certified:
    df/dx_s = -theta_s / D,   dtheta_j/dx_s = theta_j'(f) * df/dx_s < 0,
    D = 1 + sum_t x_t theta_t'(f) > 0  (the reduced residual Jacobian, a 1x1 M-matrix).
So adding competitor abundance de-represses shared partners with a certified (negative) sign, for
ANY increasing bounded occupancy.

DEMOTION NOTE (honesty): because the single-pool residual Jacobian is DIAGONAL with entries D >= 1,
it is a nonsingular M-matrix by inspection and the sign is unconditional. The counts "M-matrix holds
N/N" and "sign holds N/N" therefore CANNOT fail and validate nothing; they are recorded only as the
single-pool corollary (trivial_by_construction) and are EXCLUDED from the evidence. The two
NON-VACUOUS validations this script reports are: (1) the closed-form comparative static agrees with a
numerical finite-difference re-solve of the equilibrium (relative error ~1e-7 to 1e-6), which
validates the analytic FORMULA; and (2) that agreement is occupancy-form-independent (Langmuir, Hill
h=2, saturating-exponential). The boundary where the sign could actually flip is the MULTI-pool
shared-AGO coupling of the phase probe, which single-transfection data does not contain.

scanMiR 1.16.0 + scanMiRData are installed and DO contain these miRNAs (they are scanMiR's source
McGeary AGO-RBNS data). The verdict is provably K-source-independent (holds for any K>0), so the
real per-transcript affinities from the corpus (real seed complementarity + McGeary site-type Kd)
carry it; scanMiR per-site Kd would give quantitatively similar K and the identical verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, "/home/shadi/mirna_baselines/scripts")
from offtarget_phase_probe import solve_f  # noqa: E402

RESULTS = REPO / "results"
OUT = RESULTS / "theory_realdata_verification.json"
K_BY_SITE = {"8mer": 0.02, "7mer-m8": 0.06, "7mer-A1": 0.10, "6mer": 0.40}
CANONICAL = set(K_BY_SITE)


# Occupancy forms: theta(f, K) increasing in f, and dtheta/df (all > 0).
def occ_langmuir(f, K):
    return f / (K + f), K / (K + f) ** 2


def occ_hill2(f, K):
    d = K**2 + f**2
    return f**2 / d, 2.0 * f * K**2 / d**2


def occ_satexp(f, K):
    e = np.exp(-f / K)
    return 1.0 - e, e / K


OCC = {"langmuir": occ_langmuir, "hill_h2": occ_hill2, "satexp": occ_satexp}


def solve_general(M, x, K, occ, tol=1e-13, max_iter=200):
    """Solve M = f + sum_t x_t theta(f, K_t) for f > 0 (monotone; guarded Newton)."""
    lo, hi, f = 0.0, float(M), 0.5 * float(M)
    for _ in range(max_iter):
        th, dth = occ(f, K)
        g = f + float(np.sum(x * th)) - M
        d = 1.0 + float(np.sum(x * dth))
        hi, lo = (f, lo) if g > 0 else (hi, f)
        fn = f - g / d
        if not (lo < fn < hi):
            fn = 0.5 * (lo + hi)
        if abs(fn - f) <= tol * (1.0 + abs(fn)):
            return fn
        f = fn
    return f


def verify_mirna(K, x, M, occ_name):
    """Verify the certified ceRNA comparative static (closed-form vs numerical) for one miRNA."""
    occ = OCC[occ_name]
    f0 = solve_f(M, x, K) if occ_name == "langmuir" else solve_general(M, x, K, occ)
    th0, dth0 = occ(f0, K)
    D = 1.0 + float(np.sum(x * dth0))  # reduced residual Jacobian (1x1); D > 0 => M-matrix

    # Perturb the single highest-abundance competitor s; measure de-repression of all partners.
    s = int(np.argmax(x))
    eps = 1e-5 * max(x[s], 1.0)
    x2 = x.copy()
    x2[s] += eps
    f1 = solve_f(M, x2, K) if occ_name == "langmuir" else solve_general(M, x2, K, occ)
    th1, _ = occ(f1, K)
    dtheta_num = (th1 - th0) / eps  # numerical dtheta_j/dx_s
    dfdx = -th0[s] / D  # closed form df/dx_s
    dtheta_cf = dth0 * dfdx  # closed form dtheta_j/dx_s

    partners = np.arange(len(K)) != s
    certified_sign = bool(np.all(dtheta_num[partners] <= 1e-15))  # all partners de-repressed
    m_matrix = bool(D > 0.0)  # reduced Jacobian M-matrix
    # closed form matches numerical (relative error on the response vector)
    denom = np.abs(dtheta_cf) + 1e-30
    max_rel = float(np.max(np.abs(dtheta_num - dtheta_cf) / denom))
    sign_match = bool(np.all(np.sign(dtheta_num[partners]) == np.sign(dtheta_cf[partners])))
    return {
        "n_targets": int(len(K)),
        "f": f0,
        "D": D,
        "m_matrix": m_matrix,
        "certified_sign": certified_sign,
        "closed_form_sign_match": sign_match,
        "max_rel_err_cf_vs_num": max_rel,
        "mean_derepression": float(np.mean(dtheta_num[partners])),
    }


def run() -> dict:
    from cel.experiments.ot_train_transfer import load_corpus

    corpus = load_corpus()
    corpus = corpus[corpus["best_site_type"].isin(CANONICAL)].copy()
    corpus["K"] = corpus["best_site_type"].map(K_BY_SITE)
    corpus["x"] = np.exp(
        corpus["log_tpm_repression"] - corpus["log_tpm_repression"].median()
    )  # real abundance

    mirnas = sorted(corpus["mirna"].unique())
    langmuir_rows, form_rows = {}, {f: {} for f in OCC}
    for mi in mirnas:
        sub = corpus[corpus["mirna"] == mi]
        K = sub["K"].to_numpy(dtype=float)
        x = sub["x"].to_numpy(dtype=float)
        if len(K) < 5:
            continue
        M = 0.2 * float(x.sum())  # subsaturating single-pool guide amount
        langmuir_rows[mi] = verify_mirna(K, x, M, "langmuir")
        for fname in OCC:
            form_rows[fname][mi] = verify_mirna(K, x, M, fname)

    N = len(langmuir_rows)
    D_values = [r["D"] for r in langmuir_rows.values()]

    # NON-VACUOUS validation 1: closed-form comparative static vs numerical finite difference.
    rel = np.array([r["max_rel_err_cf_vs_num"] for r in langmuir_rows.values()])
    cf_all_agree = bool(np.all(rel < 1e-3))
    cf_disagreements = {
        mi: r["max_rel_err_cf_vs_num"]
        for mi, r in langmuir_rows.items()
        if r["max_rel_err_cf_vs_num"] >= 1e-3
    }

    # NON-VACUOUS validation 2: form-independence (Langmuir, Hill h=2, saturating-exponential).
    form_frac = {}
    for fname in OCC:
        rr = form_rows[fname]
        rel_f = np.array([r["max_rel_err_cf_vs_num"] for r in rr.values()])
        form_frac[fname] = {
            "n": len(rr),
            "cf_vs_fd_max_rel_err": float(rel_f.max()),
            "cf_vs_fd_all_agree_below_1e-3": bool(np.all(rel_f < 1e-3)),
        }
    form_independent = all(form_frac[f]["cf_vs_fd_all_agree_below_1e-3"] for f in OCC)

    return {
        "verification": "certified comparative-statics theory on the real McGeary corpus at scale",
        "dataset": "McGeary GSE140217 HeLa (real 3'UTR seed complementarity + measured log-TPM)",
        "GSE140218_HEK293": "NOT on disk; logged as unavailable, not substituted",
        "N_mirnas": N,
        "scanMiR_note": "scanMiR 1.16.0 + scanMiRData installed and DO contain these miRNAs (their "
        "McGeary AGO-RBNS source); verdict is provably K-source-independent (holds for any K>0), so "
        "the corpus per-transcript affinities carry it. K by McGeary site-type ordering.",
        "trivial_by_construction": {
            "label": "Corollary (single-pool), UNCONDITIONAL; NOT an empirical verification, EXCLUDED from evidence",
            "statement": "In the single-pool model the residual Jacobian is DIAGONAL with entries "
            "D = 1 + sum_s x_s sigma'(f) >= 1 > 0, so it is a nonsingular M-matrix by inspection for "
            "every miRNA and any K > 0; the ceRNA de-repression sign dtheta_j/dx_s = "
            "sigma_j'(f) * (-theta_s / D) < 0 is likewise unconditional. A 'M-matrix holds N/N' or "
            "'sign holds N/N' count cannot fail and validates nothing. It is recorded here only as "
            "the single-pool corollary and is excluded from the evidence tables.",
            "D_min": float(min(D_values)),
            "D_all_ge_1": bool(all(d >= 1.0 for d in D_values)),
        },
        "validation_closed_form_vs_finite_diff": {
            "description": "NON-VACUOUS. The closed-form comparative static dtheta_j/dx_s = "
            "sigma_j'(f) * (-theta_s / D) is checked against a numerical finite-difference re-solve "
            "of the single-pool equilibrium. This validates the analytic FORMULA, not the sign.",
            "n_mirnas": N,
            "relative_error_min": float(rel.min()),
            "relative_error_median": float(np.median(rel)),
            "relative_error_max": float(rel.max()),
            "all_agree_below_1e-3": cf_all_agree,
            "disagreements_rel_err_ge_1e-3": cf_disagreements,
        },
        "validation_form_independence": {
            "description": "NON-VACUOUS. The closed-form-vs-finite-difference agreement holds under "
            "Langmuir, Hill (h=2), and saturating-exponential occupancy, so the derivation is not "
            "Langmuir-specific.",
            "by_form": form_frac,
            "form_independent_all_forms_agree": form_independent,
        },
        "boundary_note": "The sign could only actually flip under the MULTI-pool shared-AGO coupling "
        "(phase probe Layer 2c); single-miRNA transfection data does not contain it. Absent a clean "
        "measured guide-vs-guide loading dataset, no coupled run is attempted here.",
        "per_mirna_langmuir": langmuir_rows,
    }


def main() -> int:
    r = run()
    OUT.write_text(json.dumps(r, indent=2), encoding="utf-8")
    N = r["N_mirnas"]
    print("Certified comparative-statics theory on the real McGeary corpus (GSE140217 HeLa)")
    print(f"  N miRNAs = {N}  (GSE140218 HEK293 unavailable on disk)")
    print(f"  {'miRNA':<10}{'n_tgt':>7}{'D':>10}{'cf_relerr':>12}{'mean_derep':>12}")
    for mi, row in r["per_mirna_langmuir"].items():
        print(
            f"  {mi:<10}{row['n_targets']:>7}{row['D']:>10.3g}"
            f"{row['max_rel_err_cf_vs_num']:>12.2e}{row['mean_derepression']:>12.2e}"
        )
    tc = r["trivial_by_construction"]
    print(
        f"  TRIVIAL-BY-CONSTRUCTION (excluded from evidence): single-pool Jacobian is diagonal, "
        f"D_min={tc['D_min']:.3g} (all >= 1). 'M-matrix holds N/N' / 'sign holds N/N' cannot fail; not a verification."
    )
    v1 = r["validation_closed_form_vs_finite_diff"]
    print(
        f"  VALIDATION 1 (closed-form vs finite-diff): rel err {v1['relative_error_min']:.1e} to "
        f"{v1['relative_error_max']:.1e} (median {v1['relative_error_median']:.1e}); all agree < 1e-3 = {v1['all_agree_below_1e-3']}"
    )
    v2 = r["validation_form_independence"]
    print(
        "  VALIDATION 2 (form-independence, cf-vs-fd max rel err): "
        + ", ".join(f"{k}={v['cf_vs_fd_max_rel_err']:.1e}" for k, v in v2["by_form"].items())
        + f"  all-forms-agree={v2['form_independent_all_forms_agree']}"
    )
    dis = v1["disagreements_rel_err_ge_1e-3"]
    print(f"  CF disagreements (rel err >= 1e-3): {dis if dis else 'none'}")
    print(f"[theory-realdata] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
