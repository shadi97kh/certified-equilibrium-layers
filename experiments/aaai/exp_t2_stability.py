#!/usr/bin/env python3
"""Theorem 2: P_alpha contraction / stability on a REAL signed, directed GRN.

On the real signed/directed GRN (TRRUST v2 by default, or DoRothEA A-C) this:

1. computes the spectral radius rho(S) of the row-normalized signed transition and
   checks the Neumann series for P_alpha = alpha (I - (1-alpha) S)^{-1} converges,
   i.e. (1-alpha) rho(S) < 1;
2. numerically verifies the contraction bound by sampling many x, x' and measuring
   ||P_alpha x - P_alpha x'|| / ||x - x'||, and computes the TRUE operator norm
   ||P_alpha||_2 (the worst-case Lipschitz constant) by power iteration, comparing
   both against the certified constants;
3. characterizes oversmoothing vs alpha (Dirichlet energy of P_alpha x) and the
   signed spectral gap;
4. compares the symmetric-normalized variant (the undirected case, built from the
   same real edges; STRING if --string-links is given) to the signed/directed case
   to expose the bound's dependence on the directed / negative-edge spectrum.

The certified constant alpha / (1 - (1-alpha) rho(S)) is the EXACT, tight L2 bound
only for a symmetric S (||S||_2 = rho). For a signed, directed S, ||S||_2 >= rho,
and the rho-based L2 constant can be exceeded (non-normality). The valid L2 bound
uses ||S||_2 (regime alpha > 1 - 1/||S||_2); the inf-norm certified constant uses
||S||_inf <= 1 and holds for every alpha (P_alpha non-expansive in the inf-norm).
The acceptance reports which normalization / regime holds.

HARD DATA POLICY (OUTLINE.md): REAL DATA ONLY. The GRN comes from a real loader
that RAISES if absent (run scripts/download_grn.sh). No graph is generated.

Output results/aaai/t2_stability.json and results/aaai/t2_stability.png. Accepts
--seed or --seeds (multi-seed, aggregated conservatively: the certified bound must
hold on every seed) and logs them. No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.verify import stability as st  # noqa: E402

OUT = REPO / "results" / "aaai"
DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TOL = 0.02  # relative tolerance for "never exceeded"


def _json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def analyze_graph(S, label, alphas, samples, power_iters, probes, seed) -> dict:
    rho = st.spectral_radius(S)
    norm2 = st.operator_2norm(S)
    norm_inf = st.inf_operator_norm(S)
    gap = st.signed_spectral_gap(S)
    alpha_min_l2 = max(0.0, 1.0 - 1.0 / norm2) if norm2 > 0 else 0.0

    per_alpha = []
    for a in alphas:
        opn = st.resolvent_2norm(S, a, iters=power_iters, seed=seed)
        l2 = st.measure_lipschitz(S, a, n_samples=samples, seed=seed, ord=2)
        linf = st.measure_lipschitz(S, a, n_samples=samples, seed=seed, ord=float("inf"))
        dvar = st.dirichlet_smoothing(S, a, n_probes=probes, seed=seed)
        per_alpha.append(
            {
                "alpha": a,
                "neumann_converges": st.neumann_converges(rho, a),
                "certified_l2_rho": st.certified_constant(rho, a),
                "certified_l2_norm": st.certified_constant(norm2, a),
                "certified_inf": st.certified_constant(norm_inf, a),
                "operator_2norm": opn,
                "measured_l2_max_ratio": l2["max_ratio"],
                "measured_inf_max_ratio": linf["max_ratio"],
                "dirichlet_in": dvar["dirichlet_in"],
                "dirichlet_out": dvar["dirichlet_out"],
                "dirichlet_ratio": dvar["ratio"],
            }
        )
    return {
        "label": label,
        "n_nodes": int(S.shape[0]),
        "n_edges": int(S.nnz),
        "rho": rho,
        "operator_2norm": norm2,
        "inf_norm": norm_inf,
        "signed_spectral_gap": gap,
        "alpha_min_l2_norm_regime": alpha_min_l2,
        "per_alpha": per_alpha,
    }


def acceptance(directed: dict, symmetric: dict) -> dict:
    conv = [r for r in directed["per_alpha"] if r["neumann_converges"]]
    l2_rho_holds = all(r["operator_2norm"] <= r["certified_l2_rho"] * (1 + TOL) for r in conv)

    norm_regime = [r for r in directed["per_alpha"] if math.isfinite(r["certified_l2_norm"])]
    l2_norm_holds = (
        all(r["operator_2norm"] <= r["certified_l2_norm"] * (1 + TOL) for r in norm_regime)
        if norm_regime
        else None
    )

    inf_nonexpansive = directed["inf_norm"] <= 1.0 + 1e-9
    inf_holds = all(r["measured_inf_max_ratio"] <= r["certified_inf"] * (1 + TOL) for r in conv)

    sym_conv = [r for r in symmetric["per_alpha"] if r["neumann_converges"]]
    sym_rho_holds = all(r["operator_2norm"] <= r["certified_l2_rho"] * (1 + TOL) for r in sym_conv)

    max_op2 = max((r["operator_2norm"] for r in conv), default=None)
    if l2_rho_holds:
        flag = "PASS"
    elif inf_nonexpansive and inf_holds and sym_rho_holds:
        flag = "RESTRICTED"
    else:
        flag = "FAIL"
    return {
        "flag": flag,
        "l2_rho_constant_holds_directed": bool(l2_rho_holds),
        "l2_norm_constant_holds_directed_in_regime": (
            None if l2_norm_holds is None else bool(l2_norm_holds)
        ),
        "inf_norm_nonexpansive_directed": bool(inf_nonexpansive),
        "inf_norm_constant_holds_directed": bool(inf_holds),
        "symmetric_l2_rho_constant_holds": bool(sym_rho_holds),
        "max_operator_2norm_directed": max_op2,
        "rho_directed": directed["rho"],
        "operator_2norm_directed": directed["operator_2norm"],
        "inf_norm_directed": directed["inf_norm"],
        "alpha_min_l2_norm_regime": directed["alpha_min_l2_norm_regime"],
        "note": (
            "The rho-based L2 constant alpha/(1-(1-alpha)rho) is exact only for the symmetric "
            "(undirected) operator, where it holds tightly. On the real signed/directed graph "
            "||S||_2 >> rho (non-normality from directed regulation and negative edges), so the "
            "true ||P_alpha||_2 can exceed the rho-based constant. A certified bound DOES hold: in "
            "the inf-norm P_alpha is non-expansive for every alpha (||S||_inf <= 1), and the L2 "
            "||S||_2-based constant holds for alpha > 1 - 1/||S||_2. Reported, not hidden."
        ),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _mean_or_none(xs: list) -> float | None:
    vals = [x for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else None


def _aggregate_analyses(analyses: list[dict], seeds: list[int]) -> dict:
    """Merge per-seed analyses conservatively: worst-case (max) norms / ratios per alpha.

    The certified constants and graph invariants are seed-independent; the true
    operator norm and the sampled ratios are taken at their MAX across seeds so the
    acceptance check is "never exceeded on any seed".
    """
    first = analyses[0]
    per_alpha = []
    for i in range(len(first["per_alpha"])):
        rows = [a["per_alpha"][i] for a in analyses]
        per_alpha.append(
            {
                "alpha": rows[0]["alpha"],
                "neumann_converges": rows[0]["neumann_converges"],
                "certified_l2_rho": rows[0]["certified_l2_rho"],
                "certified_l2_norm": rows[0]["certified_l2_norm"],
                "certified_inf": rows[0]["certified_inf"],
                "operator_2norm": max(r["operator_2norm"] for r in rows),
                "measured_l2_max_ratio": max(r["measured_l2_max_ratio"] for r in rows),
                "measured_inf_max_ratio": max(r["measured_inf_max_ratio"] for r in rows),
                "dirichlet_in": _mean([r["dirichlet_in"] for r in rows]),
                "dirichlet_out": _mean([r["dirichlet_out"] for r in rows]),
                "dirichlet_ratio": _mean_or_none([r["dirichlet_ratio"] for r in rows]),
            }
        )
    agg = dict(first)
    agg["per_alpha"] = per_alpha
    agg["seeds"] = list(seeds)
    return agg


def _plot(directed: dict, symmetric: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[t2] matplotlib not installed; skipped figure")
        return

    alphas = [r["alpha"] for r in directed["per_alpha"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(
        alphas,
        [r["operator_2norm"] for r in directed["per_alpha"]],
        "o-",
        label="||P_a||_2 directed (true)",
    )
    ax1.plot(
        alphas,
        [r["certified_l2_rho"] for r in directed["per_alpha"]],
        "k--",
        label="certified L2 (rho)",
    )
    ax1.plot(
        alphas,
        [min(r["certified_inf"], 5) for r in directed["per_alpha"]],
        "g:",
        label="certified inf-norm (<=1)",
    )
    ax1.plot(
        alphas,
        [r["operator_2norm"] for r in symmetric["per_alpha"]],
        "s-",
        color="tab:blue",
        label="||P_a||_2 symmetric",
    )
    ax1.axvline(
        directed["alpha_min_l2_norm_regime"], color="r", ls=":", label="alpha_min (L2 norm regime)"
    )
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("L2 operator norm")
    ax1.set_title(
        f"P_alpha contraction: directed vs symmetric\n(rho={directed['rho']:.2f}, ||S||_2={directed['operator_2norm']:.2f})"
    )
    ax1.legend(fontsize=7)

    ax2.plot(alphas, [r["dirichlet_ratio"] for r in directed["per_alpha"]], "o-", label="directed")
    ax2.plot(
        alphas,
        [r["dirichlet_ratio"] for r in symmetric["per_alpha"]],
        "s-",
        color="tab:blue",
        label="symmetric",
    )
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("Dirichlet energy ratio E(P_a x)/E(x)")
    ax2.set_title("Oversmoothing vs alpha")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[t2] wrote {path}")


def _load_signed_grn(which: str, species: str):
    from cel.data import load_dorothea, load_trrust

    if which == "trrust":
        return load_trrust(species=species)
    if which == "dorothea":
        return load_dorothea()
    raise ValueError(f"--grn must be 'trrust' or 'dorothea'; got {which!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=0, help="single RNG seed for the probe vectors (logged)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="multiple seeds; aggregated conservatively (bound must hold on every seed)",
    )
    parser.add_argument("--grn", choices=("trrust", "dorothea"), default="trrust")
    parser.add_argument("--species", default="human")
    parser.add_argument(
        "--string-links",
        type=str,
        default=None,
        help="optional real STRING file for the symmetric comparison",
    )
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--power-iters", type=int, default=60)
    parser.add_argument("--probes", type=int, default=12)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    print(f"[t2] seeds={seeds} grn={args.grn} species={args.species} alphas={args.alphas}")

    grn = _load_signed_grn(args.grn, args.species)  # RAISES if the real file is absent
    print(f"[t2] real signed/directed GRN: {grn.n_nodes} nodes, {grn.n_edges} signed edges")
    sym_S = st.symmetric_normalized_variant(grn.S)

    def _runs(S, label):
        return [
            analyze_graph(S, label, args.alphas, args.samples, args.power_iters, args.probes, s)
            for s in seeds
        ]

    directed_runs = _runs(grn.S, f"{args.grn}_signed_directed")
    symmetric_runs = _runs(sym_S, f"{args.grn}_symmetric_normalized")
    directed = _aggregate_analyses(directed_runs, seeds)
    symmetric = _aggregate_analyses(symmetric_runs, seeds)
    per_seed_flags = [
        acceptance(directed_runs[i], symmetric_runs[i])["flag"] for i in range(len(seeds))
    ]

    string_block = {"status": "not_requested"}
    if args.string_links:
        from cel.data import load_string_undirected

        try:
            sgrn = load_string_undirected(args.string_links)
            string_block = _aggregate_analyses(_runs(sgrn.S, "string_undirected"), seeds)
            string_block["status"] = "ok"
        except FileNotFoundError as exc:
            string_block = {"status": "pending_real_data", "source": str(exc)}

    # Conservative verdict: acceptance on the worst-case (max-across-seeds) analyses.
    acc = acceptance(directed, symmetric)
    acc["seeds"] = list(seeds)
    acc["n_seeds"] = len(seeds)
    acc["per_seed_flags"] = per_seed_flags
    numbers = {
        "acceptance": acc,
        "seeds": seeds,
        "grn": grn.source,
        "neumann_condition": "(1-alpha) rho(S) < 1",
        "certified_constant_formula": "alpha / (1 - (1-alpha) spec); spec = rho(S) (symmetric, tight) "
        "or ||S||_2 (valid L2 bound) or ||S||_inf <= 1 (inf-norm, all alpha)",
        "signed_directed": directed,
        "symmetric_normalized": symmetric,
        "string_undirected": string_block,
        "comparison_note": "rho is equal for both graphs (row/symmetric normalization), but "
        "||S||_2 differs: the directed/signed graph is non-normal (||S||_2 >> rho), the symmetric "
        "variant has ||S||_2 = rho. This is the bound's dependence on the negative-edge spectrum.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "t2_stability.json"
    out_path.write_text(json.dumps(_json_safe(numbers), indent=2), encoding="utf-8")
    _plot(directed, symmetric, args.out_dir / "t2_stability.png")

    print(
        f"[t2] rho={directed['rho']:.4f} ||S||_2={directed['operator_2norm']:.4f} ||S||_inf={directed['inf_norm']:.4f}"
    )
    print(
        f"[t2] ACCEPTANCE {acc['flag']} (seeds {seeds}, per-seed {acc['per_seed_flags']}): "
        f"L2-rho holds={acc['l2_rho_constant_holds_directed']}, "
        f"inf-norm non-expansive={acc['inf_norm_nonexpansive_directed']}, "
        f"symmetric L2-rho holds={acc['symmetric_l2_rho_constant_holds']}"
    )
    print(f"[t2] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
