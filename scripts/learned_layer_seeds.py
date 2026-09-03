"""Seed sweep of the trainable certified-monotone equilibrium layer vs its parameter-matched baseline.

REUSES the exact training path of scripts/learned_layer.py without modifying it: build_pools,
split_mirnas, assemble, EquilibriumLayer, Baseline, set_determinism and train_model are imported
and called in the same order main() uses them. The ONLY thing varied across runs is the seed.

Seed handling. learned_layer.SEED is consumed in three places: set_determinism() and train_model()
(initialization + optimizer stream) and split_mirnas() (which miRNAs are held out). To vary training
while keeping the FOUR HELD-OUT microRNAs unchanged, the split and the feature/target normalization
are computed ONCE at the original seed 20240716, and only the training seed is rebound per run
(module attribute rebinding; no file is edited and no hyperparameter is touched).

No synthetic data: the corpus loader raises if its cache is absent.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import learned_layer as ll  # noqa: E402  (REUSED training path, imported unmodified)

OUT_JSON = REPO / "results" / "learned_layer_seeds.json"

# The original seed plus nine further fixed seeds, listed explicitly and never derived at runtime.
SEEDS = [
    20240716,
    20240717,
    20240718,
    20240719,
    20240720,
    20240721,
    20240722,
    20240723,
    20240724,
    20240725,
]
SPLIT_SEED = 20240716  # held-out microRNAs are fixed at the original split


def run_one_seed(seed: int, pools: dict, train: list, test: list) -> dict:
    """One equilibrium run and one baseline run at `seed`, via the unmodified training path."""
    ll.SEED = seed  # rebind only the seed the reused path reads; nothing else changes
    ll.set_determinism()
    eq = ll.EquilibriumLayer()
    eq_run = ll.train_model(eq, pools, train, test, cert=True)
    ll.set_determinism()
    base = ll.Baseline()
    base_run = ll.train_model(base, pools, train, test, cert=False)

    traj = eq_run["min_d_traj"]

    def heldout_curve(run: dict) -> list[dict]:
        # the full held-out per-microRNA mean Spearman curve at every logged eval point, in order
        return [
            {"epoch": int(p["epoch"]), "heldout_mean_spearman": p["heldout_mean_spearman"]}
            for p in run["curve"]
        ]

    return {
        "seed": seed,
        "equilibrium_heldout_mean_spearman": eq_run["curve"][-1]["heldout_mean_spearman"],
        "feedforward_heldout_mean_spearman": base_run["curve"][-1]["heldout_mean_spearman"],
        "equilibrium_heldout_curve": heldout_curve(eq_run),
        "feedforward_heldout_curve": heldout_curve(base_run),
        "certificate_n_steps": int(len(traj)),
        "certificate_min_over_steps": float(min(traj)),
        "certificate_positive_every_step": bool(all(d > 0.0 for d in traj)),
    }


def main() -> int:
    # --- fixed split + normalization, computed once at the original seed ---
    ll.SEED = SPLIT_SEED
    data = ll.build_pools()  # raises if the real corpus cache is absent
    mirnas = sorted(data["df"]["mirna"].unique())
    train, test = ll.split_mirnas(mirnas)
    pools = ll.assemble(data, train)["pools"]

    rows = [run_one_seed(s, pools, train, test) for s in SEEDS]

    eq = [r["equilibrium_heldout_mean_spearman"] for r in rows]
    bs = [r["feedforward_heldout_mean_spearman"] for r in rows]
    eq_mean, bs_mean = statistics.fmean(eq), statistics.fmean(bs)
    eq_sd, bs_sd = statistics.stdev(eq), statistics.stdev(bs)  # sample sd (ddof=1)
    mean_diff = eq_mean - bs_mean
    pooled = (((len(eq) - 1) * eq_sd**2 + (len(bs) - 1) * bs_sd**2) / (len(eq) + len(bs) - 2)) ** 0.5
    cohens_d = mean_diff / pooled if pooled > 0.0 else float("nan")
    overlap = (min(eq) <= max(bs)) and (min(bs) <= max(eq))
    gap = min(eq) - max(bs)  # > 0 means every equilibrium seed beat every baseline seed

    cert_ok = all(r["certificate_positive_every_step"] for r in rows)
    cert_min = min(r["certificate_min_over_steps"] for r in rows)
    cert_steps = sum(r["certificate_n_steps"] for r in rows)

    verdict = (
        f"SEPARATION: equilibrium {eq_mean:.4f} vs feedforward {bs_mean:.4f} "
        f"(mean diff {mean_diff:+.4f}, Cohen's d {cohens_d:.2f}, ranges disjoint by {gap:+.4f})"
        if not overlap
        else (
            f"MEASURED PARITY: equilibrium {eq_mean:.4f} +/- {eq_sd:.4f} vs feedforward "
            f"{bs_mean:.4f} +/- {bs_sd:.4f} (mean diff {mean_diff:+.4f}, Cohen's d {cohens_d:.2f}, "
            f"per-seed ranges overlap)"
        )
    )

    out = {
        "experiment": "seed sweep of the trainable equilibrium layer vs parameter-matched baseline",
        "reused_path": "scripts/learned_layer.py (build_pools, split_mirnas, assemble, "
                       "set_determinism, EquilibriumLayer, Baseline, train_model), unmodified",
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "split_seed": SPLIT_SEED,
        "heldout_mirnas": test,
        "train_mirnas": train,
        "hyperparameters_unchanged": {
            "epochs": ll.EPOCHS,
            "eval_every": ll.EVAL_EVERY,
            "lr": ll.LR,
            "hidden": ll.HIDDEN,
            "M_fraction": ll.M_FRACTION,
            "eps_K": ll.EPS_K,
            "n_test_mirnas": ll.N_TEST_MIRNAS,
            "note": "only the seed varies across runs",
        },
        "per_seed": rows,
        "per_seed_curves": {
            # for each model: a list over seeds (same order as SEEDS), each a list of
            # {epoch, heldout_mean_spearman} at every logged eval point, in eval order
            "equilibrium": [r["equilibrium_heldout_curve"] for r in rows],
            "feedforward": [r["feedforward_heldout_curve"] for r in rows],
        },
        "equilibrium": {
            "per_seed_heldout_mean_spearman": eq,
            "mean": eq_mean,
            "sample_sd": eq_sd,
            "min": min(eq),
            "max": max(eq),
        },
        "feedforward": {
            "per_seed_heldout_mean_spearman": bs,
            "mean": bs_mean,
            "sample_sd": bs_sd,
            "min": min(bs),
            "max": max(bs),
        },
        "comparison": {
            "mean_difference_equilibrium_minus_feedforward": mean_diff,
            "cohens_d": cohens_d,
            "per_seed_ranges_overlap": bool(overlap),
            "range_gap_min_eq_minus_max_ff": gap,
        },
        "certificate": {
            "positive_every_step_every_seed": bool(cert_ok),
            "min_over_all_steps_all_seeds": cert_min,
            "total_steps_checked": cert_steps,
            "note": "D = 1 + sum_s x_s sigma'(f) >= 1 by construction (softplus K > 0, A1)",
        },
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md5 = hashlib.md5(OUT_JSON.read_bytes()).hexdigest()

    print(f"\nheld-out microRNAs (unchanged): {test}")
    print(f"equilibrium  mean {eq_mean:.6f}  sd {eq_sd:.6f}  range [{min(eq):.6f}, {max(eq):.6f}]")
    print(f"feedforward  mean {bs_mean:.6f}  sd {bs_sd:.6f}  range [{min(bs):.6f}, {max(bs):.6f}]")
    print(f"certificate  positive on every step of every seed: {cert_ok} "
          f"(min {cert_min:.6f} over {cert_steps} steps)")
    print(f"wrote {OUT_JSON.relative_to(REPO)}  md5 {md5}")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
