"""GFlowNet guide proposal with a T4-certified CASCADE-efficacy reward (BioSeq-GFN-AL).

The reward for a 19nt guide is its CASCADE predicted on-target efficacy GATED by the T4
certificate: reward(g) = efficacy(g) if the certified off-target upper bound U(g) <= tau,
else 0. A GFlowNet (BioSeq-GFN-AL, Jain et al. 2022) trained with this reward samples
DIVERSE high-efficacy guides that all satisfy the off-target certificate, exploring the
full 4^19 guide space rather than the scaffold-locked 4^7 seed enumeration of Step 6
(cel.design.t5_certified_select). The script compares diversity-at-fixed-certified
-bound between the two.

What is exact vs learned: the gate U(g) <= tau is an EXACT certificate (it uses the same
||P_alpha(mouse TRRUST)||_2 ||e_seed||_2 upper bound as Step 6), so every guide the
GFlowNet is rewarded for is certified off-target <= tau. The efficacy is a CASCADE
prediction (real Huesken-trained), and the GFlowNet's diversity / coverage is a property
of the learned sampler, not a guarantee.

REAL DATA ONLY. The reward oracle (efficacy model + certified U) is built entirely from
real data: real Huesken efficacy, real mouse TRRUST, real GRCm39/104 3'UTRs, real
ViennaRNA. BioSeq-GFN-AL is imported lazily from external/BioSeq-GFN-AL; when it is not
cloned / importable the GFlowNet arm is recorded pending (no synthetic fallback) while the
enumeration baseline still runs. Writes results/t5_gflownet.json. No em-dashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from cel.design.t5_certified_select import (  # noqa: E402
    ALPHA,
    TTR_D1,
    TTR_SEED,
    _mouse_node_utrs,
    _revcomp_rna,
    _site_dna,
    _site_sumsq,
    efficacy_predictor,
    enumerate_seeds,
    guide_for,
)

RESULTS = REPO / "results"
EXTERNAL_GFN = REPO / "external" / "BioSeq-GFN-AL"
GUIDE_LEN = 19
SEED_START = 1  # 0-indexed start of the seed (guide positions 2-8)
SEED_LEN = 7
ALPHABET = "ACGU"
DEFAULT_ETA = 0.7
DEFAULT_N_SAMPLES = 2000


def _seed_U(seeds: list[str], p_norm: float, s2: dict) -> np.ndarray:
    """Certified off-target upper bound U for each seed (same formula as Step 6)."""
    import RNA

    u = np.zeros(len(seeds))
    for i, sd in enumerate(seeds):
        dg = float(RNA.duplexfold(sd, _revcomp_rna(sd)).energy)
        u[i] = p_norm * (-min(dg, 0.0)) * float(np.sqrt(s2.get(_site_dna(sd), 0.0)))
    return u


def build_context() -> dict:
    """Build the reward components once: efficacy model + off-target machinery (real data).

    Returns a context with predict / p_norm / s2 (the mouse TRRUST P_alpha norm and the
    GRCm39/104 3'UTR seed-site abundance). Building this once keeps per-guide scoring to a
    duplexfold plus a featurization.
    """
    from cel.data import load_trrust
    from cel.verify import offtarget_bound as ob

    grn = load_trrust(species="mouse")
    p_norm = float(ob.p_alpha_norm(grn.S, ALPHA))
    s2 = _site_sumsq(_mouse_node_utrs(grn))
    predict, _cal_pred, _cal_y = efficacy_predictor()
    return {"predict": predict, "p_norm": p_norm, "s2": s2}


def make_reward(ctx: dict, tau: float):
    """The certified-efficacy reward: efficacy gated by U(g) <= tau (0 where U(g) > tau)."""

    def reward(guides: list[str]) -> np.ndarray:
        guides = list(guides)
        seeds = [g[SEED_START : SEED_START + SEED_LEN] for g in guides]
        u = _seed_U(seeds, ctx["p_norm"], ctx["s2"])
        eff = ctx["predict"](guides)
        return np.where(u <= tau, eff, 0.0)

    return reward


def build_reward(tau: float):
    """Convenience: (reward, ctx) for a given tau (builds the context once)."""
    ctx = build_context()
    return make_reward(ctx, tau), ctx


def _hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b, strict=True))


def diversity(guides: list[str], max_pairs: int = 5000, seed: int = 0) -> dict:
    """Sequence diversity: unique count, distinct seeds, mean pairwise Hamming distance."""
    uniq = sorted(set(guides))
    n = len(uniq)
    n_seeds = len({g[SEED_START : SEED_START + SEED_LEN] for g in uniq})
    if n < 2:
        return {
            "n_guides": len(guides),
            "n_unique": n,
            "n_distinct_seeds": n_seeds,
            "mean_pairwise_hamming": 0.0,
        }
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        dists = [_hamming(uniq[i], uniq[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        rng = np.random.default_rng(seed)
        dists = []
        while len(dists) < max_pairs:
            i, j = int(rng.integers(n)), int(rng.integers(n))
            if i != j:
                dists.append(_hamming(uniq[i], uniq[j]))
    return {
        "n_guides": len(guides),
        "n_unique": n,
        "n_distinct_seeds": n_seeds,
        "mean_pairwise_hamming": float(np.mean(dists)),
    }


def enumeration_baseline(ctx: dict, tau: float, eta: float, n_seeds: int | None = None) -> dict:
    """Step 6 enumeration: certified-feasible seeds on the Ttr scaffold and their diversity."""
    seeds = enumerate_seeds(n_seeds)
    guides = [guide_for(TTR_D1, s) for s in seeds]
    u = _seed_U(seeds, ctx["p_norm"], ctx["s2"])
    eff = ctx["predict"](guides)
    within_bound = u <= tau
    feasible = within_bound & (eff >= eta)
    return {
        "n_candidates": len(seeds),
        "n_within_certified_bound": int(within_bound.sum()),
        "n_certified_feasible": int(feasible.sum()),
        "diversity_within_bound": diversity([guides[i] for i in np.where(within_bound)[0]]),
        "diversity_certified_feasible": diversity([guides[i] for i in np.where(feasible)[0]]),
        "note": "scaffold-locked (Ttr D1): guides differ only in the 7nt seed, so the maximum "
        "pairwise Hamming distance is 7 (the seed positions).",
    }


def gflownet_propose(reward, n_samples: int, n_train_steps: int = 5000, seed: int = 0) -> list[str]:
    """Sample diverse 19nt guides from BioSeq-GFN-AL trained with the reward oracle.

    Thin adapter over BioSeq-GFN-AL (cloned to external/BioSeq-GFN-AL). The repo is research
    code, so confirm the trainer / generator entry point against your checkout and wire it
    in the marked block; the reward callable scores fixed-length ACGU token strings. Raises
    ImportError when the repo or its modules are not importable (the GFlowNet arm is then
    recorded pending; no synthetic fallback).
    """
    if not EXTERNAL_GFN.exists():
        raise ImportError(
            f"BioSeq-GFN-AL not found at {EXTERNAL_GFN}; clone it "
            "(git clone https://github.com/MJ10/BioSeq-GFN-AL.git) into external/"
        )
    if str(EXTERNAL_GFN) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_GFN))
    try:
        import torch  # noqa: F401

        # Wire your BioSeq-GFN-AL trainer here: construct a GFlowNet over the ACGU
        # vocabulary with max_len=GUIDE_LEN, train it for n_train_steps against `reward`
        # (the certified-efficacy oracle), then draw n_samples and return the token
        # strings. The exact module / class names depend on the checkout, so they are
        # imported (and will raise ImportError if the entry point differs) rather than
        # hard-coded against a single version.
        from lib.generator import build_gflownet_sampler  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"could not import the BioSeq-GFN-AL trainer from {EXTERNAL_GFN} ({exc}); "
            "adapt gflownet_propose to your checkout's generator entry point"
        ) from exc

    sampler = build_gflownet_sampler(
        alphabet=ALPHABET, max_len=GUIDE_LEN, reward=reward, n_train_steps=n_train_steps, seed=seed
    )
    return [str(g) for g in sampler.sample(n_samples)]


def run(
    tau: float | None = None,
    eta: float = DEFAULT_ETA,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_train_steps: int = 5000,
    n_seeds: int | None = None,
) -> dict:
    # Default tau: the parent Ttr guide's certified U (off-target no worse than the parent).
    ctx = build_context()
    u_parent = float(_seed_U([TTR_SEED], ctx["p_norm"], ctx["s2"])[0])
    tau = float(u_parent if tau is None else tau)
    reward = make_reward(ctx, tau)

    enum = enumeration_baseline(ctx, tau, eta, n_seeds)

    try:
        sampled = gflownet_propose(reward, n_samples, n_train_steps)
        seeds = [g[SEED_START : SEED_START + SEED_LEN] for g in sampled]
        u = _seed_U(seeds, ctx["p_norm"], ctx["s2"])
        eff = ctx["predict"](sampled)
        within = u <= tau
        within_guides = [sampled[i] for i in np.where(within)[0]]
        gfn = {
            "status": "ok",
            "n_sampled": len(sampled),
            "n_within_certified_bound": int(within.sum()),
            "mean_efficacy_within_bound": float(eff[within].mean()) if within.any() else None,
            "diversity_within_bound": diversity(within_guides),
        }
        e_div = enum["diversity_within_bound"]["mean_pairwise_hamming"]
        g_div = gfn["diversity_within_bound"]["mean_pairwise_hamming"]
        comparison = {
            "enumeration_mean_pairwise_hamming": e_div,
            "gflownet_mean_pairwise_hamming": g_div,
            "gflownet_more_diverse_at_fixed_bound": bool(g_div > e_div),
            "note": "both restricted to U(g) <= tau; the enumeration is scaffold-locked "
            "(Hamming <= 7), the GFlowNet explores the full 19nt space.",
        }
    except ImportError as exc:
        gfn = {"status": "pending_real_data", "source": str(exc)}
        comparison = None

    return {
        "status": "ok",
        "scaffold": TTR_D1,
        "guide_len": GUIDE_LEN,
        "reward": "CASCADE predicted efficacy gated by the T4 certificate: 0 if U(g) > tau",
        "assumptions": {
            "tau": tau,
            "tau_note": "parent Ttr guide certified U" if tau == u_parent else "user tau",
            "eta": eta,
            "U_definition": "U(g) = ||P_alpha(mouse TRRUST)||_2 ||e_seed||_2, exact upper bound",
        },
        "enumeration_baseline": enum,
        "gflownet": gfn,
        "comparison_diversity_at_fixed_bound": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--n-train-steps", type=int, default=5000)
    parser.add_argument(
        "--n-seeds", type=int, default=None, help="enumeration subset for a fast run"
    )
    parser.add_argument("--out", type=Path, default=RESULTS / "t5_gflownet.json")
    args = parser.parse_args()

    numbers = run(
        tau=args.tau,
        eta=args.eta,
        n_samples=args.n_samples,
        n_train_steps=args.n_train_steps,
        n_seeds=args.n_seeds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    enum = numbers["enumeration_baseline"]
    print(
        f"[t5-gflownet] tau={numbers['assumptions']['tau']:.3f} eta={numbers['assumptions']['eta']}"
    )
    print(
        f"[t5-gflownet] enumeration: {enum['n_within_certified_bound']} within bound, "
        f"{enum['n_certified_feasible']} certified-feasible | within-bound mean Hamming "
        f"{enum['diversity_within_bound']['mean_pairwise_hamming']:.3f} over "
        f"{enum['diversity_within_bound']['n_unique']} guides"
    )
    gfn = numbers["gflownet"]
    if gfn["status"] == "ok":
        print(
            f"[t5-gflownet] GFlowNet: {gfn['n_within_certified_bound']}/{gfn['n_sampled']} within "
            f"bound | mean Hamming {gfn['diversity_within_bound']['mean_pairwise_hamming']:.3f} | "
            f"more diverse at fixed bound: "
            f"{numbers['comparison_diversity_at_fixed_bound']['gflownet_more_diverse_at_fixed_bound']}"
        )
    else:
        print(f"[t5-gflownet] GFlowNet: PENDING ({gfn['source'][:80]}...)")
    print(f"[t5-gflownet] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
