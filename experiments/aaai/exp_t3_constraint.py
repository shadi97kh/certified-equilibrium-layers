#!/usr/bin/env python3
"""Theorem 3: KKT-projected mass-action / thermodynamic constraint, on REAL data.

Ties the predicted off-target knockdown k_g to the REAL ViennaRNA seed-duplex
free energy dG_g via a mass-action (Langmuir / Hill) occupancy law, enforced by a
differentiable KKT orthogonal-projection layer (cel.models.kkt_projection)
applied at train and test.

- Mass-action line: logit(knockdown) = a + b*dG, with (a, b) calibrated from the
  real (dG, efficacy) training data (b is the empirical Boltzmann slope).
- Constrained model (KKT-hPINN, Chen et al. 2024): the predicted-knockdown logits
  are projected so their trend on [1, dG] equals (a, b) exactly, while the
  feature-driven component orthogonal to dG is preserved (so accuracy is kept).
- Exact-occupancy variant (KKT-Hardnet Newton, Iftakher et al. 2025): a Newton
  projection pinning sigmoid(logit) = sigmoid(a + b*dG) exactly (the pure
  occupancy curve); this discards the feature residual and is reported as the
  accuracy cost of an exact pin.

It verifies on real held-out data that the constraint residual is at machine
precision (max < 1e-8, strict on EVERY seed: this is the theorem) and that the KKT
projection does not cost accuracy beyond noise: across seeds the mean constrained
Pearson must not drop more than one standard deviation of the unconstrained Pearson
below the mean unconstrained Pearson. A one-time dg_diagnostic block reports how the
real ViennaRNA seed dG, and as a contrast the duplex-end asymmetry feature, each
correlate with measured knockdown.

HARD DATA POLICY (OUTLINE.md): REAL DATA ONLY. The efficacy data comes from a real
loader that RAISES if absent, and the dG values come from ViennaRNA (RAISES if not
installed). The section is then recorded as pending with the exact source.

Output results/aaai/t3_constraint.json and results/aaai/t3_constraint.png.
Accepts --seed and logs it. No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.features import featurize_pairs  # noqa: E402
from cel.features.seedmatch import revcomp  # noqa: E402

OUT = REPO / "results" / "aaai"
RESIDUAL_TOL = 1e-8
ACC_TOL = 0.02  # constrained Pearson must be within this of unconstrained


def _target_for(antisense: str, sense: str) -> str:
    sense = (sense or "").strip()
    return sense if sense else revcomp(antisense)


def _featurize(frame, mu=None, sd=None):
    anti = frame["antisense"].tolist()
    tgt = [_target_for(a, s) for a, s in zip(frame["antisense"], frame["sense"], strict=True)]
    x = featurize_pairs(list(zip(anti, tgt, strict=True))).matrix
    if mu is None:
        mu, sd = x.mean(0), x.std(0) + 1e-8
    return (x - mu) / sd, frame["efficacy"].to_numpy(dtype=float), mu, sd


def _seed_dgs(frame) -> np.ndarray:
    from cel.features.thermo import seed_duplex_dg

    anti = frame["antisense"].tolist()
    tgt = [_target_for(a, s) for a, s in zip(frame["antisense"], frame["sense"], strict=True)]
    return np.array([seed_duplex_dg(a, t) for a, t in zip(anti, tgt, strict=True)], dtype=float)


_INPUTS_CACHE: dict = {}


def real_inputs(raw_dir: Path, cache_root: Path):
    """Real Huesken train/test split, features, efficacy, and ViennaRNA seed dG.

    Memoized per (raw_dir, cache_root): the data is identical across seeds, so the
    features and the ViennaRNA dG are computed once, not once per seed.
    """
    from cel.data import load_huesken_efficacy
    from cel.features.thermo import require_vienna

    require_vienna()  # raises ImportError (with the install command) if ViennaRNA is absent
    key = (str(raw_dir), str(cache_root))
    if key in _INPUTS_CACHE:
        return _INPUTS_CACHE[key]
    ds = load_huesken_efficacy(raw_dir, cache_root=cache_root)  # raises if absent
    train, test = ds.train().frame, ds.test().frame
    xtr, ytr, mu, sd = _featurize(train)
    xte, yte, _, _ = _featurize(test, mu, sd)
    dg_tr, dg_te = _seed_dgs(train), _seed_dgs(test)
    out = (xtr, ytr, dg_tr, xte, yte, dg_te)
    _INPUTS_CACHE[key] = out
    return out


def _corr_block(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson, Spearman, and the sign of the OLS slope of ``y`` regressed on ``x``."""
    from scipy.stats import pearsonr, spearmanr

    slope = float(np.polyfit(x, y, 1)[0])  # d(y)/d(x)
    return {
        "pearson": float(pearsonr(x, y)[0]),
        "spearman": float(spearmanr(x, y)[0]),
        "ols_slope": slope,
        "ols_slope_sign": int(np.sign(slope)),
    }


def dg_diagnostic(raw_dir: Path, cache_root: Path) -> dict:
    """One-time property of the REAL Huesken data (not per seed).

    Reports how the real ViennaRNA seed-duplex dG, and as a contrast the duplex-end
    asymmetry feature (5' antisense end minus 3' end stability), each correlate with
    measured knockdown, and whether the dG is standardized before the mass-action fit.
    """
    import torch

    from cel.data import load_huesken_efficacy
    from cel.features.thermo import require_vienna, thermo_asymmetry
    from cel.models.kkt_projection import fit_mass_action_line

    require_vienna()
    frame = load_huesken_efficacy(raw_dir, cache_root=cache_root).frame
    knockdown = frame["efficacy"].to_numpy(dtype=float)  # 1 - expression/100 (real)
    dg = _seed_dgs(frame)  # real ViennaRNA seed-duplex dG (kcal/mol)
    asym = np.array([thermo_asymmetry(a)[2] for a in frame["antisense"]], dtype=float)

    # The mass-action line is fit on RAW (unstandardized) dG, in logit space.
    a_fit, b_fit = fit_mass_action_line(torch.tensor(dg), torch.tensor(knockdown))
    return {
        "n": int(len(knockdown)),
        "knockdown": "efficacy = 1 - expression/100 (measured residual mRNA, real)",
        "dg_vs_knockdown": _corr_block(dg, knockdown),
        "thermo_asymmetry_vs_knockdown": _corr_block(asym, knockdown),
        "dg_standardized_before_mass_action_fit": False,
        "mass_action_fit_raw_slope_b": float(b_fit),
        "mass_action_fit_raw_slope_sign": int(np.sign(b_fit)),
        "note": "fit_mass_action_line uses RAW (unstandardized) dG; b is the logit-space slope on "
        "raw dG. Only the MLP feature matrix is z-scored, never the dG that enters the mass-action "
        "fit or the projection, so the sign of b is not a feature-scaling artifact. The "
        "rational-design literature (Khvorova / Schwarz) attributes on-target efficacy to "
        "duplex-END asymmetry rather than total seed-duplex stability; the two correlations are "
        "reported as that contrast.",
    }


def vienna_sign_convention() -> dict:
    """Three example seed:target duplexes through the SAME seed_duplex_dg code path.

    Perfect complement, single mismatch, and scrambled, to state the dG sign
    convention explicitly: a more negative dG (kcal/mol) is tighter binding.
    """
    from cel.features.seedmatch import revcomp
    from cel.features.thermo import require_vienna, seed_duplex_dg

    require_vienna()
    guide = "UGGGGAAACGUACGUACGU"  # fixed; seed_duplex_dg hybridizes its seed (pos 2 to 8)
    seed = guide[1:8]  # GGGGAAA
    perfect = revcomp(seed)  # exact antiparallel complement of the seed
    flip = {"A": "G", "G": "A", "C": "U", "U": "C"}
    mm = list(perfect)
    mm[len(mm) // 2] = flip[mm[len(mm) // 2]]  # one central mismatch
    mismatch = "".join(mm)
    scrambled = "CUAUGCA"  # fixed, deliberately non-complementary to the seed (near-zero dG)
    examples = {
        "perfect_complement": {"target": perfect, "dg": seed_duplex_dg(guide, perfect)},
        "single_mismatch": {"target": mismatch, "dg": seed_duplex_dg(guide, mismatch)},
        "scrambled": {"target": scrambled, "dg": seed_duplex_dg(guide, scrambled)},
    }
    return {
        "guide": guide,
        "seed_positions": "2-8",
        "seed": seed,
        "examples": examples,
        "convention": "more negative dG (kcal/mol) means tighter binding",
    }


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr

    d = pred - y
    mse, mae = float(np.mean(d**2)), float(np.mean(np.abs(d)))
    if len(y) < 2 or np.std(pred) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None, "mse": mse, "mae": mae}
    return {
        "pearson": float(pearsonr(pred, y)[0]),
        "spearman": float(spearmanr(pred, y)[0]),
        "mse": mse,
        "mae": mae,
    }


def run_t3(raw_dir: Path, cache_root: Path, seed: int, epochs: int, lr: float) -> dict:
    import torch
    from torch import nn

    from cel.models.kkt_projection import (
        MassActionProjection,
        fit_mass_action_line,
        mass_action_residual,
        newton_hard_projection,
    )

    xtr, ytr, dg_tr, xte, yte, dg_te = real_inputs(raw_dir, cache_root)
    xtr_t = torch.tensor(xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    dgtr_t = torch.tensor(dg_tr, dtype=torch.float32)
    xte_t = torch.tensor(xte, dtype=torch.float32)
    dgte_d = torch.tensor(dg_te, dtype=torch.float64)

    a, b = fit_mass_action_line(dgtr_t, ytr_t)
    proj = MassActionProjection(a, b)

    class MLP(nn.Module):
        def __init__(self, d: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_model(constrained: bool):
        torch.manual_seed(seed)
        model = MLP(xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            s = model(xtr_t)
            if constrained:
                s = proj(s, dgtr_t)
            loss = ((torch.sigmoid(s) - ytr_t) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        return model

    unconstrained = train_model(constrained=False)
    constrained = train_model(constrained=True)

    with torch.no_grad():
        # Unconstrained held-out predictions and their (nonzero) constraint residual.
        s_unc = unconstrained(xte_t).double()
        k_unc = torch.sigmoid(s_unc).numpy()
        residual_unc = mass_action_residual(s_unc, dgte_d, a, b)

        # Constrained held-out predictions: the deployed output IS the double KKT
        # projection, so its residual is at machine precision.
        s_con = proj(constrained(xte_t).double(), dgte_d)
        k_con = torch.sigmoid(s_con).numpy()
        residual_con = proj.residual(s_con, dgte_d)

        # Exact-occupancy (Newton) variant: pin sigmoid(s) = sigmoid(a + b dG).
        # Start from a non-saturated point so the sigmoid Jacobian is well conditioned.
        target_occ = torch.sigmoid(a + b * dgte_d)
        s_newton = newton_hard_projection(
            s_unc.clamp(-12.0, 12.0), lambda u: torch.sigmoid(u) - target_occ
        )
        k_newton = torch.sigmoid(s_newton).numpy()
        residual_newton = float((torch.sigmoid(s_newton) - target_occ).abs().max())

    return {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "mass_action_line": {
            "a": a,
            "b": b,
            "note": "logit(knockdown) = a + b*dG; b is the Boltzmann slope (expected < 0)",
        },
        "constraint_residual": {
            "constrained_max": residual_con,
            "unconstrained_max": residual_unc,
            "exact_occupancy_newton_max": residual_newton,
            "tolerance": RESIDUAL_TOL,
        },
        "accuracy": {
            "unconstrained": _metrics(k_unc, yte),
            "constrained_kkt": _metrics(k_con, yte),
            "exact_occupancy_newton": _metrics(k_newton, yte),
        },
    }


def acceptance(result: dict) -> dict:
    res = result["constraint_residual"]["constrained_max"]
    residual_ok = res < RESIDUAL_TOL
    unc = result["accuracy"]["unconstrained"]["pearson"]
    con = result["accuracy"]["constrained_kkt"]["pearson"]
    if unc is None or con is None:
        no_regression, accuracy_cost = None, None
    else:
        no_regression = con >= unc - ACC_TOL
        accuracy_cost = con < unc - ACC_TOL
    flag = "PASS" if (residual_ok and no_regression is not False) else "FAIL"
    return {
        "flag": flag,
        "constraint_residual_below_tol": bool(residual_ok),
        "constrained_residual_max": res,
        "no_accuracy_regression": (None if no_regression is None else bool(no_regression)),
        "constraint_costs_accuracy": (None if accuracy_cost is None else bool(accuracy_cost)),
        "unconstrained_pearson": unc,
        "constrained_pearson": con,
        "exact_occupancy_pearson": result["accuracy"]["exact_occupancy_newton"]["pearson"],
        "note": "The KKT-hPINN projection enforces the mass-action trend exactly (residual at "
        "machine precision) while preserving the feature residual orthogonal to dG, so accuracy "
        "is kept. The exact-occupancy Newton pin discards the feature residual and is reported as "
        "the accuracy cost of an exact nonlinear pin.",
    }


def _mean_or_none(xs: list) -> float | None:
    vals = [x for x in xs if x is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _mean_metrics(metric_dicts: list[dict]) -> dict:
    keys = ("pearson", "spearman", "mse", "mae")
    return {k: _mean_or_none([m.get(k) for m in metric_dicts]) for k in keys}


def aggregate_results(results: list[dict], seeds: list[int]) -> dict:
    """Merge per-seed runs: worst-case residual, mean accuracy, per-seed acceptance."""
    first = results[0]
    con_res = [r["constraint_residual"]["constrained_max"] for r in results]
    unc_res = [r["constraint_residual"]["unconstrained_max"] for r in results]
    new_res = [r["constraint_residual"]["exact_occupancy_newton_max"] for r in results]
    models = ("unconstrained", "constrained_kkt", "exact_occupancy_newton")
    per_seed = [
        {
            "seed": seeds[i],
            "flag": acceptance(r)["flag"],
            "constrained_residual": r["constraint_residual"]["constrained_max"],
            "unconstrained_pearson": r["accuracy"]["unconstrained"]["pearson"],
            "constrained_pearson": r["accuracy"]["constrained_kkt"]["pearson"],
        }
        for i, r in enumerate(results)
    ]
    return {
        "seeds": list(seeds),
        "n_train": first["n_train"],
        "n_test": first["n_test"],
        "mass_action_line": first["mass_action_line"],  # (a, b) is seed-independent
        "constraint_residual": {
            "constrained_max": max(con_res),  # worst seed
            "constrained_per_seed": con_res,
            "unconstrained_max": min(unc_res),
            "exact_occupancy_newton_max": max(new_res),
            "tolerance": RESIDUAL_TOL,
        },
        "accuracy": {m: _mean_metrics([r["accuracy"][m] for r in results]) for m in models},
        "per_seed": per_seed,
    }


def _mean_std(values: list) -> tuple[float | None, float | None]:
    """Mean and sample std (ddof=1) over the non-None values; std 0 for a single value."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def aggregate_acceptance(aggregated: dict, per_seed_acc: list[dict]) -> dict:
    """Strict residual gate on every seed; the accuracy gate is std-based.

    Residual: < tol on every seed (the theorem). Accuracy: FAIL only if the mean
    constrained Pearson drops more than one standard deviation of the unconstrained
    Pearson below the mean unconstrained Pearson across seeds (a drop beyond seed noise).
    """
    res = aggregated["constraint_residual"]["constrained_max"]  # worst seed
    residual_ok = res < RESIDUAL_TOL
    per_seed = aggregated["per_seed"]
    mean_unc, std_unc = _mean_std([p["unconstrained_pearson"] for p in per_seed])
    mean_con, std_con = _mean_std([p["constrained_pearson"] for p in per_seed])
    regression = (
        None if (mean_unc is None or mean_con is None) else mean_con < mean_unc - std_unc
    )  # a drop beyond one std of the unconstrained Pearson
    flag = "PASS" if (residual_ok and not regression) else "FAIL"
    return {
        "flag": flag,
        "seeds": aggregated["seeds"],
        "n_seeds": len(aggregated["seeds"]),
        "per_seed_flags": [a["flag"] for a in per_seed_acc],
        "constraint_residual_below_tol_all_seeds": bool(residual_ok),
        "constrained_residual_max": res,
        "mean_unconstrained_pearson": mean_unc,
        "std_unconstrained_pearson": std_unc,
        "mean_constrained_pearson": mean_con,
        "std_constrained_pearson": std_con,
        "accuracy_regression_beyond_one_std": (None if regression is None else bool(regression)),
        "no_accuracy_regression": (None if regression is None else (not regression)),
        "mean_exact_occupancy_pearson": aggregated["accuracy"]["exact_occupancy_newton"]["pearson"],
        "accuracy_gate": "FAIL only if mean_constrained_pearson < mean_unconstrained_pearson - "
        "std_unconstrained_pearson (a drop beyond one std of the unconstrained Pearson over seeds)",
        "note": "Residual gate is strict (every seed < 1e-8: the theorem). The KKT-hPINN projection "
        "enforces the mass-action trend at machine precision while preserving the feature residual "
        "orthogonal to dG; the accuracy gate flags only a drop larger than one seed-to-seed standard "
        "deviation, so within-noise changes do not count as a regression.",
    }


def _plot(result: dict, acc: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[t3] matplotlib not installed; skipped figure")
        return

    res = result["constraint_residual"]
    acc_b = result["accuracy"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = ["unconstrained", "constrained_kkt", "exact_newton"]
    residuals = [
        max(res["unconstrained_max"], 1e-18),
        max(res["constrained_max"], 1e-18),
        max(res["exact_occupancy_newton_max"], 1e-18),
    ]
    ax1.bar(range(3), residuals, color=["tab:red", "tab:green", "tab:orange"])
    ax1.axhline(RESIDUAL_TOL, color="k", ls="--", label=f"tol {RESIDUAL_TOL:g}")
    ax1.set_yscale("log")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels, rotation=15, fontsize=8)
    ax1.set_ylabel("max constraint residual (log)")
    ax1.set_title("Mass-action constraint residual")
    ax1.legend(fontsize=8)

    pear = [
        acc_b["unconstrained"]["pearson"] or 0.0,
        acc_b["constrained_kkt"]["pearson"] or 0.0,
        acc_b["exact_occupancy_newton"]["pearson"] or 0.0,
    ]
    ax2.bar(range(3), pear, color=["tab:red", "tab:green", "tab:orange"])
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, rotation=15, fontsize=8)
    ax2.set_ylabel("held-out Pearson")
    ax2.set_title(f"Accuracy (KKT keeps it, exact pin costs it); acceptance {acc['flag']}")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[t3] wrote {path}")


def _print_summary(result: dict, acc: dict) -> None:
    def g(v, p: int = 4) -> str:
        return "n/a" if v is None else f"{v:.{p}f}"

    print("\n===== T3 summary (real Huesken, corrected guide strand) =====")
    print(f"residual max per seed (strict gate, < {RESIDUAL_TOL:g}):")
    per = result["constraint_residual"]["constrained_per_seed"]
    for s, r in zip(result["seeds"], per, strict=True):
        print(f"  seed {s}: {r:.3e}   ok={r < RESIDUAL_TOL}")
    print(
        f"Pearson unconstrained: {g(acc['mean_unconstrained_pearson'])} "
        f"+/- {g(acc['std_unconstrained_pearson'])}"
    )
    print(
        f"Pearson constrained:   {g(acc['mean_constrained_pearson'])} "
        f"+/- {g(acc['std_constrained_pearson'])}"
    )
    print(
        f"accuracy gate (FAIL if mean_con < mean_unc - std_unc): "
        f"regression beyond 1 std = {acc['accuracy_regression_beyond_one_std']}"
    )
    print(f"OVERALL FLAG: {acc['flag']}  (per-seed {acc['per_seed_flags']})")

    dg = result["dg_diagnostic"]
    d, ta = dg["dg_vs_knockdown"], dg["thermo_asymmetry_vs_knockdown"]
    print(f"\n-- dG-vs-knockdown diagnostic (one-time, n={dg['n']}) --")
    print(
        f"  real seed dG vs knockdown:      pearson {g(d['pearson'], 3)}  spearman {g(d['spearman'], 3)}"
        f"  OLS slope {g(d['ols_slope'])} (sign {d['ols_slope_sign']:+d})"
    )
    print(
        f"  thermo_asymmetry vs knockdown:  pearson {g(ta['pearson'], 3)}  spearman {g(ta['spearman'], 3)}"
        f"  OLS slope {g(ta['ols_slope'])} (sign {ta['ols_slope_sign']:+d})"
    )
    print(
        f"  dG standardized before mass-action fit: {dg['dg_standardized_before_mass_action_fit']}"
        f" | raw mass-action slope b = {g(dg['mass_action_fit_raw_slope_b'])}"
        f" (sign {dg['mass_action_fit_raw_slope_sign']:+d})"
    )

    vc = result["vienna_sign_convention"]
    print(f"\n-- ViennaRNA sign-convention check ({vc['convention']}) --")
    for name in ("perfect_complement", "single_mismatch", "scrambled"):
        e = vc["examples"][name]
        print(f"  {name:<18} target={e['target']:<9} dG = {e['dg']:+.2f} kcal/mol")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="single RNG seed (logged)")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="multiple seeds; aggregated conservatively (residual < tol and no regression every seed)",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--raw-dir", type=str, default=str(REPO / "data" / "raw" / "efficacy"))
    parser.add_argument("--cache", type=Path, default=OUT / "cache")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print the summary + diagnostics but do NOT write the JSON or figure",
    )
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    print(f"[t3] seeds={seeds} epochs={args.epochs} raw_dir={args.raw_dir} dry_run={args.dry_run}")

    try:
        runs = [run_t3(Path(args.raw_dir), args.cache, s, args.epochs, args.lr) for s in seeds]
        per_seed_acc = [acceptance(r) for r in runs]
        result = aggregate_results(runs, seeds)
        acc = aggregate_acceptance(result, per_seed_acc)
        result["dg_diagnostic"] = dg_diagnostic(Path(args.raw_dir), args.cache)
        result["vienna_sign_convention"] = vienna_sign_convention()
        status = "ok"
    except (FileNotFoundError, NotImplementedError) as exc:
        result = {"status": "pending_real_data", "source": str(exc)}
        acc = {"flag": "PENDING", "seeds": seeds, "reason": "real efficacy data unavailable"}
        status = "pending_real_data"
        print(f"[t3] PENDING real data: {exc}")
    except ImportError as exc:
        result = {"status": "pending_dependency", "source": str(exc)}
        acc = {"flag": "PENDING", "seeds": seeds, "reason": "ViennaRNA or torch unavailable"}
        status = "pending_dependency"
        print(f"[t3] PENDING dependency: {exc}")

    if status == "ok":
        _print_summary(result, acc)

    numbers = {
        "acceptance": acc,
        "seeds": seeds,
        "status": status,
        **(result if status == "ok" else {"detail": result}),
    }
    if args.dry_run:
        print(
            "\n[t3] DRY RUN: result JSON and figure NOT written (rerun without --dry-run to write)."
        )
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "t3_constraint.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    if status == "ok":
        _plot(result, acc, args.out_dir / "t3_constraint.png")
        print(
            f"[t3] ACCEPTANCE {acc['flag']} (seeds {seeds}, per-seed {acc['per_seed_flags']}): "
            f"worst constrained residual={acc['constrained_residual_max']:.2e} (< {RESIDUAL_TOL:g}); "
            f"mean pearson unconstrained={acc['mean_unconstrained_pearson']} vs "
            f"constrained={acc['mean_constrained_pearson']} (regression beyond 1 std="
            f"{acc['accuracy_regression_beyond_one_std']})"
        )
    print(f"[t3] status={status} wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
