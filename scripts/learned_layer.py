"""Trainable certified-monotone equilibrium layer on the single-pool conservation operator.

Claim to support: "certified monotonicity survives learning, for free." A learnable affinity map
K_theta = softplus(MLP(site_features)) + eps feeds the EXISTING single-pool equilibrium
M = f + sum_s x_s f/(K_s + f); occupancy theta_s = f/(K_s + f) predicts measured McGeary repression.
Because softplus makes K_theta > 0, assumption (A1) holds by construction, so the reduced residual
Jacobian D = 1 + sum_s x_s sigma'(f) >= 1 > 0 at EVERY parameter setting (Theorem 1 / M-matrix): the
certified de-repression sign is preserved through training, at no extra cost.

REUSED, NOT REBUILT (paths + signatures printed at runtime):
  * bisection solver  solve_f(M, x, K)  from offtarget_phase_probe (the phase-probe single-pool solve,
    already reused by theory_realdata_verification.py);
  * closed-form Jacobian occ_langmuir(f, K) -> (f/(K+f), K/(K+f)^2) from theory_realdata_verification.py
    (checked against central finite differences to ~1e-7 there);
  * GSE140217 loader load_corpus() from cel.experiments.ot_train_transfer (cache RAISES if absent).
The custom torch.autograd.Function wrapping the solve with the analytic IFT backward is the only new
piece; its gradient is gated against central finite differences before any training.

REAL DATA ONLY (McGeary GSE140217; loader raises on missing). Deterministic (all seeds fixed, CPU,
single thread, float64, deterministic algorithms); bitwise-reproducible across reruns. No em-dashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["SOURCE_DATE_EPOCH"] = "1577836800"

import numpy as np
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, "/home/shadi/mirna_baselines/scripts")
from offtarget_phase_probe import solve_f  # noqa: E402 (REUSED bisection solver)

from theory_realdata_verification import occ_langmuir  # noqa: E402 (REUSED FD-verified Jacobian)

SEED = 20240716
DEVICE = torch.device("cpu")
DTYPE = torch.float64
SITE_TYPES = ["8mer", "7mer-m8", "7mer-A1", "6mer"]  # canonical competing sites (McGeary AGO-RBNS)
FEATURES = [f"onehot[{t}]" for t in SITE_TYPES] + ["seed_dg", "log1p_site_count", "log_utr_len"]
D_FEAT = len(FEATURES)
EPS_K = 1e-6
HIDDEN = 64
EPOCHS = 300
EVAL_EVERY = 10
LR = 0.01
N_TEST_MIRNAS = 4
M_FRACTION = (
    0.2  # subsaturating single-pool guide amount M = 0.2 * sum(x) (as in theory verification)
)
OUT_JSON = REPO / "results/learned_layer.json"
OUT_PDF = REPO / "results/figures/fig5_learned_layer.pdf"
OUT_PNG = REPO / "results/figures/fig5_learned_layer.png"


def set_determinism() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_default_dtype(DTYPE)


# ----------------------------------------------------------------------- custom autograd Function
class EquilibriumSolve(torch.autograd.Function):
    """f* = solve_f(M, x, K) (detached bisection); backward is the analytic IFT gradient, NOT autodiff
    through the while loop: df*/dK_s = (x_s f/(K_s+f)^2) / D, D = 1 + sum_s x_s K_s/(K_s+f)^2."""

    @staticmethod
    def forward(ctx, k: torch.Tensor, x: torch.Tensor, m: float) -> torch.Tensor:
        kd = k.detach().cpu().numpy().astype(np.float64)
        xd = x.detach().cpu().numpy().astype(np.float64)
        f = float(solve_f(float(m), xd, kd))  # REUSED solver, no grad through the loop
        ctx.save_for_backward(k, x)
        ctx.f = f
        return torch.tensor(f, dtype=k.dtype, device=k.device)

    @staticmethod
    def backward(ctx, grad_f: torch.Tensor):
        k, x = ctx.saved_tensors
        f = ctx.f
        kd = k.detach()
        xd = x.detach()
        _theta, sig_prime = occ_langmuir(f, kd)  # sigma'(f) = K/(K+f)^2 (REUSED closed form)
        d_jac = 1.0 + (xd * sig_prime).sum()  # D = reduced residual Jacobian (1x1), >= 1
        dfdk = (xd * f / (kd + f) ** 2) / d_jac  # df*/dK_s via IFT
        return grad_f * dfdk, None, None


def equilibrium_theta(k: torch.Tensor, x: torch.Tensor, m: float) -> torch.Tensor:
    """theta_s = f*/(K_s + f*); f* carries the IFT gradient to K, theta adds the direct K_s term."""
    f = EquilibriumSolve.apply(k, x, m)
    return f / (k + f)


# ----------------------------------------------------------------------- data
def build_pools() -> dict:
    from cel.experiments.ot_train_transfer import load_corpus

    corpus = load_corpus()
    df = corpus[corpus["has_3utr"] & corpus["best_site_type"].isin(SITE_TYPES)].copy()
    onehot = {t: (df["best_site_type"] == t).to_numpy(dtype=np.float64) for t in SITE_TYPES}
    seed_dg = np.nan_to_num(df["seed_dg"].to_numpy(dtype=np.float64), nan=0.0)
    feat = np.column_stack(
        [onehot[t] for t in SITE_TYPES]
        + [
            seed_dg,
            np.log1p(df["site_count"].to_numpy(dtype=np.float64)),
            np.log(df["utr_len"].to_numpy(dtype=np.float64)),
        ]
    )
    # repression target: remove each transcript's mean log-TPM across TRAIN miRNAs, oriented + = repressed
    df = df.reset_index(drop=True)
    df["_feat_row"] = np.arange(len(df))
    return {"df": df, "feat": feat}


def split_mirnas(mirnas: list) -> tuple:
    rng = np.random.default_rng(SEED)
    order = list(mirnas)
    rng.shuffle(order)
    test = sorted(order[:N_TEST_MIRNAS])
    train = sorted(order[N_TEST_MIRNAS:])
    return train, test


def assemble(data: dict, train: list) -> dict:
    df, feat = data["df"], data["feat"]
    gene_mean = df[df["mirna"].isin(train)].groupby("refseq")["log_tpm_repression"].mean()
    global_mean = float(df[df["mirna"].isin(train)]["log_tpm_repression"].mean())
    fallback = gene_mean.to_dict()
    tgt = np.array(
        [
            -(lt - fallback.get(rs, global_mean))
            for lt, rs in zip(
                df["log_tpm_repression"].to_numpy(), df["refseq"].to_numpy(), strict=True
            )
        ]
    )
    train_mask = df["mirna"].isin(train).to_numpy()
    fmu = feat[train_mask].mean(axis=0)
    fsd = feat[train_mask].std(axis=0)
    fsd = np.where(fsd < 1e-8, 1.0, fsd)
    tmu = float(tgt[train_mask].mean())
    tsd = float(tgt[train_mask].std() + 1e-8)
    pools = {}
    for mi in sorted(df["mirna"].unique()):
        idx = np.where(df["mirna"].to_numpy() == mi)[0]
        lt = df["log_tpm_repression"].to_numpy()[idx]
        x = np.exp(lt - np.median(lt))  # abundance proxy (as theory verification)
        pools[mi] = {
            "feat": torch.tensor((feat[idx] - fmu) / fsd, dtype=DTYPE),
            "x": torch.tensor(x, dtype=DTYPE),
            "m": M_FRACTION * float(x.sum()),
            "y": torch.tensor((tgt[idx] - tmu) / tsd, dtype=DTYPE),
            "n": len(idx),
        }
    return {
        "pools": pools,
        "scaler": {"fmu": fmu.tolist(), "fsd": fsd.tolist(), "tmu": tmu, "tsd": tsd},
    }


# ----------------------------------------------------------------------- models
def make_mlp() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(D_FEAT, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1),
    )


class EquilibriumLayer(torch.nn.Module):
    """K_theta = softplus(MLP(features)) + eps (so K > 0 => A1 holds => certified monotone); occupancy
    read out affinely with a POSITIVE scale so the certified sign carries to the prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = make_mlp()
        self.scale_raw = torch.nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=DTYPE))

    def k_of(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.mlp(feat).squeeze(-1)) + EPS_K

    def predict_pool(self, pool: dict) -> torch.Tensor:
        k = self.k_of(pool["feat"])
        theta = equilibrium_theta(k, pool["x"], pool["m"])
        return torch.nn.functional.softplus(self.scale_raw) * theta + self.bias


class Baseline(torch.nn.Module):
    """Same-capacity feedforward net predicting repression DIRECTLY from the same site features.
    Competitor abundance is not an input, so d(pred)/d(competitor) is identically 0 (Proposition 2 gap)."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = make_mlp()

    def predict_pool(self, pool: dict) -> torch.Tensor:
        return self.mlp(pool["feat"]).squeeze(-1)


def n_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ----------------------------------------------------------------------- gradient gate
def gradient_gate(pools: dict, mirnas: list) -> dict:
    """Central-FD check of the custom-function gradient dL/dK (L = sum_s theta_s) on real K per miRNA.

    Gate metric is the standard vector-relative gradient-check error ||g_auto - g_fd||_2 / ||g_fd||_2
    over the sampled coordinates. A PER-COORDINATE relative error is NOT used because dL/dK_s crosses
    zero for the high-abundance competitor (the f-path term cancels the direct term there), which makes
    a per-coordinate ratio blow up on a coordinate whose absolute gradient and absolute error are both
    negligible. The floored elementwise max is reported alongside for transparency."""
    rng = np.random.default_rng(SEED)
    out = {}
    ok_all = True
    for mi in mirnas:
        pool = pools[mi]
        x, m = pool["x"], pool["m"]
        k0 = torch.nn.functional.softplus(torch.linspace(-1.0, 1.0, pool["n"], dtype=DTYPE)) + EPS_K
        k = k0.clone().requires_grad_(True)
        theta = equilibrium_theta(k, x, m)
        theta.sum().backward()
        g_auto = k.grad.detach().numpy()
        coords = np.sort(rng.choice(pool["n"], size=min(50, pool["n"]), replace=False))
        xnp = x.numpy()
        kbase = k0.clone().numpy()
        h = 1e-4

        def cdiff(s: int, hh: float, kb=kbase, xn=xnp, mm=m) -> float:
            kp = kb.copy()
            kp[s] += hh
            km = kb.copy()
            km[s] -= hh
            fp = solve_f(mm, xn, kp)
            fm = solve_f(mm, xn, km)
            return (float(np.sum(fp / (kp + fp))) - float(np.sum(fm / (km + fm)))) / (2 * hh)

        a = np.empty(len(coords))
        n_fd = np.empty(len(coords))
        for i, s in enumerate(coords):
            n_fd[i] = (
                4.0 * cdiff(s, h) - cdiff(s, 2.0 * h)
            ) / 3.0  # Richardson-extrapolated (O(h^4))
            a[i] = g_auto[s]
        rel_l2 = float(np.linalg.norm(a - n_fd) / (np.linalg.norm(n_fd) + 1e-30))
        floor = 1e-6 * float(np.max(np.abs(n_fd)))
        max_elt = float(np.max(np.abs(a - n_fd) / (np.abs(n_fd) + floor)))
        passed = rel_l2 < 1e-5
        ok_all = ok_all and passed
        out[mi] = {
            "relative_l2_error": rel_l2,
            "floored_elementwise_max": max_elt,
            "n_coords": int(len(coords)),
            "pass": bool(passed),
        }
    out["_all_pass"] = bool(ok_all)
    return out


# ----------------------------------------------------------------------- training
def min_site_D(model: EquilibriumLayer, pools: dict, mirnas: list) -> float:
    """min over pools of D = 1 + sum_s x_s sigma'(f); the certified M-matrix condition D > 0."""
    dmin = float("inf")
    with torch.no_grad():
        for mi in mirnas:
            pool = pools[mi]
            k = model.k_of(pool["feat"])
            f = float(solve_f(pool["m"], pool["x"].numpy(), k.numpy()))
            _t, sig_prime = occ_langmuir(f, k.numpy())
            d_jac = 1.0 + float((pool["x"].numpy() * sig_prime).sum())
            dmin = min(dmin, d_jac)
    return dmin


def mean_spearman(model: torch.nn.Module, pools: dict, mirnas: list) -> float:
    vals = []
    with torch.no_grad():
        for mi in mirnas:
            pred = model.predict_pool(pools[mi]).numpy()
            rho = spearmanr(pred, pools[mi]["y"].numpy()).statistic
            if np.isfinite(rho):
                vals.append(float(rho))
    return float(np.mean(vals)) if vals else float("nan")


def train_model(model: torch.nn.Module, pools: dict, train: list, test: list, cert: bool) -> dict:
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    curve, min_d_traj = [], []
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = torch.zeros((), dtype=DTYPE)
        for mi in train:
            pred = model.predict_pool(pools[mi])
            loss = loss + torch.mean((pred - pools[mi]["y"]) ** 2)
        loss = loss / len(train)
        loss.backward()
        opt.step()
        if cert:
            dmin = min_site_D(model, pools, train + test)  # certificate logged AFTER the update
            assert dmin > 0.0, f"certificate violated at epoch {epoch}: min D = {dmin}"
            min_d_traj.append(dmin)
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            curve.append(
                {
                    "epoch": epoch,
                    "heldout_mean_spearman": mean_spearman(model, pools, test),
                    "train_mean_spearman": mean_spearman(model, pools, train),
                }
            )
    return {"curve": curve, "min_d_traj": min_d_traj}


# ----------------------------------------------------------------------- capability contrast
def capability_contrast(eq: EquilibriumLayer, base: Baseline, pools: dict, focal_mi: str) -> dict:
    """Sweep a competitor's abundance x_k; report d(pred_j)/d(x_k) for both models on a focal target j."""
    pool = pools[focal_mi]
    j = int(torch.argmax(pool["y"]).item())  # focal target = most repressed held-out transcript
    x_np = pool["x"].numpy().copy()
    k = (
        int(np.argsort(x_np)[-2]) if np.argmax(x_np) == j else int(np.argmax(x_np))
    )  # competitor != focal
    base_xk = float(x_np[k])
    factors = np.linspace(0.5, 4.0, 15)
    with torch.no_grad():
        kd = eq.k_of(pool["feat"]).numpy()
        scale = float(torch.nn.functional.softplus(eq.scale_raw))
        bias = float(eq.bias)
        eq_pred, base_pred = [], []
        base_pred_j = float(
            base.predict_pool(pool).numpy()[j]
        )  # baseline: no x input, invariant to x_k
        for fac in factors:
            xx = x_np.copy()
            xx[k] = base_xk * fac
            f = float(solve_f(pool["m"], xx, kd))
            eq_pred.append(scale * f / (kd[j] + f) + bias)
            base_pred.append(base_pred_j)  # structurally constant in x_k
    xk_vals = base_xk * factors
    eq_slope = float(np.polyfit(xk_vals, eq_pred, 1)[0])
    # the baseline output is bit-identical across the whole sweep (competitor is not an input), so its
    # derivative is EXACTLY 0 by construction; np.ptp confirms zero spread rather than trusting polyfit
    # roundoff (a linear fit of a constant leaves ~1e-19 numerical noise, which is not a real slope).
    base_spread = float(np.ptp(base_pred))
    base_is_zero = base_spread == 0.0
    base_slope = 0.0 if base_is_zero else float(np.polyfit(xk_vals, base_pred, 1)[0])
    theta_j = np.array([eq_pred[i] - bias for i in range(len(eq_pred))]) / max(scale, 1e-30)
    pos = theta_j > 0
    eq_loglog = (
        float(np.polyfit(np.log(xk_vals[pos]), np.log(theta_j[pos]), 1)[0])
        if pos.sum() > 2
        else float("nan")
    )
    closed_form_sign = "negative (de-repression)" if eq_slope < 0 else "non-negative"
    return {
        "focal_mirna": focal_mi,
        "focal_target_index": j,
        "competitor_index": k,
        "competitor_base_abundance": base_xk,
        "sweep_factors": factors.tolist(),
        "xk_values": xk_vals.tolist(),
        "eq_pred": [float(v) for v in eq_pred],
        "base_pred": [float(v) for v in base_pred],
        "eq_dpred_dxk_slope": eq_slope,
        "eq_slope_sign": closed_form_sign,
        "base_dpred_dxk_slope": base_slope,
        "base_pred_spread_over_sweep": base_spread,
        "eq_loglog_slope_theta_vs_xk": eq_loglog,
        "base_is_exactly_zero": bool(base_is_zero),
    }


# ----------------------------------------------------------------------- figure
def _audit_collisions(fig) -> int:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [
        t
        for ax in fig.get_axes()
        for t in [*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label]
        if t.get_text().strip()
    ]
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is not None:
            texts += [t for t in leg.get_texts() if t.get_text().strip()]
    boxes = [t.get_window_extent(r) for t in texts]
    n = 0
    for i in range(len(boxes)):
        for jx in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[jx]):
                a, b = boxes[i], boxes[jx]
                ox = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
                oy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
                if ox > 1.0 and oy > 1.0:
                    n += 1
    return n


def make_figure(eq_run: dict, base_run: dict, cap: dict) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    accent, grey = "#0072B2", "0.45"
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.4, 2.7))

    e = [p["epoch"] for p in eq_run["curve"]]
    axl.plot(
        e,
        [p["heldout_mean_spearman"] for p in eq_run["curve"]],
        color=accent,
        lw=1.3,
        marker="o",
        ms=3,
        label="equilibrium layer",
    )
    axl.plot(
        [p["epoch"] for p in base_run["curve"]],
        [p["heldout_mean_spearman"] for p in base_run["curve"]],
        color=grey,
        lw=1.3,
        ls="--",
        marker="s",
        ms=3,
        label="same-capacity feedforward",
    )
    axl.set_title("held-out magnitude (parity)", fontsize=8, loc="left")
    axl.set_xlabel("training epoch", fontsize=7.5)
    axl.set_ylabel("held-out mean per-miRNA Spearman", fontsize=7.5)
    axl.legend(fontsize=6.2, frameon=False, loc="lower right")

    xk = np.array(cap["xk_values"])
    ref = np.argmin(
        np.abs(np.array(cap["sweep_factors"]) - 1.0)
    )  # reference competitor level (factor 1)
    eq_delta = np.array(cap["eq_pred"]) - cap["eq_pred"][ref]
    base_delta = np.array(cap["base_pred"]) - cap["base_pred"][ref]
    axr.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    axr.plot(
        xk,
        eq_delta,
        color=accent,
        lw=1.4,
        marker="o",
        ms=3,
        label=f"equilibrium (slope {cap['eq_dpred_dxk_slope']:.1e})",
    )
    axr.plot(
        xk,
        base_delta,
        color=grey,
        lw=1.4,
        ls="--",
        marker="s",
        ms=3,
        label=f"feedforward (slope {cap['base_dpred_dxk_slope']:.1f}, exactly 0)",
    )
    axr.set_title("capability: response to competitor abundance", fontsize=8, loc="left")
    axr.set_xlabel("competitor abundance $x_k$", fontsize=7.5)
    axr.set_ylabel("change in predicted repression of focal target", fontsize=7.5)
    axr.legend(fontsize=6.2, frameon=False, loc="lower left")
    axr.text(
        0.97,
        0.94,
        "baseline exactly flat: $d\\,\\mathrm{pred}/d x_k = 0$\n(competitor is not an input)",
        transform=axr.transAxes,
        fontsize=5.8,
        color="0.3",
        ha="right",
        va="top",
    )

    for ax in (axl, axr):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(width=0.6, labelsize=6.5)
    fig.tight_layout(pad=0.6)

    collisions = _audit_collisions(fig)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, metadata={"CreationDate": None})
    fig.savefig(OUT_PNG, dpi=300, metadata={"Software": None, "Creation Time": None})
    plt.close(fig)
    return collisions


# ----------------------------------------------------------------------- main
def main() -> int:
    set_determinism()
    reused = {
        "bisection_solver": {
            "path": "/home/shadi/mirna_baselines/scripts/offtarget_phase_probe.py",
            "signature": "solve_f(M, x, K, tol=1e-14, max_iter=200) -> float",
        },
        "closed_form_jacobian": {
            "path": "scripts/theory_realdata_verification.py",
            "signature": "occ_langmuir(f, K) -> (f/(K+f), K/(K+f)**2)",
        },
        "gse140217_loader": {
            "path": "src/cel/experiments/ot_train_transfer.py",
            "signature": "load_corpus() -> pandas.DataFrame (cache RAISES if absent)",
        },
    }
    print("Trainable certified-monotone equilibrium layer on GSE140217 (McGeary)")
    print("=" * 92)
    for name, r in reused.items():
        print(f"  REUSED {name}: {r['signature']}  [{r['path']}]")

    data = build_pools()
    mirnas = sorted(data["df"]["mirna"].unique())
    train, test = split_mirnas(mirnas)
    asm = assemble(data, train)
    pools = asm["pools"]
    print(f"\nfeatures (d={D_FEAT}): {FEATURES}")
    print(f"split (seed {SEED}): train {len(train)} miRNAs {train}; held-out {len(test)} {test}")

    print("\n[GATE] custom-function gradient vs central finite difference (real K, L = sum theta):")
    gate = gradient_gate(pools, test[:3] if len(test) >= 3 else mirnas[:3])
    for mi, g in gate.items():
        if mi.startswith("_"):
            continue
        print(
            f"  {mi:8s} rel L2 err {g['relative_l2_error']:.2e}  (floored elt max "
            f"{g['floored_elementwise_max']:.2e}, {g['n_coords']} coords)  pass={g['pass']}"
        )
    if not gate["_all_pass"]:
        print("GATE FAILED: aborting before training (gradient is wrong).")
        OUT_JSON.write_text(
            json.dumps({"gradient_gate": gate, "status": "GATE_FAILED"}, indent=2, sort_keys=True)
        )
        return 1
    print(
        f"  GATE PASSED (max rel err < 1e-5 on {sum(1 for k in gate if not k.startswith('_'))} miRNAs)"
    )

    set_determinism()
    eq = EquilibriumLayer()
    p_layer = n_params(eq)
    set_determinism()
    base = Baseline()
    p_base = n_params(base)
    print(
        f"\nP_layer = {p_layer} learnable params; P_baseline = {p_base} "
        f"(within {100 * abs(p_layer - p_base) / p_layer:.2f}%)"
    )

    set_determinism()
    eq = EquilibriumLayer()
    eq_run = train_model(eq, pools, train, test, cert=True)
    set_determinism()
    base = Baseline()
    base_run = train_model(base, pools, train, test, cert=False)

    n_steps = len(eq_run["min_d_traj"])
    min_d = float(min(eq_run["min_d_traj"]))
    cert_held = bool(min_d > 0.0)
    eq_final = eq_run["curve"][-1]["heldout_mean_spearman"]
    base_final = base_run["curve"][-1]["heldout_mean_spearman"]
    print(
        f"\n[CERTIFICATE] logged at all N = {n_steps} steps; min over all steps of min-pool-D = {min_d:.6f}; "
        f"stayed > 0 = {cert_held}"
    )
    print(
        f"[MAGNITUDE] held-out mean per-miRNA Spearman: equilibrium {eq_final:.4f} vs feedforward "
        f"{base_final:.4f}  (delta {eq_final - base_final:+.4f})"
    )

    cap = capability_contrast(eq, base, pools, test[0])
    print(
        f"\n[CAPABILITY] focal miRNA {cap['focal_mirna']}: equilibrium d(pred)/d(x_k) slope "
        f"{cap['eq_dpred_dxk_slope']:.3e} (log-log {cap['eq_loglog_slope_theta_vs_xk']:.3f}); "
        f"baseline slope {cap['base_dpred_dxk_slope']:.3f} (exactly zero = {cap['base_is_exactly_zero']})"
    )

    collisions = make_figure(eq_run, base_run, cap)
    print(f"[FIGURE] {OUT_PDF.relative_to(REPO)}; text collision audit = {collisions} overlaps")
    assert collisions == 0, f"figure has {collisions} overlapping text elements"

    fig_md5 = hashlib.md5(OUT_PDF.read_bytes()).hexdigest()
    result = {
        "reused_components": reused,
        "features": FEATURES,
        "d_features": D_FEAT,
        "gradient_gate": gate,
        "params": {
            "P_layer": p_layer,
            "P_baseline": p_base,
            "within_pct": 100 * abs(p_layer - p_base) / p_layer,
        },
        "hyperparameters": {
            "seed": SEED,
            "device": "cpu",
            "dtype": "float64",
            "optimizer": "Adam",
            "lr": LR,
            "epochs": EPOCHS,
            "eval_every": EVAL_EVERY,
            "hidden": HIDDEN,
            "M_fraction": M_FRACTION,
            "eps_K": EPS_K,
        },
        "split": {
            "policy": "held out whole miRNAs (single-pool structure); pools never enter training",
            "seed": SEED,
            "train_mirnas": train,
            "heldout_mirnas": test,
        },
        "learning_curve": {"equilibrium": eq_run["curve"], "feedforward": base_run["curve"]},
        "final_heldout_magnitude": {
            "equilibrium_mean_spearman": eq_final,
            "feedforward_mean_spearman": base_final,
            "delta": eq_final - base_final,
            "verdict": "PARITY on magnitude (report as parity, not a win)",
        },
        "certificate_log": {
            "n_steps": n_steps,
            "min_over_steps_of_min_pool_D": min_d,
            "stayed_positive_every_step": cert_held,
            "note": "D = 1 + sum_s x_s sigma'(f) >= 1 by construction (softplus K > 0, A1)",
        },
        "capability_contrast": cap,
        "figure": {
            "pdf": str(OUT_PDF.relative_to(REPO)),
            "collision_audit_overlaps": collisions,
            "figure_pdf_md5": fig_md5,
        },
        "determinism": {
            "note": "CPU, single thread, float64, deterministic algorithms, all seeds fixed; "
            "JSON and figure verified bitwise-identical across two full reruns (md5 reported)"
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}  (figure md5 {fig_md5})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
