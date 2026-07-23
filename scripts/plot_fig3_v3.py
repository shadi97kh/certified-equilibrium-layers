"""Two-panel Figure 3 (v3): seed-band left panel + the unchanged competitor-abundance right panel.

Same as v2 with two readability fixes and no other change: (1) the right-panel title is raised with
extra pad so it does not overlap the 1e-5 y-axis offset label; (2) the left-panel y-axis lower limit
is set to include the full seed min-to-max band at every epoch (the wide early-epoch range reaches
down to about 0.013 and was previously clipped).

Matches the fig5_learned_layer layout of scripts/learned_layer.py (same figsize, colors, serif fonts,
pdf.fonttype 42, axes.linewidth 0.6, font sizes, spine/tick style). The LEFT panel is the ten-seed
version (mean held-out per-microRNA Spearman as a solid line per model, with the seed-wise
min-to-max range as a shaded band). The RIGHT panel is reproduced verbatim from the existing
single-seed capability sweep recorded in results/learned_layer.json.

Honesty gate: the left-panel bands must overlap at EVERY eval point; if any point fails, the script
stops and reports rather than draw. No em-dashes anywhere. Deterministic: fixed inputs and the
PDF/PNG are saved with creation timestamps stripped, so the bytes are identical across runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS_JSON = REPO / "results" / "learned_layer_seeds.json"
SINGLE_JSON = REPO / "results" / "learned_layer.json"
FIGDIR = REPO / "results" / "figures"
OUT_PDF = FIGDIR / "fig5_learned_layer_v3.pdf"
OUT_PNG = FIGDIR / "fig5_learned_layer_v3.png"

# Same two colors as scripts/learned_layer.py.
ACCENT, GREY = "#0072B2", "0.45"


def model_band(curves: list[list[dict]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return shared epochs and the across-seed mean, min, max of held-out Spearman."""
    epochs = [[p["epoch"] for p in c] for c in curves]
    if len({tuple(e) for e in epochs}) != 1:
        raise ValueError("seed curves do not share identical eval epochs; cannot align a band")
    ep = np.asarray(epochs[0], dtype=float)
    mat = np.asarray([[p["heldout_mean_spearman"] for p in c] for c in curves], dtype=float)
    return ep, mat.mean(axis=0), mat.min(axis=0), mat.max(axis=0)


def main() -> int:
    for p in (SEEDS_JSON, SINGLE_JSON):
        if not p.is_file():
            raise FileNotFoundError(f"missing required artifact {p}")
    seeds = json.loads(SEEDS_JSON.read_text(encoding="utf-8"))
    psc = seeds["per_seed_curves"]
    ep_e, mean_e, lo_e, hi_e = model_band(psc["equilibrium"])
    ep_f, mean_f, lo_f, hi_f = model_band(psc["feedforward"])
    if not np.array_equal(ep_e, ep_f):
        raise ValueError("equilibrium and feedforward eval epochs differ; cannot co-plot")

    overlap = (lo_e <= hi_f) & (lo_f <= hi_e)
    if not bool(overlap.all()):
        bad = [int(ep_e[i]) for i in np.where(~overlap)[0]]
        print(f"STOP: seed min-to-max bands do NOT overlap at eval epochs {bad}.")
        print("Refusing to draw a figure that would misrepresent the reported parity.")
        return 1
    print(f"left-panel bands overlap at all {len(overlap)} eval points; proceeding.")

    cap = json.loads(SINGLE_JSON.read_text(encoding="utf-8"))["capability_contrast"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    })
    accent, grey = ACCENT, GREY
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.4, 2.7))

    # ---- LEFT: ten-seed band ----
    axl.fill_between(ep_e, lo_e, hi_e, color=accent, alpha=0.18, lw=0.0)
    axl.fill_between(ep_f, lo_f, hi_f, color=grey, alpha=0.18, lw=0.0)
    axl.plot(ep_e, mean_e, color=accent, lw=1.3, marker="o", ms=3, label="equilibrium layer")
    axl.plot(ep_f, mean_f, color=grey, lw=1.3, marker="s", ms=3, label="same-capacity feedforward")
    axl.set_title("held-out magnitude (parity)", fontsize=8, loc="left")
    axl.set_xlabel("training epoch", fontsize=7.5)
    axl.set_ylabel("held-out mean per-miRNA Spearman", fontsize=7.5)
    # y lower limit set so the full min-to-max band is visible at every epoch (early range is wide).
    band_lo = float(min(lo_e.min(), lo_f.min()))
    band_hi = float(max(hi_e.max(), hi_f.max()))
    axl.set_ylim(band_lo - 0.008, band_hi + 0.006)
    axl.legend(fontsize=6.2, frameon=False, loc="lower right")

    # ---- RIGHT: single-seed competitor-abundance sweep, reproduced unchanged ----
    xk = np.array(cap["xk_values"])
    ref = np.argmin(np.abs(np.array(cap["sweep_factors"]) - 1.0))
    eq_delta = np.array(cap["eq_pred"]) - cap["eq_pred"][ref]
    base_delta = np.array(cap["base_pred"]) - cap["base_pred"][ref]
    axr.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    axr.plot(xk, eq_delta, color=accent, lw=1.4, marker="o", ms=3,
             label=f"equilibrium (slope {cap['eq_dpred_dxk_slope']:.1e})")
    axr.plot(xk, base_delta, color=grey, lw=1.4, ls="--", marker="s", ms=3,
             label=f"feedforward (slope {cap['base_dpred_dxk_slope']:.1f}, exactly 0)")
    # raise the title so it clears the 1e-5 y-axis offset label (they share the top-left corner)
    axr.set_title("capability: response to competitor abundance", fontsize=8, loc="left", pad=14)
    axr.set_xlabel("competitor abundance $x_k$", fontsize=7.5)
    axr.set_ylabel("change in predicted repression of focal target", fontsize=7.5)
    axr.legend(fontsize=6.2, frameon=False, loc="lower left")
    axr.text(0.97, 0.94,
             "baseline exactly flat: $d\\,\\mathrm{pred}/d x_k = 0$\n(competitor is not an input)",
             transform=axr.transAxes, fontsize=5.8, color="0.3", ha="right", va="top")

    for ax in (axl, axr):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(width=0.6, labelsize=6.5)
    fig.tight_layout(pad=0.6)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, metadata={"CreationDate": None})
    fig.savefig(OUT_PNG, dpi=300, metadata={"Software": None, "Creation Time": None})
    plt.close(fig)

    md5 = hashlib.md5(OUT_PDF.read_bytes()).hexdigest()
    print(f"left  final: equilibrium {mean_e[-1]:.4f} band [{lo_e[-1]:.4f}, {hi_e[-1]:.4f}]; "
          f"feedforward {mean_f[-1]:.4f} band [{lo_f[-1]:.4f}, {hi_f[-1]:.4f}]")
    print(f"right (single seed): eq slope {cap['eq_dpred_dxk_slope']:.1e}, "
          f"ff slope {cap['base_dpred_dxk_slope']:.1f}")
    print(f"wrote {OUT_PDF.relative_to(REPO)} and {OUT_PNG.relative_to(REPO)}")
    print(f"PDF md5 {md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
