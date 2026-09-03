"""Figure 2: the certified-direction ECDF panel (five arms, one row).

The ECDF curves come from the SAME shared-universe KS functions that produce the panel numbers
(certified_panel_filtered.compute_contrast for the two GNA contrasts;
gse108823_dose_enumeration.arm_enrichment for the RISC-block arm) via their return_arrays option, so the
figure and the table cannot diverge. For each panel we RE-COMPUTE the KS D as the maximum vertical gap
between the two plotted ECDFs and ASSERT it equals the printed D to within 1e-9, raising otherwise.

Panels (each = one arm; two ECDFs on the same shared universe: seed-matched vs non-matched background):
  1 GSE184929 Ttr parent          2 GSE184929 Ttr (S)-GNA g7
  3 GSE108823 Hao1 parent         4 GSE108823 Hao1 (S)-GNA g7      5 GSE108823 RISC-block (iB)
Brackets over 1-2 and 3-4 carry the attenuation (1 - D_mod/D_parent).

x = log2 fold change, common clip [-1.5, 1.0] across all panels (view only; D is on the FULL arrays).
Deterministic and bitwise-identical on rerun. No em-dashes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["SOURCE_DATE_EPOCH"] = "1577836800"  # fixed -> reproducible PDF/PNG timestamps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import gse108823_dose_enumeration as g8  # noqa: E402

from certified_panel_filtered import CONTRASTS, compute_contrast  # noqa: E402
from seedmod_loader import _load_fasta  # noqa: E402

ACCENT = "#0072B2"  # seed-matched curve
GREY = "0.55"  # background curve
XLIM = (-1.5, 1.0)
OUT_PDF = REPO / "results/figures/fig2_certified_panel.pdf"
OUT_PNG = REPO / "results/figures/fig2_certified_panel.png"


def ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(arr, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def max_gap(matched: np.ndarray, background: np.ndarray) -> tuple[float, float, float, float]:
    """D+ = max_x (F_matched(x) - F_background(x)), and the x and the two ECDF heights there.
    This is exactly scipy's one-sided ks_2samp(alternative='greater') statistic, recomputed from the
    plotted arrays so we can assert the drawn gap equals the printed D."""
    ms = np.sort(np.asarray(matched, float))
    bs = np.sort(np.asarray(background, float))
    allx = np.sort(np.concatenate([ms, bs]))
    fm = np.searchsorted(ms, allx, side="right") / len(ms)
    fb = np.searchsorted(bs, allx, side="right") / len(bs)
    gap = fm - fb
    i = int(np.argmax(gap))
    return float(gap[i]), float(allx[i]), float(fb[i]), float(fm[i])


def panel(ax, arm: dict, label: str, assay: str) -> dict:
    matched, background = arm["matched"], arm["background"]
    D, xstar, fb, fm = max_gap(matched, background)
    if abs(D - arm["ks_D"]) >= 1e-9:
        raise AssertionError(
            f"{label}: max-gap D from plotted ECDFs ({D!r}) != printed D ({arm['ks_D']!r})"
        )
    for a, color, lw in ((background, GREY, 1.0), (matched, ACCENT, 1.3)):
        xs, ys = ecdf(a)
        ax.plot(xs, ys, drawstyle="steps-post", color=color, lw=lw, solid_capstyle="butt")
    # draw the KS D as a physical vertical distance at x* (where the ECDFs are furthest apart):
    # an I-beam (two short horizontal ticks at the two ECDF heights joined by a thin vertical line).
    # As D -> 0 this degrades to a hairline, which is exactly the message for the RISC-block panel.
    tick, mid = 0.055, (fm + fb) / 2
    ax.plot([xstar, xstar], [fb, fm], color="0.15", lw=0.8, solid_capstyle="butt")
    ax.plot([xstar - tick, xstar + tick], [fb, fb], color="0.15", lw=0.8, solid_capstyle="butt")
    ax.plot([xstar - tick, xstar + tick], [fm, fm], color="0.15", lw=0.8, solid_capstyle="butt")
    ax.annotate(
        f"$D$={D:.3f}",
        xy=(xstar + tick, mid),
        xytext=(xstar + 0.18, mid),
        fontsize=6,
        va="center",
        color="0.15",
        arrowprops={"arrowstyle": "-", "color": "0.15", "lw": 0.5, "shrinkA": 1, "shrinkB": 0},
    )
    # header band: title and assay line sit in the blank band ABOVE the curves (data y > 1.0), so
    # nothing is drawn over the data. The assay line matters: the two GNA brackets differ in assay
    # format AND dose, and GSE108823 has no dose variation and no in vivo arm, so guide and assay
    # cannot be separated; keep that visible in-figure.
    ax.text(0.05, 0.99, label, transform=ax.transAxes, fontsize=6.5, va="top")
    ax.text(0.05, 0.90, assay, transform=ax.transAxes, fontsize=5, va="top", color="0.5")
    # p and n stay inside the axes at upper-left, over the flat near-zero part of the curves
    ax.text(
        0.05,
        0.72,
        f"$p$={arm['ks_p']:.1e}\n$n$={arm['n_matched']}",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.spines["left"].set_bounds(
        0, 1.0
    )  # y-axis line stops at 1.0; blank band above holds the header
    ax.tick_params(width=0.6, labelsize=6)
    ax.set_xlim(*XLIM)
    ax.set_ylim(0, 1.22)  # blank band above the curves for the title + assay line
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.plot([0.0, 0.0], [0.0, 1.0], color="0.85", lw=0.5, zorder=0)  # x=0 guide, only over the data
    ax.set_xlabel("log2 fold change", fontsize=6.5)
    return {
        "label": label,
        "D": D,
        "p": float(arm["ks_p"]),
        "n_matched": int(arm["n_matched"]),
        "xstar": xstar,
    }


def bracket(fig, ax_left, ax_right, text: str) -> None:
    bl, br = ax_left.get_position(), ax_right.get_position()
    xl, xr = (bl.x0 + bl.x1) / 2, (br.x0 + br.x1) / 2
    y, yt = 0.865, 0.83
    fig.add_artist(
        Line2D([xl, xl, xr, xr], [yt, y, y, yt], transform=fig.transFigure, color="0.15", lw=0.8)
    )
    fig.text((xl + xr) / 2, 0.885, text, ha="center", va="bottom", fontsize=7)


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    )
    utr = _load_fasta(REPO / "data/raw/reference/rat_3utr.fa")
    ttr = compute_contrast(
        next(c for c in CONTRASTS if c["series"] == "GSE184929"), return_arrays=True
    )
    hao = compute_contrast(
        next(c for c in CONTRASTS if c["series"] == "GSE108823"), return_arrays=True
    )
    ib = g8.arm_enrichment(g8.load_deseq(g8.ARMS["siRNA-5-iB"]), utr, return_arrays=True)

    assay_iv = "rat liver, in vivo, 30 mg/kg"  # panels 1-2 (GSE184929)
    assay_ic = "rat hepatocytes, in vitro, 10 nM"  # panels 3-5 (GSE108823)
    spec = [
        (ttr["measured"]["parent"], "Ttr parent", assay_iv),
        (ttr["measured"]["modified"], "Ttr (S)-GNA g7", assay_iv),
        (hao["measured"]["parent"], "Hao1 parent", assay_ic),
        (hao["measured"]["modified"], "Hao1 (S)-GNA g7", assay_ic),
        (ib, "RISC-block (iB)", assay_ic),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.2), sharey=True)
    built = [panel(ax, arm, lab, assay) for ax, (arm, lab, assay) in zip(axes, spec, strict=True)]
    axes[0].set_ylabel("cumulative fraction", fontsize=7)
    # colour key belongs in the caption (accent = seed-matched, grey = background); an in-axes legend
    # here reads as two data curves, so it is omitted.

    fig.tight_layout(rect=(0, 0, 1, 0.80), pad=0.4, w_pad=0.5)
    bracket(fig, axes[0], axes[1], f"attenuation {ttr['attenuation']:.2f}")
    bracket(fig, axes[2], axes[3], f"attenuation {hao['attenuation']:.3f}")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, metadata={"CreationDate": None})
    fig.savefig(OUT_PNG, dpi=300, metadata={"Software": None, "Creation Time": None})
    plt.close(fig)

    print("Figure 2 written:", OUT_PDF.relative_to(REPO), "and", OUT_PNG.name)
    print("Numbers the figure was built from (assert: max-gap D == printed D within 1e-9):")
    for b in built:
        print(
            f"  {b['label']:22s} D={b['D']:.6f}  p={b['p']:.3e}  n_matched={b['n_matched']}  "
            f"x*={b['xstar']:+.4f}"
        )
    print(
        f"  attenuation Ttr (panels 1->2) = {ttr['attenuation']:.4f}; "
        f"Hao1 (panels 3->4) = {hao['attenuation']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
