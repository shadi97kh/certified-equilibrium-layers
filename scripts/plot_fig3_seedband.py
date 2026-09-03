"""Figure 3 left panel, seed-band version: mean held-out Spearman with the seed min-to-max band.

Reads results/learned_layer_seeds.json (per_seed_curves) and redraws the left panel of Figure 3.
For each model the mean held-out per-microRNA Spearman across the ten seeds is a solid line over
training epochs, and the seed-wise min-to-max range is a shaded band at the matching color and low
alpha. Style matches scripts/learned_layer.py (same two colors, serif fonts, thin lines, no title).

Honesty gate: the two shaded bands must visibly overlap, consistent with the reported parity. If the
bands do not overlap at the final eval point, the script stops and reports that instead of drawing a
misleading figure. No em-dashes anywhere (submission constraint). Deterministic: fixed data, and the
PDF/PNG are saved with creation timestamps stripped so the bytes are identical across runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "learned_layer_seeds.json"
FIGDIR = REPO / "results" / "figures"
OUT_PDF = FIGDIR / "fig3_seedband.pdf"
OUT_PNG = FIGDIR / "fig3_seedband.png"

# Same two colors used for equilibrium vs feedforward in scripts/learned_layer.py.
ACCENT, GREY = "#0072B2", "0.45"


def model_band(curves: list[list[dict]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From a list-over-seeds of {epoch, heldout_mean_spearman} curves, return epochs, mean, min, max.

    Every seed shares the same eval epochs (the training path logs them identically), verified here.
    """
    epochs = [[p["epoch"] for p in c] for c in curves]
    if len({tuple(e) for e in epochs}) != 1:
        raise ValueError("seed curves do not share identical eval epochs; cannot align a band")
    ep = np.asarray(epochs[0], dtype=float)
    mat = np.asarray([[p["heldout_mean_spearman"] for p in c] for c in curves], dtype=float)
    return ep, mat.mean(axis=0), mat.min(axis=0), mat.max(axis=0)


def main() -> int:
    if not SRC.is_file():
        raise FileNotFoundError(f"missing seed-sweep artifact at {SRC}; run learned_layer_seeds first")
    data = json.loads(SRC.read_text(encoding="utf-8"))
    psc = data["per_seed_curves"]

    ep_e, mean_e, lo_e, hi_e = model_band(psc["equilibrium"])
    ep_f, mean_f, lo_f, hi_f = model_band(psc["feedforward"])
    if not np.array_equal(ep_e, ep_f):
        raise ValueError("equilibrium and feedforward eval epochs differ; cannot co-plot")

    # Honesty gate: require the min-to-max bands to overlap at the final eval point.
    overlap_per_epoch = (lo_e <= hi_f) & (lo_f <= hi_e)
    if not bool(overlap_per_epoch[-1]):
        print("STOP: the seed min-to-max bands do NOT overlap at the final eval point "
              f"(equilibrium [{lo_e[-1]:.4f}, {hi_e[-1]:.4f}], "
              f"feedforward [{lo_f[-1]:.4f}, {hi_f[-1]:.4f}]).")
        print("Refusing to produce a figure that would misrepresent the reported parity.")
        return 1
    n_overlap = int(overlap_per_epoch.sum())
    print(f"bands overlap at {n_overlap}/{len(overlap_per_epoch)} eval points "
          f"(including the final point); proceeding.")

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

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.7))
    ax.fill_between(ep_e, lo_e, hi_e, color=ACCENT, alpha=0.18, lw=0.0)
    ax.fill_between(ep_f, lo_f, hi_f, color=GREY, alpha=0.18, lw=0.0)
    ax.plot(ep_e, mean_e, color=ACCENT, lw=1.3, marker="o", ms=2.5, label="equilibrium layer")
    ax.plot(ep_f, mean_f, color=GREY, lw=1.3, marker="s", ms=2.5, label="feedforward baseline")

    ax.set_xlabel("training epoch", fontsize=7.5)
    ax.set_ylabel("held-out mean per-microRNA Spearman", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    fig.tight_layout()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, metadata={"CreationDate": None})
    fig.savefig(OUT_PNG, dpi=300, metadata={"Software": None, "Creation Time": None})
    plt.close(fig)

    md5 = hashlib.md5(OUT_PDF.read_bytes()).hexdigest()
    print(f"equilibrium mean final {mean_e[-1]:.4f}  band [{lo_e[-1]:.4f}, {hi_e[-1]:.4f}]")
    print(f"feedforward mean final {mean_f[-1]:.4f}  band [{lo_f[-1]:.4f}, {hi_f[-1]:.4f}]")
    print(f"wrote {OUT_PDF.relative_to(REPO)} and {OUT_PNG.relative_to(REPO)}")
    print(f"PDF md5 {md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
