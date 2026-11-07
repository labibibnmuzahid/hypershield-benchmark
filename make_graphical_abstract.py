"""Graphical abstract (supplement/Graphical_Abstract.tif/.pdf/.png) from the stored results.

Elsevier asks for at least 1328 x 531 pixels (w x h), readable at 13 x 5 cm.
The figure is 10 x 4 in at 300 dpi (3000 x 1200 px). Numbers are read from the
result files, so the picture always matches the paper.
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "supplement")

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.dpi": 300})


def numbers():
    rows = list(csv.DictReader(open(os.path.join(RES, "tables", "summary_all_cells.csv"))))
    cell = {(r["maneuver"], r["sensing"], r["estimator"]): r for r in rows}
    ospa = {s: float(cell["medium", s, "IMM-CV/CA"]["OSPA"]) for s in ("ir", "radar", "fused")}
    ttft = {s: float(cell["medium", s, "IMM-CV/CA"]["TTFT_median"]) for s in ("ir", "radar", "fused")}
    cond = json.load(open(os.path.join(RES, "conditional.json")))
    est = ("EKF-CV", "UKF-CV", "EKF-CA", "IMM-CV/CA")
    after = [cond[k]["ANEES_after_false"] for k in est]
    other = [cond[k]["ANEES_otherwise"] for k in est]
    return ospa, ttft, (min(after), max(after)), (min(other), max(other))


def schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # Earth and atmosphere
    ax.add_patch(Circle((5, -38), 40.0, color="#dfe9f5", zorder=0))
    ax.add_patch(Circle((5, -38), 40.6, color="#eef3fa", zorder=-1))
    # radar and its horizon-limited coverage
    rx, ry = 8.6, 1.95
    ax.add_patch(Wedge((rx, ry), 2.1, 90, 180, color=C[1], alpha=0.16, zorder=1))
    ax.add_patch(Polygon([[rx - 0.18, ry], [rx + 0.18, ry], [rx, ry + 0.38]], color=C[1], zorder=4))
    ax.text(rx + 0.05, ry - 0.42, "Ground radar\n(late, horizon-limited)", ha="center", va="top", fontsize=10.5,
            color=C[1], fontweight="bold")
    # maneuvering glide trajectory
    t = np.linspace(0, 1, 200)
    x = 0.6 + 7.4 * t
    y = 2.55 + 0.25 * np.sin(2 * np.pi * 2.2 * t) + 0.18 * t
    ax.plot(x, y, color=INK, lw=2.4, zorder=3)
    ax.annotate("", xy=(x[-1] + 0.25, y[-1] - 0.02), xytext=(x[-6], y[-6]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2), zorder=3)
    ax.text(0.55, 2.0, "Maneuvering glide target,\nMach 5–8, 20–60 km", fontsize=10.5, color=INK, va="top")
    # LEO IR satellites with angle-only lines of sight
    sats = [(1.3, 5.3), (3.5, 5.6), (5.6, 5.35)]
    tgt = (x[95], y[95])
    for sx, sy in sats:
        ax.plot([sx, tgt[0]], [sy, tgt[1]], color=C[0], lw=1.6, ls=(0, (4, 3)), zorder=2)
        ax.add_patch(FancyBboxPatch((sx - 0.22, sy - 0.11), 0.44, 0.22, boxstyle="round,pad=0.02",
                                    color=C[0], zorder=4))
        ax.plot([sx - 0.55, sx + 0.55], [sy, sy], color=C[0], lw=3, zorder=4)
    ax.text(8.15, 5.3, "LEO infrared layer\n48 satellites,\nangle-only", ha="center", va="center", fontsize=10.5,
            color=C[0], fontweight="bold")


def main():
    ospa, ttft, after, other = numbers()
    fig = plt.figure(figsize=(10, 4))
    fig.text(0.012, 0.955, "Space–ground fusion benchmark for tracking maneuvering hypersonic glide targets",
             fontsize=14.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.865, "EKF, UKF and IMM filters in identical track management, 1500 held-out runs per configuration",
             fontsize=10.5, color=INK2, va="top")

    ax0 = fig.add_axes([0.0, 0.0, 0.44, 0.80])
    schematic(ax0)

    # OSPA by sensing configuration
    ax1 = fig.add_axes([0.505, 0.21, 0.19, 0.48])
    labels = ["IR only", "Radar\nonly", "Fused"]
    vals = [ospa["ir"], ospa["radar"], ospa["fused"]]
    bars = ax1.bar(labels, vals, color=[C[0], C[1], C[2]], width=0.65)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=11,
                 color=INK, fontweight="bold")
    ax1.set_ylim(0, 8.2)
    ax1.set_ylabel("Mean OSPA (km)", fontsize=10.5)
    ax1.set_title("Fusion cuts error", fontsize=12, fontweight="bold", color=INK, loc="left")
    fig.text(0.505, 0.025, f"Median time to first track: {ttft['fused']:.0f} s fused vs {ttft['radar']:.0f} s radar only",
             fontsize=9.5, color=INK2, va="bottom")

    # credibility by association status
    ax2 = fig.add_axes([0.765, 0.21, 0.19, 0.48])
    lab = ["After a false\nassociation", "Other\nepochs"]
    hi = [after[1], other[1]]
    lo = [after[0], other[0]]
    bars = ax2.bar(lab, hi, color=[C[1], C[2]], width=0.6)
    for b, l, h, nd in zip(bars, lo, hi, (1, 2)):
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.25, f"{l:.{nd}f}–{h:.{nd}f}", ha="center", va="bottom",
                 fontsize=11, color=INK, fontweight="bold")
    ax2.axhline(1.0, color=INK, ls="--", lw=1.2)
    ax2.set_ylim(0, 11.5)
    ax2.set_ylabel("Normalized ANEES (1 = consistent)", fontsize=10)
    ax2.set_title("False associations\ncause overconfidence", fontsize=11.5, fontweight="bold", color=INK,
                  loc="left")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(OUT, f"Graphical_Abstract.{ext}"), dpi=300, facecolor="white", **kw)
    plt.close(fig)
    print("graphical abstract written (3000 x 1200 px)")


if __name__ == "__main__":
    main()
