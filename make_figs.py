"""Figures 1-9 of the paper from the stored results (PNG at 300 dpi + vector PDF).

Colour follows the entity in every figure (fixed categorical order, validated
palette); marker shape and line style carry identity too, for print and CVD.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hsbench import metrics
from hsbench.config import ESTIMATORS, MANEUVER_LEVELS, RE, SENSING, SOUND, default_config
from hsbench.geometry import great_circle_distance, initial_bearing, radar_los_range

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
LEVELS = list(MANEUVER_LEVELS)
LVL_LABEL = {"low": "Low\n(0.5 g)", "medium": "Medium\n(2 g)", "high": "High\n(4 g)"}

# validated reference palette, slots in fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EST_STYLE = {"IMM-CV/CA": (C[0], "o", "-"), "EKF-CV": (C[1], "s", "--"),
             "UKF-CV": (C[2], "^", ":"), "EKF-CA": (C[3], "D", "-.")}
EST_ORDER = ["IMM-CV/CA", "EKF-CV", "UKF-CV", "EKF-CA"]
SEN_STYLE = {"ir": (C[0], "o", "IR only"), "radar": (C[1], "s", "Radar only"), "fused": (C[2], "^", "Fused")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0,
    "savefig.dpi": 300, "figure.dpi": 110, "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontweight="bold", color=INK)


def load_cell(level, sensing, kind):
    return dict(np.load(os.path.join(RES, "main", f"{level}_{sensing}_{kind.replace('/', '')}.npz")))


# ------------------------------------------------------------------ summaries
def cell_stats():
    cfg = default_config()
    out = {}
    counts = None
    for lv in LEVELS:
        for se in SENSING:
            for k in ESTIMATORS:
                s = metrics.run_sums(load_cell(lv, se, k), cfg.metrics)
                if counts is None:
                    counts = metrics.bootstrap_counts(len(s["ospa"]), cfg.metrics.n_boot, seed=2024)
                sm = metrics.summarize(s, counts)
                tt = s["ttft"]
                med = np.array([np.median(np.repeat(tt, c.astype(int))) for c in counts[:500]])
                sm["TTFT_CI"] = tuple(np.percentile(med, [2.5, 97.5]))
                out[lv, se, k] = sm
    return out


def bars(ax, groups, series, val, ci, styles, labels, width=0.24):
    x = np.arange(len(groups))
    n = len(series)
    for i, s in enumerate(series):
        col = styles[s][0]
        v = np.array([val(g, s) for g in groups])
        lo = np.array([ci(g, s)[0] for g in groups])
        hi = np.array([ci(g, s)[1] for g in groups])
        pos = x + (i - (n - 1) / 2) * (width + 0.02)
        ax.bar(pos, v, width, color=col, label=labels[s], edgecolor="white", linewidth=1.0, zorder=2)
        ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LVL_LABEL[g] for g in groups])
    ax.grid(axis="x", visible=False)


# ----------------------------------------------------------------- figure 1
def fig1(K):
    h = np.linspace(0, 80, 400)
    d = radar_los_range(0.010, h)
    lo, hi = K["altitude_range_km"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axs[0]
    ax.axvspan(lo, hi, color=GRID, alpha=0.8, lw=0, label="Generated altitudes")
    ax.plot(h, d, color=C[0], label="Radar line of sight (10 m mast)")
    for hh, off in ((20, (-6, 9)), (30, (4, -13)), (60, (-10, 9))):
        dd = radar_los_range(0.010, hh)
        ax.plot(hh, dd, "o", color=C[0], ms=5, mec="white", mew=1)
        ax.annotate(f"{dd:.0f} km", (hh, dd), textcoords="offset points", xytext=off, fontsize=8, color=INK)
    ax.axhline(3707, color=INK2, ls="--", lw=1.2)
    ax.text(2, 3707 - 260, "LEO sensor horizon at 1000 km: 3707 km", fontsize=8, color=INK2)
    ax.set_xlabel("Target altitude (km)")
    ax.set_ylabel("Line-of-sight range (km)")
    ax.set_ylim(0, 4100)
    ax.set_xlim(0, 80)
    panel(ax, "a")
    ax.legend(loc="center right")
    ax = axs[1]
    ax.axvspan(lo, hi, color=GRID, alpha=0.8, lw=0)
    for m, (col, mk, ls) in zip((5, 8), ((C[0], "o", "-"), (C[1], "s", "--"))):
        t = d / (m * SOUND) / 60
        ax.plot(h, t, color=col, ls=ls, label=f"Mach {m}")
        for hh in (20, 30):
            tt = radar_los_range(0.010, hh) / (m * SOUND) / 60
            ax.plot(hh, tt, mk, color=col, ms=5, mec="white", mew=1)
    ax.set_xlabel("Target altitude (km)")
    ax.set_ylabel("Time from radar horizon to site (min)")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 13)
    panel(ax, "b")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "Figure_1")


# ----------------------------------------------------------------- figure 2
def fig2():
    ex = dict(np.load(os.path.join(RES, "example.npz")))
    cfg = default_config()
    latr, lonr = np.radians(cfg.scenario.radar_lat_deg), np.radians(cfg.scenario.radar_lon_deg)
    lat, lon, t = ex["lat"], ex["lon"], ex["times"]
    rho = great_circle_distance(latr, lonr, lat, lon)
    az = initial_bearing(latr, lonr, lat, lon)
    x, y = rho * np.sin(az), rho * np.cos(az)
    out, cov = ex["outage"].astype(bool), ex["coverage"].astype(bool)
    enter = int(np.argmax(cov)) if cov.any() else None
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    # radar horizon circle (ground range) at the mean altitude of the run
    los = radar_los_range(0.010, float(np.mean(ex["alt"])))
    ground = RE * np.arcsin(np.clip(los / (RE + np.mean(ex["alt"])), 0, 1))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(ground * np.sin(th), ground * np.cos(th), color=INK2, ls="--", lw=1.2,
            label=f"Radar horizon at {np.mean(ex['alt']):.0f} km altitude")
    ax.plot(x, y, color=C[0], lw=2, label="Ground track (IR layer available)")
    seg = np.where(out, y, np.nan)
    ax.plot(x, seg, color=C[1], lw=3.2, solid_capstyle="butt", label="IR layer outage")
    ax.plot(0, 0, marker="^", color=INK, ms=8, ls="none", label="Ground radar")
    ax.plot(x[0], y[0], "o", color=C[0], ms=7, mec="white", mew=1.2)
    ax.annotate("start", (x[0], y[0]), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    if enter is not None:
        ax.plot(x[enter], y[enter], "D", color=C[2], ms=7, mec="white", mew=1.2, ls="none",
                label=f"Radar coverage entry (t = {enter} s)")
    ax.set_aspect("equal")
    ax.set_xlabel("East of radar (km)")
    ax.set_ylabel("North of radar (km)")
    panel(ax, "a")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=7.5)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, rho, color=INK)
    ax.set_ylabel("Range to radar (km)")
    ax.tick_params(labelbottom=False)
    panel(ax, "b")
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax)
    ax2.fill_between(t, 1.1, 1.9, where=~out, color=C[0], step="mid", lw=0, label="IR layer available")
    ax2.fill_between(t, 1.1, 1.9, where=out, color=C[1], step="mid", lw=0, label="IR layer outage")
    ax2.fill_between(t, 0.1, 0.9, where=cov, color=C[2], step="mid", lw=0, label="In radar coverage")
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(["Radar", "Space IR"])
    ax2.set_ylim(0, 2)
    ax2.grid(False)
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, t[-1])
    panel(ax2, "c")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=7.5)
    fig.tight_layout()
    save(fig, "Figure_2")
    return enter


# ----------------------------------------------------------------- figure 3
def fig3():
    ex = dict(np.load(os.path.join(RES, "example.npz")))
    t = ex["times"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    col, _, lab = SEN_STYLE["ir"]
    ax.plot(t, ex["err_ir"], color=col, lw=3.6, alpha=0.45, solid_capstyle="butt", label=f"{lab}: position error")
    col, _, lab = SEN_STYLE["radar"]
    ax.plot(t, ex["err_radar"], color=col, lw=1.6, label=f"{lab}: position error")
    col, _, lab = SEN_STYLE["fused"]
    ax.plot(t, ex["err_fused"], color=col, lw=1.3, label=f"{lab}: position error")
    cov = ex["coverage"].astype(bool)
    if cov.any():
        ent = int(np.argmax(cov))
        ax.axvline(ent, color=INK2, lw=1, ls="--")
        ax.text(ent + 3, 18, f"radar coverage\nfrom t = {ent} s", fontsize=7.5, color=INK2, va="top")
    ax.plot(t, ex["sigma_fused"], color=C[2], lw=1.2, ls=":", label=r"Fused: stated $\sqrt{\mathrm{tr}(P_{\mathrm{pos}})}$")
    out = ex["outage"].astype(bool)
    ax.fill_between(t, 1e-3, 1e3, where=out, color=GRID, step="mid", lw=0, label="IR layer outage", zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 30)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (km)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    save(fig, "Figure_3")


# ----------------------------------------------------------------- figure 4
def fig4(S):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
    k = "IMM-CV/CA"
    labels = {s: SEN_STYLE[s][2] for s in SENSING}
    st = {s: SEN_STYLE[s] for s in SENSING}
    bars(axs[0], LEVELS, SENSING, lambda g, s: S[g, s, k]["OSPA"], lambda g, s: S[g, s, k]["OSPA_CI"], st, labels)
    axs[0].set_ylabel("Mean OSPA (km)")
    panel(axs[0], "a")
    bars(axs[1], LEVELS, SENSING, lambda g, s: 100 * S[g, s, k]["custody"],
         lambda g, s: tuple(100 * v for v in S[g, s, k]["custody_CI"]), st, labels)
    axs[1].set_ylabel("Custody (% of time)")
    axs[1].set_ylim(0, 100)
    panel(axs[1], "b")
    bars(axs[2], LEVELS, SENSING, lambda g, s: S[g, s, k]["TTFT_median"], lambda g, s: S[g, s, k]["TTFT_CI"], st, labels)
    axs[2].set_yscale("log")
    axs[2].set_ylim(1, 1000)
    axs[2].set_ylabel("Median time to first track (s)")
    panel(axs[2], "c")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Figure_4")


# ----------------------------------------------------------------- figure 5
def fig5(S):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
    st = {k: EST_STYLE[k] for k in EST_ORDER}
    lab = {k: k for k in EST_ORDER}
    w = 0.19
    bars(axs[0], LEVELS, EST_ORDER, lambda g, k: S[g, "fused", k]["OSPA"], lambda g, k: S[g, "fused", k]["OSPA_CI"], st, lab, w)
    axs[0].set_ylabel("Mean OSPA (km)")
    panel(axs[0], "a")
    bars(axs[1], LEVELS, EST_ORDER, lambda g, k: S[g, "fused", k]["ANEES"], lambda g, k: S[g, "fused", k]["ANEES_CI"], st, lab, w)
    axs[1].axhline(1.0, color=INK, ls="--", lw=1)
    axs[1].set_ylabel("Normalized ANEES")
    panel(axs[1], "b")
    bars(axs[2], LEVELS, EST_ORDER, lambda g, k: S[g, "fused", k]["C95"], lambda g, k: S[g, "fused", k]["C95_CI"], st, lab, w)
    axs[2].axhline(0.95, color=INK, ls="--", lw=1)
    axs[2].set_ylim(0.8, 1.0)
    axs[2].set_ylabel("Coverage of nominal 95% ellipsoid")
    panel(axs[2], "c")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Figure_5")


# ----------------------------------------------------------------- figure 6
def fig6():
    sr = dict(np.load(os.path.join(RES, "conditional_series.npz")))
    a = sr["cov_curve_alpha"]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Ideal (consistent)")
    for k in EST_ORDER:
        col, mk, ls = EST_STYLE[k]
        y = sr[f"cov_curve_{k.replace('/', '')}"]
        ax.plot(a, y, color=col, ls=ls, lw=1.8, label=k)
        pts = [49, 67, 89, 94, 98]
        ax.plot(a[pts], y[pts], mk, color=col, ms=5, mec="white", mew=0.8)
    ax.set_xlabel(r"Nominal level $\alpha$")
    ax.set_ylabel(r"Empirical coverage $C_\alpha$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "Figure_6")


# ----------------------------------------------------------------- figure 7
def fig7():
    sr = dict(np.load(os.path.join(RES, "conditional_series.npz")))
    t = np.arange(len(sr["accept_lo"]))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.fill_between(t, sr["accept_lo"], sr["accept_hi"], color=GRID, lw=0, label="95% acceptance region (per epoch)")
    for k in EST_ORDER:
        col, mk, ls = EST_STYLE[k]
        ax.plot(t, sr[f"anees_t_smooth_{k.replace('/', '')}"], color=col, ls=ls, lw=1.6, label=k)
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 30)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized ANEES (9 s moving mean)")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    save(fig, "Figure_7")


# ----------------------------------------------------------------- figure 8
def fig8():
    cd = json.load(open(os.path.join(RES, "conditional.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(EST_ORDER))
    for ax, key, ylab, ref in ((axs[0], "ANEES", "Normalized ANEES", 1.0), (axs[1], "C95", "Coverage of nominal 95% ellipsoid", 0.95)):
        v1 = [cd[k][f"{key}_after_false"] for k in EST_ORDER]
        v0 = [cd[k][f"{key}_otherwise"] for k in EST_ORDER]
        ax.bar(x - 0.2, v1, 0.38, color=C[1], edgecolor="white", label="False association in preceding 10 s")
        ax.bar(x + 0.2, v0, 0.38, color=C[0], edgecolor="white", label="All other epochs")
        for xi, (a, b) in enumerate(zip(v1, v0)):
            ax.text(xi - 0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
            ax.text(xi + 0.2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.axhline(ref, color=INK, ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(EST_ORDER, fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axs[1].set_ylim(0.5, 1.06)
    panel(axs[0], "a"); panel(axs[1], "b")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "Figure_8")


# ----------------------------------------------------------------- figure 9
def fig9():
    rows = json.load(open(os.path.join(RES, "sweeps.json")))
    params = [("ir_revisit", "IR revisit interval (s)", 1.0),
              ("outage_fraction", "IR outage fraction (%)", 100.0),
              ("ir_sigma_urad", r"IR angle noise ($\mu$rad)", 1e6),
              ("radar_pd", "Radar detection probability", 1.0),
              ("false_alarms", "False alarms per scan", 1.0)]
    fig, axs = plt.subplots(2, 5, figsize=(7.4, 3.9), sharey="row")
    for j, (p, lab, sc) in enumerate(params):
        for k in ("IMM-CV/CA", "EKF-CV"):
            col, mk, _ = EST_STYLE[k]
            for se, ls in (("fused", "-"), ("ir", "--")):
                if p == "radar_pd" and se == "ir":
                    continue
                rr = sorted([r for r in rows if r["parameter"] == p and r["estimator"] == k and r["sensing"] == se],
                            key=lambda r: r["value"])
                xv = np.array([r["value"] * sc for r in rr])
                for i, m in enumerate(("OSPA", "ANEES")):
                    yv = np.array([np.nan if r[m] is None else r[m] for r in rr], float)
                    axs[i, j].plot(xv, yv, color=col, ls=ls, marker=mk, ms=4, lw=1.5,
                                   label=f"{k}, {'fused' if se == 'fused' else 'IR only'}")
        vals = sorted({r["value"] * sc for r in rows if r["parameter"] == p})
        for i in range(2):
            axs[i, j].set_xticks(vals)
            axs[i, j].set_xticklabels([f"{v:g}" for v in vals], fontsize=6.5)
        axs[1, j].set_xlabel(lab, fontsize=7.5)
        axs[1, j].axhline(1.0, color=INK2, ls=":", lw=1)
        axs[0, j].tick_params(labelsize=7); axs[1, j].tick_params(labelsize=7)
    axs[0, 0].set_ylabel("Mean OSPA (km)")
    axs[1, 0].set_ylabel("Normalized ANEES")
    axs[1, 0].set_yscale("log")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "Figure_9")


def main():
    K = json.load(open(os.path.join(RES, "key_numbers.json")))
    S = cell_stats()
    fig1(K)
    fig2()
    fig3()
    fig4(S)
    fig5(S)
    fig6()
    fig7()
    fig8()
    if os.path.exists(os.path.join(RES, "sweeps.json")):
        fig9()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
