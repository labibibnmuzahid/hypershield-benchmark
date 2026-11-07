"""Supplementary tables S1-S12 and figures S1-S2 from the stored results.

Writes
  supplement/generated/tab_S*.tex   LaTeX tables included by supplement/Supplementary_Material.tex
  supplement/Figure_S1.(pdf|png), supplement/Figure_S2.(pdf|png)
  results/supplement/*.csv          the same tables as CSV files

Run after run_experiments.py, analysis_tables.py and analysis_conditional.py; then
build the PDF with `pdflatex Supplementary_Material.tex` (twice) inside supplement/.
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hsbench.config import ESTIMATORS, G0, MANEUVER_LEVELS, SENSING, default_config

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
SUP = os.path.join(ROOT, "supplement")
GEN = os.path.join(SUP, "generated")
CSV = os.path.join(RES, "supplement")
LEVELS = list(MANEUVER_LEVELS)
LVL = {"low": "Low (0.5 g)", "medium": "Medium (2 g)", "high": "High (4 g)"}
SEN = {"ir": "IR only", "radar": "Radar only", "fused": "Fused"}
EST_TEX = {"EKF-CV": "EKF-CV", "UKF-CV": "UKF-CV", "EKF-CA": "EKF-CA", "IMM-CV/CA": "IMM-CV/CA"}
MCFG = default_config().metrics

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEN_STYLE = {"ir": (C[0], "-", "IR only"), "radar": (C[1], "--", "Radar only"), "fused": (C[2], "-", "Fused")}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.8,
    "savefig.dpi": 300, "figure.dpi": 110, "pdf.fonttype": 42,
})


# ------------------------------------------------------------------ helpers
def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def load_cell(level, sensing, kind):
    return dict(np.load(os.path.join(RES, "main", f"{level}_{sensing}_{kind.replace('/', '')}.npz")))


def write_csv(name, header, rows):
    os.makedirs(CSV, exist_ok=True)
    with open(os.path.join(CSV, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_tex(name, text):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name), "w") as f:
        f.write(text)


def f(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pct(x, nd=1):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def ci(s):
    lo, hi = (float(v) for v in s.strip("[]").split(","))
    return lo, hi


def sci(p):
    """p-value in LaTeX: 2 significant digits, scientific below 1e-3."""
    if p >= 1e-3:
        return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def pow10(x):
    """1e-05 -> 10^{-5}, 3e-05 -> 3\\times10^{-5}."""
    m, e = f"{x:.0e}".split("e")
    return f"10^{{{int(e)}}}" if m == "1" else f"{m}\\times10^{{{int(e)}}}"


def ttft(x):
    return "never" if not np.isfinite(x) else (f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}")


# --------------------------------------------------------- S1-S3 all cells
def tables_all_cells():
    rows = list(csv.DictReader(open(os.path.join(RES, "tables", "summary_all_cells.csv"))))
    out_csv = []
    for n, level in enumerate(LEVELS, start=1):
        body = []
        for s in SENSING:
            body.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\textit{{{SEN[s]}}}}} \\")
            for k in ESTIMATORS:
                r = next(x for x in rows if x["maneuver"] == level and x["sensing"] == s and x["estimator"] == k)
                v = {key: float(r[key]) for key in ("OSPA", "custody", "continuity", "segments", "TTFT_median",
                                                     "ever_tracked", "RMSE", "ANEES", "C50", "C95")}
                o_lo, o_hi = ci(r["OSPA_CI"])
                a_lo, a_hi = ci(r["ANEES_CI"])
                body.append(" & ".join([
                    EST_TEX[k], f"{v['OSPA']:.2f} [{o_lo:.2f}, {o_hi:.2f}]", pct(v["custody"]),
                    pct(v["continuity"]), f"{v['segments']:.2f}", ttft(v["TTFT_median"]), pct(v["ever_tracked"]),
                    f"{v['RMSE']:.2f}", f"{v['ANEES']:.2f} [{a_lo:.2f}, {a_hi:.2f}]", f"{v['C50']:.3f}",
                    f"{v['C95']:.3f}"]) + r" \\")
                out_csv.append([level, s, k, v["OSPA"], o_lo, o_hi, v["custody"], v["continuity"], v["segments"],
                                v["TTFT_median"], v["ever_tracked"], v["RMSE"], v["ANEES"], a_lo, a_hi,
                                v["C50"], v["C95"]])
            body.append(r"\midrule" if s != SENSING[-1] else "")
        tex = rf"""\begin{{table}}[!htbp]
\caption{{All sensing configurations and estimators at the {LVL[level].lower()} maneuver level (500 held-out runs each).}}
\label{{tab:S{n}}}
\centering\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lcccccccccc@{{}}}}
\toprule
Estimator & OSPA (km) & Custody & Contin. & Segments & TTFT & Ever & RMSE & ANEES & $C_{{50}}$ & $C_{{95}}$ \\
 & [95\% CI] & (\%) & (\%) & per run & (s) & tracked (\%) & (km) & [95\% CI] & & \\
\midrule
{chr(10).join(b for b in body if b)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize OSPA: mean OSPA distance, cut-off 10 km. Custody: share of epochs with a confirmed track within 1 km. Contin.: share of epochs with a confirmed track. Segments: confirmed track segments per run. TTFT: median time to first track within 5 km (runs never tracked count as infinite; a median over an even number of runs can end in .5, which the paper rounds to whole seconds). Ever tracked: share of runs with at least one such track. RMSE: over epochs with a track. ANEES: normalized (ideal 1). $C_\alpha$: empirical coverage of the nominal $\alpha$ ellipsoid. Confidence intervals: run-level percentile bootstrap, 2000 resamples.}}
\end{{table}}
"""
        write_tex(f"tab_S{n}.tex", tex)
    write_csv("S1-S3_all_cells.csv",
              ["maneuver", "sensing", "estimator", "OSPA_km", "OSPA_CI_low", "OSPA_CI_high", "custody", "continuity",
               "segments_per_run", "TTFT_median_s", "ever_tracked", "RMSE_km", "ANEES", "ANEES_CI_low",
               "ANEES_CI_high", "C50", "C95"], out_csv)


# --------------------------------------------------- S4-S5 paired tests
def tables_paired():
    pa = json.load(open(os.path.join(RES, "tables", "paired_all.json")))
    body, out = [], []
    for level in LEVELS:
        for s in SENSING:
            for r in [x for x in pa["estimators"] if x["maneuver"] == level and x["sensing"] == s]:
                body.append(" & ".join([LVL[level].split()[0], SEN[s], f"{r['first']} vs {r['second']}",
                                        f"${r['delta']:+.4f}$", f"${100 * r['rel']:+.2f}$", f"{100 * r['better']:.1f}",
                                        sci(r["p"]), sci(r["p_holm"])]) + r" \\")
                out.append([level, s, r["first"], r["second"], r["delta"], r["rel"], r["better"], r["p"], r["p_holm"]])
            body.append(r"\addlinespace[2pt]")
        body.append(r"\midrule" if level != LEVELS[-1] else "")
    write_tex("tab_S4.tex", rf"""\begin{{table}}[!htbp]
\caption{{All 36 paired estimator comparisons of per-run OSPA (one Holm family).}}
\label{{tab:S4}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lllrrrrr@{{}}}}
\toprule
Maneuver & Sensing & Comparison & $\Delta$ OSPA (km) & $\Delta$ (\%) & Better (\%) & $p$ & Holm $p$ \\
\midrule
{chr(10).join(b for b in body if b)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize $\Delta$: mean per-run OSPA difference, first minus second estimator; $\Delta$ (\%): relative to the second estimator. Better: share of runs in which the first estimator had the lower OSPA. $p$: two-sided Wilcoxon signed-rank test on 500 paired runs; Holm $p$: adjusted over all 36 comparisons.}}
\end{{table}}
""")
    write_csv("S4_paired_estimators.csv",
              ["maneuver", "sensing", "first", "second", "delta_OSPA_km", "relative", "share_first_better",
               "p_wilcoxon", "p_holm"], out)

    body, out = [], []
    for level in LEVELS:
        for r in [x for x in pa["fusion"] if x["maneuver"] == level]:
            body.append(" & ".join([LVL[level].split()[0], r["estimator"], f"Fused vs {SEN[r['baseline']]}",
                                    f"${r['delta']:+.3f}$", f"${100 * r['rel']:+.1f}$", f"{100 * r['better']:.1f}",
                                    sci(r["p"]), sci(r["p_holm"])]) + r" \\")
            out.append([level, r["estimator"], r["baseline"], r["delta"], r["rel"], r["better"], r["p"], r["p_holm"]])
        body.append(r"\midrule" if level != LEVELS[-1] else "")
    write_tex("tab_S5.tex", rf"""\begin{{table}}[!htbp]
\caption{{All 24 paired comparisons of fused against single-sensor tracking (per-run OSPA, one Holm family).}}
\label{{tab:S5}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lllrrrrr@{{}}}}
\toprule
Maneuver & Estimator & Comparison & $\Delta$ OSPA (km) & $\Delta$ (\%) & Better (\%) & $p$ & Holm $p$ \\
\midrule
{chr(10).join(b for b in body if b)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize $\Delta$: mean per-run OSPA difference, fused minus single-sensor; $\Delta$ (\%): relative to the single-sensor value. Better: share of runs in which fused tracking had the lower OSPA. $p$: two-sided Wilcoxon signed-rank test; Holm $p$: adjusted over the 24 comparisons.}}
\end{{table}}
""")
    write_csv("S5_paired_fusion.csv",
              ["maneuver", "estimator", "baseline", "delta_OSPA_km", "relative", "share_fused_better",
               "p_wilcoxon", "p_holm"], out)


# ------------------------------------------- S6-S7 credibility diagnostics
def tables_credibility():
    c = load_json("conditional.json")
    E = list(ESTIMATORS)
    spec = [
        ("Epochs within 10 s after a false association (\\%)", lambda d: pct(d["false_window_share_of_epochs"])),
        ("Normalized ANEES after a false association", lambda d: f(d["ANEES_after_false"])),
        ("Normalized ANEES, all other epochs", lambda d: f(d["ANEES_otherwise"])),
        ("$C_{95}$ after a false association", lambda d: f(d["C95_after_false"], 3)),
        ("$C_{95}$, all other epochs", lambda d: f(d["C95_otherwise"], 3)),
        ("Share of summed NEES after a false association (\\%)", lambda d: pct(d["false_window_share_of_NEES"])),
        ("Share of updates that used a clutter detection (\\%)", lambda d: pct(d["false_update_share"], 2)),
        ("Median normalized NEES", lambda d: f(d["median_normalized_NEES"])),
        ("Share of summed NEES in the worst 5\\% of epochs (\\%)", lambda d: pct(d["worst5pct_share_of_NEES"])),
    ] + [(f"$C_{{{a.replace('0.', '')}}}$ (nominal {a})", (lambda a: lambda d: f(d["coverage"][a], 3))(a))
         for a in ("0.50", "0.68", "0.90", "0.95", "0.99")]
    body = [" & ".join([lab] + [fn(c[k]) for k in E]) + r" \\" for lab, fn in spec]
    write_tex("tab_S6.tex", rf"""\begin{{table}}[!htbp]
\caption{{Credibility diagnostics of the stated position covariance, fused configuration, medium maneuver level.}}
\label{{tab:S6}}
\centering\footnotesize\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Quantity & {' & '.join(E)} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize A false association is an update (or a track initiation) that used a clutter detection. Normalized NEES is NEES/3, so its consistent mean is 1. $C_\alpha$: empirical coverage of the nominal $\alpha$ ellipsoid over all scored epochs.}}
\end{{table}}
""")
    keys = ["false_window_share_of_epochs", "ANEES_after_false", "ANEES_otherwise", "C95_after_false",
            "C95_otherwise", "false_window_share_of_NEES", "false_update_share", "median_normalized_NEES",
            "worst5pct_share_of_NEES"]
    write_csv("S6_credibility_fused_medium.csv", ["quantity"] + E,
              [[key] + [c[k][key] for k in E] for key in keys]
              + [[f"coverage_{a}"] + [c[k]["coverage"][a] for k in E] for a in ("0.50", "0.68", "0.90", "0.95", "0.99")])

    bins = c[E[0]]["acceleration_bins"]
    body, out = [], []
    for i, b in enumerate(bins):
        lo, hi = b["range_g"]
        lab = (f"{lo:g}--{hi:g} g ({lo * G0 * 1e3:.1f}--{hi * G0 * 1e3:.1f} m/s$^2$)" if hi is not None
               else f"$>${lo:g} g ($>${lo * G0 * 1e3:.1f} m/s$^2$)")
        row = [lab, pct(b["share"])] + [f(c[k]["acceleration_bins"][i]["ANEES"]) for k in E]
        body.append(" & ".join(row) + r" \\")
        out.append([lo, hi, b["share"]] + [c[k]["acceleration_bins"][i]["ANEES"] for k in E])
    write_tex("tab_S7.tex", rf"""\begin{{table}}[!htbp]
\caption{{Normalized ANEES by the target's instantaneous lateral acceleration, fused configuration, medium maneuver level.}}
\label{{tab:S7}}
\centering\footnotesize\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Lateral acceleration & Epochs (\%) & {' & '.join(E)} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize Epochs (\%): share of scored epochs (IMM-CV/CA) in each bin. Compare with Table~\ref{{tab:S6}}: the spread across acceleration bins is far smaller than the difference between epochs with and without a recent false association.}}
\end{{table}}
""")
    write_csv("S7_ANEES_by_acceleration.csv", ["accel_low_g", "accel_high_g", "share_of_epochs"] + E, out)


# ---------------------------------------------------- S8-S9 sensitivity
SWEEP_LABEL = {"ir_revisit": ("IR revisit interval", "s", lambda v: f"{v:g}"),
               "outage_fraction": ("IR outage fraction", "\\%", lambda v: f"{100 * v:g}"),
               "ir_sigma_urad": ("IR angle noise", "$\\mu$rad", lambda v: f"{v * 1e6:g}"),
               "radar_pd": ("Radar detection probability", "", lambda v: f"{v:g}"),
               "false_alarms": ("False alarms per scan", "", lambda v: f"{v:g}")}
NOMINAL = {"ir_revisit": 2.0, "outage_fraction": 0.2, "ir_sigma_urad": 100e-6, "radar_pd": 0.9, "false_alarms": 1.0}


def tables_sweeps():
    rows = load_json("sweeps.json")
    E = list(ESTIMATORS)
    out = []
    for n, sensing in ((8, "fused"), (9, "ir")):
        body = []
        for name, (lab, unit, fmt) in SWEEP_LABEL.items():
            vals = sorted({r["value"] for r in rows if r["parameter"] == name})
            body.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\textit{{{lab}{(' (' + unit + ')') if unit else ''}}}}} \\")
            for v in vals:
                cell = {r["estimator"]: r for r in rows if r["parameter"] == name and r["value"] == v
                        and r["sensing"] == sensing}
                mark = "$^\\ast$" if np.isclose(v, NOMINAL[name]) else ""
                fa = cell["IMM-CV/CA"]["false_update_share"]
                body.append(" & ".join([fmt(v) + mark] + [f(cell[k]["OSPA"]) for k in E]
                                       + [f(cell[k]["ANEES"]) for k in E] + [pct(fa, 2)]) + r" \\")
                for k in E:
                    r = cell[k]
                    out.append([sensing, name, v, k, r["OSPA"], r["custody"], r["ANEES"], r["C95"],
                                r["false_update_share"]])
            body.append(r"\addlinespace[2pt]")
        title = "fused configuration" if sensing == "fused" else "IR-only configuration"
        extra = (" Radar detection probability does not affect this configuration. At 10 s revisit no IR-only track is confirmed, so ANEES is undefined."
                 if sensing == "ir" else "")
        write_tex(f"tab_S{n}.tex", rf"""\begin{{table}}[!htbp]
\caption{{One-at-a-time sensitivity analysis, {title} (300 medium-maneuver runs with trajectories fixed across parameter values).}}
\label{{tab:S{n}}}
\centering\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lrrrrrrrrr@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{Mean OSPA (km)}} & \multicolumn{{4}}{{c}}{{Normalized ANEES}} & False \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{6-9}}
Value & EKF-CV & UKF-CV & EKF-CA & IMM & EKF-CV & UKF-CV & EKF-CA & IMM & upd. (\%) \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize $^\ast$ Nominal value (Table 3 of the paper). IMM: IMM-CV/CA. False upd.: share of IMM-CV/CA updates that used a clutter detection.{extra}}}
\end{{table}}
""")
    write_csv("S8-S9_sensitivity.csv", ["sensing", "parameter", "value", "estimator", "OSPA_km", "custody",
                                        "ANEES", "C95", "false_update_share"], out)


# ------------------------------------------------------------ S10 tuning
def table_tuning():
    rows = load_json("tuning.json")
    body, out = [], []
    prev = None
    for r in rows:
        k = r["estimator"]
        if prev is not None and k != prev:
            body.append(r"\midrule")
        prev = k
        if k == "EKF-CV":
            sett = f"$q_v = {pow10(r['qv_cv'])}$"
        elif k == "EKF-CA":
            sett = f"$q_a = {pow10(r['qa_ca'])}$"
        else:
            sett = f"$q_v = {pow10(r['imm_qv'])}$, $q_a = {pow10(r['imm_qa'])}$, $p_s = {r['imm_ps']:.2f}$"
        cells = [k, sett, f(r["OSPA"]), pct(r["custody"]), f(r["ANEES"]), f(r["C95"], 3)]
        if r["selected"]:
            cells = [f"\\textbf{{{x}}}" if not x.startswith("$") else x for x in cells] + ["selected"]
        else:
            cells += [""]
        body.append(" & ".join(cells) + r" \\")
        out.append([k, r.get("qv_cv", r.get("imm_qv")), r.get("qa_ca", r.get("imm_qa")), r.get("imm_ps"),
                    r["OSPA"], r["custody"], r["ANEES"], r["C95"], r["selected"]])
    write_tex("tab_S10.tex", rf"""\begin{{table}}[!htbp]
\caption{{Complete process-noise tuning grid (fused configuration, medium maneuvers, 200 tuning runs disjoint from all evaluation runs).}}
\label{{tab:S10}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llrrrrl@{{}}}}
\toprule
Estimator & Setting & Mean OSPA (km) & Custody (\%) & ANEES & $C_{{95}}$ & \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize $q_v$ in km$^2$/s$^3$, $q_a$ in km$^2$/s$^5$. The setting with the lowest mean OSPA was selected for each estimator; UKF-CV uses the EKF-CV value. Rows in bold are the selected settings.}}
\end{{table}}
""")
    write_csv("S10_tuning.csv", ["estimator", "q_v_or_none", "q_a_or_none", "p_s", "OSPA_km", "custody",
                                 "ANEES", "C95", "selected"], out)


# ---------------------------------------------------- S11 scenario stats
def table_scenarios():
    stats = {}
    for level in LEVELS:
        d = np.load(os.path.join(RES, "main", f"{level}_scenario.npz"))
        pk = d["peak_alat_g"]
        cov = d["coverage"]
        entry = np.array([np.argmax(r) if r.any() else np.nan for r in cov], float)
        stats[level] = {
            "peak_med": np.median(pk), "peak_p5": np.percentile(pk, 5), "peak_p95": np.percentile(pk, 95),
            "alt_min": float(d["alt"].min()), "alt_max": float(d["alt"].max()),
            "vis": float(d["n_visible"].mean()), "outage": float(d["outage"].mean()),
            "cover": float(cov.mean()), "never": float((~cov.any(1)).mean()),
            "entry_med": float(np.nanmedian(entry)), "entry_p5": float(np.nanpercentile(entry, 5)),
            "entry_p95": float(np.nanpercentile(entry, 95)),
        }
    s = stats
    lines = [
        ("Peak lateral acceleration, median (g)", [f"{s[l]['peak_med']:.2f}" for l in LEVELS]),
        ("Peak lateral acceleration, 5th--95th percentile (g)", [f"{s[l]['peak_p5']:.2f}--{s[l]['peak_p95']:.2f}" for l in LEVELS]),
        ("Peak lateral acceleration, median (m/s$^2$)", [f"{s[l]['peak_med'] * G0 * 1e3:.1f}" for l in LEVELS]),
        ("Altitude range (km)", [f"{s[l]['alt_min']:.1f}--{s[l]['alt_max']:.1f}" for l in LEVELS]),
        ("Mean number of visible satellites", [f"{s[l]['vis']:.2f}" for l in LEVELS]),
        ("Realized IR outage fraction (\\%)", [pct(s[l]["outage"]) for l in LEVELS]),
        ("Time inside radar coverage (\\%)", [pct(s[l]["cover"]) for l in LEVELS]),
        ("Runs never inside radar coverage (\\%)", [pct(s[l]["never"]) for l in LEVELS]),
        ("Radar entry time, median (s)", [f"{s[l]['entry_med']:.0f}" for l in LEVELS]),
        ("Radar entry time, 5th--95th percentile (s)", [f"{s[l]['entry_p5']:.0f}--{s[l]['entry_p95']:.0f}" for l in LEVELS]),
    ]
    body = [" & ".join([lab] + v) + r" \\" for lab, v in lines]
    write_tex("tab_S11.tex", rf"""\begin{{table}}[!htbp]
\caption{{Statistics of the held-out scenarios (500 runs per maneuver level).}}
\label{{tab:S11}}
\centering\footnotesize\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
Quantity & Low (0.5 g) & Medium (2 g) & High (4 g) \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize $g = 9.80665$ m/s$^2$. Visible satellites: line of sight above the Earth's surface and range below 4500 km, averaged over IR scan times. Radar entry time: first epoch inside radar coverage, over the runs that enter it.}}
\end{{table}}
""")
    write_csv("S11_scenario_statistics.csv", ["quantity"] + LEVELS,
              [[k] + [s[l][k] for l in LEVELS] for k in s[LEVELS[0]]])


# ----------------------------------------------------- S12 timing
def table_timing():
    t = load_json("timing.json")
    body, out = [], []
    for sensing in SENSING:
        for k in ESTIMATORS:
            tag = lambda l: f"{l}_{sensing}_{k.replace('/', '')}"
            body.append(" & ".join([SEN[sensing], k] + [f"{t[tag(l)]['us_per_cycle_run']:.1f}" for l in LEVELS]
                                   + [f"{t[tag(l)]['cpu_s']:.2f}" for l in LEVELS]) + r" \\")
            out.append([sensing, k] + [t[tag(l)]["us_per_cycle_run"] for l in LEVELS] + [t[tag(l)]["cpu_s"] for l in LEVELS])
        body.append(r"\midrule" if sensing != SENSING[-1] else "")
    total = sum(v["cpu_s"] for v in t.values())
    write_tex("tab_S12.tex", rf"""\begin{{table}}[!htbp]
\caption{{Computational cost of each cell of the main evaluation.}}
\label{{tab:S12}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
 & & \multicolumn{{3}}{{c}}{{$\mu$s per filter cycle and run}} & \multicolumn{{3}}{{c}}{{Tracker CPU time (s)}} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}
Sensing & Estimator & Low & Medium & High & Low & Medium & High \\
\midrule
{chr(10).join(b for b in body if b)}
\bottomrule
\end{{tabular}}

\smallskip
\parbox{{\textwidth}}{{\footnotesize Process CPU time of the tracker for 500 runs $\times$ 481 cycles, measured on one core with the vectorized Python implementation (scenario and detection generation excluded). Total over the 36 cells: {total:.1f} s. These figures compare estimators within one implementation and are not real-time performance claims.}}
\end{{table}}
""")
    write_csv("S12_timing.csv", ["sensing", "estimator"] + [f"us_per_cycle_run_{l}" for l in LEVELS]
              + [f"cpu_s_{l}" for l in LEVELS], out)


# ------------------------------------------------------------ figures
def per_run_ospa(ep):
    c = MCFG.ospa_cutoff
    e = np.nan_to_num(ep["err"].astype(float), nan=c)
    return np.where(ep["conf"], np.minimum(e, c), c)


def fig_S1():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    t = None
    for s in SENSING:
        ep = load_cell("medium", s, "IMM-CV/CA")
        o = per_run_ospa(ep)
        t = np.arange(o.shape[1])
        col, ls, lab = SEN_STYLE[s]
        axes[0].plot(t, o.mean(0), color=col, ls=ls, label=lab)
        axes[1].plot(t, 100 * ep["conf"].mean(0), color=col, ls=ls, label=lab)
    sc = np.load(os.path.join(RES, "main", "medium_scenario.npz"))
    axes[0].set_ylabel("Mean OSPA (km)")
    axes[0].set_ylim(0, 10.5)
    axes[1].set_ylabel("Runs with a confirmed track (%)")
    axes[1].set_ylim(0, 102)
    cover = 100 * sc["coverage"].mean(0)
    axes[1].plot(t, cover, color=INK2, ls=":", lw=1.4, label="Target inside radar coverage")
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, t[-1])
        ax.set_title(f"({letter})", loc="left", fontweight="bold", color=INK)
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="lower right")
    save(fig, "Figure_S1")


def fig_S2():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True, sharey=True)
    for ax, level, letter in zip(axes, LEVELS, "abc"):
        a = per_run_ospa(load_cell(level, "fused", "IMM-CV/CA")).mean(1)
        b = per_run_ospa(load_cell(level, "fused", "EKF-CV")).mean(1)
        d = a - b
        lim = np.percentile(np.abs(d), 99)
        ax.hist(np.clip(d, -lim, lim), bins=np.linspace(-lim, lim, 41), color=C[0], alpha=0.85)
        ax.axvline(0, color=INK, lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_title(f"({letter}) {LVL[level]}", loc="left", fontweight="bold", color=INK)
        ax.text(0.03, 0.95, f"IMM lower in\n{100 * (d < 0).mean():.1f}% of runs", transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color=INK)
    axes[0].set_ylabel("Runs")
    fig.supxlabel("Per-run mean OSPA, IMM-CV/CA minus EKF-CV (km)", fontsize=9, color=INK)
    save(fig, "Figure_S2")


def save(fig, name):
    os.makedirs(SUP, exist_ok=True)
    fig.savefig(os.path.join(SUP, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(SUP, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    tables_all_cells()
    tables_paired()
    tables_credibility()
    tables_sweeps()
    table_tuning()
    table_scenarios()
    table_timing()
    fig_S1()
    fig_S2()
    print("supplementary tables:", sorted(os.listdir(GEN)))
    print("CSV files:", sorted(os.listdir(CSV)))


if __name__ == "__main__":
    main()
