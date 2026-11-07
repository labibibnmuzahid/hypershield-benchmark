"""Tables and headline numbers from the stored experiment outputs.

Writes results/tables/*.csv, *.tex, *.md and results/key_numbers.json.
"""
from __future__ import annotations

import csv
import itertools
import json
import os

import numpy as np

from hsbench import metrics
from hsbench.config import ESTIMATORS, MANEUVER_LEVELS, SENSING, default_config
from hsbench.observability import table1

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
LEVELS = list(MANEUVER_LEVELS)
SNAME = {"ir": "IR only", "radar": "Radar only", "fused": "Fused"}


def load_cell(level, sensing, kind):
    return dict(np.load(os.path.join(RES, "main", f"{level}_{sensing}_{kind.replace('/', '')}.npz")))


def all_sums(mcfg):
    S = {}
    for lv, s, k in itertools.product(LEVELS, SENSING, ESTIMATORS):
        S[lv, s, k] = metrics.run_sums(load_cell(lv, s, k), mcfg)
    return S


def fmt_ci(v, ci, d=2):
    return f"{v:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def p_fmt(p):
    if p == 0:
        return "<1e-300"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m} x 10^{int(e)}"
    return f"{p:.3f}"


def main():
    os.makedirs(TAB, exist_ok=True)
    cfg = default_config()
    mc = cfg.metrics
    S = all_sums(mc)
    N = len(next(iter(S.values()))["ospa"])
    T = next(iter(S.values()))["T"]
    counts = metrics.bootstrap_counts(N, mc.n_boot, seed=2024)

    # ----------------------------------------------------- summary of every cell
    summ = {}
    rows = []
    for key, s in S.items():
        sm = metrics.summarize(s, counts)
        summ[key] = sm
        lv, se, k = key
        rows.append({"maneuver": lv, "sensing": se, "estimator": k, **{
            kk: (vv if not isinstance(vv, tuple) else f"[{vv[0]:.4f}, {vv[1]:.4f}]") for kk, vv in sm.items()}})
    with open(os.path.join(TAB, "summary_all_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---------------------------------------------- Table 5 (medium maneuvers)
    t5 = ["| Sensing | Estimator | OSPA (km) | Custody (%) | Continuity (%) | TTFT (s) | RMSE (km) | ANEES | C95 |",
          "|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for se in SENSING:
        for k in ESTIMATORS:
            m = summ["medium", se, k]
            ttft = "never" if not np.isfinite(m["TTFT_median"]) else f"{m['TTFT_median']:.0f}"
            cells = [SNAME[se], k, fmt_ci(m["OSPA"], m["OSPA_CI"]), f"{100*m['custody']:.1f}",
                     f"{100*m['continuity']:.1f}", ttft, f"{m['RMSE']:.2f}",
                     fmt_ci(m["ANEES"], m["ANEES_CI"]), f"{m['C95']:.3f}"]
            t5.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(cells).replace("[", "[").replace("%", r"\%") + r" \\")
        tex.append(r"\midrule")
    open(os.path.join(TAB, "table5_medium.md"), "w").write("\n".join(t5) + "\n")
    open(os.path.join(TAB, "table5_medium.tex"), "w").write("\n".join(tex[:-1]) + "\n")

    # ------------------------------------------------ fusion paired comparisons
    per_run = {k: s["ospa"] / T for k, s in S.items()}
    fus = []
    for lv, k in itertools.product(LEVELS, ESTIMATORS):
        for base in ("ir", "radar"):
            r = metrics.paired(per_run[lv, "fused", k], per_run[lv, base, k])
            fus.append({"maneuver": lv, "estimator": k, "baseline": base, **r})
    adj = metrics.holm([r["p"] for r in fus])
    for r, a in zip(fus, adj):
        r["p_holm"] = float(a)

    # ---------------------------------------------- estimator comparisons (36)
    comps = [("IMM-CV/CA", "EKF-CV"), ("IMM-CV/CA", "UKF-CV"), ("IMM-CV/CA", "EKF-CA"), ("UKF-CV", "EKF-CV")]
    est = []
    for lv, se in itertools.product(LEVELS, SENSING):
        for a, b in comps:
            r = metrics.paired(per_run[lv, se, a], per_run[lv, se, b])
            est.append({"maneuver": lv, "sensing": se, "first": a, "second": b, **r})
    adj = metrics.holm([r["p"] for r in est])
    for r, a in zip(est, adj):
        r["p_holm"] = float(a)
    t6 = ["| Maneuver | Comparison | Delta OSPA (km) | Better (%) | Adjusted p |", "|---|---|---|---|---|"]
    for r in est:
        if r["sensing"] != "fused" or (r["first"], r["second"]) == ("IMM-CV/CA", "UKF-CV"):
            continue
        d = "< 0.001" if abs(r["delta"]) < 5e-4 else f"{r['delta']:+.3f}"
        t6.append(f"| {r['maneuver'].capitalize()} | {r['first']} vs {r['second']} | {d} | "
                  f"{100*r['better']:.1f} | {p_fmt(r['p_holm'])} |")
    open(os.path.join(TAB, "table6_paired.md"), "w").write("\n".join(t6) + "\n")
    with open(os.path.join(TAB, "paired_all.json"), "w") as f:
        json.dump({"fusion": fus, "estimators": est}, f, indent=2)

    # ------------------------------------------------------- tuning (Table A.1)
    tun = json.load(open(os.path.join(RES, "tuning.json")))
    t7 = ["| Estimator | Setting | Mean OSPA (km) | Custody (%) | ANEES | C95 | Selected |", "|---|---|---|---|---|---|---|"]
    for r in tun:
        sett = ", ".join(f"{k}={v:g}" for k, v in r.items() if k in ("qv_cv", "qa_ca", "imm_qv", "imm_qa", "imm_ps"))
        t7.append(f"| {r['estimator']} | {sett} | {r['OSPA']:.2f} | {100*r['custody']:.1f} | {r['ANEES']:.2f} | "
                  f"{r['C95']:.3f} | {'yes' if r['selected'] else ''} |")
    open(os.path.join(TAB, "tableA1_tuning.md"), "w").write("\n".join(t7) + "\n")

    # ------------------------------------------------------------ headline numbers
    K = {}
    def rng_(vals, d=1):
        return [round(float(min(vals)), d), round(float(max(vals)), d)]

    red_ir = [100 * (1 - summ[lv, "fused", k]["OSPA"] / summ[lv, "ir", k]["OSPA"]) for lv in LEVELS for k in ESTIMATORS]
    red_rad = [100 * (1 - summ[lv, "fused", k]["OSPA"] / summ[lv, "radar", k]["OSPA"]) for lv in LEVELS for k in ESTIMATORS]
    K["fusion_OSPA_reduction_vs_IR_pct"] = rng_(red_ir)
    K["fusion_OSPA_reduction_vs_radar_pct"] = rng_(red_rad)
    K["TTFT_fused_s"] = rng_([summ[lv, "fused", k]["TTFT_median"] for lv in LEVELS for k in ESTIMATORS], 0)
    K["TTFT_radar_s"] = rng_([summ[lv, "radar", k]["TTFT_median"] for lv in LEVELS for k in ESTIMATORS], 0)
    K["fused_lower_than_IR_share_pct"] = rng_([100 * r["better"] for r in fus if r["baseline"] == "ir"])
    K["fused_lower_than_radar_share_pct"] = rng_([100 * r["better"] for r in fus if r["baseline"] == "radar"])
    K["fusion_max_p_holm"] = float(max(r["p_holm"] for r in fus))
    imm_cv = [r for r in est if r["sensing"] == "fused" and r["first"] == "IMM-CV/CA" and r["second"] == "EKF-CV"]
    imm_ca = [r for r in est if r["sensing"] == "fused" and r["first"] == "IMM-CV/CA" and r["second"] == "EKF-CA"]
    K["IMM_vs_EKFCV_fused_rel_reduction_pct"] = {r["maneuver"]: round(-100 * r["rel"], 2) for r in imm_cv}
    K["IMM_vs_EKFCV_fused_better_pct"] = {r["maneuver"]: round(100 * r["better"], 1) for r in imm_cv}
    K["IMM_vs_EKFCA_fused_rel_reduction_pct"] = {r["maneuver"]: round(-100 * r["rel"], 2) for r in imm_ca}
    K["IMM_max_p_holm_fused"] = float(max(r["p_holm"] for r in imm_cv + imm_ca))
    ukf = [abs(summ[lv, se, "UKF-CV"]["OSPA"] - summ[lv, se, "EKF-CV"]["OSPA"]) for lv in LEVELS for se in SENSING]
    K["UKF_EKF_max_abs_OSPA_diff_km"] = float(max(ukf))
    K["IMM_lowest_OSPA_every_cell"] = bool(all(
        min(ESTIMATORS, key=lambda k: summ[lv, se, k]["OSPA"]) == "IMM-CV/CA" for lv in LEVELS for se in SENSING))
    K["EKFCA_highest_OSPA_every_cell"] = bool(all(
        max(ESTIMATORS, key=lambda k: summ[lv, se, k]["OSPA"]) == "EKF-CA" for lv in LEVELS for se in SENSING))
    K["ANEES_fused_range"] = rng_([summ[lv, "fused", k]["ANEES"] for lv in LEVELS for k in ESTIMATORS], 2)
    K["C95_fused_range"] = rng_([summ[lv, "fused", k]["C95"] for lv in LEVELS for k in ESTIMATORS], 3)
    single = ("EKF-CV", "UKF-CV", "EKF-CA")
    K["IMM_ANEES_reduction_vs_single_fused_pct"] = rng_(
        [100 * (1 - summ[lv, "fused", "IMM-CV/CA"]["ANEES"] / summ[lv, "fused", k]["ANEES"]) for lv in LEVELS for k in single])
    K["custody_fused_IMM_pct"] = rng_([100 * summ[lv, "fused", "IMM-CV/CA"]["custody"] for lv in LEVELS])
    K["custody_ir_IMM_pct"] = rng_([100 * summ[lv, "ir", "IMM-CV/CA"]["custody"] for lv in LEVELS])
    m = {se: summ["medium", se, "IMM-CV/CA"] for se in SENSING}
    K["medium_IMM"] = {se: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in m[se].items()} for se in SENSING}
    K["medium_IMM_reduction_vs_IR_pct"] = 100 * (1 - m["fused"]["OSPA"] / m["ir"]["OSPA"])
    K["medium_IMM_reduction_vs_radar_pct"] = 100 * (1 - m["fused"]["OSPA"] / m["radar"]["OSPA"])

    sc = {lv: dict(np.load(os.path.join(RES, "main", f"{lv}_scenario.npz"))) for lv in LEVELS}
    K["peak_lateral_accel_median_g"] = {lv: round(float(np.median(sc[lv]["peak_alat_g"])), 2) for lv in LEVELS}
    K["mean_visible_satellites"] = round(float(np.mean([sc[lv]["n_visible"].mean() for lv in LEVELS])), 2)
    K["radar_coverage_mean_pct"] = round(100 * float(np.mean([sc[lv]["coverage"].mean() for lv in LEVELS])), 1)
    K["never_in_radar_coverage_medium_pct"] = round(100 * float((~sc["medium"]["coverage"].any(1)).mean()), 1)
    K["realized_outage_fraction"] = round(float(np.mean([sc[lv]["outage"].mean() for lv in LEVELS])), 3)
    K["altitude_range_km"] = rng_([sc[lv]["alt"].min() for lv in LEVELS] + [sc[lv]["alt"].max() for lv in LEVELS])

    tim = json.load(open(os.path.join(RES, "timing.json")))
    per = {}
    for k in ESTIMATORS:
        v = [tim[f"{lv}_{se}_{k.replace('/', '')}"]["us_per_cycle_run"] for lv in LEVELS for se in ("ir", "fused")]
        per[k] = rng_(v, 1)
    K["us_per_cycle_per_run"] = per
    K["main_cpu_seconds_total"] = round(sum(v["cpu_s"] for v in tim.values()), 1)
    K["tuned"] = json.load(open(os.path.join(RES, "tuned.json")))
    K["table1"] = table1()
    K["n_runs_per_level"] = N
    with open(os.path.join(RES, "key_numbers.json"), "w") as f:
        json.dump(K, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in K.items() if k not in ("medium_IMM", "table1")}, indent=1, default=float))


if __name__ == "__main__":
    main()
