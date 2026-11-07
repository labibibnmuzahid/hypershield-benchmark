"""Credibility diagnostics (Section 7.4): where the excess NEES comes from.

Writes results/conditional.json and results/conditional_series.npz (for Figs. 6-8).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import chi2

from hsbench.config import ESTIMATORS, MANEUVER_LEVELS, default_config

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")


def load(level, sensing, kind):
    return dict(np.load(os.path.join(RES, "main", f"{level}_{sensing}_{kind.replace('/', '')}.npz")))


def moving_mean(x, w=9):
    k = np.ones(w) / w
    valid = np.isfinite(x)
    num = np.convolve(np.where(valid, x, 0.0), k, mode="same")
    den = np.convolve(valid.astype(float), k, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def main():
    cfg = default_config()
    alphas = np.array(cfg.metrics.alphas)
    out, series = {}, {}
    sc = dict(np.load(os.path.join(RES, "main", "medium_scenario.npz")))
    acc_g = sc["a_lat"] / 9.80665e-3
    for kind in ESTIMATORS:
        tag = kind.replace("/", "")
        d = load("medium", "fused", kind)
        conf, nees, rf = d["conf"], d["nees"].astype(float), d["recent_false"]
        v = nees[conf]
        fa = rf[conf]
        n_norm = v / 3.0
        srt = np.sort(v)[::-1]
        top = srt[: max(1, int(np.ceil(0.05 * len(srt))))].sum() / srt.sum()
        thr95 = chi2.ppf(0.95, 3)
        acc = acc_g[conf]
        bins = [0, 0.5, 1.0, 1.5, np.inf]
        acc_bins = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (acc >= lo) & (acc < hi)
            acc_bins.append({"range_g": [lo, None if not np.isfinite(hi) else hi],
                             "share": float(m.mean()), "ANEES": float(v[m].mean() / 3) if m.any() else None})
        out[kind] = {
            "median_normalized_NEES": float(np.median(n_norm)),
            "worst5pct_share_of_NEES": float(top),
            "false_window_share_of_epochs": float(fa.mean()),
            "ANEES_after_false": float(v[fa].mean() / 3),
            "ANEES_otherwise": float(v[~fa].mean() / 3),
            "C95_after_false": float((v[fa] <= thr95).mean()),
            "C95_otherwise": float((v[~fa] <= thr95).mean()),
            "false_window_share_of_NEES": float(v[fa].sum() / v.sum()),
            "false_update_share": float(d["n_false_updates"].sum() / max(d["n_updates"].sum(), 1)),
            "acceleration_bins": acc_bins,
            "coverage": {f"{a:.2f}": float((v <= chi2.ppf(a, 3)).mean()) for a in alphas},
        }
        # time-resolved ANEES and its acceptance region
        nk = conf.sum(0)
        an = np.where(nk > 0, np.nansum(np.where(conf, nees, 0.0), 0) / np.maximum(nk, 1) / 3, np.nan)
        series[f"anees_t_{tag}"] = an
        series[f"anees_t_smooth_{tag}"] = moving_mean(an, 9)
        series[f"nk_{tag}"] = nk
        cov_curve_a = np.linspace(0.01, 0.99, 99)
        series[f"cov_curve_{tag}"] = np.array([(v <= chi2.ppf(a, 3)).mean() for a in cov_curve_a])
        series["cov_curve_alpha"] = cov_curve_a
        if kind == "IMM-CV/CA":
            lo = chi2.ppf(0.025, 3 * np.maximum(nk, 1)) / (3 * np.maximum(nk, 1))
            hi = chi2.ppf(0.975, 3 * np.maximum(nk, 1)) / (3 * np.maximum(nk, 1))
            series["accept_lo"], series["accept_hi"] = lo, hi
    # ANEES change with maneuver level, IMM fused
    out["IMM_fused_ANEES_by_level"] = {}
    for lv in MANEUVER_LEVELS:
        d = load(lv, "fused", "IMM-CV/CA")
        out["IMM_fused_ANEES_by_level"][lv] = float(np.nanmean(d["nees"][d["conf"]]) / 3)
    with open(os.path.join(RES, "conditional.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(RES, "conditional_series.npz"), **series)
    for k in ESTIMATORS:
        o = out[k]
        print(f"{k:10s} med={o['median_normalized_NEES']:.2f} top5={o['worst5pct_share_of_NEES']:.2f} "
              f"falsewin={o['false_window_share_of_epochs']:.3f} ANEES fa={o['ANEES_after_false']:.2f} "
              f"other={o['ANEES_otherwise']:.2f} C95 fa={o['C95_after_false']:.2f} other={o['C95_otherwise']:.2f} "
              f"NEESshare={o['false_window_share_of_NEES']:.2f}")


if __name__ == "__main__":
    main()
