"""Evaluation metrics and statistics (Section 6).

Per-epoch arrays (runs x epochs) are reduced to per-run sums so that pooled
statistics and their run-level bootstrap intervals are exact ratios of sums.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, wilcoxon


def per_epoch(result: dict, truth_pos: np.ndarray) -> dict:
    """Position error norm and NEES at every epoch (NaN where no confirmed track)."""
    conf = result["confirmed"]
    err = result["est"] - truth_pos
    e = np.linalg.norm(err, axis=-1)
    nees = np.full(conf.shape, np.nan)
    P = result["Ppos"][conf]
    ev = err[conf]
    nees[conf] = np.einsum("ni,ni->n", ev, np.linalg.solve(P, ev[..., None])[..., 0])
    e = np.where(conf, e, np.nan)
    sig = np.where(conf, np.sqrt(np.trace(np.nan_to_num(result["Ppos"]), axis1=-2, axis2=-1)), np.nan)
    return {"err": e.astype(np.float32), "nees": nees.astype(np.float32), "conf": conf,
            "recent_false": result["recent_false"], "sigma": sig.astype(np.float32),
            "segments": result["segments"], "n_updates": result["n_updates"],
            "n_false_updates": result["n_false_updates"]}


def run_sums(ep: dict, mcfg) -> dict:
    """Per-run numerators/denominators for every pooled metric."""
    conf = ep["conf"]
    e = np.nan_to_num(ep["err"].astype(float), nan=0.0)
    nees = np.nan_to_num(ep["nees"].astype(float), nan=0.0)
    T = conf.shape[1]
    c = mcfg.ospa_cutoff
    ospa = np.where(conf, np.minimum(e, c), c)
    thr = chi2.ppf(np.array(mcfg.alphas), 3)
    within = (nees[..., None] <= thr) & conf[..., None]
    ok5 = conf & (e <= mcfg.ttft_radius)
    ttft = np.where(ok5.any(1), ok5.argmax(1).astype(float), np.inf)
    return {
        "T": T,
        "ospa": ospa.sum(1),
        "custody": (conf & (e <= mcfg.custody_radius)).sum(1),
        "continuity": conf.sum(1),
        "n_conf": conf.sum(1),
        "sq_err": (e ** 2 * conf).sum(1),
        "nees": (nees * conf).sum(1),
        "cover": within.sum(1),             # (N, n_alpha)
        "ttft": ttft,
        "segments": ep["segments"],
    }


def point_metrics(s: dict) -> dict:
    T = s["T"]
    N = len(s["ospa"])
    nc = s["n_conf"].sum()
    out = {
        "OSPA": s["ospa"].sum() / (N * T),
        "custody": s["custody"].sum() / (N * T),
        "continuity": s["continuity"].sum() / (N * T),
        "segments": s["segments"].mean(),
        "TTFT_median": float(np.median(s["ttft"])),
        "ever_tracked": float(np.isfinite(s["ttft"]).mean()),
        "RMSE": np.sqrt(s["sq_err"].sum() / nc) if nc else np.nan,
        "ANEES": s["nees"].sum() / nc / 3.0 if nc else np.nan,
    }
    cov = s["cover"].sum(0) / nc if nc else np.full(s["cover"].shape[1], np.nan)
    for a, v in zip((50, 68, 90, 95, 99), cov):
        out[f"C{a}"] = v
    return out


def bootstrap_counts(N: int, B: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(float)


def bootstrap_ci(num, den, counts, level=0.95):
    """Percentile interval for sum(num)/sum(den) with runs resampled."""
    stat = (counts @ num) / np.maximum(counts @ den, 1e-300)
    lo, hi = np.percentile(stat, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def summarize(s: dict, counts) -> dict:
    out = point_metrics(s)
    N, T = len(s["ospa"]), s["T"]
    ones = np.full(N, float(T))
    out["OSPA_CI"] = bootstrap_ci(s["ospa"], ones, counts)
    out["custody_CI"] = bootstrap_ci(s["custody"].astype(float), ones, counts)
    out["ANEES_CI"] = tuple(v / 3 for v in bootstrap_ci(s["nees"], s["n_conf"].astype(float), counts))
    out["C95_CI"] = bootstrap_ci(s["cover"][:, 3].astype(float), s["n_conf"].astype(float), counts)
    return out


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def paired(a_run_ospa, b_run_ospa):
    """Mean difference (a-b), share of runs where a < b, Wilcoxon signed-rank p."""
    d = a_run_ospa - b_run_ospa
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(a_run_ospa, b_run_ospa, zero_method="wilcox").pvalue)
    return {"delta": float(d.mean()), "better": float((d < 0).mean()),
            "rel": float(d.mean() / b_run_ospa.mean()), "p": p}
