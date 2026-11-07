"""Run the benchmark experiments.

    python run_experiments.py tune      # process-noise grid search (seed 11, 200 runs)
    python run_experiments.py main      # 3 sensing x 4 estimators x 3 maneuver levels (500 runs each)
    python run_experiments.py sweeps    # one-at-a-time sensitivity (seed 33, 300 runs)
    python run_experiments.py example   # illustrative run (seed 44)
    python run_experiments.py all

Options: --workers N (default 2), --quick (small run counts for a smoke test).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from hsbench import metrics, sensors, tracker
from hsbench.config import (ESTIMATORS, MANEUVER_LEVELS, N_RUNS, SEEDS, SENSING,
                            default_config)

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
_DATA = {}


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _cell(args):
    """Run one (sensing, estimator) cell on the dataset held in _DATA (fork-shared)."""
    key, sensing, kind, cfg = args
    data = _DATA[key]
    t0 = time.process_time()
    r = tracker.run_tracker(data, sensing, kind, cfg)
    cpu = time.process_time() - t0
    ep = metrics.per_epoch(r, data["traj"]["pos"])
    return sensing, kind, ep, cpu


def _pool_map(jobs, workers):
    if workers <= 1:
        return [_cell(j) for j in jobs]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_cell, jobs, chunksize=1)


def load_tuned(cfg):
    path = os.path.join(RES, "tuned.json")
    if not os.path.exists(path):
        raise SystemExit("results/tuned.json missing: run `python run_experiments.py tune` first")
    with open(path) as f:
        t = json.load(f)
    for k, v in t.items():
        setattr(cfg.tracker, k, v)
    return cfg


# ------------------------------------------------------------------ tuning
TUNE_GRID = {
    "EKF-CV": [{"qv_cv": q} for q in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)],
    "EKF-CA": [{"qa_ca": q} for q in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)],
    "IMM-CV/CA": [{"imm_qv": qv, "imm_qa": qa, "imm_ps": ps}
                  for qv in (1e-5, 1e-4) for qa in (1e-5, 1e-4, 1e-3) for ps in (0.90, 0.95, 0.98)],
}


def tune(n_runs, workers):
    cfg = default_config()
    cfg.scenario.amax_g = MANEUVER_LEVELS["medium"]
    _DATA["tune"] = sensors.generate_dataset(cfg, n_runs, SEEDS["tune"])
    jobs, settings = [], []
    for kind, grid in TUNE_GRID.items():
        for g in grid:
            c = cfg.copy()
            for k, v in g.items():
                setattr(c.tracker, k, v)
            jobs.append(("tune", "fused", kind, c))
            settings.append((kind, g))
    out = _pool_map(jobs, workers)
    rows = []
    for (kind, g), (_, _, ep, _) in zip(settings, out):
        pm = metrics.point_metrics(metrics.run_sums(ep, cfg.metrics))
        rows.append({"estimator": kind, **g, **{k: pm[k] for k in ("OSPA", "custody", "ANEES", "C95")}})
    best = {}
    for kind in TUNE_GRID:
        cand = [r for r in rows if r["estimator"] == kind]
        b = min(cand, key=lambda r: r["OSPA"])
        for k in TUNE_GRID[kind][0]:
            best[k] = b[k]
        for r in cand:
            r["selected"] = r is b
    _ensure(RES)
    with open(os.path.join(RES, "tuning.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(RES, "tuned.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("selected:", best)
    return best


# -------------------------------------------------------------------- main
def main_eval(n_runs, workers):
    cfg = load_tuned(default_config())
    out_dir = _ensure(os.path.join(RES, "main"))
    cfg.save(os.path.join(RES, "defaults.json"))
    timing = {}
    for level, amax in MANEUVER_LEVELS.items():
        c = cfg.copy()
        c.scenario.amax_g = amax
        t0 = time.time()
        data = sensors.generate_dataset(c, n_runs, SEEDS[level])
        tr, ir, rd = data["traj"], data["ir"], data["radar"]
        np.savez_compressed(
            os.path.join(out_dir, f"{level}_scenario.npz"),
            peak_alat_g=tr["peak_alat_g"], a_lat=np.abs(tr["a_lat"]).astype(np.float32),
            alt=tr["alt"].astype(np.float32), coverage=rd["coverage"], outage=ir["outage"],
            n_visible=ir["n_visible"])
        _DATA[level] = data
        jobs = [(level, s, k, c) for s in SENSING for k in ESTIMATORS]
        res = _pool_map(jobs, workers)
        for sensing, kind, ep, cpu in res:
            tag = f"{level}_{sensing}_{kind.replace('/', '')}"
            np.savez_compressed(os.path.join(out_dir, tag + ".npz"), **ep)
            timing[tag] = {"cpu_s": cpu, "us_per_cycle_run": 1e6 * cpu / (n_runs * tr["pos"].shape[1])}
        del _DATA[level]
        print(f"{level}: {time.time() - t0:.1f} s")
    with open(os.path.join(RES, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)


# ------------------------------------------------------------------ sweeps
SWEEPS = {
    "ir_revisit": ("ir.revisit", [1.0, 2.0, 5.0, 10.0]),
    "outage_fraction": ("ir.outage_fraction", [0.0, 0.1, 0.2, 0.35, 0.5]),
    "ir_sigma_urad": ("ir.sigma", [50e-6, 100e-6, 150e-6, 200e-6]),
    "radar_pd": ("radar.pd", [0.5, 0.6, 0.7, 0.8, 0.9]),
    "false_alarms": ("fa_rate", [0.0, 0.5, 1.0, 2.0, 3.0]),
}


def _apply(cfg, path, value):
    if path == "fa_rate":
        cfg.ir.fa_rate = value
        cfg.radar.fa_rate = value
        return
    sec, name = path.split(".")
    setattr(getattr(cfg, sec), name, value)


def sweeps(n_runs, workers):
    cfg = load_tuned(default_config())
    cfg.scenario.amax_g = MANEUVER_LEVELS["medium"]
    base = sensors.generate_dataset(cfg, n_runs, SEEDS["sweeps"])
    rows = []
    for name, (path, values) in SWEEPS.items():
        for v in values:
            c = cfg.copy()
            _apply(c, path, v)
            key = f"{name}={v}"
            _DATA[key] = sensors.regenerate_sensors(base, c)
            jobs = [(key, s, k, c) for s in ("ir", "fused") for k in ESTIMATORS]
            for sensing, kind, ep, _ in _pool_map(jobs, workers):
                pm = metrics.point_metrics(metrics.run_sums(ep, c.metrics))
                nu = ep["n_updates"].sum()
                rows.append({"parameter": name, "value": v, "sensing": sensing, "estimator": kind,
                             "false_update_share": float(ep["n_false_updates"].sum() / max(nu, 1)),
                             **{k: (None if not np.isfinite(pm[k]) else pm[k]) for k in pm}})
            del _DATA[key]
        print("sweep", name, "done")
    _ensure(RES)
    with open(os.path.join(RES, "sweeps.json"), "w") as f:
        json.dump(rows, f, indent=2)


# ----------------------------------------------------------------- example
def example(pool_size=50):
    """Illustrative run. Selection rule: the first of 50 seed-44 medium-maneuver
    runs that enters radar coverage between 250 and 350 s and has at least three
    IR outages before radar entry (chosen to show every phase of a typical run)."""
    cfg = load_tuned(default_config())
    cfg.scenario.amax_g = MANEUVER_LEVELS["medium"]
    data = sensors.generate_dataset(cfg, pool_size, SEEDS["example"])
    cov, out = data["radar"]["coverage"], data["ir"]["outage"]
    pick = 0
    for i in range(pool_size):
        if not cov[i].any():
            continue
        enter = int(np.argmax(cov[i]))
        n_out = int(np.sum(np.diff(out[i, :enter].astype(int)) == 1) + out[i, 0])
        if 250 <= enter <= 350 and n_out >= 3:
            pick = i
            break
    sub = {"traj": {k: (v[pick:pick + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == pool_size else v)
                    for k, v in data["traj"].items()}}
    for s in ("ir", "radar"):
        sub[s] = {k: (v[pick:pick + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == pool_size else v)
                  for k, v in data[s].items()}
    save = {"run_index": pick, "lat": sub["traj"]["lat"][0], "lon": sub["traj"]["lon"][0],
            "alt": sub["traj"]["alt"][0], "coverage": sub["radar"]["coverage"][0],
            "outage": sub["ir"]["outage"][0], "times": data["traj"]["times"]}
    for sensing in SENSING:
        r = tracker.run_tracker(sub, sensing, "IMM-CV/CA", cfg)
        ep = metrics.per_epoch(r, sub["traj"]["pos"])
        save[f"err_{sensing}"] = ep["err"][0]
        save[f"sigma_{sensing}"] = ep["sigma"][0]
    _ensure(RES)
    np.savez_compressed(os.path.join(RES, "example.npz"), **save)
    print("example run index", pick, "radar entry", int(np.argmax(save["coverage"])), "s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tune", "main", "sweeps", "example", "all"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    nt, nm, ns = (N_RUNS["tune"], N_RUNS["main"], N_RUNS["sweeps"]) if not a.quick else (40, 60, 40)
    t0 = time.time()
    if a.what in ("tune", "all"):
        tune(nt, a.workers)
    if a.what in ("main", "all"):
        main_eval(nm, a.workers)
    if a.what in ("sweeps", "all"):
        sweeps(ns, a.workers)
    if a.what in ("example", "all"):
        example()
    print(f"total {time.time() - t0:.1f} s")
