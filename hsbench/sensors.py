"""Space IR layer and ground radar detection generators (Sections 4.2-4.3).

Detections are generated once per Monte Carlo run, before any tracker runs, so
all estimators and sensing configurations see identical data. Every random
quantity is drawn per second of the scenario from its own stream, with coupled
(inverse-CDF / threshold) draws, so sweeping one parameter changes as little of
the realisation as possible (common random numbers).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from .config import RE, IRCfg, RadarCfg, ScenarioCfg
from .geometry import (enu_basis, geodetic_to_ecef, ir_frame, ir_measure,
                       radar_los_range, radar_measure, segment_clears_earth,
                       walker_positions)

MAX_FA = 15  # cap on false alarms per scan (P(Poisson(3) > 15) ~ 1e-7)


def radar_site(scn: ScenarioCfg, rad: RadarCfg):
    lat, lon = np.radians(scn.radar_lat_deg), np.radians(scn.radar_lon_deg)
    r0 = geodetic_to_ecef(lat, lon, rad.mast)
    M = enu_basis(lat, lon)
    return r0, M


def _pack(vals, valid, is_tgt):
    """Move valid entries to the front along axis -2 (detection axis)."""
    order = np.argsort(~valid, axis=-1, kind="stable")
    vals = np.take_along_axis(vals, order[..., None], axis=-2)
    valid = np.take_along_axis(valid, order, axis=-1)
    is_tgt = np.take_along_axis(is_tgt, order, axis=-1)
    dmax = max(int(valid.sum(-1).max()), 1)
    return vals[..., :dmax, :], valid[..., :dmax], is_tgt[..., :dmax]


def draw_randoms(seed_seq: np.random.SeedSequence, n_runs: int, n_t: int):
    """Pre-draw all sensor random numbers (independent of sensor parameters)."""
    ss = seed_seq.spawn(4)
    g_const, g_out, g_ir, g_rad = (np.random.default_rng(s) for s in ss)
    K = 3
    R = {
        "raan0": g_const.uniform(0, 2 * np.pi, n_runs),
        "u0": g_const.uniform(0, 2 * np.pi, n_runs),
        "out_u": g_out.uniform(size=(n_runs, n_t)),
        "ir_det_u": g_ir.uniform(size=(n_runs, n_t, K)).astype(np.float32),
        "ir_noise": g_ir.standard_normal((n_runs, n_t, K, 2)).astype(np.float32),
        "ir_fa_u": g_ir.uniform(size=(n_runs, n_t, K)).astype(np.float32),
        "ir_fa_pos": g_ir.uniform(-1, 1, (n_runs, n_t, K, MAX_FA, 2)).astype(np.float32),
        "rad_det_u": g_rad.uniform(size=(n_runs, n_t)).astype(np.float32),
        "rad_noise": g_rad.standard_normal((n_runs, n_t, 3)).astype(np.float32),
        "rad_fa_u": g_rad.uniform(size=(n_runs, n_t)).astype(np.float32),
        "rad_fa_pos": g_rad.uniform(-1, 1, (n_runs, n_t, MAX_FA, 3)).astype(np.float32),
        "init_key": g_rad.uniform(size=(n_runs, n_t)),
    }
    return R


def outage_states(u, fraction, mean_dur):
    """Two-state Markov outage chain, one step per second. Returns bool (N, T)."""
    N, T = u.shape
    out = np.zeros((N, T), dtype=bool)
    if fraction <= 0:
        return out
    p_exit = 1.0 / mean_dur
    p_enter = fraction / (1.0 - fraction) * p_exit
    state = u[:, 0] < fraction
    out[:, 0] = state
    for k in range(1, T):
        state = np.where(state, u[:, k] >= p_exit, u[:, k] < p_enter)
        out[:, k] = state
    return out


def generate_ir(traj, cfg: IRCfg, R) -> dict:
    pos = traj["pos"]
    N, T, _ = pos.shape
    times = traj["times"]
    rev = int(round(cfg.revisit))
    scan_k = np.arange(0, T, rev)
    K = cfg.n_tasked
    out = outage_states(R["out_u"], cfg.outage_fraction, cfg.outage_mean)
    lam = cfg.fa_rate
    n_scan = len(scan_k)
    sat_pos = np.zeros((N, n_scan, K, 3))
    slot_ok = np.zeros((N, n_scan, K), dtype=bool)
    n_vis = np.zeros((N, n_scan), dtype=int)
    raw = np.zeros((N, n_scan, K, 1 + MAX_FA, 2))
    valid = np.zeros((N, n_scan, K, 1 + MAX_FA), dtype=bool)
    is_tgt = np.zeros((N, n_scan, K, 1 + MAX_FA), dtype=bool)
    inc = np.radians(cfg.inclination_deg)
    rows = np.arange(N)
    for i, k in enumerate(scan_k):
        s_all = walker_positions(times[k:k + 1], cfg.n_sats, cfg.n_planes, cfg.phasing,
                                 cfg.altitude, inc, R["raan0"], R["u0"])[:, 0]   # (N, S, 3)
        p = pos[:, k]
        rel = p[:, None, :] - s_all
        rng_ = np.linalg.norm(rel, axis=-1)
        vis = segment_clears_earth(s_all, np.broadcast_to(p[:, None, :], s_all.shape)) \
            & (rng_ < cfg.max_range)
        n_vis[:, i] = vis.sum(1)
        # D-optimal greedy tasking on the summed angle-only information, Eq. (7)
        u = rel / rng_[..., None]
        Ji = (np.eye(3) - u[..., :, None] * u[..., None, :]) / (rng_ ** 2)[..., None, None]
        avail = vis.copy()
        Jsum = np.zeros((N, 3, 3))
        chosen = np.full((N, K), -1)
        first = np.where(avail, rng_, np.inf).argmin(1)
        has = avail[rows, first]
        chosen[:, 0] = np.where(has, first, -1)
        Jsum[has] += Ji[rows, first][has]
        avail[rows[has], first[has]] = False
        for j in range(1, K):
            score = np.linalg.det(Jsum[:, None] + Ji)
            score = np.where(avail, score + 1e-30 / rng_, -np.inf)   # tie-break: nearer first
            nxt = score.argmax(1)
            has = avail[rows, nxt]
            chosen[:, j] = np.where(has, nxt, -1)
            Jsum[has] += Ji[rows, nxt][has]
            avail[rows[has], nxt[has]] = False
        ok = (chosen >= 0) & ~out[:, k][:, None]
        s_sel = s_all[rows[:, None], np.maximum(chosen, 0)]             # (N, K, 3)
        sat_pos[:, i] = s_sel
        slot_ok[:, i] = ok
        z_true = ir_measure(p[:, None, :], s_sel)                       # (N, K, 2)
        det = ok & (R["ir_det_u"][:, k] < cfg.pd)
        raw[:, i, :, 0] = z_true + cfg.sigma * R["ir_noise"][:, k]
        valid[:, i, :, 0] = det
        is_tgt[:, i, :, 0] = True
        n_fa = poisson.ppf(R["ir_fa_u"][:, k], lam).astype(int) if lam > 0 else np.zeros((N, K), int)
        n_fa = np.minimum(n_fa, MAX_FA) * ok
        raw[:, i, :, 1:] = z_true[:, :, None, :] + cfg.fa_halfwidth * R["ir_fa_pos"][:, k]
        valid[:, i, :, 1:] = np.arange(MAX_FA)[None, None, :] < n_fa[..., None]
    raw, valid, is_tgt = _pack(raw, valid, is_tgt)
    raw = (raw + np.pi) % (2 * np.pi) - np.pi
    return {"scan_k": scan_k, "sat_pos": sat_pos, "slot_ok": slot_ok, "z": raw,
            "valid": valid, "is_tgt": is_tgt, "outage": out, "n_visible": n_vis}


def generate_radar(traj, scn: ScenarioCfg, cfg: RadarCfg, R) -> dict:
    pos = traj["pos"]
    N, T, _ = pos.shape
    r0, M = radar_site(scn, cfg)
    z_true = radar_measure(pos, r0, M)                                   # (N, T, 3)
    los = radar_los_range(cfg.mast, traj["alt"], cfg.k_factor)
    cover = (z_true[..., 0] <= los) & (z_true[..., 0] < cfg.max_range)
    rev = int(round(cfg.revisit))
    scan = np.zeros(T, dtype=bool)
    scan[::rev] = True
    ok = cover & scan[None, :]
    sig = np.array([cfg.sigma_r, cfg.sigma_ang, cfg.sigma_ang])
    half = np.array([cfg.fa_half_r, cfg.fa_half_ang, cfg.fa_half_ang])
    raw = np.zeros((N, T, 1 + MAX_FA, 3))
    valid = np.zeros((N, T, 1 + MAX_FA), dtype=bool)
    is_tgt = np.zeros((N, T, 1 + MAX_FA), dtype=bool)
    raw[:, :, 0] = z_true + sig * R["rad_noise"]
    valid[:, :, 0] = ok & (R["rad_det_u"] < cfg.pd)
    is_tgt[:, :, 0] = True
    lam = cfg.fa_rate
    n_fa = poisson.ppf(R["rad_fa_u"], lam).astype(int) if lam > 0 else np.zeros((N, T), int)
    n_fa = np.minimum(n_fa, MAX_FA) * ok
    raw[:, :, 1:] = z_true[:, :, None, :] + half * R["rad_fa_pos"]
    valid[:, :, 1:] = np.arange(MAX_FA)[None, None, :] < n_fa[..., None]
    raw, valid, is_tgt = _pack(raw, valid, is_tgt)
    raw[..., 1:] = (raw[..., 1:] + np.pi) % (2 * np.pi) - np.pi
    return {"r0": r0, "M": M, "z": raw, "valid": valid, "is_tgt": is_tgt,
            "coverage": cover, "init_key": R["init_key"]}


def generate_dataset(cfg, n_runs: int, entropy: int) -> dict:
    """Trajectories plus detections for one experiment cell (common random numbers)."""
    from . import trajectory
    root = np.random.SeedSequence(entropy)
    ss_traj, ss_sens = root.spawn(2)
    traj = trajectory.generate(cfg.scenario, n_runs, np.random.default_rng(ss_traj))
    R = draw_randoms(ss_sens, n_runs, len(traj["times"]))
    return {"traj": traj,
            "ir": generate_ir(traj, cfg.ir, R),
            "radar": generate_radar(traj, cfg.scenario, cfg.radar, R),
            "randoms": R}


def regenerate_sensors(data: dict, cfg) -> dict:
    """New sensor realisation for changed sensor parameters, same trajectories/randoms."""
    traj, R = data["traj"], data["randoms"]
    return {"traj": traj, "ir": generate_ir(traj, cfg.ir, R),
            "radar": generate_radar(traj, cfg.scenario, cfg.radar, R), "randoms": R}
