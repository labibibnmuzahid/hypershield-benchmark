"""Randomly maneuvering glide trajectories over a spherical Earth (Section 4.1).

All N runs are integrated together with a 0.05 s Euler step and sampled at 1 Hz.
"""
from __future__ import annotations

import numpy as np

from .config import G0, RE, SOUND, ScenarioCfg
from .geometry import destination, enu_basis, geodetic_to_ecef, initial_bearing, wrap


def generate(cfg: ScenarioCfg, n_runs: int, rng: np.random.Generator) -> dict:
    N = n_runs
    lat_r, lon_r = np.radians(cfg.radar_lat_deg), np.radians(cfg.radar_lon_deg)

    # ---- per-run random parameters (drawn in a fixed order, independent of amax)
    dist0 = rng.uniform(*cfg.start_dist, N)
    bearing_from_radar = rng.uniform(-np.pi, np.pi, N)
    offset = np.radians(rng.uniform(-cfg.heading_offset_deg, cfg.heading_offset_deg, N))
    mach = rng.uniform(*cfg.mach, N)
    decel = rng.uniform(*cfg.decel, N)
    h0 = rng.uniform(*cfg.h0, N)
    amp = rng.uniform(*cfg.alt_amp, N)
    period = rng.uniform(*cfg.alt_period, N)
    phase = rng.uniform(0.0, 2 * np.pi, N)
    sink = rng.uniform(*cfg.sink, N)
    n_seg = int(np.ceil(cfg.duration / cfg.hold[0])) + 2
    holds = rng.uniform(*cfg.hold, (N, n_seg))
    cmd_unit = rng.uniform(-1.0, 1.0, (N, n_seg))          # scaled by amax below
    amax = cfg.amax_g * G0
    cmds = cmd_unit * amax
    switch_t = np.cumsum(holds, axis=1)                     # end time of each segment

    lat0, lon0 = destination(np.full(N, lat_r), np.full(N, lon_r), bearing_from_radar, dist0)
    psi0 = initial_bearing(lat0, lon0, np.full(N, lat_r), np.full(N, lon_r)) + offset
    v0 = mach * SOUND

    # plane of the initial great circle (for the heading-hold reference)
    B0 = enu_basis(lat0, lon0)
    t0 = np.cos(psi0)[:, None] * B0[:, 1] + np.sin(psi0)[:, None] * B0[:, 0]
    pole = np.cross(B0[:, 2], t0)

    def alt(t):
        return h0 + amp * (np.sin(2 * np.pi * t / period + phase) - np.sin(phase)) - sink * t

    def alt_rate(t):
        return amp * (2 * np.pi / period) * np.cos(2 * np.pi * t / period + phase) - sink

    dt = cfg.dt_int
    n_steps = int(round(cfg.duration / dt))
    every = int(round(1.0 / dt))
    n_out = n_steps // every + 1

    lat, lon, psi = lat0.copy(), lon0.copy(), psi0.copy()
    a_f = np.zeros(N)
    seg = np.zeros(N, dtype=int)
    rows = np.arange(N)

    out_lat = np.empty((N, n_out)); out_lon = np.empty((N, n_out)); out_psi = np.empty((N, n_out))
    out_alat = np.empty((N, n_out)); out_cmd = np.empty((N, n_out))

    for step in range(n_steps + 1):
        t = step * dt
        seg = seg + (t >= switch_t[rows, seg])
        a_c = cmds[rows, seg]
        h = alt(t)
        v = v0 - decel * t
        hd = alt_rate(t)
        vh = np.sqrt(np.maximum(v * v - hd * hd, 1e-9))
        # heading of the initial great circle at the current position
        B = enu_basis(lat, lon)
        tang = np.cross(pole, B[:, 2])
        psi_ref = np.arctan2(np.sum(tang * B[:, 0], -1), np.sum(tang * B[:, 1], -1))
        a_lat = a_f - cfg.k_psi * v * wrap(psi - psi_ref)
        if step % every == 0:
            k = step // every
            out_lat[:, k], out_lon[:, k], out_psi[:, k] = lat, lon, psi
            out_alat[:, k], out_cmd[:, k] = a_lat, a_c
        if step == n_steps:
            break
        r = RE + h
        dlat = vh * np.cos(psi) / r
        dlon = vh * np.sin(psi) / (r * np.cos(lat))
        dpsi = a_lat / v + vh * np.sin(psi) * np.tan(lat) / r
        lat = lat + dt * dlat
        lon = lon + dt * dlon
        psi = psi + dt * dpsi
        a_f = a_f + dt * (a_c - a_f) / cfg.lag_tau

    times = np.arange(n_out) * (every * dt)
    tt = times[None, :]
    h0, amp, period, phase, sink = (a[:, None] for a in (h0, amp, period, phase, sink))
    h_out = alt(tt)
    hd_out = alt_rate(tt)
    v_out = v0[:, None] - decel[:, None] * times[None, :]
    vh_out = np.sqrt(np.maximum(v_out ** 2 - hd_out ** 2, 1e-9))
    pos = geodetic_to_ecef(out_lat, out_lon, h_out)
    Bo = enu_basis(out_lat, out_lon)
    vel = (vh_out[..., None] * (np.cos(out_psi)[..., None] * Bo[..., 1, :]
                                 + np.sin(out_psi)[..., None] * Bo[..., 0, :])
           + hd_out[..., None] * Bo[..., 2, :])
    return {
        "times": times, "pos": pos, "vel": vel, "lat": out_lat, "lon": out_lon,
        "alt": h_out, "heading": out_psi, "a_lat": out_alat, "a_cmd": out_cmd,
        "mach0": mach, "peak_alat_g": np.max(np.abs(out_alat), axis=1) / G0,
    }
