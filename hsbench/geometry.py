"""Spherical-Earth geometry, sensor frames and measurement functions.

Frame: Earth-centred Cartesian (non-rotating), km. All functions broadcast over
leading batch dimensions; the last axis holds xyz components.
"""
from __future__ import annotations

import numpy as np

from .config import MU, RE


# ----------------------------------------------------------------- basics
def wrap(a):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def geodetic_to_ecef(lat, lon, h):
    r = RE + h
    cl = np.cos(lat)
    return np.stack([r * cl * np.cos(lon), r * cl * np.sin(lon), r * np.sin(lat)], axis=-1)


def enu_basis(lat, lon):
    """Unit east, north, up vectors (shape (..., 3, 3), rows E, N, U)."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    e = np.stack([-so, co, np.zeros_like(lat)], axis=-1)
    n = np.stack([-sl * co, -sl * so, cl], axis=-1)
    u = np.stack([cl * co, cl * so, sl], axis=-1)
    return np.stack([e, n, u], axis=-2)


def destination(lat, lon, bearing, dist):
    """Point reached from (lat, lon) along a great circle (dist in km)."""
    d = dist / RE
    lat2 = np.arcsin(np.sin(lat) * np.cos(d) + np.cos(lat) * np.sin(d) * np.cos(bearing))
    lon2 = lon + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(lat),
                            np.cos(d) - np.sin(lat) * np.sin(lat2))
    return lat2, lon2


def initial_bearing(lat1, lon1, lat2, lon2):
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.arctan2(y, x)


def great_circle_distance(lat1, lon1, lat2, lon2):
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2.0 * RE * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def radar_los_range(h_radar, h_target, k=4.0 / 3.0):
    """Line-of-sight range under the k-Earth refraction model, Eq. (2)."""
    return np.sqrt(2 * k * RE * h_radar) + np.sqrt(2 * k * RE * np.maximum(h_target, 0.0))


def satellite_horizon(h_sat):
    """Geometric horizon distance of a satellite, Eq. (3)."""
    return np.sqrt((RE + h_sat) ** 2 - RE ** 2)


def segment_clears_earth(a, b, radius=RE):
    """True when the straight segment a-b stays above the sphere of `radius`."""
    d = b - a
    dd = np.sum(d * d, axis=-1)
    t = np.clip(-np.sum(a * d, axis=-1) / np.maximum(dd, 1e-12), 0.0, 1.0)
    closest = a + t[..., None] * d
    return np.linalg.norm(closest, axis=-1) > radius


# ----------------------------------------------------- Walker constellation
def walker_positions(times, n_sats, n_planes, phasing, alt, inc, raan0, u0):
    """Walker-delta i:T/P/F positions.

    times: (nt,), raan0/u0: (N,) per-run randomisation.
    Returns (N, nt, n_sats, 3).
    """
    a = RE + alt
    n = np.sqrt(MU / a ** 3)
    s_per = n_sats // n_planes
    p = np.repeat(np.arange(n_planes), s_per)                 # plane index
    s = np.tile(np.arange(s_per), n_planes)                    # slot in plane
    raan = raan0[:, None] + 2 * np.pi * p[None, :] / n_planes  # (N, S)
    phase = (2 * np.pi * s / s_per + 2 * np.pi * phasing * p / n_sats)[None, :]
    u = u0[:, None, None] + phase[:, None, :] + n * times[None, :, None]   # (N, nt, S)
    cu, su = np.cos(u), np.sin(u)
    ci, si = np.cos(inc), np.sin(inc)
    cO, sO = np.cos(raan)[:, None, :], np.sin(raan)[:, None, :]
    x = a * (cO * cu - sO * su * ci)
    y = a * (sO * cu + cO * su * ci)
    z = a * (su * si) * np.ones_like(cO)
    return np.stack([x, y, z], axis=-1)


# --------------------------------------------------------- IR measurement
def ir_frame(s):
    """Nadir-referenced sensor axes e1, e2, e3 for satellite position s (..., 3)."""
    e1 = -s / np.linalg.norm(s, axis=-1, keepdims=True)
    zhat = np.zeros_like(s)
    zhat[..., 2] = 1.0
    e2 = np.cross(zhat, e1)
    e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
    e3 = np.cross(e1, e2)
    return e1, e2, e3


def ir_measure(p, s, frame=None):
    """Two line-of-sight angles, Eq. (8) (noise-free). p, s broadcastable (..., 3)."""
    e1, e2, e3 = ir_frame(s) if frame is None else frame
    u = p - s
    a = np.sum(u * e1, -1)
    b = np.sum(u * e2, -1)
    c = np.sum(u * e3, -1)
    return np.stack([np.arctan2(b, a), np.arctan2(c, np.hypot(a, b))], axis=-1)


def ir_jacobian(p, s, frame=None):
    """d z_IR / d p, shape (..., 2, 3)."""
    e1, e2, e3 = ir_frame(s) if frame is None else frame
    u = p - s
    a = np.sum(u * e1, -1)[..., None]
    b = np.sum(u * e2, -1)[..., None]
    c = np.sum(u * e3, -1)[..., None]
    rab2 = a * a + b * b
    rab = np.sqrt(rab2)
    d1 = (a * e2 - b * e1) / rab2
    d2 = (rab * e3 - c * (a * e1 + b * e2) / rab) / (rab2 + c * c)
    return np.stack([d1, d2], axis=-2)


def ir_los_unit(z, frame):
    """Unit line of sight from measured angles (inverse of Eq. (8))."""
    e1, e2, e3 = frame
    t1, t2 = z[..., 0:1], z[..., 1:2]
    return np.cos(t2) * np.cos(t1) * e1 + np.cos(t2) * np.sin(t1) * e2 + np.sin(t2) * e3


# ------------------------------------------------------ radar measurement
def radar_measure(p, r0, M):
    """Range, azimuth, elevation in the radar ENU frame (M rows = E, N, U)."""
    d = np.einsum("ij,...j->...i", M, p - r0)
    rng = np.linalg.norm(d, axis=-1)
    az = np.arctan2(d[..., 0], d[..., 1])
    el = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1]))
    return np.stack([rng, az, el], axis=-1)


def radar_jacobian(p, r0, M):
    """d z_radar / d p, shape (..., 3, 3)."""
    d = np.einsum("ij,...j->...i", M, p - r0)
    dE, dN, dU = d[..., 0], d[..., 1], d[..., 2]
    rh2 = dE ** 2 + dN ** 2
    rh = np.sqrt(rh2)
    rho2 = rh2 + dU ** 2
    rho = np.sqrt(rho2)
    z = np.zeros_like(dE)
    J_rng = np.stack([dE / rho, dN / rho, dU / rho], -1)
    J_az = np.stack([dN / rh2, -dE / rh2, z], -1)
    J_el = np.stack([-dU * dE / (rh * rho2), -dU * dN / (rh * rho2), rh / rho2], -1)
    Jd = np.stack([J_rng, J_az, J_el], axis=-2)
    return Jd @ M


def radar_to_cartesian(z, r0, M):
    """Polar-to-Cartesian conversion and its Jacobian d p / d z."""
    rng, az, el = z[..., 0], z[..., 1], z[..., 2]
    ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
    d = np.stack([rng * ce * sa, rng * ce * ca, rng * se], -1)
    p = r0 + np.einsum("ji,...j->...i", M, d)
    dd = np.stack([
        np.stack([ce * sa, rng * ce * ca, -rng * se * sa], -1),
        np.stack([ce * ca, -rng * ce * sa, -rng * se * ca], -1),
        np.stack([se, np.zeros_like(rng), rng * ce], -1),
    ], axis=-2)                                   # d d_enu / d (r, az, el)
    J = np.einsum("ji,...jk->...ik", M, dd)       # M^T dd
    return p, J
