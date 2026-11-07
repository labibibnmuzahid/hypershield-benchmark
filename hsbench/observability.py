"""Section 3: why the target is hard to observe (Table 1, Fig. 1)."""
from __future__ import annotations

import numpy as np

from .config import SOUND
from .geometry import radar_los_range, satellite_horizon

T_INF_32KM = 228.65   # K, U.S. Standard Atmosphere 1976 at 32 km
GAMMA = 1.4
WIEN = 2898.0         # um K


def stagnation_temperature(mach, t_inf=T_INF_32KM, gamma=GAMMA):
    """Calorically perfect gas, Eq. (1)."""
    return t_inf * (1 + 0.5 * (gamma - 1) * np.asarray(mach, float) ** 2)


def wien_peak_um(T):
    return WIEN / np.asarray(T, float)


def plasma_critical_density(f_hz):
    """Electron density whose plasma frequency equals f: f_p ~ 8.98 sqrt(n_e)."""
    return (np.asarray(f_hz, float) / 8.98) ** 2


def warning_time_s(h_target, mach, mast=0.010):
    return radar_los_range(mast, h_target) / (np.asarray(mach, float) * SOUND)


def table1() -> dict:
    return {
        "stagnation_K": stagnation_temperature([5, 7, 8]).tolist(),
        "wien_um": wien_peak_um([800, 1500, 2500]).tolist(),
        "wien_earth_um": float(wien_peak_um(300)),
        "critical_density": plasma_critical_density([0.45e9, 3e9, 10e9, 35e9]).tolist(),
        "radar_los_km": radar_los_range(0.010, np.array([20.0, 30.0, 60.0])).tolist(),
        "sat_horizon_km": float(satellite_horizon(1000.0)),
        "warning_min_30km": (warning_time_s(30.0, np.array([5.0, 8.0])) / 60).tolist(),
        "warning_min_20km": (warning_time_s(20.0, np.array([5.0, 8.0])) / 60).tolist(),
    }
