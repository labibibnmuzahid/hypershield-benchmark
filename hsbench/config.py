"""Benchmark configuration.

All values reproduce Tables 2-4 of the paper. Units: km, s, rad unless noted.
Every experiment starts from ``default_config()`` and overrides single fields,
so the archived ``results/defaults.json`` fully documents a run.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

G0 = 9.80665e-3          # standard gravity, km/s^2
RE = 6371.0              # Earth radius, km
MU = 398600.4418         # Earth gravitational parameter, km^3/s^2
SOUND = 0.303            # speed of sound used for Mach conversion, km/s


@dataclass
class ScenarioCfg:
    duration: float = 480.0               # s
    dt_int: float = 0.05                  # Euler integration step, s
    radar_lat_deg: float = 30.0
    radar_lon_deg: float = 0.0
    start_dist: tuple = (1250.0, 1450.0)  # great-circle distance to radar, km
    heading_offset_deg: float = 15.0      # +/- offset from bearing to radar
    mach: tuple = (5.0, 8.0)
    decel: tuple = (0.1e-3, 0.5e-3)       # km/s^2 (0.1-0.5 m/s^2)
    h0: tuple = (28.0, 52.0)              # km
    alt_amp: tuple = (0.5, 3.0)           # km
    alt_period: tuple = (60.0, 150.0)     # s
    sink: tuple = (0.0, 5.0e-3)           # km/s (0-5 m/s)
    hold: tuple = (15.0, 45.0)            # lateral-command hold time, s
    lag_tau: float = 4.0                  # s
    k_psi: float = 0.02                   # heading-hold gain, 1/s
    amax_g: float = 2.0                   # maneuver level (0.5 / 2 / 4 g)


@dataclass
class IRCfg:
    enabled: bool = True
    n_sats: int = 48
    n_planes: int = 6
    phasing: int = 1
    altitude: float = 1000.0              # km
    inclination_deg: float = 55.0
    max_range: float = 4500.0             # km
    n_tasked: int = 3
    revisit: float = 2.0                  # s
    sigma: float = 100e-6                 # rad per axis
    pd: float = 0.9
    outage_fraction: float = 0.2
    outage_mean: float = 25.0             # s
    fa_rate: float = 1.0                  # mean false alarms per scan per satellite
    fa_halfwidth: float = 2e-3            # rad, uniform box around target LOS


@dataclass
class RadarCfg:
    enabled: bool = True
    mast: float = 0.010                   # km (10 m)
    k_factor: float = 4.0 / 3.0
    max_range: float = 1500.0             # km
    revisit: float = 1.0                  # s
    sigma_r: float = 0.050                # km
    sigma_ang: float = 1e-3               # rad (azimuth and elevation)
    pd: float = 0.9
    fa_rate: float = 1.0                  # mean false alarms per scan
    fa_half_r: float = 3.0                # km
    fa_half_ang: float = 6e-3             # rad


@dataclass
class TrackerCfg:
    cycle: float = 1.0
    qv_cv: float = 1e-3                   # EKF-CV / UKF-CV accel PSD, km^2/s^3
    qa_ca: float = 1e-5                   # EKF-CA jerk PSD, km^2/s^5
    imm_qv: float = 1e-4
    imm_qa: float = 1e-5
    imm_ps: float = 0.95
    sigma_v0: float = 2.5                 # km/s per axis
    sigma_a0: float = 0.05                # km/s^2 per axis
    gate_prob: float = 0.999
    confirm_hits: int = 3
    confirm_window: float = 10.0          # s
    coast_limit: float = 20.0             # s without update
    max_pos_std: float = 30.0             # km, sqrt(tr(P_pos))
    assoc_ratio_min: float = 0.4
    assoc_smoothing: float = 0.3
    stereo_max_miss: float = 2.0          # km
    stereo_alt: tuple = (5.0, 120.0)      # km
    gauss_newton_iters: int = 3


@dataclass
class MetricCfg:
    ospa_cutoff: float = 10.0             # km
    custody_radius: float = 1.0           # km
    ttft_radius: float = 5.0              # km
    alphas: tuple = (0.50, 0.68, 0.90, 0.95, 0.99)
    false_assoc_window: float = 10.0      # s
    n_boot: int = 2000


@dataclass
class Config:
    scenario: ScenarioCfg = field(default_factory=ScenarioCfg)
    ir: IRCfg = field(default_factory=IRCfg)
    radar: RadarCfg = field(default_factory=RadarCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    metrics: MetricCfg = field(default_factory=MetricCfg)

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


MANEUVER_LEVELS = {"low": 0.5, "medium": 2.0, "high": 4.0}
SENSING = ("ir", "radar", "fused")
ESTIMATORS = ("EKF-CV", "UKF-CV", "EKF-CA", "IMM-CV/CA")

# NumPy SeedSequence entropy values (Appendix B of the paper)
SEEDS = {"tune": 11, "low": 221, "medium": 222, "high": 223, "sweeps": 33, "example": 44}
N_RUNS = {"tune": 200, "main": 500, "sweeps": 300}


def default_config() -> Config:
    return Config()
