"""Section 3 quantities and geometry primitives."""
import numpy as np
import pytest

from hsbench import geometry as g
from hsbench.config import RE, SOUND
from hsbench.observability import table1


def test_radar_line_of_sight_values():
    for h, d in ((20, 596), (30, 727), (60, 1023)):
        assert g.radar_los_range(0.010, h) == pytest.approx(d, abs=1.0)


def test_satellite_horizon():
    assert g.satellite_horizon(1000.0) == pytest.approx(3707, abs=1.0)


def test_table1_values():
    t = table1()
    assert [round(v) for v in t["stagnation_K"]] == [1372, 2469, 3155]
    assert [round(v, 2) for v in t["wien_um"]] == [3.62, 1.93, 1.16]
    assert t["wien_earth_um"] == pytest.approx(9.66, abs=0.01)
    assert [float(f"{v:.1e}") for v in t["critical_density"]] == [2.5e15, 1.1e17, 1.2e18, 1.5e19]
    assert [round(v, 1) for v in t["warning_min_30km"]] == [8.0, 5.0]
    assert [round(v, 1) for v in t["warning_min_20km"]] == [6.6, 4.1]


def test_destination_and_distance_roundtrip():
    rng = np.random.default_rng(0)
    lat = np.radians(30.0) * np.ones(50)
    lon = np.zeros(50)
    b = rng.uniform(-np.pi, np.pi, 50)
    d = rng.uniform(100, 2000, 50)
    lat2, lon2 = g.destination(lat, lon, b, d)
    assert np.allclose(g.great_circle_distance(lat, lon, lat2, lon2), d, atol=1e-6)


def test_enu_basis_orthonormal():
    B = g.enu_basis(np.radians(30.0), np.radians(10.0))
    assert np.allclose(B @ B.T, np.eye(3), atol=1e-12)
    assert np.allclose(np.cross(B[0], B[1]), B[2])


def test_segment_clears_earth():
    a = np.array([RE + 1000, 0, 0.0])
    assert g.segment_clears_earth(a, np.array([RE + 30, 500, 0.0]))
    assert not g.segment_clears_earth(a, np.array([-(RE + 30), 0, 0.0]))


def test_walker_constellation_shape():
    t = np.array([0.0, 100.0])
    s = g.walker_positions(t, 48, 6, 1, 1000.0, np.radians(55), np.zeros(1), np.zeros(1))
    assert s.shape == (1, 2, 48, 3)
    assert np.allclose(np.linalg.norm(s, axis=-1), RE + 1000.0)
    # orbit normals have 55 deg inclination
    h = np.cross(s[0, 0], s[0, 1])
    inc = np.degrees(np.arccos(h[:, 2] / np.linalg.norm(h, axis=-1)))
    assert np.allclose(inc, 55.0, atol=1e-6)
    assert len(np.unique(np.round(s[0, 0], 3), axis=0)) == 48


def _num_jac(f, p, eps=1e-4):
    cols = []
    for i in range(3):
        dp = np.zeros(3); dp[i] = eps
        cols.append(g.wrap(f(p + dp) - f(p - dp)) / (2 * eps))
    return np.stack(cols, -1)


def test_ir_jacobian_matches_numeric():
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = g.geodetic_to_ecef(rng.uniform(-1, 1), rng.uniform(-3, 3), 1000.0)
        p = g.geodetic_to_ecef(rng.uniform(-1, 1), rng.uniform(-3, 3), 40.0)
        if not g.segment_clears_earth(s, p):
            continue
        J = g.ir_jacobian(p, s)
        Jn = _num_jac(lambda q: g.ir_measure(q, s), p)
        assert np.allclose(J, Jn, rtol=1e-5, atol=1e-10)


def test_radar_jacobian_and_inverse():
    r0 = g.geodetic_to_ecef(np.radians(30), 0.0, 0.01)
    M = g.enu_basis(np.radians(30), 0.0)
    p = g.geodetic_to_ecef(np.radians(33), np.radians(2), 35.0)
    J = g.radar_jacobian(p, r0, M)
    Jn = _num_jac(lambda q: g.radar_measure(q, r0, M), p)
    assert np.allclose(J, Jn, rtol=1e-5, atol=1e-10)
    z = g.radar_measure(p, r0, M)
    p2, Jc = g.radar_to_cartesian(z, r0, M)
    assert np.allclose(p2, p, atol=1e-9)
    assert np.allclose(Jc @ J, np.eye(3), atol=1e-8)


def test_ir_los_unit_inverts_measurement():
    s = g.geodetic_to_ecef(0.3, 0.2, 1000.0)
    p = g.geodetic_to_ecef(0.4, 0.25, 30.0)
    fr = g.ir_frame(s)
    u = g.ir_los_unit(g.ir_measure(p, s), fr)
    assert np.allclose(u, (p - s) / np.linalg.norm(p - s))


def test_mach_conversion():
    assert 8 * SOUND == pytest.approx(2.424)
