"""Scenario/sensor generators, track management, metrics and reproducibility."""
import numpy as np
import pytest

from hsbench import geometry as g
from hsbench import metrics, sensors, tracker, trajectory
from hsbench.config import G0, SOUND, default_config
from hsbench.tracker import _stereo_init


@pytest.fixture(scope="module")
def small():
    cfg = default_config()
    return cfg, sensors.generate_dataset(cfg, 40, 12345)


def test_trajectory_envelope(small):
    cfg, d = small
    tr = d["traj"]
    sp = np.linalg.norm(tr["vel"], axis=-1)
    assert sp[:, 0].min() >= 5 * SOUND - 1e-6 and sp[:, 0].max() <= 8 * SOUND + 1e-6
    assert tr["alt"].min() > 15 and tr["alt"].max() < 60
    fd = np.linalg.norm(np.diff(tr["pos"], axis=1), axis=-1)
    assert np.allclose(fd, 0.5 * (sp[:, 1:] + sp[:, :-1]), atol=2e-3)
    # commanded |a_c| <= amax; the heading-hold term may add a little on top
    assert np.median(tr["peak_alat_g"]) <= cfg.scenario.amax_g
    assert np.all(tr["peak_alat_g"] <= cfg.scenario.amax_g * 1.5)


def test_maneuver_level_changes_only_lateral_command():
    cfg = default_config()
    a = trajectory.generate(cfg.scenario, 5, np.random.default_rng(1))
    cfg.scenario.amax_g = 0.0
    b = trajectory.generate(cfg.scenario, 5, np.random.default_rng(1))
    assert np.allclose(a["alt"], b["alt"]) and np.allclose(a["pos"][:, 0], b["pos"][:, 0])
    assert np.abs(b["a_cmd"]).max() == 0


def test_outage_chain_statistics():
    rng = np.random.default_rng(0)
    out = sensors.outage_states(rng.uniform(size=(400, 4000)), 0.2, 25.0)
    assert out.mean() == pytest.approx(0.2, abs=0.01)
    starts = np.sum(np.diff(out.astype(int), axis=1) == 1)
    assert out.sum() / starts == pytest.approx(25.0, rel=0.06)


def test_detection_statistics(small):
    cfg, d = small
    ir, rd = d["ir"], d["radar"]
    ok = ir["slot_ok"]
    tgt = (ir["valid"] & ir["is_tgt"]).sum(-1)[ok]
    fa = (ir["valid"] & ~ir["is_tgt"]).sum(-1)[ok]
    assert tgt.mean() == pytest.approx(0.9, abs=0.02)
    assert fa.mean() == pytest.approx(1.0, abs=0.05)
    assert 4 < ir["n_visible"].mean() < 7
    rcov = rd["coverage"]
    assert ((rd["valid"] & rd["is_tgt"]).sum(-1)[rcov]).mean() == pytest.approx(0.9, abs=0.03)
    assert not rd["valid"][~rcov].any()


def test_common_random_numbers_across_sweeps(small):
    cfg, d = small
    c = cfg.copy()
    c.radar.pd = 0.5
    d2 = sensors.regenerate_sensors(d, c)
    assert np.array_equal(d2["ir"]["z"], d["ir"]["z"])
    # detections at pd 0.5 are a subset of those at pd 0.9
    t1 = d["radar"]["valid"][..., 0] & d["radar"]["is_tgt"][..., 0]
    t2 = d2["radar"]["valid"][..., 0] & d2["radar"]["is_tgt"][..., 0]
    assert np.all(t1 | ~t2)


def test_stereo_initiation_noise_free():
    cfg = default_config()
    p = g.geodetic_to_ecef(np.radians(32), np.radians(3), 35.0)
    sats = np.stack([g.geodetic_to_ecef(np.radians(40), np.radians(-5), 1000.0),
                     g.geodetic_to_ecef(np.radians(25), np.radians(12), 1000.0),
                     g.geodetic_to_ecef(np.radians(45), np.radians(15), 1000.0)])[None]
    Z = g.ir_measure(p, sats[0])[None, :, None, :]
    V = np.ones((1, 3, 1), dtype=bool)
    succ, p0, P0, _ = _stereo_init(Z, V, sats, cfg)
    assert succ[0]
    assert np.linalg.norm(p0[0] - p) < 1e-6
    assert np.all(np.linalg.eigvalsh(P0[0]) > 0)


def test_tracker_deterministic_and_consistent(small):
    cfg, d = small
    r1 = tracker.run_tracker(d, "fused", "IMM-CV/CA", cfg)
    r2 = tracker.run_tracker(d, "fused", "IMM-CV/CA", cfg)
    assert np.array_equal(r1["confirmed"], r2["confirmed"])
    assert np.allclose(np.nan_to_num(r1["est"]), np.nan_to_num(r2["est"]))
    ep = metrics.per_epoch(r1, d["traj"]["pos"])
    pm = metrics.point_metrics(metrics.run_sums(ep, cfg.metrics))
    assert pm["OSPA"] < 4.0 and pm["custody"] > 0.6
    # confirmed implies a covariance; unconfirmed epochs carry no estimate
    assert np.all(np.isfinite(ep["nees"][ep["conf"]]))
    assert np.all(np.isnan(ep["err"][~ep["conf"]]))


def test_no_clutter_gives_credible_covariance():
    cfg = default_config()
    cfg.ir.fa_rate = 0.0
    cfg.radar.fa_rate = 0.0
    d = sensors.generate_dataset(cfg, 40, 777)
    r = tracker.run_tracker(d, "fused", "EKF-CV", cfg)
    pm = metrics.point_metrics(metrics.run_sums(metrics.per_epoch(r, d["traj"]["pos"]), cfg.metrics))
    assert pm["ANEES"] < 1.5
    assert r["n_false_updates"].sum() == 0


def test_dataset_reproducible():
    cfg = default_config()
    a = sensors.generate_dataset(cfg, 6, 99)
    b = sensors.generate_dataset(cfg, 6, 99)
    for s in ("ir", "radar"):
        assert np.array_equal(a[s]["z"], b[s]["z"])
        assert np.array_equal(a[s]["valid"], b[s]["valid"])


def test_ospa_and_holm():
    cfg = default_config()
    conf = np.array([[True, True, False, True]])
    err = np.array([[0.5, 20.0, np.nan, 2.0]], dtype=np.float32)
    ep = {"conf": conf, "err": err, "nees": np.where(conf, 3.0, np.nan), "segments": np.array([1])}
    s = metrics.run_sums(ep, cfg.metrics)
    assert s["ospa"][0] / 4 == pytest.approx((0.5 + 10 + 10 + 2) / 4)
    assert metrics.point_metrics(s)["ANEES"] == pytest.approx(1.0)
    adj = metrics.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(0)
    num = rng.uniform(1, 2, 300)
    den = np.ones(300)
    counts = metrics.bootstrap_counts(300, 2000)
    lo, hi = metrics.bootstrap_ci(num, den, counts)
    assert lo < num.mean() < hi
    assert hi - lo == pytest.approx(2 * 1.96 * num.std() / np.sqrt(300), rel=0.15)
