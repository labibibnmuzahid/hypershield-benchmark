"""Motion models, update primitives and estimator consistency."""
import numpy as np
import pytest
from scipy.integrate import quad_vec

from hsbench import filters as F


def _psd(M):
    return np.all(np.linalg.eigvalsh(0.5 * (M + M.T)) > -1e-14)


@pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
def test_process_noise_equals_integral(T):
    """Q = int_0^T F(t) G q G^T F(t)^T dt for white acceleration / white jerk."""
    q = 0.7
    _, Qcv = F.cv_matrices(T, q)
    def cv(t):
        Ft, _ = F.cv_matrices(t, 0.0)
        G = np.vstack([np.zeros((3, 3)), np.eye(3)])
        return Ft @ G @ G.T @ Ft.T * q
    assert np.allclose(Qcv, quad_vec(cv, 0, T)[0], atol=1e-10)
    _, Qca = F.ca_matrices(T, q)
    def ca(t):
        Ft, _ = F.ca_matrices(t, 0.0)
        G = np.vstack([np.zeros((6, 3)), np.eye(3)])
        return Ft @ G @ G.T @ Ft.T * q
    assert np.allclose(Qca, quad_vec(ca, 0, T)[0], atol=1e-10)
    assert _psd(Qcv) and _psd(Qca)


class LinearPosModel:
    """z = p + v, used to test filters against the exact Kalman filter."""
    m = 3
    wrap_mask = np.array([False, False, False])

    def __init__(self, r):
        self.R = r * np.eye(3)

    def h(self, p):
        return p.copy()

    def H(self, p):
        return np.broadcast_to(np.eye(3), p.shape[:-1] + (3, 3)).copy()

    def subset(self, sel):
        return self


def test_ekf_and_ukf_exact_for_linear_measurement():
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((4, n, n))
    P = A @ np.swapaxes(A, -1, -2) + np.eye(n)
    x = rng.standard_normal((4, n))
    z = rng.standard_normal((4, 3))
    mdl = LinearPosModel(0.3)
    zh, S, H = F.ekf_prep(x, P, mdl)
    xe, Pe = F.ekf_apply(x, P, H, S, mdl.R, F.innovation(z, zh, mdl.wrap_mask))
    zu, Su, Pxz = F.ukf_prep(x, P, mdl)
    xu, Pu = F.ukf_apply(x, P, Pxz, Su, F.innovation(z, zu, mdl.wrap_mask))
    # textbook KF
    Hm = np.hstack([np.eye(3), np.zeros((3, 3))])
    for i in range(4):
        Sk = Hm @ P[i] @ Hm.T + mdl.R
        K = P[i] @ Hm.T @ np.linalg.inv(Sk)
        xk = x[i] + K @ (z[i] - Hm @ x[i])
        Pk = (np.eye(n) - K @ Hm) @ P[i]
        assert np.allclose(xe[i], xk) and np.allclose(Pe[i], Pk)
        assert np.allclose(xu[i], xk) and np.allclose(Pu[i], Pk)


def test_ukf_sigma_points_are_cubature_rule():
    """alpha=1, beta=0, kappa=0: 2n points at +-sqrt(n) chol columns, weights 1/(2n)."""
    n = 6
    P = np.diag(np.arange(1.0, n + 1))[None]
    x = np.zeros((1, n))
    mdl = LinearPosModel(1.0)
    zh, S, Pxz = F.ukf_prep(x, P, mdl)
    assert np.allclose(S[0], P[0, :3, :3] + np.eye(3))    # exact second moment
    assert np.allclose(Pxz[0], P[0, :, :3])


def _simulate_cv(N, steps, q, r, rng):
    Fm, Q = F.cv_matrices(1.0, q)
    L = np.linalg.cholesky(Q)
    x = np.zeros((N, 6)); x[:, 3:] = rng.normal(0, 0.5, (N, 3))
    xs, zs = [], []
    for _ in range(steps):
        x = x @ Fm.T + rng.standard_normal((N, 6)) @ L.T
        xs.append(x.copy())
        zs.append(x[:, :3] + np.sqrt(r) * rng.standard_normal((N, 3)))
    return np.array(xs), np.array(zs)


@pytest.mark.parametrize("kind", ["EKF-CV", "UKF-CV"])
def test_filter_is_consistent_on_matched_model(kind):
    rng = np.random.default_rng(3)
    N, steps, q, r = 400, 60, 1e-3, 0.04
    xs, zs = _simulate_cv(N, steps, q, r, rng)
    f = F.SingleModelFilter(N, kind, 1.0, q, 0.5, 0.05)
    idx = np.arange(N)
    mdl = LinearPosModel(r)
    f.init(idx, zs[0], r * np.tile(np.eye(3), (N, 1, 1)))
    nees = []
    for k in range(1, steps):
        f.predict(idx)
        zh, S, aux = f.gate_prep(idx, mdl)
        f.update(idx, mdl, zs[k], zh, S, aux)
        p, P = f.output(idx)
        e = p - xs[k, :, :3]
        nees.append(np.einsum("ni,ni->n", e, np.linalg.solve(P, e[..., None])[..., 0]))
    anees = np.mean(nees[10:]) / 3
    assert 0.95 < anees < 1.05


def test_imm_probabilities_and_consistency():
    rng = np.random.default_rng(5)
    N, steps, q, r = 300, 60, 1e-4, 0.04
    xs, zs = _simulate_cv(N, steps, q, r, rng)
    f = F.IMMFilter(N, 1.0, q, 1e-5, 0.95, 0.5, 0.05)
    idx = np.arange(N)
    mdl = LinearPosModel(r)
    f.init(idx, zs[0], r * np.tile(np.eye(3), (N, 1, 1)))
    nees = []
    for k in range(1, steps):
        f.predict(idx)
        zh, S, aux = f.gate_prep(idx, mdl)
        f.update(idx, mdl, zs[k], zh, S, aux)
        f.end_cycle(idx)
        assert np.allclose(f.mu.sum(1), 1.0)
        p, P = f.output(idx)
        assert np.all(np.linalg.eigvalsh(P) > 0)
        e = p - xs[k, :, :3]
        nees.append(np.einsum("ni,ni->n", e, np.linalg.solve(P, e[..., None])[..., 0]))
    # CV truth: the CV mode should dominate and the combination should be credible
    assert f.mu[:, 0].mean() > 0.6
    assert 0.8 < np.mean(nees[10:]) / 3 < 1.2


def test_innovation_wraps_angles():
    nu = F.innovation(np.array([np.pi - 0.01]), np.array([-np.pi + 0.01]), np.array([True]))
    assert nu[0] == pytest.approx(-0.02)
