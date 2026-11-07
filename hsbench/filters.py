"""Motion models, measurement models and the four estimators (Section 5).

Every estimator holds the state of one (possible) track per Monte Carlo run and
operates on index subsets of runs, so a single time loop advances all runs.
"""
from __future__ import annotations

import numpy as np

from .geometry import ir_frame, ir_jacobian, ir_measure, radar_jacobian, radar_measure, wrap

LOG2PI = np.log(2.0 * np.pi)


# ------------------------------------------------------------ motion models
def cv_matrices(T: float, q: float):
    I = np.eye(3)
    F = np.block([[I, T * I], [np.zeros((3, 3)), I]])
    Q = q * np.block([[T ** 3 / 3 * I, T ** 2 / 2 * I], [T ** 2 / 2 * I, T * I]])
    return F, Q


def ca_matrices(T: float, q: float):
    I = np.eye(3)
    Z = np.zeros((3, 3))
    F = np.block([[I, T * I, T ** 2 / 2 * I], [Z, I, T * I], [Z, Z, I]])
    B = np.array([[T ** 5 / 20, T ** 4 / 8, T ** 3 / 6],
                  [T ** 4 / 8, T ** 3 / 3, T ** 2 / 2],
                  [T ** 3 / 6, T ** 2 / 2, T]])
    return F, q * np.kron(B, I)


def cv_in_ca_space(T: float, q: float, eps: float = 1e-12):
    """CV mode of the IMM in the nine-state space: acceleration reset to zero."""
    F6, Q6 = cv_matrices(T, q)
    F = np.zeros((9, 9)); F[:6, :6] = F6
    Q = np.zeros((9, 9)); Q[:6, :6] = Q6; Q[6:, 6:] = eps * np.eye(3)
    return F, Q


# ------------------------------------------------------- measurement models
class IRModel:
    m = 2
    wrap_mask = np.array([True, True])

    def __init__(self, s, sigma):
        self.s = s
        self.frame = ir_frame(s)
        self.R = sigma ** 2 * np.eye(2)

    def _f(self, p):
        if p.ndim == 3:
            return self.s[:, None], tuple(e[:, None] for e in self.frame)
        return self.s, self.frame

    def h(self, p):
        s, fr = self._f(p)
        return ir_measure(p, s, fr)

    def H(self, p):
        s, fr = self._f(p)
        return ir_jacobian(p, s, fr)

    def subset(self, sel):
        out = IRModel.__new__(IRModel)
        out.s = self.s[sel]
        out.frame = tuple(e[sel] for e in self.frame)
        out.R = self.R
        return out


class RadarModel:
    m = 3
    wrap_mask = np.array([False, True, True])

    def __init__(self, r0, M, sigma_r, sigma_ang):
        self.r0, self.M = r0, M
        self.R = np.diag([sigma_r ** 2, sigma_ang ** 2, sigma_ang ** 2])

    def h(self, p):
        return radar_measure(p, self.r0, self.M)

    def H(self, p):
        return radar_jacobian(p, self.r0, self.M)

    def subset(self, sel):
        return self


def innovation(z, zhat, wrap_mask):
    d = z - zhat
    return np.where(wrap_mask, wrap(d), d)


def nis(nu, S):
    return np.einsum("...i,...i->...", nu, np.linalg.solve(S, nu[..., None])[..., 0])


def gauss_loglik(nu, S):
    m = nu.shape[-1]
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * (nis(nu, S) + logdet + m * LOG2PI)


def sym(P):
    return 0.5 * (P + np.swapaxes(P, -1, -2))


def safe_sqrtm(P):
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(sym(P))
        return V * np.sqrt(np.clip(w, 1e-15, None))[..., None, :]


# ---------------------------------------------------- update primitives
def ekf_prep(x, P, model):
    p = x[:, :3]
    zhat = model.h(p)
    Hp = model.H(p)
    H = np.zeros((x.shape[0], model.m, x.shape[1]))
    H[:, :, :3] = Hp
    S = sym(H @ P @ np.swapaxes(H, -1, -2) + model.R)
    return zhat, S, H


def ekf_apply(x, P, H, S, R, nu):
    """Extended Kalman update with the Joseph-form covariance."""
    PHt = P @ np.swapaxes(H, -1, -2)
    K = np.swapaxes(np.linalg.solve(S, np.swapaxes(PHt, -1, -2)), -1, -2)
    x = x + np.einsum("nij,nj->ni", K, nu)
    IKH = np.eye(x.shape[1]) - K @ H
    P = IKH @ P @ np.swapaxes(IKH, -1, -2) + K @ R @ np.swapaxes(K, -1, -2)
    return x, sym(P)


def ukf_prep(x, P, model):
    """Unscented transform, 2n equally weighted points (alpha=1, beta=0, kappa=0)."""
    n = x.shape[1]
    L = safe_sqrtm(P) * np.sqrt(n)
    dX = np.concatenate([np.swapaxes(L, -1, -2), -np.swapaxes(L, -1, -2)], axis=1)  # (N, 2n, n)
    X = x[:, None, :] + dX
    Z = model.h(X[..., :3])
    ref = Z[:, :1]
    dZ = innovation(Z, ref, model.wrap_mask)
    zhat = ref[:, 0] + dZ.mean(1)
    zhat = np.where(model.wrap_mask, wrap(zhat), zhat)
    dZ = innovation(Z, zhat[:, None], model.wrap_mask)
    w = 1.0 / (2 * n)
    S = sym(w * np.einsum("nki,nkj->nij", dZ, dZ) + model.R)
    Pxz = w * np.einsum("nki,nkj->nij", dX, dZ)
    return zhat, S, Pxz


def ukf_apply(x, P, Pxz, S, nu):
    K = np.swapaxes(np.linalg.solve(S, np.swapaxes(Pxz, -1, -2)), -1, -2)
    x = x + np.einsum("nij,nj->ni", K, nu)
    P = P - K @ S @ np.swapaxes(K, -1, -2)
    return x, sym(P)


# --------------------------------------------------------------- estimators
class SingleModelFilter:
    """EKF-CV, UKF-CV or EKF-CA."""

    def __init__(self, N, kind, T, q, sigma_v0, sigma_a0):
        self.kind = kind
        self.ukf = kind == "UKF-CV"
        if kind in ("EKF-CV", "UKF-CV"):
            self.F, self.Q = cv_matrices(T, q)
        elif kind == "EKF-CA":
            self.F, self.Q = ca_matrices(T, q)
        else:
            raise ValueError(kind)
        self.nx = self.F.shape[0]
        self.x = np.zeros((N, self.nx))
        self.P = np.tile(np.eye(self.nx), (N, 1, 1))
        self.sv0, self.sa0 = sigma_v0, sigma_a0

    def init(self, idx, p0, Ppos0):
        self.x[idx] = 0.0
        self.x[idx, :3] = p0
        P = np.zeros((len(idx), self.nx, self.nx))
        P[:, :3, :3] = Ppos0
        P[:, 3:6, 3:6] = self.sv0 ** 2 * np.eye(3)
        if self.nx == 9:
            P[:, 6:, 6:] = self.sa0 ** 2 * np.eye(3)
        self.P[idx] = P

    def predict(self, idx):
        self.x[idx] = self.x[idx] @ self.F.T
        self.P[idx] = sym(self.F @ self.P[idx] @ self.F.T + self.Q)

    def gate_prep(self, idx, model):
        x, P = self.x[idx], self.P[idx]
        if self.ukf:
            zhat, S, aux = ukf_prep(x, P, model)
        else:
            zhat, S, aux = ekf_prep(x, P, model)
        return zhat, S, aux

    def update(self, idx, model, z, zhat, S, aux):
        nu = innovation(z, zhat, model.wrap_mask)
        if self.ukf:
            x, P = ukf_apply(self.x[idx], self.P[idx], aux, S, nu)
        else:
            x, P = ekf_apply(self.x[idx], self.P[idx], aux, S, model.R, nu)
        self.x[idx], self.P[idx] = x, P

    def end_cycle(self, idx):
        pass

    def output(self, idx):
        return self.x[idx, :3], self.P[idx, :3, :3]


class IMMFilter:
    """Two-model IMM (CV with acceleration reset, CA) in the nine-state space."""

    kind = "IMM-CV/CA"

    def __init__(self, N, T, qv, qa, ps, sigma_v0, sigma_a0):
        F0, Q0 = cv_in_ca_space(T, qv)
        F1, Q1 = ca_matrices(T, qa)
        self.F = np.stack([F0, F1])
        self.Q = np.stack([Q0, Q1])
        self.Pi = np.array([[ps, 1 - ps], [1 - ps, ps]])
        self.nx = 9
        self.xm = np.zeros((N, 2, 9))
        self.Pm = np.tile(np.eye(9), (N, 2, 1, 1))
        self.mu = np.full((N, 2), 0.5)
        self.c = np.full((N, 2), 0.5)
        self.L = np.zeros((N, 2))
        self.sv0, self.sa0 = sigma_v0, sigma_a0

    def init(self, idx, p0, Ppos0):
        x = np.zeros((len(idx), 9)); x[:, :3] = p0
        P = np.zeros((len(idx), 9, 9))
        P[:, :3, :3] = Ppos0
        P[:, 3:6, 3:6] = self.sv0 ** 2 * np.eye(3)
        P[:, 6:, 6:] = self.sa0 ** 2 * np.eye(3)
        self.xm[idx] = x[:, None]
        self.Pm[idx] = P[:, None]
        self.mu[idx] = 0.5
        self.c[idx] = 0.5
        self.L[idx] = 0.0

    @staticmethod
    def _combine(xm, Pm, w):
        x = np.einsum("nj,nji->ni", w, xm)
        d = xm - x[:, None]
        P = np.einsum("nj,njab->nab", w, Pm + d[..., :, None] * d[..., None, :])
        return x, sym(P)

    def predict(self, idx):
        mu = self.mu[idx]
        c = mu @ self.Pi                                  # predicted mode probabilities
        mix = (self.Pi[None] * mu[:, :, None]) / c[:, None, :]   # mu_{i|j}
        xm, Pm = self.xm[idx], self.Pm[idx]
        x0 = np.einsum("nij,nia->nja", mix, xm)
        d = xm[:, :, None, :] - x0[:, None, :, :]            # (n, i, j, a)
        P0 = np.einsum("nij,niab->njab", mix, Pm) + \
            np.einsum("nij,nija,nijb->njab", mix, d, d)
        x1 = np.einsum("jab,njb->nja", self.F, x0)
        P1 = self.F[None] @ P0 @ np.swapaxes(self.F, -1, -2)[None] + self.Q[None]
        self.xm[idx], self.Pm[idx] = x1, sym(P1)
        self.c[idx] = c
        self.L[idx] = 0.0

    def _weights(self, idx):
        lw = np.log(np.maximum(self.c[idx], 1e-300)) + self.L[idx]
        lw -= lw.max(1, keepdims=True)
        w = np.exp(lw)
        return w / w.sum(1, keepdims=True)

    def gate_prep(self, idx, model):
        x, P = self._combine(self.xm[idx], self.Pm[idx], self._weights(idx))
        zhat, S, H = ekf_prep(x, P, model)
        return zhat, S, None

    def update(self, idx, model, z, zhat, S, aux):
        for j in range(2):
            xj, Pj = self.xm[idx, j], self.Pm[idx, j]
            zh, Sj, Hj = ekf_prep(xj, Pj, model)
            nu = innovation(z, zh, model.wrap_mask)
            self.L[idx, j] += gauss_loglik(nu, Sj)
            xj, Pj = ekf_apply(xj, Pj, Hj, Sj, model.R, nu)
            self.xm[idx, j], self.Pm[idx, j] = xj, Pj

    def end_cycle(self, idx):
        self.mu[idx] = np.clip(self._weights(idx), 1e-12, None)
        self.mu[idx] /= self.mu[idx].sum(1, keepdims=True)

    def output(self, idx):
        x, P = self._combine(self.xm[idx], self.Pm[idx], self.mu[idx])
        return x[:, :3], P[:, :3, :3]


def make_filter(kind, N, tcfg):
    T = tcfg.cycle
    if kind == "IMM-CV/CA":
        return IMMFilter(N, T, tcfg.imm_qv, tcfg.imm_qa, tcfg.imm_ps, tcfg.sigma_v0, tcfg.sigma_a0)
    q = tcfg.qa_ca if kind == "EKF-CA" else tcfg.qv_cv
    return SingleModelFilter(N, kind, T, q, tcfg.sigma_v0, tcfg.sigma_a0)
