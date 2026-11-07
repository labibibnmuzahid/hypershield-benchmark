"""Single-target track management with sequential space-ground fusion (Sections 5.3-5.4).

One call to :func:`run_tracker` advances all Monte Carlo runs of one
(sensing configuration, estimator) cell through the same 1 Hz time loop.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .config import RE, Config
from .filters import IRModel, RadarModel, innovation, make_filter, nis
from .geometry import (ir_frame, ir_jacobian, ir_los_unit, ir_measure,
                       radar_to_cartesian, wrap)

PAIRS = ((0, 1), (0, 2), (1, 2))


def _stereo_init(Z, V, s, cfg: Config):
    """IR stereo initiation for n runs.

    Z (n, K, D, 2) angles, V (n, K, D) valid, s (n, K, 3) satellite positions.
    Returns success (n,), position (n, 3), covariance (n, 3, 3), (slot, det) of both.
    """
    tc = cfg.tracker
    n, K, D, _ = Z.shape
    fr = ir_frame(s)
    U = ir_los_unit(Z, tuple(e[:, :, None, :] for e in fr))          # (n, K, D, 3)
    best = np.full(n, np.inf)
    best_mid = np.zeros((n, 3))
    best_ij = np.zeros((n, 4), dtype=int)
    for a, b in PAIRS:
        if a >= K or b >= K:
            continue
        s1 = s[:, a, None, None, :]
        s2 = s[:, b, None, None, :]
        d1 = U[:, a, :, None, :]
        d2 = U[:, b, None, :, :]
        w0 = s1 - s2
        bb = np.sum(d1 * d2, -1)
        dd = np.sum(d1 * w0, -1)
        ee = np.sum(d2 * w0, -1)
        den = 1.0 - bb * bb
        den_s = np.where(den > 1e-12, den, 1.0)
        t1 = (bb * ee - dd) / den_s
        t2 = (ee - bb * dd) / den_s
        P1 = s1 + t1[..., None] * d1
        P2 = s2 + t2[..., None] * d2
        miss = np.linalg.norm(P1 - P2, axis=-1)
        mid = 0.5 * (P1 + P2)
        alt = np.linalg.norm(mid, axis=-1) - RE
        ok = (V[:, a, :, None] & V[:, b, None, :] & (den > 1e-12) & (t1 > 0) & (t2 > 0)
              & (miss < tc.stereo_max_miss) & (alt > tc.stereo_alt[0]) & (alt < tc.stereo_alt[1]))
        score = np.where(ok, miss, np.inf).reshape(n, -1)
        j = score.argmin(1)
        sc = score[np.arange(n), j]
        better = sc < best
        best = np.where(better, sc, best)
        best_mid[better] = mid.reshape(n, -1, 3)[better, j[better]]
        best_ij[better] = np.stack([np.full(n, a), j // D, np.full(n, b), j % D], 1)[better]
    succ = np.isfinite(best)
    rows = np.arange(n)
    sa, da, sb, db = best_ij.T
    z1, z2 = Z[rows, sa, da], Z[rows, sb, db]
    S1, S2 = s[rows, sa], s[rows, sb]
    f1, f2 = ir_frame(S1), ir_frame(S2)
    p = best_mid.copy()
    for _ in range(tc.gauss_newton_iters):
        r = np.concatenate([wrap(z1 - ir_measure(p, S1, f1)), wrap(z2 - ir_measure(p, S2, f2))], -1)
        H = np.concatenate([ir_jacobian(p, S1, f1), ir_jacobian(p, S2, f2)], -2)
        HtH = np.swapaxes(H, -1, -2) @ H
        HtH_s = np.where(succ[:, None, None], HtH, np.eye(3))
        step = np.linalg.solve(HtH_s, np.einsum("nki,nk->ni", H, r)[..., None])[..., 0]
        p = p + np.where(succ[:, None], step, 0.0)
    H = np.concatenate([ir_jacobian(p, S1, f1), ir_jacobian(p, S2, f2)], -2)
    info = np.swapaxes(H, -1, -2) @ H / cfg.ir.sigma ** 2
    info = np.where(succ[:, None, None], info, np.eye(3))
    cov = np.linalg.inv(info)
    return succ, p, cov, best_ij


def run_tracker(data: dict, sensing: str, kind: str, cfg: Config, record_cycles=False) -> dict:
    tc = cfg.tracker
    traj, ir, rd = data["traj"], data["ir"], data["radar"]
    N, T, _ = traj["pos"].shape
    use_ir = sensing in ("ir", "fused") and cfg.ir.enabled
    use_rad = sensing in ("radar", "fused") and cfg.radar.enabled
    flt = make_filter(kind, N, tc)
    g_ir = chi2.ppf(tc.gate_prob, 2)
    g_rad = chi2.ppf(tc.gate_prob, 3)
    rmodel = RadarModel(rd["r0"], rd["M"], cfg.radar.sigma_r, cfg.radar.sigma_ang)
    ir_map = {int(k): i for i, k in enumerate(ir["scan_k"])}
    K = ir["slot_ok"].shape[2]

    state = np.zeros(N, dtype=np.int8)          # 0 none, 1 tentative, 2 confirmed
    t_init = np.zeros(N, int)
    last_upd = np.zeros(N, int)
    hit_cycles = np.zeros(N, int)
    q = np.ones(N)
    last_false = np.full(N, -10 ** 9)
    segments = np.zeros(N, int)
    n_upd = np.zeros(N, int)
    n_false_upd = np.zeros(N, int)

    conf = np.zeros((N, T), dtype=bool)
    est = np.full((N, T, 3), np.nan)
    Ppos = np.full((N, T, 3, 3), np.nan)
    recent_false = np.zeros((N, T), dtype=bool)
    since_false = np.full((N, T), np.inf)

    rows = np.arange(N)
    for k in range(T):
        idx_a = np.nonzero(state > 0)[0]
        if k > 0 and idx_a.size:
            flt.predict(idx_a)

        scans = []
        if use_ir and k in ir_map:
            i = ir_map[k]
            for j in range(K):
                av = ir["slot_ok"][:, i, j] & ir["valid"][:, i, j, 0]
                scans.append(("ir", i, j, av))
        if use_rad:
            scans.append(("radar", k, None, rd["valid"][:, k, 0]))

        # ------------------------------------------------ track initiation
        new = np.zeros(N, dtype=bool)
        none = state == 0
        if use_rad:
            cand = np.nonzero(none & rd["valid"][:, k, 0])[0]
            if cand.size:
                nd = rd["valid"][cand, k].sum(-1)
                j = np.minimum((rd["init_key"][cand, k] * nd).astype(int), nd - 1)
                z = rd["z"][cand, k, j]
                p0, J = radar_to_cartesian(z, rd["r0"], rd["M"])
                P0 = J @ rmodel.R @ np.swapaxes(J, -1, -2)
                flt.init(cand, p0, P0)
                new[cand] = True
                clutter = ~rd["is_tgt"][cand, k, j]
                last_false[cand] = np.where(clutter, k, -10 ** 9)
        if use_ir and k in ir_map:
            i = ir_map[k]
            cand = np.nonzero(none & ~new)[0]
            if cand.size:
                okslot = ir["slot_ok"][cand, i] & ir["valid"][cand, i, :, 0]
                cand = cand[okslot.sum(1) >= 2]
            if cand.size:
                V = ir["valid"][cand, i] & ir["slot_ok"][cand, i][..., None]
                succ, p0, P0, ij = _stereo_init(ir["z"][cand, i], V, ir["sat_pos"][cand, i], cfg)
                c2 = cand[succ]
                if c2.size:
                    flt.init(c2, p0[succ], P0[succ])
                    new[c2] = True
                    ijs = ij[succ]
                    t1 = ir["is_tgt"][c2, i, ijs[:, 0], ijs[:, 1]]
                    t2 = ir["is_tgt"][c2, i, ijs[:, 2], ijs[:, 3]]
                    last_false[c2] = np.where(t1 & t2, -10 ** 9, k)
        if new.any():
            state[new] = 1
            t_init[new] = k
            last_upd[new] = k
            hit_cycles[new] = 1
            q[new] = 1.0

        # ------------------------------------- gating, association, update
        hits = np.zeros(N, int)
        navail = np.zeros(N, int)
        for kind_s, a, j, av in scans:
            sel = idx_a[av[idx_a]]
            if sel.size == 0:
                continue
            navail[sel] += 1
            if kind_s == "ir":
                model = IRModel(ir["sat_pos"][sel, a, j], cfg.ir.sigma)
                Z = ir["z"][sel, a, j]
                Vd = ir["valid"][sel, a, j]
                tgt = ir["is_tgt"][sel, a, j]
                gate = g_ir
            else:
                model = rmodel
                Z = rd["z"][sel, a]
                Vd = rd["valid"][sel, a]
                tgt = rd["is_tgt"][sel, a]
                gate = g_rad
            zhat, S, aux = flt.gate_prep(sel, model)
            nu = innovation(Z, zhat[:, None], model.wrap_mask)
            d2 = nis(nu, S[:, None])
            d2 = np.where(Vd, d2, np.inf)
            b = d2.argmin(1)
            ok = d2[np.arange(sel.size), b] < gate
            if not ok.any():
                continue
            g = sel[ok]
            bo = b[ok]
            flt.update(g, model.subset(ok), Z[ok, bo], zhat[ok], S[ok],
                       None if aux is None else aux[ok])
            hits[g] += 1
            n_upd[g] += 1
            fa = ~tgt[ok, bo]
            n_false_upd[g[fa]] += 1
            last_false[g[fa]] = k
        if idx_a.size:
            flt.end_cycle(idx_a)

        # --------------------------------------------------- bookkeeping
        old = (state > 0) & ~new
        hit = hits > 0
        last_upd[old & hit] = k
        tent = old & (state == 1)
        hit_cycles[tent & hit] += 1
        confirm = tent & (hit_cycles >= tc.confirm_hits)
        state[confirm] = 2
        segments[confirm] += 1
        upd_q = old & (navail > 0)
        q[upd_q] = (1 - tc.assoc_smoothing) * q[upd_q] + tc.assoc_smoothing * hits[upd_q] / navail[upd_q]

        idx_live = np.nonzero(state > 0)[0]
        if idx_live.size:
            pos, P = flt.output(idx_live)
            std = np.sqrt(np.trace(P, axis1=-2, axis2=-1))
            dead = np.zeros(N, dtype=bool)
            dead[idx_live] = std > tc.max_pos_std
            dead |= old & (k - last_upd >= tc.coast_limit)
            dead |= (state == 2) & (q < tc.assoc_ratio_min)
            dead |= (state == 1) & (k - t_init >= tc.confirm_window)
            state[dead] = 0
            keep = state[idx_live] == 2
            ic = idx_live[keep]
            conf[ic, k] = True
            est[ic, k] = pos[keep]
            Ppos[ic, k] = P[keep]
            since = k - last_false[ic]
            since_false[ic, k] = np.where(since >= 0, since, np.inf)
            recent_false[ic, k] = (since >= 0) & (since <= cfg.metrics.false_assoc_window)

    return {"sensing": sensing, "estimator": kind, "confirmed": conf, "est": est, "Ppos": Ppos,
            "recent_false": recent_false, "since_false": since_false, "segments": segments,
            "n_updates": n_upd, "n_false_updates": n_false_upd}
