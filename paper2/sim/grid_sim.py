# -*- coding: utf-8 -*-
"""Kare dikimli (izgara) bahcede kablolu drone platformlariyla secmeli hasat.

Bahce (tohuma bagli kusurlar): kare/dikdortgene yakin dikim (iki yonde
%5-10 farkli aralik), hafif egrilen siralar (dusuk frekansli yer degistirme),
agac basina konum sapmasi, rastgele ve kumelenmis eksik agaclar, duzensiz
sinir (OSM gercek sinir ya da kosesi kesilmis dikdortgen), bilinmeyen
yonelim acisi. Platform agac aralarindaki koridorlarda iki yonde ilerler;
yurutmede yol = GERCEK cercevede L1 + donus cezasi. Planlayicilar cerceveyi
agac konumlarindan kestirir (gridplan.grid_angle).
Meyve, olgunluk, tespit, toplama, kasa ve yeniden planlama modeli
platform_sim ile aynidir (dugum = agac; platform agacin yanindaki koridorda
durur, drone'lar tacin tumunu toplar).
Planlayicilar: kdsteal (k-d bolge + uc devralma), kd (k-d bolge, devralma yok),
psb (kuresel PSB, her donuste yeniden plan), ortools<T>, greedy, rowblock
(bugunku pratik: bitisik sira bantlari, yeniden planlama yok).
Birimler: cm, cm/s, s.
"""
from __future__ import annotations

import heapq
import math
import os
import sys
import time

import numpy as np
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path.insert(0, P2)
sys.path.insert(0, HERE)
from mtsp import gridplan as G, mt  # noqa: E402

P = dict(
    A_U=400.0, A_V=400.0, A_JIT=(0.95, 1.10), BEND_AMP=(20.0, 60.0), BEND_LEN=(4000.0, 9000.0),
    SIGMA=25.0, MISSING=0.06, PATCHES=(1, 3), PATCH_R=(500.0, 1000.0), PATCH_P=0.5, INSET=200.0,
    MU_FRUIT=300.0, K_FRUIT=5.0,
    CULTIVARS=(("red", 0.148), ("yellow", -0.202), ("green", -0.602)),
    BLOCK_ROWS=5, TREE_SD=0.35, CANOPY=0.45, FRUIT_SD=0.35,
    SIG_PRIOR=0.6, FP_PRIOR=2.0, R_MIN=5,
    RHO=1000.0, P_VIS=0.60,
    REC=0.85, PREC=0.88, P_SUCC=0.80, T_PICK=6.0, M_DRONES=4, T_STOP=20.0,
    V_PLAT=100.0, T_TURN=20.0,
    Q=2300.0, T_BIN=120.0,
)


# --------------------------------------------------------------------------
# Bahce
# --------------------------------------------------------------------------
def make_trees(poly, phi_deg, seed, p=P):
    rng = np.random.default_rng(seed * 7919 + 11)
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    w = np.array([-u[1], u[0]])
    c = poly.mean(0)
    pu, pw = (poly - c) @ u, (poly - c) @ w
    au = p["A_U"] * rng.uniform(*p["A_JIT"])
    aw = p["A_V"] * rng.uniform(*p["A_JIT"])
    us = np.arange(pu.min() + p["INSET"], pu.max() - p["INSET"] + 1e-9, au)
    ws = np.arange(pw.min() + p["INSET"], pw.max() - p["INSET"] + 1e-9, aw)
    UU, WW = np.meshgrid(us, ws)
    UU, WW = UU.ravel(), WW.ravel()
    # egrilen siralar: dusuk frekansli yer degistirme (iki yonde)
    A1, L1, f1 = rng.uniform(*p["BEND_AMP"]), rng.uniform(*p["BEND_LEN"]), rng.uniform(0, 2 * math.pi)
    A2, L2, f2 = 0.5 * rng.uniform(*p["BEND_AMP"]), rng.uniform(*p["BEND_LEN"]), rng.uniform(0, 2 * math.pi)
    WW = WW + A1 * np.sin(2 * math.pi * UU / L1 + f1)
    UU = UU + A2 * np.sin(2 * math.pi * WW / L2 + f2)
    Pt = c + UU[:, None] * u + WW[:, None] * w
    inside = Path(poly).contains_points(Pt)
    A, B = poly, np.roll(poly, -1, axis=0)
    d = np.full(len(Pt), np.inf)
    for a, b in zip(A, B):
        ab = b - a
        t = np.clip(((Pt - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        d = np.minimum(d, np.hypot(*(Pt - (a + t[:, None] * ab)).T))
    keep = inside & (d >= p["INSET"] - 1e-6) & (rng.random(len(Pt)) >= p["MISSING"])
    # kumelenmis eksik agaclar (kurumus / yenilenmis yamalar)
    for _ in range(int(rng.integers(p["PATCHES"][0], p["PATCHES"][1] + 1))):
        q = Pt[rng.integers(len(Pt))]
        r = rng.uniform(*p["PATCH_R"])
        m = np.hypot(*(Pt - q).T) < r
        keep &= ~(m & (rng.random(len(Pt)) < p["PATCH_P"]))
    Pt = Pt[keep] + rng.normal(0, p["SIGMA"], (keep.sum(), 2))
    row = np.rint((WW[keep] - ws.min()) / aw).astype(int)
    return Pt, row, dict(au=au, aw=aw, bend=(A1, L1, A2, L2))


def rect_with_notch(L_m, W_m, seed):
    rng = np.random.default_rng(seed * 31 + 5)
    L, W = L_m * 100.0, W_m * 100.0
    nx, ny = rng.uniform(0.15, 0.35) * L, rng.uniform(0.15, 0.35) * W
    pts = np.array([[0, 0], [L, 0], [L, W - ny], [L - nx, W - ny], [L - nx, W], [0, W]], float)
    return pts - np.array([L, W]) / 2


def build(poly, phi_deg, seed=1, p=P):
    rng = np.random.default_rng(seed + 17)
    T, row, geo = make_trees(poly, phi_deg, seed, p)
    n = len(T)
    blocks = (row - row.min()) // p["BLOCK_ROWS"]
    cob = rng.permutation(np.resize(np.arange(len(p["CULTIVARS"])), blocks.max() + 1))
    cult = cob[blocks]
    shift = np.array([s for _, s in p["CULTIVARS"]])[cult]
    from ripeness_sim import ripeness_field
    f = ripeness_field(T[:, 0], T[:, 1], seed=seed, noise=0.0)
    teff = p["TREE_SD"] * (f - f.mean()) / (f.std() + 1e-12)
    N = rng.negative_binomial(p["K_FRUIT"], p["K_FRUIT"] / (p["K_FRUIT"] + p["MU_FRUIT"]), n)
    R = np.zeros(n, int)
    U = np.zeros(n, int)
    for i in range(n):
        if N[i]:
            m = shift[i] + teff[i] + p["CANOPY"] * (rng.random(N[i]) - 0.5) + rng.normal(0, p["FRUIT_SD"], N[i])
            R[i] = int((m >= 0).sum())
            U[i] = int(N[i] - R[i])
    prior = np.where(R > 0, R * np.exp(rng.normal(0, p["SIG_PRIOR"], n)), rng.poisson(p["FP_PRIOR"], n))
    # istasyon: gercek cercevede bas tarla ortasi (en kucuk u tarafi), 5 m disarida
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    w = np.array([-u[1], u[0]])
    c = T.mean(0)
    tu, tw = (T - c) @ u, (T - c) @ w
    st = c + u * (tu.min() - 500.0) + w * 0.5 * (tw.min() + tw.max())
    turn = p["T_TURN"] * p["V_PLAT"]
    Ftrue = G.Frame(T[:, 0], T[:, 1], st, turn, theta=((phi_deg + 45.0) % 90.0) - 45.0)
    Fest = G.Frame(T[:, 0], T[:, 1], st, turn)
    return dict(trees=T, row=row, cult=cult, R=R, U=U, est=prior.astype(float), Ftrue=Ftrue, Fest=Fest,
                station=st, phi=phi_deg, geo=geo, n_trees=n)


# --------------------------------------------------------------------------
# Tahminler
# --------------------------------------------------------------------------
def svc_est(est, p=P):
    return p["T_STOP"] + p["T_PICK"] * est * p["REC"] / p["PREC"] / p["M_DRONES"]


def dem_est(est, p=P):
    return est * p["REC"] * p["P_SUCC"] / p["PREC"]


def rowblock_trips(F, idx, s, d, k, Q):
    """Bugunku pratik: bitisik sira bantlari (tahmini is esit), yilan sirasi,
    kasa doldukca istasyon; yeniden planlama yok."""
    g = G.serpentine(F, idx, None, (F.su0, F.sv0))
    cum = np.cumsum(s[g]) / max(s[g].sum(), 1e-9)
    grp = np.minimum((cum * k - 1e-9).astype(int), k - 1)
    out = []
    for j in range(k):
        seq = g[grp == j]
        trips, cur, load = [], [], 0.0
        for q in seq:
            if cur and load + d[q] > Q:
                trips.append(np.array(cur, np.int64))
                cur, load = [], 0.0
            cur.append(int(q))
            load += d[q]
        if cur:
            trips.append(np.array(cur, np.int64))
        out.append(trips)
    return out


# --------------------------------------------------------------------------
# Simulasyon
# --------------------------------------------------------------------------
def simulate(o, k, planner, seed=1, p=P, log=False, oracle_frame=False):
    """planner 'ad@oracle': gercek cerceve verilir; 'ad@axis0': cerceve kestirilmez
    (theta = 0, harita ekseni)."""
    rng = np.random.default_rng(seed + 1000)
    Ft, Fe = o["Ftrue"], (o["Ftrue"] if oracle_frame else o["Fest"])
    if "@" in planner:
        planner, fr = planner.split("@")
        if fr == "oracle":
            Fe = o["Ftrue"]
        elif fr == "axis0":
            Fe = G.Frame(o["trees"][:, 0], o["trees"][:, 1], o["station"], Ft.turn, theta=0.0)
    R, U = o["R"].copy(), o["U"].copy()
    n = len(R)
    est = o["est"].copy()
    done = np.zeros(n, bool)
    reserved = np.zeros(n, bool)
    v, Q, tb = p["V_PLAT"], p["Q"], p["T_BIN"]
    st = dict(picked=0, unripe_picked=0, plans=0, plan_time=0.0, wait_plan=0.0, trips=0, partial=0,
              travel_s=0.0, service_s=0.0, steals=0, fallback=0)
    trace = [[] for _ in range(k)] if log else None
    picks = [] if log else None
    obs, ev = [], []
    sched = [[] for _ in range(k)]
    state = ["depot"] * k
    ret = np.zeros(k)
    cur_rem = np.zeros(k)                    # ucustaki seferin kalan tahmini hizmeti (devralma dengesi)
    pfree = 0.0
    from scipy.spatial import cKDTree
    kdt = cKDTree(o["trees"])
    ripe_total = int(R.sum())
    kd = None
    failed = [False]

    def alive():
        return ~done & ~reserved & (est >= p["R_MIN"])

    def apply_obs(t):
        while obs and obs[0][0] <= t:
            _, f, val = heapq.heappop(obs)
            if not done[f]:
                est[f] = val

    def timed(fn):
        nonlocal pfree
        t0 = time.perf_counter()
        r = fn()
        dt = time.perf_counter() - t0
        st["plans"] += 1
        st["plan_time"] += dt
        return r, dt

    def plan(now, ids):
        """Doner: plan bitis zamani; sched[j] guncellenir."""
        nonlocal pfree, kd
        apply_obs(now)
        start = max(now, pfree)
        a = alive()
        s_all = svc_est(est, p)
        d_all = dem_est(est, p)
        if planner in ("kdsteal", "kd", "kdsteal2", "kd2"):
            if kd is None:
                v2 = planner.endswith("2")
                if v2:
                    # agirlik: hizmet + kasa turu basina istasyon gidis-donusunun payi
                    dst = (np.abs(Fe.U - Fe.su0) + np.abs(Fe.V - Fe.sv0)) / v
                    wgt = np.where(a, s_all + 2.0 * dst * d_all / Q, 0.0)
                else:
                    wgt = np.where(a, s_all, 0.0)
                (kd_, dt) = timed(lambda: G.KDSteal(Fe, np.arange(n), wgt, k,
                                                    steal=planner.startswith("kdsteal"),
                                                    axis=("field" if v2 else None),
                                                    split=((v, Q, tb) if v2 else None)))
                kd = kd_
            else:
                dt = 0.0
            for j in ids:
                pend = ~done & ~reserved
                tr, dt2 = timed(lambda: kd.next_trip(j, s_all, d_all, Q, a, pend, cur_rem))
                dt += dt2
                sched[j] = [np.array(tr, np.int64)] if tr else []
                for q in tr:
                    a[q] = False
            end = start + dt
            pfree = end
            return end
        idx = np.where(a)[0]
        if planner == "greedy" or len(idx) == 0:
            for j in ids:
                sched[j] = greedy(idx[~reserved[idx]] if len(idx) else idx, s_all, d_all)
                for r in sched[j]:
                    reserved[r] = True
            for j in ids:
                for r in sched[j]:
                    reserved[r] = False
            return start
        av = np.array([max(0.0, (ret[j] + tb if state[j] == "flying" else now) - start) for j in range(k)])
        if planner == "rowblock":
            res, dt = timed(lambda: rowblock_trips(Fe, idx, s_all, d_all, k, Q))
        elif planner == "psb":
            res, dt = timed(lambda: G.psb_global(Fe, idx, s_all, d_all, v, Q, k, av, tb))
        elif planner.startswith("ortools"):
            T = float(planner[7:])

            def _ot():
                from mtsp import ortools_vrp
                M = np.zeros((len(idx) + 1, len(idx) + 1))
                uu = np.concatenate([[Fe.su0], Fe.U[idx]])
                vv = np.concatenate([[Fe.sv0], Fe.V[idx]])
                du = np.abs(uu[:, None] - uu[None, :])
                dv = np.abs(vv[:, None] - vv[None, :])
                M = (du + dv + Fe.turn * ((du > Fe.tol) & (dv > Fe.tol))) / v
                r = ortools_vrp.cvrp(M, s_all[idx], d_all[idx], Q, time_limit=T)
                if r["sorties"] is None:
                    return None
                trips = [idx[t] for t in r["sorties"]]
                durs = [G.rtime(Fe, t, v, s_all) for t in trips]
                own, _ = mt.assign(durs, k, av, tb)
                return [[trips[q] for q in sorted(own[j])] for j in range(k)]
            res, dt = timed(_ot)
        else:
            raise ValueError(planner)
        end = start + dt
        pfree = end
        if res is None:
            # planlayici plan donduremedi: plan isteyen platformlara cakismayan
            # acgozlu seferler, sonraki turda yeniden dene
            st["fallback"] += 1
            failed[0] = True
            held = []
            for j in ids:
                sched[j] = greedy(np.array([q for q in idx if not reserved[q]], np.int64), s_all, d_all)
                for r in sched[j]:
                    reserved[r] = True
                    held.append(r)
            for r in held:
                reserved[r] = False
            return end
        failed[0] = False
        for j in range(k):
            sched[j] = [np.asarray(r, np.int64) for r in res[j]]
        return end

    def greedy(pool, s_all, d_all):
        if not len(pool):
            return []
        left = list(map(int, pool))
        r, load = [], 0.0
        cu, cv = Fe.su0, Fe.sv0
        while left:
            L = np.array(left)
            dd = np.abs(Fe.U[L] - cu) + np.abs(Fe.V[L] - cv)
            q = int(L[np.argmin(dd)])
            if r and load + d_all[q] > Q:
                break
            r.append(q)
            load += d_all[q]
            left.remove(q)
            cu, cv = Fe.U[q], Fe.V[q]
        return [np.array(r, np.int64)]

    def tdist(a, b):
        """Gercek cercevede L1 + donus; -1 istasyon."""
        ua, va = (Ft.su0, Ft.sv0) if a < 0 else (Ft.U[a], Ft.V[a])
        ub, vb = (Ft.su0, Ft.sv0) if b < 0 else (Ft.U[b], Ft.V[b])
        return G._d(ua, va, ub, vb, Ft.turn, Ft.tol)

    def depart(j, t):
        while sched[j]:
            route = [int(f) for f in sched[j].pop(0) if not done[f] and not reserved[f]]
            if route:
                break
        else:
            state[j] = "idle"
            return
        reserved[route] = True
        st["trips"] += 1
        cur_rem[j] = float(np.sum(svc_est(est[route], p)))
        tt, load, cur = 0.0, 0.0, -1
        pth = [(t, -1)]
        for f in route:
            if load >= Q:
                break
            dd = tdist(cur, f)
            tt += dd / v
            st["travel_s"] += dd / v
            cur = f
            for h in kdt.query_ball_point(o["trees"][f], p["RHO"]):
                if not done[h] and h != f:
                    heapq.heappush(obs, (t + tt, h, float(rng.binomial(R[h], p["P_VIS"]) / p["P_VIS"])))
            det = rng.binomial(R[f], p["REC"])
            q_fp = min(1.0, det * (1 - p["PREC"]) / p["PREC"] / max(U[f], 1))
            fp = rng.binomial(U[f], q_fp) if U[f] else 0
            ok_r, ok_u = rng.binomial(det, p["P_SUCC"]), rng.binomial(fp, p["P_SUCC"])
            room, frac = Q - load, 1.0
            if ok_r + ok_u > room:
                frac = room / max(ok_r + ok_u, 1)
                ok_r = int(round(ok_r * frac))
                ok_u = int(min(room - ok_r, round(ok_u * frac)))
                st["partial"] += 1
            att = int(round((det + fp) * frac))
            dt = p["T_STOP"] + p["T_PICK"] * att / p["M_DRONES"]
            pth.append((t + tt, f))
            tt += dt
            st["service_s"] += dt
            R[f] -= ok_r
            U[f] -= ok_u
            st["picked"] += ok_r
            st["unripe_picked"] += ok_u
            load += ok_r + ok_u
            if picks is not None:
                picks.append((round(t + tt, 1), int(f), int(ok_r), int(ok_u), int(j)))
            pth.append((t + tt, f))
            if frac < 1.0:
                heapq.heappush(obs, (t + tt, f, float(R[f])))
                break
            done[f] = True
            est[f] = 0.0
        back = tdist(cur, -1) if cur >= 0 else 0.0
        tt += back / v
        st["travel_s"] += back / v
        pth.append((t + tt, -1))
        if trace is not None:
            trace[j].append(pth)
        for f in route:
            reserved[f] = False
        state[j] = "flying"
        ret[j] = t + tt
        heapq.heappush(ev, (t + tt, j))

    end0 = plan(0.0, list(range(k)))
    st["first_plan"] = end0
    for j in range(k):
        depart(j, end0)
    static = planner == "rowblock"
    t_now = end0
    while ev:
        t_now, j = heapq.heappop(ev)
        state[j] = "depot"
        cur_rem[j] = 0.0
        if static:
            dep = t_now + tb
        else:
            end = plan(t_now, [j])
            dep = max(t_now + tb, end)
            st["wait_plan"] += max(0.0, end - (t_now + tb))
        depart(j, dep)
        if not static:
            for i in range(k):
                if state[i] == "idle":
                    end = t_now
                    if planner in ("kdsteal", "kd", "kdsteal2", "kd2", "greedy"):
                        end = plan(t_now, [i])
                    elif failed[0]:
                        a_ = alive()
                        sched[i] = greedy(np.where(a_)[0], svc_est(est, p), dem_est(est, p))
                    if sched[i]:
                        state[i] = "depot"
                        depart(i, max(t_now, end))
    st["mission_s"] = float(max([t_now] + list(ret)))
    st["ripe_total"] = ripe_total
    st["recall"] = st["picked"] / max(1, ripe_total)
    st["precision"] = st["picked"] / max(1, st["picked"] + st["unripe_picked"])
    st["steals"] = kd.steals if kd is not None else 0
    st["theta_err"] = float(abs(((o["Fest"].theta - o["Ftrue"].theta) + 45) % 90 - 45))
    st["nodes"] = n
    if log:
        st["trace"] = trace
        st["picks"] = picks
    return st
