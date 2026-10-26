# -*- coding: utf-8 -*-
"""Sofralik elma icin secmeli hasat: kablolu drone tasiyan yer platformlari.

Senaryo (bir toplama gecisi; Tevel tipi uygulama): k platform bas tarladaki
kasa istasyonundan calisir. Platform sira araliginda bir duraga (DURAK = aralik
boyunca SEG uzunlugunda hucre) gelir; uzerindeki M_DRONES kablolu drone iki
yandaki agac yuzlerinde olgun meyveyi tespit eder, koparir ve platformdaki kasaya
koyar (yere dusurme yok). Kasa dolunca (Q meyve) platform istasyona doner,
kasa degisir (T_BIN). Ham meyve agacta kalir (sonraki gecis).

Model ozeti (degerler ve kaynaklar: data/harvest_params.md):
  * Bahce: Tall Spindle 0.9 m x 3.3 m (gercek OSM bahce sinirlari icinde ya da
    dikdortgen blok), %5 eksik agac, cesit bloklari (BLOCK_ROWS sira).
  * Meyve: agac basina NegBin(MU_FRUIT, K_FRUIT); yuzlere esit; olgunluk
    m = cesit kaymasi + agac etkisi (uzamsal) + tac konumu (dis/ust erken) + gurultu.
  * On bilgi: durak basina olgun meyve tahmini = gercek * lognormal(0, SIG_PRIOR)
    (+ olgun meyvesi olmayan durakta yanlis pozitif).
  * Gozlem: platform kameralari ayni araliktaki RHO icindeki duraklari gorur;
    tahmin = Bin(R, P_VIS) / P_VIS.
  * Toplama: olgun meyvenin REC'i tespit edilir; kesinlik PREC'e gore ham meyve
    de yanlislikla denenir; her deneme T_PICK s, basari P_SUCC; M_DRONES drone
    paralel calisir; durakta sabit T_STOP. Kasa dolarsa durak yarim kalir.
  * Hareket: platform hizi V_PLAT; sira degisimi bas tarladan, donus cezasi T_TURN.
  * Yeniden planlama: her kasa degisiminde kalan is tum filo icin; tek
    planlayici; platform max(donus + T_BIN, plan bitisi) aninda kalkar.
Planlayicilar: psb (PSB-MT/R), psb_lns1, ortools<T>, greedy, rowblock (bugunku
ekip pratigi: sira bloklari, yeniden planlama yok), static (PSB, yeniden
planlama yok), delay<L> (PSB + yapay planlama gecikmesi L s).
Birimler: konum cm, hiz cm/s, sure s.
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
from mtsp import mtr, mt  # noqa: E402

P = dict(
    ROW=330.0, TREE=90.0, INSET=150.0, MISSING=0.05, SIGMA=10.0,   # Tall Spindle (cm)
    SEG=360.0,            # cm, durak hucresi (4 agac)
    MU_FRUIT=130.0, K_FRUIT=8.0,
    CULTIVARS=(("red", 0.30), ("yellow", -0.05), ("green", -0.45)),
    BLOCK_ROWS=6, TREE_SD=0.35, CANOPY=0.45, FRUIT_SD=0.35,
    SIG_PRIOR=0.6, FP_PRIOR=2.0, R_MIN=5,
    RHO=1000.0, P_VIS=0.60,
    REC=0.85, PREC=0.88, P_SUCC=0.80, T_PICK=6.0, M_DRONES=4, T_STOP=20.0,
    V_PLAT=100.0, T_TURN=20.0,
    Q=2300.0, T_BIN=120.0,
)


# --------------------------------------------------------------------------
# Bahce: agaclar, sira araliklari, duraklar
# --------------------------------------------------------------------------
def trees_in_polygon(poly, phi_deg, p=P, seed=1):
    rng = np.random.default_rng(seed)
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    w = np.array([-u[1], u[0]])
    c = poly.mean(0)
    pu, pw = (poly - c) @ u, (poly - c) @ w
    us = np.arange(pu.min() + p["INSET"], pu.max() - p["INSET"] + 1e-9, p["TREE"])
    ws = np.arange(pw.min() + p["INSET"], pw.max() - p["INSET"] + 1e-9, p["ROW"])
    G = np.array([c + a * u + b * w for b in ws for a in us])
    G = G[Path(poly).contains_points(G)]
    # sinirdan INSET icerde
    A, B = poly, np.roll(poly, -1, axis=0)
    d = np.full(len(G), np.inf)
    for a, b in zip(A, B):
        ab = b - a
        t = np.clip(((G - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        d = np.minimum(d, np.hypot(*(G - (a + t[:, None] * ab)).T))
    G = G[d >= p["INSET"] - 1e-6]
    G = G[rng.random(len(G)) >= p["MISSING"]]
    return G + rng.normal(0, p["SIGMA"], G.shape)


def rect_block(rows, length_m, p=P):
    L = length_m * 100.0
    W = rows * p["ROW"]
    return np.array([[0, 0], [L, 0], [L, W], [0, W]], float) - np.array([L, W]) / 2 + 1e-3


def build(poly, phi_deg, depot, seed=1, p=P):
    """Doner: dict(trees, field (mtr.Field), dugum verisi, depo)."""
    rng = np.random.default_rng(seed + 17)
    T = trees_in_polygon(poly, phi_deg, p, 1)          # bahce geometrisi tohumdan bagimsiz
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    w = np.array([-u[1], u[0]])
    c = T.mean(0)
    tu, tw = (T - c) @ u, (T - c) @ w
    if depot is None:                                 # bas tarla ortasi, bir agac araligi disarida
        dd = c + u * (tu.min() - 300.0) + w * 0.5 * (tw.min() + tw.max())
        depot = (float(dd[0]), float(dd[1]))
    row = np.rint((tw - tw.min()) / p["ROW"]).astype(int)
    w0 = tw.min()
    nrow = row.max() + 1
    # cesit bloklari
    blocks = row // p["BLOCK_ROWS"]
    cob = rng.permutation(np.resize(np.arange(len(p["CULTIVARS"])), blocks.max() + 1))
    cult = cob[blocks]
    shift = np.array([s for _, s in p["CULTIVARS"]])[cult]
    from ripeness_sim import ripeness_field
    f = ripeness_field(T[:, 0], T[:, 1], seed=seed, noise=0.0)
    teff = p["TREE_SD"] * (f - f.mean()) / (f.std() + 1e-12)
    nT = len(T)
    N = rng.negative_binomial(p["K_FRUIT"], p["K_FRUIT"] / (p["K_FRUIT"] + p["MU_FRUIT"]), nT)
    # yuz: 0 = +w yani aralik row, 1 = -w yani aralik row-1
    R = np.zeros((nT, 2), int)
    U = np.zeros((nT, 2), int)
    for i in range(nT):
        if N[i] == 0:
            continue
        side = rng.integers(0, 2, N[i])
        pos = rng.random(N[i])
        m = shift[i] + teff[i] + p["CANOPY"] * (pos - 0.5) + rng.normal(0, p["FRUIT_SD"], N[i])
        ripe = m >= 0
        for sd in (0, 1):
            R[i, sd] = int(np.sum(ripe & (side == sd)))
            U[i, sd] = int(np.sum(~ripe & (side == sd)))
    # araliklar: a = -1 .. nrow-1 (a, row a ile a+1 arasi); uc noktalari
    alleys = {}
    for a in range(-1, nrow):
        sel = (row == a) | (row == a + 1)
        if not sel.any():
            continue
        s0, s1 = tu[sel].min() - 100.0, tu[sel].max() + 100.0
        wa = w0 + (a + 0.5) * p["ROW"]
        alleys[a] = (s0, s1, wa)
    akeys = sorted(alleys)
    aidx = {a: q for q, a in enumerate(akeys)}
    E0 = np.zeros((len(akeys), 3))
    E1 = np.zeros((len(akeys), 3))
    for a, q in aidx.items():
        s0, s1, wa = alleys[a]
        E0[q, :2] = c + s0 * u + wa * w
        E1[q, :2] = c + s1 * u + wa * w
        E0[q, 2], E1[q, 2] = s0, s1
    # duraklar: aralik boyunca SEG hucreleri; her hucre iki yandaki yuzleri toplar
    nodeA, nodeS, nR, nU, nTr = [], [], [], [], []
    for a, q in aidx.items():
        s0, s1, wa = alleys[a]
        up = np.where(row == a)[0]           # aralik a'ya +w yuzu bakan agaclar (yuz 0)
        lo = np.where(row == a + 1)[0]       # aralik a'ya -w yuzu bakan agaclar (yuz 1)
        nseg = int(math.ceil((s1 - s0) / p["SEG"]))
        for b in range(nseg):
            a0, a1 = s0 + b * p["SEG"], s0 + (b + 1) * p["SEG"]
            t_up = up[(tu[up] >= a0) & (tu[up] < a1)]
            t_lo = lo[(tu[lo] >= a0) & (tu[lo] < a1)]
            if len(t_up) + len(t_lo) == 0:
                continue
            nodeA.append(q)
            nodeS.append(0.5 * (a0 + a1))
            nR.append(int(R[t_up, 0].sum() + R[t_lo, 1].sum()))
            nU.append(int(U[t_up, 0].sum() + U[t_lo, 1].sum()))
            nTr.append((t_up.tolist(), t_lo.tolist()))
    nodeA = np.array(nodeA, np.int64)
    nodeS = np.array(nodeS, float)
    Rn = np.array(nR, int)
    Un = np.array(nU, int)
    prior = np.where(Rn > 0, Rn * np.exp(rng.normal(0, p["SIG_PRIOR"], len(Rn))), rng.poisson(p["FP_PRIOR"], len(Rn)))
    F = mtr.Field(nodeA, nodeS, E0, E1, depot[0], depot[1], p["T_TURN"] * p["V_PLAT"])
    XY = np.array([c + s * u + (w0 + (akeys[q] + 0.5) * p["ROW"]) * w for q, s in zip(nodeA, nodeS)])
    return dict(trees=T, cult=cult, R_tree=R, U_tree=U, F=F, R=Rn, U=Un, est=prior.astype(float),
                node_trees=nTr, node_xy=XY, alley_keys=akeys, u=u, w=w, c=c, depot=np.asarray(depot, float),
                n_trees=nT, area_ha=None)


# --------------------------------------------------------------------------
# Tahminler ve planlayicilar
# --------------------------------------------------------------------------
def svc_est(est, p=P):
    att = est * p["REC"] / p["PREC"]
    return p["T_STOP"] + p["T_PICK"] * att / p["M_DRONES"]


def dem_est(est, p=P):
    return est * p["REC"] * p["P_SUCC"] / p["PREC"]


def make_planner(name, p=P):
    v, Q, tb = p["V_PLAT"], p["Q"], p["T_BIN"]
    if name in ("psb", "static") or name.startswith("delay"):
        return lambda F, s, d, k, av: mtr.solve(F, s, d, v, Q, k, av, tb, 0.0)[0]
    if name.startswith("psb_lns"):
        b = float(name[7:])
        return lambda F, s, d, k, av: mtr.solve(F, s, d, v, Q, k, av, tb, b)[0]
    if name.startswith("ortools"):
        from mtsp import ortools_vrp
        T = float(name[7:])

        def plan(F, s, d, k, av):
            r = ortools_vrp.cvrp(F.matrix(v), s, d, Q, time_limit=T)
            if r["sorties"] is None:
                return None
            return mtr.schedule(F, r["sorties"], v, s, k, av, tb)[0]
        return plan
    if name in ("greedy", "rowblock"):
        return None
    raise ValueError(name)


def rowblock_plan(F, idx, s, d, k, p=P):
    """Bugunku pratik: tum araliklar yilan sirasinda dolasilir; bu sira tahmini
    is esit olacak sekilde k bitisik bolume ayrilir (bolum siniri aralik icine
    dusebilir); her platform kendi bolumunu sirayla toplar, kasa doldukca
    istasyona doner. Yeniden planlama yok. Doner: platform basina sefer listesi."""
    sub = F.sub(idx)
    g_all = mtr.serpentine(sub, None, 0)
    cum = np.cumsum(s[g_all]) / max(s.sum(), 1e-9)
    grp = np.minimum((cum * k - 1e-9).astype(int), k - 1)
    out = []
    for j in range(k):
        g = g_all[grp == j]
        if not len(g):
            out.append([])
            continue
        trips, cur, load = [], [], 0.0
        for q in g:
            if cur and load + d[q] > p["Q"]:
                trips.append(np.array(cur, np.int64))
                cur, load = [], 0.0
            cur.append(int(q))
            load += d[q]
        if cur:
            trips.append(np.array(cur, np.int64))
        out.append(trips)
    return out


def greedy_trip(F, pool, s, d, p=P):
    if not len(pool):
        return []
    sub = F.sub(pool)
    left = list(range(len(pool)))
    r, load = [], 0.0
    cur = None
    while left:
        if cur is None:
            dd = [mtr._dist_depot(q, sub.A, sub.S, sub.E0, sub.E1, sub.SE) for q in left]
        else:
            dd = [mtr._dist(cur, q, sub.A, sub.S, sub.E0, sub.E1, sub.turn, sub.EE) for q in left]
        q = left[int(np.argmin(dd))]
        if r and load + d[pool[q]] > p["Q"]:
            break
        r.append(int(pool[q]))
        load += d[pool[q]]
        left.remove(q)
        cur = q
    return [np.array(r, np.int64)]


# --------------------------------------------------------------------------
# Simulasyon
# --------------------------------------------------------------------------
def simulate(o, k, planner_name, seed=1, p=P, max_time=24 * 3600.0, log=False):
    rng = np.random.default_rng(seed + 1000)
    F = o["F"]
    R, U = o["R"].copy(), o["U"].copy()
    n = len(R)
    est = o["est"].copy()
    done = np.zeros(n, bool)
    reserved = np.zeros(n, bool)
    v, Q, tb = p["V_PLAT"], p["Q"], p["T_BIN"]
    planner = make_planner(planner_name, p)
    delay = float(planner_name[5:]) if planner_name.startswith("delay") else 0.0
    static = planner_name in ("static", "rowblock")
    obs, ev = [], []
    sched = [[] for _ in range(k)]
    state = ["depot"] * k
    ret = np.zeros(k)
    pfree = 0.0
    st = dict(picked=0, unripe_picked=0, attempts=0, plans=0, plan_time=0.0, wait_plan=0.0, trips=0,
              partial=0, fallback=0, travel_s=0.0, service_s=0.0, idle_end=0.0)
    trace = [[] for _ in range(k)] if log else None
    picks = [] if log else None
    ripe_total = int(R.sum())
    failed = [False]
    byalley = {}
    for i in range(n):
        byalley.setdefault(int(F.A[i]), []).append(i)

    def apply_obs(t):
        while obs and obs[0][0] <= t:
            _, f, val = heapq.heappop(obs)
            if not done[f]:
                est[f] = val

    def pool():
        return np.where(~done & ~reserved & (est >= p["R_MIN"]))[0]

    def plan(now, ids):
        nonlocal pfree
        apply_obs(now)
        idx = pool()
        start = max(now, pfree)
        if planner_name == "greedy" or len(idx) == 0:
            held = []
            for j in ids:
                sched[j] = greedy_trip(F, pool(), svc_est(est, p), dem_est(est, p), p) if len(idx) else []
                for r in sched[j]:
                    reserved[r] = True
                    held.append(r)
            for r in held:
                reserved[r] = False
            return start
        av = np.array([max(0.0, (ret[j] + tb if state[j] == "flying" else now) - start) for j in range(k)])
        t0 = time.perf_counter()
        if planner_name == "rowblock":
            res = rowblock_plan(F, idx, svc_est(est[idx], p), dem_est(est[idx], p), k, p)
        else:
            res = planner(F.sub(idx), svc_est(est[idx], p), dem_est(est[idx], p), k, av)
        t_p = time.perf_counter() - t0 + delay
        st["plans"] += 1
        st["plan_time"] += t_p
        end = start + t_p
        pfree = end
        if res is None:
            # planlayici bu turda plan donduremedi: plan isteyen platformlara
            # acgozlu seferler (birbirine cakismayan), sonraki turda yeniden dene
            st["fallback"] += 1
            failed[0] = True
            held = []
            for j in ids:
                sched[j] = greedy_trip(F, pool(), svc_est(est, p), dem_est(est, p), p)
                for r in sched[j]:
                    reserved[r] = True
                    held.append(r)
            for r in held:
                reserved[r] = False
            return end
        failed[0] = False
        for j in range(k):
            sched[j] = [idx[np.asarray(r, int)] for r in res[j]]
        return end

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
        tt, load, cur = 0.0, 0.0, -1
        pth = [(t, -1)]
        for f in route:
            dist = mtr._dist_depot(f, F.A, F.S, F.E0, F.E1, F.SE) if cur < 0 else \
                mtr._dist(cur, f, F.A, F.S, F.E0, F.E1, F.turn, F.EE)
            if load >= Q:
                break
            tt += dist / v
            st["travel_s"] += dist / v
            cur = f
            # kameralar: ayni aralikta RHO icindeki duraklar
            for h in byalley[int(F.A[f])]:
                if not done[h] and abs(F.S[h] - F.S[f]) <= p["RHO"] and h != f:
                    heapq.heappush(obs, (t + tt, h, float(rng.binomial(R[h], p["P_VIS"]) / p["P_VIS"])))
            det = rng.binomial(R[f], p["REC"])
            q_fp = min(1.0, det * (1 - p["PREC"]) / p["PREC"] / max(U[f], 1))
            fp = rng.binomial(U[f], q_fp) if U[f] else 0
            ok_r = rng.binomial(det, p["P_SUCC"])
            ok_u = rng.binomial(fp, p["P_SUCC"])
            room = Q - load
            frac = 1.0
            if ok_r + ok_u > room:                       # kasa doldu: durak yarim kalir
                frac = room / max(ok_r + ok_u, 1)
                ok_r = int(round(ok_r * frac))
                ok_u = int(min(room - ok_r, round(ok_u * frac)))
                st["partial"] += 1
            att = int(round((det + fp) * frac))
            dt = p["T_STOP"] + p["T_PICK"] * att / p["M_DRONES"]
            pth.append((t + tt, f))
            tt += dt
            st["service_s"] += dt
            st["attempts"] += att
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
        back = mtr._dist_depot(cur, F.A, F.S, F.E0, F.E1, F.SE) if cur >= 0 else 0.0
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
    t_now = end0
    while ev:
        t_now, j = heapq.heappop(ev)
        if t_now > max_time:
            break
        state[j] = "depot"
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
                    if planner_name == "greedy" or failed[0]:
                        sched[i] = greedy_trip(F, pool(), svc_est(est, p), dem_est(est, p), p)
                    if sched[i]:
                        state[i] = "depot"
                        depart(i, max(t_now, pfree))
    st["mission_s"] = float(max([t_now] + list(ret)))
    st["ripe_total"] = ripe_total
    st["recall"] = st["picked"] / max(1, ripe_total)
    st["precision"] = st["picked"] / max(1, st["picked"] + st["unripe_picked"])
    st["nodes"] = n
    if log:
        st["trace"] = trace
        st["picks"] = picks
    return st
