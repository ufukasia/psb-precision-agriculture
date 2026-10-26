# -*- coding: utf-8 -*-
"""Sanayilik elma icin drone ile secmeli hasat: olay tabanli simulasyon.

Senaryo (bir hasat gecisi): k drone tek bir bas tarla istasyonundan calisir.
Drone sira arasinda agacin bir yuzune (face) gelir, olgun meyveleri tespit
eder, saplarini keser; meyve yere duser ve sonradan yerden toplanir (sanayilik
elmada yerden toplama uygulamasi). Ham meyve agacta kalir (sonraki gecis).

Model
  * Agac yuzleri: her agacin iki yuzu vardir (iki komsu sira araligi); hedef =
    yuz, konum = agac + sira dikine +-FACE_OFF.
  * Cesit bloklari: siralar BLOCK_ROWS'luk bloklara ayrilir, her bloga bir
    cesit atanir; cesitlerin olgunluk kaymasi farklidir (farkli hasat tarihleri).
  * Meyve: agac basina N ~ NegBin(ortalama MU_FRUIT, dagilim K_FRUIT); her meyve
    yuzlere esit olasilikla; olgunluk m = cesit kaymasi + agac etkisi (uzamsal
    alan) + tac konumu etkisi (dis/ust daha erken) + gurultu; m >= 0 olgun.
  * On bilgi (uzaktan algilama / onceki gecis): yuz basina olgun meyve tahmini
    R^ = R * lognormal(0, SIG_PRIOR) (+ yanlis pozitif Poisson).
  * Gozlem: ucus yolunun RHO yaricapindaki yuzlerin tahmini yakindan
    olcumle guncellenir: R^ = Bin(R, P_VIS) / P_VIS.
  * Hizmet: yuzde olgun meyvenin P_DET'i tespit edilip kesilir, ham meyvenin
    P_FP'si yanlislikla olgun sanilip kesilir; sure T_FACE + T_CUT * kesilen.
  * Batarya: sorti suresi <= E_USE; drone bir sonraki yuze gitmeden once
    (gidis + tahmini hizmet + donus) <= kalan sure kontrolu yapar, degilse doner;
    hizmet sirasinda donus suresi kadar pay kalinca keser (kalan meyve yuzde
    kalir, yuz havuza geri doner). Istasyonda batarya degisimi T_SWAP.
  * Yeniden planlama: her drone donusunde (olay tetiklemeli) kalan havuz tum
    filo icin yeniden planlanir; tek planlayici (tek cekirdek) siradadir; drone
    max(donus + T_SWAP, plan bitisi) aninda kalkar. Planlama suresi duvar
    saatiyle olculur ve simulasyon saatine eklenir.
Planlayicilar: psb_mt (PSB-MT), psb_mt_lns1, ortools<T>, lkh_dcvrp<T>,
greedy (donen drone'a en yakin hedeflerden sorti), static (yalniz baslangic
plani, yeniden planlama yok).
Birimler: konum cm, hiz cm/s, sure s.
"""
from __future__ import annotations

import heapq
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path.insert(0, P2)
from mtsp import mt, mm  # noqa: E402

# --------------------------------------------------------------------------
# Parametreler (kaynaklar: paper2/data/harvest_params.md)
# --------------------------------------------------------------------------
P = dict(
    SPEED=300.0,        # cm/s, sira arasi seyir
    FACE_OFF=120.0,     # cm, agac merkezinden yuz hedef noktasina
    MU_FRUIT=120.0,     # agac basina meyve (ortalama)
    K_FRUIT=8.0,        # NegBin dagilim parametresi
    CULTIVARS=(("red", 0.35), ("yellow", -0.05), ("green", -0.60)),   # (ad, olgunluk kaymasi)
    BLOCK_ROWS=6,
    TREE_SD=0.35,       # agac etkisi olcegi
    CANOPY=0.45,        # dis/ust tac etkisi
    FRUIT_SD=0.35,      # meyve gurultusu
    SIG_PRIOR=0.6,      # on tahmin lognormal hatasi
    FP_PRIOR=1.0,       # olgun meyvesi olmayan yuzde on tahmin yanlis pozitif ortalamasi
    R_MIN=3,            # ziyaret esigi (tahmini olgun meyve)
    RHO=400.0,          # cm, gozlem yaricapi
    P_VIS=0.8,          # gozlemde gorunen meyve orani
    P_DET=0.90,         # hizmette olgun meyve tespit-kesim basarisi
    P_FP=0.03,          # ham meyvenin olgun sanilip kesilme olasiligi
    T_FACE=8.0,         # s, yuze yaklasma, tarama, ayrilma
    T_CUT=3.0,          # s, meyve basina kesim
    E_USE=1200.0,       # s, kullanilabilir sorti suresi
    T_SWAP=60.0,        # s, batarya degisimi
)


# --------------------------------------------------------------------------
# Bahce ve meyve
# --------------------------------------------------------------------------
def build(X, Y, phi_deg, dx, dy, seed=1, p=P):
    rng = np.random.default_rng(seed)
    n = len(X)
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    w = np.array([-u[1], u[0]])
    P0 = np.column_stack([X, Y])
    c = P0.mean(0)
    row = np.rint(((P0 - c) @ w) / 400.0).astype(int)
    row -= row.min()
    blocks = row // p["BLOCK_ROWS"]
    cult_of_block = rng.permutation(np.resize(np.arange(len(p["CULTIVARS"])), blocks.max() + 1))
    cult = cult_of_block[blocks]
    shift = np.array([sh for _, sh in p["CULTIVARS"]])[cult]
    # agac etkisi: yumusak uzamsal alan
    from ripeness_sim import ripeness_field
    f = ripeness_field(X, Y, seed=seed, noise=0.0)
    tree_eff = p["TREE_SD"] * (f - f.mean()) / (f.std() + 1e-12)
    N = rng.negative_binomial(p["K_FRUIT"], p["K_FRUIT"] / (p["K_FRUIT"] + p["MU_FRUIT"]), n)
    R = np.zeros((n, 2), int)
    U = np.zeros((n, 2), int)
    for i in range(n):
        if N[i] == 0:
            continue
        side = rng.integers(0, 2, N[i])
        pos = rng.random(N[i])                                 # 0 ic/alt ... 1 dis/ust
        m = shift[i] + tree_eff[i] + p["CANOPY"] * (pos - 0.5) + rng.normal(0, p["FRUIT_SD"], N[i])
        ripe = m >= 0
        for sd in (0, 1):
            R[i, sd] = int(np.sum(ripe & (side == sd)))
            U[i, sd] = int(np.sum(~ripe & (side == sd)))
    FX = np.concatenate([X + w[0] * p["FACE_OFF"], X - w[0] * p["FACE_OFF"]])
    FY = np.concatenate([Y + w[1] * p["FACE_OFF"], Y - w[1] * p["FACE_OFF"]])
    Rf = np.concatenate([R[:, 0], R[:, 1]])
    Uf = np.concatenate([U[:, 0], U[:, 1]])
    prior = np.where(Rf > 0, Rf * np.exp(rng.normal(0, p["SIG_PRIOR"], 2 * n)), rng.poisson(p["FP_PRIOR"], 2 * n))
    return dict(FX=FX, FY=FY, R=Rf, U=Uf, est=prior.astype(float), cult=np.concatenate([cult, cult]),
                tree=np.concatenate([np.arange(n), np.arange(n)]), dx=dx, dy=dy, N=N, n_trees=n)


# --------------------------------------------------------------------------
# Planlayicilar: plan(FX, FY, s, k, avail, now) -> drone basina sorti listesi
# --------------------------------------------------------------------------
def service_est(est, p=P):
    return p["T_FACE"] + p["T_CUT"] * p["P_DET"] * est


def make_planner(name, p=P):
    v, E, ts = p["SPEED"], p["E_USE"], p["T_SWAP"]
    if name.startswith("delay"):                # PSB-MT + yapay planlama gecikmesi (simulate ekler)
        return lambda X, Y, dx, dy, s, k, av: mt.solve(X, Y, dx, dy, s, v, E, k, av, ts, 0.0)[0]
    if name in ("psb_mt", "static"):
        return lambda X, Y, dx, dy, s, k, av: mt.solve(X, Y, dx, dy, s, v, E, k, av, ts, 0.0)[0]
    if name.startswith("psb_mt_lns"):
        b = float(name[10:])
        return lambda X, Y, dx, dy, s, k, av: mt.solve(X, Y, dx, dy, s, v, E, k, av, ts, b)[0]
    if name.startswith("ortools"):
        from mtsp import ortools_vrp
        T = float(name[7:])

        def plan(X, Y, dx, dy, s, k, av):
            r = ortools_vrp.sorties(X, Y, dx, dy, s, v, E, time_limit=T)
            if r["sorties"] is None:
                return None
            return mt.plan_from_sorties(r["sorties"], X, Y, dx, dy, s, v, k, av, ts)[0]
        return plan
    if name.startswith("lkh_dcvrp"):
        from mtsp import lkh3
        T = float(name[9:])

        def plan(X, Y, dx, dy, s, k, av):
            r = lkh3.dcvrp(X, Y, dx, dy, s, v, E, time_limit=T)
            if r["sorties"] is None:
                return None
            return mt.plan_from_sorties([x for x in r["sorties"] if len(x)], X, Y, dx, dy, s, v, k, av, ts)[0]
        return plan
    if name == "greedy":
        return None
    raise ValueError(name)


def greedy_sortie(pool, FX, FY, dx, dy, s, p=P):
    """Donen drone icin: en yakin hedefe git, batarya yettigi surece en yakin
    komsu ile devam et."""
    v, E = p["SPEED"], p["E_USE"]
    left = set(pool)
    r = []
    x, y, t = dx, dy, 0.0
    while left:
        cand = np.fromiter(left, int)
        d = np.hypot(FX[cand] - x, FY[cand] - y)
        q = int(cand[np.argmin(d)])
        go = float(np.min(d)) / v
        back = math.hypot(FX[q] - dx, FY[q] - dy) / v
        if r and t + go + s[q] + back > E:
            break
        r.append(q)
        t += go + s[q]
        x, y = FX[q], FY[q]
        left.discard(q)
    return [np.array(r, np.int32)] if r else []


# --------------------------------------------------------------------------
# Simulasyon
# --------------------------------------------------------------------------
def simulate(o, k, planner_name, seed=1, p=P, max_time=10 * 3600.0, log_paths=False):
    rng = np.random.default_rng(seed + 1000)
    FX, FY, R, U = o["FX"], o["FY"], o["R"].copy(), o["U"].copy()
    dx, dy = o["dx"], o["dy"]
    v, E, ts = p["SPEED"], p["E_USE"], p["T_SWAP"]
    nf = len(FX)
    est = o["est"].copy()
    done = np.zeros(nf, bool)
    reserved = np.zeros(nf, bool)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([FX, FY]))
    planner = make_planner(planner_name, p)
    delay = float(planner_name[5:]) if planner_name.startswith("delay") else 0.0
    obs = []                                            # (t, face, value) gozlem olaylari
    ev = []                                             # (t, tur, drone)
    sched = [[] for _ in range(k)]
    state = ["depot"] * k
    ready = np.zeros(k)
    ret_time = np.zeros(k)
    planner_free = 0.0
    stats = dict(harvested=0, unripe_cut=0, plans=0, plan_time=0.0, wait_plan=0.0, sorties=0, truncated=0,
                 fallback=0, partial=0)
    paths = [[] for _ in range(k)] if log_paths else None
    cuts = [] if log_paths else None              # (t_bitis, yuz, kesilen olgun, kesilen ham, drone)
    ripe_total = int(R.sum())

    def apply_obs(t):
        while obs and obs[0][0] <= t:
            _, f, val = heapq.heappop(obs)
            if not done[f]:
                est[f] = val

    def pool():
        return np.where(~done & ~reserved & (est >= p["R_MIN"]))[0]

    def plan(now, drones):
        """drones: plan alacak drone'lar (depoda). Doner: plan bitis zamani."""
        nonlocal planner_free
        apply_obs(now)
        idx = pool()
        start = max(now, planner_free)
        if planner is None or len(idx) == 0:
            t_p = 0.0
            res = None
        else:
            av = np.array([max(0.0, (ready[j] if state[j] == "depot" else ret_time[j] + ts) - start) for j in range(k)])
            t0 = time.perf_counter()
            res = planner(FX[idx], FY[idx], dx, dy, service_est(est[idx], p), k, av)
            t_p = time.perf_counter() - t0 + delay
            stats["plans"] += 1
            stats["plan_time"] += t_p
        end = start + t_p
        planner_free = end
        if planner is None or (res is None and len(idx)):
            if res is None and planner is not None:
                stats["fallback"] += 1
            for j in drones:                      # aç gözlü dagitim
                sched[j] = greedy_sortie(pool(), FX, FY, dx, dy, service_est(est, p), p)
                for r in sched[j]:
                    reserved[r] = True
            return end
        for j in range(k):
            sched[j] = [idx[np.asarray(r, int)] for r in res[j]] if res is not None else []
        return end

    def depart(j, t):
        if not sched[j]:
            state[j] = "idle"
            return
        route = sched[j].pop(0)
        route = [int(f) for f in route if not done[f]]
        if not route:
            return depart(j, t)
        reserved[route] = True
        stats["sorties"] += 1
        # sortiyi yurut
        x, y, tt = dx, dy, 0.0
        visited = []
        pth = [(t, dx, dy)]
        for q, f in enumerate(route):
            go = math.hypot(FX[f] - x, FY[f] - y) / v
            back = math.hypot(FX[f] - dx, FY[f] - dy) / v
            s_est = service_est(est[f], p)
            if visited and tt + go + s_est + back > E:
                stats["truncated"] += 1
                break
            # ucus sirasinda gozlem
            L = max(1, int(go * v / (0.5 * p["RHO"])))
            for a in range(1, L + 1):
                px = x + (FX[f] - x) * a / L
                py = y + (FY[f] - y) * a / L
                for h in tree.query_ball_point((px, py), p["RHO"]):
                    if not done[h]:
                        val = rng.binomial(R[h], p["P_VIS"]) / p["P_VIS"]
                        heapq.heappush(obs, (t + tt + go * a / L, int(h), float(val)))
            tt += go
            x, y = FX[f], FY[f]
            pth.append((t + tt, x, y))
            # hizmet: tespit, kesim
            det = rng.binomial(R[f], p["P_DET"])
            fp = rng.binomial(U[f], p["P_FP"])
            room = E - tt - back - p["T_FACE"]
            can = max(0, int(room // p["T_CUT"]))
            cut = min(det + fp, can)
            cut_r = min(det, cut)
            cut_u = cut - cut_r
            tt += p["T_FACE"] + p["T_CUT"] * cut
            pth.append((t + tt, x, y))
            R[f] -= cut_r
            U[f] -= cut_u
            stats["harvested"] += cut_r
            stats["unripe_cut"] += cut_u
            if cuts is not None:
                cuts.append((round(t + tt, 2), int(f), int(cut_r), int(cut_u), int(j)))
            visited.append(f)
            if cut < det + fp:                      # batarya: yuz yarim kaldi
                stats["partial"] += 1
                est[f] = float(R[f])
                heapq.heappush(obs, (t + tt, int(f), float(R[f])))
                break
            done[f] = True
        tt += math.hypot(x - dx, y - dy) / v
        pth.append((t + tt, dx, dy))
        if paths is not None:
            paths[j].append(pth)
        for f in route:
            reserved[f] = False
        state[j] = "flying"
        ret_time[j] = t + tt
        heapq.heappush(ev, (t + tt, 1, j))

    # baslangic plani: tum drone'lar depoda bekler
    end0 = plan(0.0, list(range(k)))
    stats["first_plan"] = end0
    for j in range(k):
        ready[j] = end0
        depart(j, end0)
    static = planner_name == "static"
    t_now = end0
    while ev:
        t_now, _, j = heapq.heappop(ev)
        if t_now > max_time:
            break
        state[j] = "depot"
        ready[j] = t_now + ts
        if not static and (not sched[j] or True):
            end = plan(t_now, [j])
            dep = max(t_now + ts, end)
            stats["wait_plan"] += max(0.0, end - (t_now + ts))
        else:
            dep = t_now + ts
        depart(j, dep)
        if not static:
            for i in range(k):                     # bosta bekleyenleri yeniden devreye al
                if state[i] == "idle":
                    if planner is None:
                        sched[i] = greedy_sortie(pool(), FX, FY, dx, dy, service_est(est, p), p)
                    if sched[i]:
                        state[i] = "depot"
                        depart(i, max(t_now, planner_free))
    stats["mission_s"] = float(max([t_now] + [ret_time[j] for j in range(k)]))
    stats["ripe_total"] = ripe_total
    stats["recall"] = stats["harvested"] / max(1, ripe_total)
    stats["ripe_missed"] = ripe_total - stats["harvested"]
    stats["faces"] = int(nf)
    stats["faces_visited"] = int(done.sum())
    if paths is not None:
        stats["paths"] = paths
        stats["cuts"] = cuts
    return stats
