# -*- coding: utf-8 -*-
"""Olgunluk tabanli dinamik yeniden planlama -- ayrik olay simulasyonu.

Senaryo: k drone, tek depo (bas tarla). Her agacin GERCEK olgunlugu r_i
(uzamsal olarak iliskili alan + agac gurultusu) bilinmez; uzaktan algilama
on tahmini r^_i gurultuludur. Hedef: r_i >= tau olan (olgun) agaclari
ziyaret etmek (hasat / hedefli islem).

Tur (sorti) dongusu:
  1. Planlayici, "olgun sanilan ve henuz islenmemis" agaclar kumesi icin
     depolu min-max rota uretir. Drone'lar depoda bekler: planlama suresi
     gorev suresine DOGRUDAN eklenir.
  2. Drone'lar rotalari ucar. Ucus yolunun `rho` yaricapindaki her agac
     gozlenir: tahmin gercek degerle degisir (kamera).
  3. Ziyaret edilen agac: gercekten olgunsa islenir; degilse bosuna ziyaret.
  4. Depoya donus; yeni bilgiyle 1'e. Olgun sanilan islenmemis agac
     kalmayinca gorev biter.
Hasat: drone agaca varir, olgun elmalara dokunup saplarini keser ve elmalar
yere (toplama agina) duser: olgun agacta s_h saniye; agac olgun cikmazsa
yalniz kisa kontrol, s_u saniye. Planlayici hangisinin olgun oldugunu
bilmediginden tum hedeflere ayni hizmet suresi s_h'yi varsayar: rota maliyeti
= yol + w |rota|, w = s_h v (mmg.Metric(w), LKH-3'te acik matris).
Gercek ucus suresi rota basina yol / v + s_h (#olgun) + s_u (#olgun degil).
Gorev suresi = sum_tur (t_plan + max_j ucus_j).

Planlayici arabirimi: plan(X, Y, dx, dy, k, w) -> (rotalar, t_plan_s).
Kayit (log) Blender canlandirmasi icin JSON'a yazilir.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path.insert(0, P2)
from mtsp import mmg as mm  # noqa: E402


def ripeness_field(X, Y, seed=0, n_bumps=12, scale=None, noise=0.08):
    """Gercek olgunluk r in [0,1]: gunesli yamaclar / cesit bloklari gibi
    yumusak Gauss tumsekleri + agac basina gurultu."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)
    span = max(np.ptp(X), np.ptp(Y))
    scale = scale or 0.18 * span
    f = np.zeros(len(X))
    for _ in range(n_bumps):
        cx = rng.uniform(X.min(), X.max())
        cy = rng.uniform(Y.min(), Y.max())
        s = scale * rng.uniform(0.6, 1.4)
        a = rng.uniform(-1.0, 1.0)
        f += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    f = (f - f.mean()) / (f.std() + 1e-12)
    r = 1.0 / (1.0 + np.exp(-1.6 * f)) + rng.normal(0, noise, len(X))
    return np.clip(r, 0.0, 1.0)


def prior_estimate(r, X, Y, seed=0, sigma=0.18, smooth=None):
    """Uzaktan algilama on tahmini: gercek alanin bulaniklastirilmis hali +
    gurultu (cozunurluk kaybi ve olcum hatasi)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed + 1)
    P = np.column_stack([X, Y])
    span = max(np.ptp(X), np.ptp(Y))
    smooth = smooth or 0.04 * span
    tree = cKDTree(P)
    nb = tree.query_ball_point(P, smooth)
    rs = np.array([r[i].mean() for i in nb])
    return np.clip(rs + rng.normal(0, sigma, len(r)), 0, 1)


def _observed(route_xy, P_tree, rho):
    """Polilinin rho yaricapindaki agaclar (cKDTree + segment ornekleme)."""
    pts = []
    for (x0, y0), (x1, y1) in zip(route_xy[:-1], route_xy[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        m = max(1, int(L / (0.5 * rho)))
        t = np.linspace(0, 1, m + 1)
        pts.append(np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]))
    S = np.vstack(pts)
    hit = P_tree.query_ball_point(S, rho)
    out = set()
    for h in hit:
        out.update(h)
    return out


def simulate(X, Y, dx, dy, k, r_true, r_hat, planner, tau=0.7, rho=600.0, speed=500.0,
             s_h=6.0, s_u=1.0, max_rounds=20, log_routes=True):
    """Birimler: koordinat cm, hiz cm/s (500 = 5 m/s), rho cm.
    Doner: dict(ozet, turlar=[...])."""
    from scipy.spatial import cKDTree
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    est = np.array(r_hat, float).copy()
    known = np.zeros(n, bool)
    done = np.zeros(n, bool)          # islendi (olgun + ziyaret)
    visited_unripe = np.zeros(n, bool)
    ripe = r_true >= tau
    T = cKDTree(np.column_stack([X, Y]))
    rounds = []
    mission = 0.0
    for rd in range(max_rounds):
        tgt = np.where((est >= tau) & ~done & ~visited_unripe)[0]
        if len(tgt) == 0:
            break
        kk = int(min(k, len(tgt)))
        routes_loc, t_plan = planner(X[tgt], Y[tgt], dx, dy, kk, s_h * speed)
        routes = [tgt[np.asarray(r, int)] for r in routes_loc if len(r)]
        mk = mm.makespan(routes, X, Y, dx, dy)                  # yalniz yol
        per = [mm.route_cost(r, X, Y, dx, dy) / speed + s_h * int(ripe[r].sum()) + s_u * int((~ripe[r]).sum())
               for r in routes]
        flight = max(per)
        newly_seen = set()
        for r in routes:
            xy = [(dx, dy)] + [(X[i], Y[i]) for i in r] + [(dx, dy)]
            newly_seen |= _observed(xy, T, rho)
        seen_idx = np.fromiter(newly_seen, int) if newly_seen else np.zeros(0, int)
        new_info = seen_idx[~known[seen_idx]]
        known[seen_idx] = True
        est[seen_idx] = r_true[seen_idx]
        vis = np.concatenate(routes)
        hv = vis[ripe[vis]]
        wv = vis[~ripe[vis]]
        done[hv] = True
        visited_unripe[wv] = True
        rounds.append(dict(round=rd, n_targets=int(len(tgt)), t_plan=float(t_plan), makespan=float(mk),
                           flight_s=float(flight), harvested=int(len(hv)), wasted=int(len(wv)),
                           newly_observed=int(len(new_info)),
                           routes=[r.tolist() for r in routes] if log_routes else None,
                           est_after=est.round(3).tolist() if log_routes else None))
        mission += t_plan + flight
    summary = dict(mission_s=float(mission), rounds=len(rounds),
                   plan_s=float(sum(r["t_plan"] for r in rounds)),
                   flight_s=float(sum(r["flight_s"] for r in rounds)),
                   harvested=int(done.sum()), ripe_total=int(ripe.sum()),
                   recall=float(done.sum() / max(1, ripe.sum())),
                   wasted=int(visited_unripe.sum()))
    return dict(summary=summary, rounds=rounds)


# ---------------------------------------------------------------------------
#  Planlayicilar
# ---------------------------------------------------------------------------
def planner_psb(lns_budget=0.0, workers=8):
    def plan(X, Y, dx, dy, k, w):
        t0 = time.perf_counter()
        M = mm.Metric(len(X), w=w)
        r, _ = mm.solve(X, Y, dx, dy, k, workers=workers, M=M)
        if lns_budget > 0:
            r, _ = mm.improve(r, X, Y, dx, dy, lns_budget, M=M)
        return r, time.perf_counter() - t0
    return plan


def planner_lkh(time_limit, warm=False):
    """LKH-3 ayarli (MTSP baslangic turu, MOVE_TYPE 2), dugum agirlikli acik
    matris; warm=True: PSB cozumunden baslar (sure dahil)."""
    from mtsp import lkh3

    def plan(X, Y, dx, dy, k, w):
        t0 = time.perf_counter()
        init = None
        if warm:
            init, _ = mm.solve(X, Y, dx, dy, k, workers=1, M=mm.Metric(len(X), w=w))
        res = lkh3.mtsp_minmax_weighted(X, Y, dx, dy, k, w, time_limit=time_limit,
                                        extra=lkh3.TUNED, init_routes=init)
        routes = res["routes"]
        if routes is None:                       # LKH butcede cozum yazamadi: gecersiz plan
            raise RuntimeError("LKH-3 cozum uretmedi")
        if init is not None:
            M = mm.Metric(len(X), w=w)
            if mm.makespan(init, X, Y, dx, dy, M) < mm.makespan(routes, X, Y, dx, dy, M):
                routes = init
        return routes, time.perf_counter() - t0
    return plan
