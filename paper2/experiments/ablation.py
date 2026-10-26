# -*- coding: utf-8 -*-
"""PSB ablasyonu ve kume-once / rota-sonra tabanlari (tek atis, tek cekirdek).

Varyantlar (hepsi deterministik):
  full        : PSB (GE + 2-opt/Or-opt dev tur, optimal bolme, cilalama, dengeleme)
  no_balance  : dengeleme yok (cilalamadan sonra durur)
  no_polish   : cilalama ve dengeleme yok (yalniz bolme)
  equal_split : optimal bolme yerine dev turu esit uzunlukta k parcaya bolme
                (en iyi baslangic noktasi), sonra cilalama + dengeleme
  ge_nols     : dev turda 2-opt/Or-opt yok (yalniz GE), gerisi ayni
  nn_tour     : dev tur en yakin komsu + 2-opt/Or-opt, gerisi ayni
  kmeans_tsp  : k-means (k kume) + kume basina depolu tur (GE + 2-opt/Or-opt)
  sweep_tsp   : depo etrafinda acisal esit sayili k dilim + kume basina tur
  kmeans_bal  : kmeans_tsp + PSB dengelemesi (adil karsilastirma)

Kosum:  python paper2/experiments/ablation.py --workers 20
Cikti:  paper2/experiments/out/extra/ablation.json
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse          # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import bench as B        # noqa: E402
from mtsp import mm      # noqa: E402

OUT = os.path.join(HERE, "out", "extra", "ablation.json")
VARIANTS = ("full", "no_balance", "no_polish", "equal_split", "ge_nols", "nn_tour",
            "kmeans_tsp", "sweep_tsp", "kmeans_bal")


def nn_tour(X, Y):
    from scipy.spatial import cKDTree
    n = len(X)
    P = np.column_stack([X, Y])
    tree = cKDTree(P)
    left = np.ones(n, bool)
    t = np.empty(n, np.int32)
    cur = 0
    left[0] = False
    t[0] = 0
    for s in range(1, n):
        kq = 8
        while True:
            _, idx = tree.query(P[cur], k=min(n, kq))
            idx = np.atleast_1d(idx)
            c = idx[left[idx]]
            if len(c):
                nxt = int(c[0])
                break
            if kq >= n:
                nxt = int(np.where(left)[0][0])
                break
            kq *= 4
        left[nxt] = False
        t[s] = nxt
        cur = nxt
    return t


def equal_split(g, X, Y, dx, dy, k):
    """Dev turu esit uzunlukta k ardisik parcaya bol; n baslangic arasinda
    en kucuk makespan'i veren secilir (O(n^2 / n) ornekleme: 64 baslangic)."""
    n = len(g)
    best = None
    for s in np.linspace(0, n - 1, min(n, 64)).astype(int):
        h = np.roll(g, -s)
        seg = np.hypot(np.diff(X[h]), np.diff(Y[h]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        L = cum[-1]
        cuts = np.searchsorted(cum, L * np.arange(1, k) / k)
        parts = [p for p in np.split(h, cuts) if len(p)]
        mk = mm.makespan(parts, X, Y, dx, dy)
        if best is None or mk < best[0]:
            best = (mk, parts)
    parts = [np.asarray(p, np.int32) for p in best[1]]
    while len(parts) < k:
        parts.append(np.zeros(0, np.int32))
    return parts


def route_tour(idx, X, Y, dx, dy):
    """Kume icin depolu tur: GE + 2-opt/Or-opt (yerel), depodan baslar."""
    idx = np.asarray(idx, np.int32)
    if len(idx) < 3:
        return idx
    lx = np.concatenate([[dx], X[idx]])
    ly = np.concatenate([[dy], Y[idx]])
    nb = mm.knn(lx, ly, 10)
    t = mm.giant_tour(lx, ly, nb)
    z = int(np.where(t == 0)[0][0])
    t = np.roll(t, -z)[1:] - 1
    return idx[t]


def polish_balance(routes, X, Y, dx, dy, nbr, polish=True, balance=True):
    if not polish:
        return routes
    routes = mm.polish_routes(routes, X, Y, dx, dy, 10, 50, 1)
    best, best_r = mm.makespan(routes, X, Y, dx, dy), routes
    if not balance:
        return routes
    for _ in range(20):
        routes, mv, touched = mm.balance_routes(routes, X, Y, dx, dy, nbr)
        if mv == 0:
            break
        routes = mm.polish_routes(routes, X, Y, dx, dy, 10, 50, 1, which=touched)
        mk = mm.makespan(routes, X, Y, dx, dy)
        if mk < best - 1e-9:
            best, best_r = mk, routes
        else:
            break
    return best_r


def variant(v, X, Y, dx, dy, k):
    X = np.ascontiguousarray(X, float)
    Y = np.ascontiguousarray(Y, float)
    nbr = mm.knn(X, Y, 10)
    if v in ("kmeans_tsp", "kmeans_bal", "sweep_tsp"):
        if v == "sweep_tsp":
            ang = np.arctan2(Y - dy, X - dx)
            o = np.argsort(ang)
            # dilim siniri: en buyuk acisal bosluktan baslat
            gaps = np.diff(np.concatenate([ang[o], [ang[o][0] + 2 * np.pi]]))
            s = (int(np.argmax(gaps)) + 1) % len(o)
            o = np.roll(o, -s)
            groups = np.array_split(o, k)
        else:
            from scipy.cluster.vq import kmeans2
            P = np.column_stack([X, Y])
            _, lab = kmeans2(P, k, minit="++", seed=1)
            groups = [np.where(lab == j)[0] for j in range(k)]
        routes = [route_tour(gp, X, Y, dx, dy) for gp in groups]
        routes = [np.asarray(r, np.int32) for r in routes]
        if v == "kmeans_bal":
            routes = polish_balance(routes, X, Y, dx, dy, nbr)
        return routes
    if v == "nn_tour":
        g = nn_tour(X, Y)
        mm.ls_tour(g, X, Y, nbr, 50, True)
    else:
        g = mm.giant_tour(X, Y, nbr, ls=(v != "ge_nols"))
    if v == "equal_split":
        routes = equal_split(g, X, Y, dx, dy, k)
    else:
        routes = mm.split_minmax(g, X, Y, dx, dy, k)
        while len(routes) < k:
            routes.append(np.zeros(0, np.int32))
    return polish_balance(routes, X, Y, dx, dy, nbr, polish=(v != "no_polish"),
                          balance=(v not in ("no_balance", "no_polish")))


def run(spec):
    group, name, k = spec
    o = B.load(group, name)
    X, Y, dx, dy = o["X"], o["Y"], o["dx"], o["dy"]
    out = dict(group=group, name=name, n=len(X), k=k)
    for v in VARIANTS:
        t0 = time.perf_counter()
        r = variant(v, X, Y, dx, dy, k)
        t = time.perf_counter() - t0
        mm.check_solution(r, len(X))
        out[v] = dict(mk=float(mm.makespan(r, X, Y, dx, dy)), t=t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    specs = [s for s in B.inst_specs()]
    w = B.load("mtsplib", "eil51")
    for v in VARIANTS:                      # numba derlemesi
        variant(v, w["X"], w["Y"], w["dx"], w["dy"], 3)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, specs))
    json.dump(res, open(OUT, "w"))
    print("yazildi", OUT, len(res))


if __name__ == "__main__":
    main()
