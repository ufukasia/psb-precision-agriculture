# -*- coding: utf-8 -*-
"""Dayanim (batarya) kisitli, cok sortili min-max planlama: PSB-MT.

Problem: depo D, hedefler P (hizmet suresi s_i [s]), hiz v [birim/s], k drone;
her drone ardisik sortiler ucar, her sorti D'den baslar, D'de biter ve suresi
(yol / v + sum s_i) en fazla E [s] olur; sortiler arasinda batarya degisimi
t_swap [s]. Drone j t = a_j'de musait. Amac: son drone'un bitis zamani
(makespan) en kucuk.

PSB-MT:
  1. dev tur (GE + 2-opt/Or-opt; mm.giant_tour);
  2. dev turun, toplam sure (sorti suresi + t_swap) en kucuk olacak sekilde
     E-uygun ardisik sortilere optimal bolunmesi (dinamik programlama; sorti
     maliyeti uca gore monoton oldugundan her baslangic icin uygunluk bozulana
     dek ilerlenir; birkac dairesel baslangic denenir);
  3. sorti ici cilalama (depolu 2-opt/Or-opt);
  4. sortilerin drone'lara atanmasi: musaitlik zamanli LPT + en uzun drone'dan
     tasima/degistirme yerel aramasi;
  5. istege bagli LNS: dev tur pertuerbasyonu (cift kopru) + yeniden bolme.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from . import mm


@njit(cache=True, nogil=True)
def _split_dp(g, X, Y, dx, dy, s, v, E, t_swap):
    """Dogrusal sira g icin E-uygun sortilere toplam sureyi en kucukleyen bolme.
    Doner: (toplam, kesim noktalari dizisi). Tek basina E'yi asan hedef kendi
    sortisini alir (yurutmede kesilir)."""
    n = len(g)
    f = np.full(n + 1, np.inf)
    prv = np.full(n + 1, -1, np.int64)
    f[0] = 0.0
    for i in range(n):
        if not np.isfinite(f[i]):
            continue
        a = g[i]
        path = 0.0
        serv = 0.0
        first = math.hypot(X[a] - dx, Y[a] - dy)
        for j in range(i, n):
            b = g[j]
            if j > i:
                c = g[j - 1]
                path += math.hypot(X[b] - X[c], Y[b] - Y[c])
            serv += s[b]
            back = math.hypot(X[b] - dx, Y[b] - dy)
            dur = (first + path + back) / v + serv
            if dur > E and j > i:
                break
            tot = f[i] + dur + t_swap
            if tot < f[j + 1]:
                f[j + 1] = tot
                prv[j + 1] = i
            if dur > E:
                break
    cuts = []
    j = n
    while j > 0:
        cuts.append(j)
        j = prv[j]
    cuts.append(0)
    out = np.array(cuts[::-1], np.int64)
    return f[n], out


def sortie_time(r, X, Y, dx, dy, s, v):
    if len(r) == 0:
        return 0.0
    return mm.route_cost(np.asarray(r, np.int32), X, Y, dx, dy) / v + float(np.sum(s[r]))


def split_sorties(g, X, Y, dx, dy, s, v, E, t_swap, starts=8):
    n = len(g)
    best = None
    offs = np.unique(np.linspace(0, n - 1, min(n, starts)).astype(int)) if n else [0]
    for o in offs:
        h = np.roll(g, -int(o)).astype(np.int32)
        tot, cuts = _split_dp(h, X, Y, dx, dy, s, float(v), float(E), float(t_swap))
        if best is None or tot < best[0]:
            best = (tot, [h[cuts[q]:cuts[q + 1]] for q in range(len(cuts) - 1)])
    return best[1]


def polish_sorties(sorties, X, Y, dx, dy):
    return [mm._polish_one(np.asarray(r, np.int32), X, Y, dx, dy, 10, 50) if len(r) >= 3 else np.asarray(r, np.int32)
            for r in sorties]


def assign(durs, k, avail, t_swap, iters=2000):
    """Sortileri drone'lara ata (musaitlik a_j). Doner: (atama listesi, makespan)."""
    m = len(durs)
    order = np.argsort(-np.asarray(durs))
    load = np.array(avail, float).copy()
    own = [[] for _ in range(k)]
    for q in order:
        j = int(np.argmin(load))
        own[j].append(int(q))
        load[j] += durs[q] + t_swap
    # yerel arama: en yuklu drone'dan tasima / degistirme
    for _ in range(iters):
        jm = int(np.argmax(load))
        best = None
        for q in own[jm]:
            dq = durs[q] + t_swap
            for j in range(k):
                if j == jm:
                    continue
                nm = max(load[jm] - dq, load[j] + dq)
                if nm < load[jm] - 1e-9 and (best is None or nm < best[0]):
                    best = (nm, "mv", q, j, None)
                for p in own[j]:
                    dp = durs[p] + t_swap
                    if dp >= dq:
                        continue
                    nm2 = max(load[jm] - dq + dp, load[j] + dq - dp)
                    if nm2 < load[jm] - 1e-9 and (best is None or nm2 < best[0]):
                        best = (nm2, "sw", q, j, p)
        if best is None:
            break
        _, kind, q, j, p = best
        own[jm].remove(q)
        own[j].append(q)
        load[jm] -= durs[q] + t_swap
        load[j] += durs[q] + t_swap
        if kind == "sw":
            own[j].remove(p)
            own[jm].append(p)
            load[j] -= durs[p] + t_swap
            load[jm] += durs[p] + t_swap
    # son sortiden sonra batarya degisimi beklenmez
    mk = max((load[j] - (t_swap if own[j] else 0.0)) for j in range(k))
    return own, float(mk)


def plan_from_sorties(sorties, X, Y, dx, dy, s, v, k, avail, t_swap):
    sorties = [np.asarray(r, np.int32) for r in sorties if len(r)]
    durs = [sortie_time(r, X, Y, dx, dy, s, v) for r in sorties]
    own, mk = assign(durs, k, avail, t_swap)
    # drone basina sorti sirasi: depoya yakin olanlar once (bilgi erken toplanir)
    sched = []
    for j in range(k):
        sched.append(sorted([sorties[q] for q in own[j]],
                            key=lambda r: float(np.min(np.hypot(X[r] - dx, Y[r] - dy)))))
    return sched, mk


def solve(X, Y, dx, dy, s, v, E, k, avail=None, t_swap=60.0, budget=0.0, seed=1):
    """Doner: (cizelge: drone basina sorti listesi, makespan [s], bilgi)."""
    t0 = time.perf_counter()
    X = np.ascontiguousarray(X, float)
    Y = np.ascontiguousarray(Y, float)
    s = np.ascontiguousarray(s, float)
    avail = np.zeros(k) if avail is None else np.asarray(avail, float)
    n = len(X)
    if n == 0:
        return [[] for _ in range(k)], float(np.max(avail)), dict(t=0.0)
    nbr = mm.knn(X, Y, 10) if n > 1 else np.zeros((n, 0), np.int32)
    g = mm.giant_tour(X, Y, nbr) if n >= 3 else np.arange(n, dtype=np.int32)
    sorties = polish_sorties(split_sorties(g, X, Y, dx, dy, s, v, E, t_swap), X, Y, dx, dy)
    sched, mk = plan_from_sorties(sorties, X, Y, dx, dy, s, v, k, avail, t_swap)
    best = (mk, sched)
    rng = np.random.default_rng(seed)
    it = 0
    while budget > 0 and time.perf_counter() - t0 < budget and n >= 8:
        it += 1
        h = g.copy()
        a, b, c = np.sort(rng.choice(np.arange(1, n), 3, replace=False))
        h = np.concatenate([h[:a], h[b:c], h[a:b], h[c:]]).astype(np.int32)     # cift kopru
        mm.ls_tour(h, X, Y, nbr, 20, True)
        so = polish_sorties(split_sorties(h, X, Y, dx, dy, s, v, E, t_swap, starts=4), X, Y, dx, dy)
        sc, m2 = plan_from_sorties(so, X, Y, dx, dy, s, v, k, avail, t_swap)
        if m2 < best[0] - 1e-9:
            best = (m2, sc)
            g = h
    return best[1], best[0], dict(t=time.perf_counter() - t0, lns_iters=it, sorties=sum(len(x) for x in best[1]))
