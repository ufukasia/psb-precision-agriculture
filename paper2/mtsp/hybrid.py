# -*- coding: utf-8 -*-
"""Hibrit hat: PSB bolumlemesi + rota basina paralel LKH-3 (TSP).

Bolumleme ve dengeleme mm.solve ile aynidir; yalniz 3. adimdaki rota
cilalayici LKH-3 TSP ile degistirilir. Her rota (depo + noktalari) ayri bir
LKH sureci olarak ESZAMANLI kosar (k cekirdek).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import lkh3
from . import mm


def _lkh_route(route, X, Y, dx, dy, time_limit, max_trials, seed):
    m = len(route)
    if m < 4:
        return route
    lx = np.concatenate([[dx], X[route]])
    ly = np.concatenate([[dy], Y[route]])
    t = lkh3.tsp(lx, ly, time_limit=time_limit, seed=seed, max_trials=max_trials)
    z = int(np.where(t == 0)[0][0])
    t = np.roll(t, -z)[1:] - 1
    new = route[t]
    # LKH tamsayi yuvarlamasiyla nadiren float'ta daha kotu olabilir: koru
    if mm.route_cost(new, X, Y, dx, dy) > mm.route_cost(route, X, Y, dx, dy):
        return route
    return new


def lkh_polisher(X, Y, dx, dy, workers=8, time_limit=10.0, max_trials=None, seed=1):
    pool = ThreadPoolExecutor(workers)

    def pf(routes, which=None):
        idx = list(range(len(routes))) if which is None else list(which)
        res = list(pool.map(lambda i: _lkh_route(routes[i], X, Y, dx, dy, time_limit, max_trials, seed), idx))
        out = list(routes)
        for i, r in zip(idx, res):
            out[i] = r
        return out
    return pf


def solve(X, Y, dx, dy, k, workers=8, time_limit=10.0, max_trials=None, rounds=5, trace=None):
    X = np.ascontiguousarray(X, float)
    Y = np.ascontiguousarray(Y, float)
    pf = lkh_polisher(X, Y, dx, dy, workers, time_limit, max_trials)
    return mm.solve(X, Y, dx, dy, k, workers=workers, rounds=rounds, polish_fn=pf, trace=trace)
