# -*- coding: utf-8 -*-
"""Google OR-Tools (9.11) ile dayanim kisitli sortiler.

Her sorti bir "arac": D'den cikar, D'ye doner; zaman boyutu kapasitesi E
(yol / v + hizmet suresi). Amac toplam sure (sorti sayisi kadar arac, kullanilmayan
araclar bos kalir). Ilk cozum PATH_CHEAPEST_ARC, iyilestirme GUIDED_LOCAL_SEARCH,
sure siniri saniye. Sure matrisi 0.1 s tamsayi birimde C++ tarafina verilir.
Sortiler daha sonra mt.assign ile drone'lara atanir (PSB-MT ile ayni atama).
OR-Tools proje icindeki third_party/pylibs dizininden yuklenir.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "pylibs")
if _LIB not in sys.path:
    sys.path.append(_LIB)

UNIT = 10.0          # 1 birim = 0.1 s


def sorties(X, Y, dx, dy, s, v, E, time_limit=10.0, vehicles=None):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    t0 = time.perf_counter()
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    s = np.asarray(s, float)
    n = len(X)
    if n == 0:
        return dict(sorties=[], wall=0.0, status="empty")
    PX = np.concatenate([[dx], X])
    PY = np.concatenate([[dy], Y])
    S = np.concatenate([[0.0], s])
    if vehicles is None:
        lb = (float(np.sum(s)) + 2.0 * float(np.mean(np.hypot(X - dx, Y - dy))) / v) / E
        vehicles = int(math.ceil(1.3 * lb)) + 2
    # gecis suresi i -> j = d(i,j)/v + s_i  (hizmet cikista sayilir)
    T = np.rint((np.hypot(PX[:, None] - PX[None, :], PY[:, None] - PY[None, :]) / v + S[:, None]) * UNIT)
    T = T.astype(np.int64)
    np.fill_diagonal(T, 0)
    mgr = pywrapcp.RoutingIndexManager(n + 1, vehicles, 0)
    rt = pywrapcp.RoutingModel(mgr)
    cb = rt.RegisterTransitMatrix(T.tolist())
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    rt.AddDimension(cb, 0, int(E * UNIT), True, "time")
    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    rem = max(0.5, time_limit - (time.perf_counter() - t0))
    prm.time_limit.FromMilliseconds(int(rem * 1000))
    sol = rt.SolveWithParameters(prm)
    if sol is None:
        return dict(sorties=None, wall=time.perf_counter() - t0, status="no solution")
    out = []
    for veh in range(vehicles):
        idx = rt.Start(veh)
        r = []
        while not rt.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node:
                r.append(node - 1)
            idx = sol.Value(rt.NextVar(idx))
        if r:
            out.append(np.array(r, np.int32))
    return dict(sorties=out, wall=time.perf_counter() - t0, status="ok")


def cvrp(M, svc, dem, Q, time_limit=10.0, vehicles=None):
    """Kapasiteli seferler, verilen sure matrisiyle (M: (n+1)x(n+1), 0 = depo,
    saniye; gecis i->j = M[i,j] + svc_i). Talepler tamsayiya yuvarlanir."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    t0 = time.perf_counter()
    n = M.shape[0] - 1
    if n == 0:
        return dict(sorties=[], wall=0.0, status="empty")
    S = np.concatenate([[0.0], np.asarray(svc, float)])
    # tek durak kasayi asamaz (yurutmede kasa dolunca durak yarim kalir)
    D = np.concatenate([[0], np.minimum(np.ceil(np.asarray(dem, float)), Q).astype(np.int64)])
    if vehicles is None:
        vehicles = int(math.ceil(1.3 * D.sum() / Q)) + 2
    T = np.rint((M + S[:, None]) * UNIT).astype(np.int64)
    np.fill_diagonal(T, 0)
    mgr = pywrapcp.RoutingIndexManager(n + 1, vehicles, 0)
    rt = pywrapcp.RoutingModel(mgr)
    cb = rt.RegisterTransitMatrix(T.tolist())
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    dcb = rt.RegisterUnaryTransitVector(D.tolist())
    rt.AddDimensionWithVehicleCapacity(dcb, 0, [int(Q)] * vehicles, True, "load")
    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    rem = max(0.5, time_limit - (time.perf_counter() - t0))
    prm.time_limit.FromMilliseconds(int(rem * 1000))
    sol = rt.SolveWithParameters(prm)
    if sol is None:
        return dict(sorties=None, wall=time.perf_counter() - t0, status="no solution")
    out = []
    for veh in range(vehicles):
        idx = rt.Start(veh)
        r = []
        while not rt.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node:
                r.append(node - 1)
            idx = sol.Value(rt.NextVar(idx))
        if r:
            out.append(np.array(r, np.int64))
    return dict(sorties=out, wall=time.perf_counter() - t0, status="ok")
