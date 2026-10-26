# -*- coding: utf-8 -*-
"""Simulasyon kaydini Blender icin zaman cizelgesine cevirir.

Her senaryo (planlayici) icin:
  - tur basina planlama araligi [t, t + t_plan] (drone'lar depoda)
  - drone yollari: (t, x, y) dugumleri (dogrusal enterpolasyon)
  - agac olaylari: ilk gozlenme ani (renk tahminden gercege), islenme ani
Cikti: JSON (birimler: metre, saniye; gorev saati).

Kullanim:
  python paper2/sim/anim_data.py --rows 40 --cols 50 --k 8 --lkh 60 --out paper2/blender/anim.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, P2)
sys.path.insert(0, os.path.join(P2, "experiments"))
import instances as I        # noqa: E402
import ripeness_sim as S     # noqa: E402

CM = 0.01


def timeline(sim, X, Y, dx, dy, speed, rho, r_true, tau, s_h=6.0, s_u=1.0):
    T = cKDTree(np.column_stack([X, Y]))
    t = 0.0
    drones = None
    obs_t = np.full(len(X), np.inf)
    done_t = np.full(len(X), np.inf)
    phases = []
    for rd in sim["rounds"]:
        phases.append(dict(kind="plan", t0=t, t1=t + rd["t_plan"], round=rd["round"], n=rd["n_targets"]))
        t += rd["t_plan"]
        routes = rd["routes"]
        if drones is None:
            drones = [[] for _ in range(len(routes))]
        while len(drones) < len(routes):
            drones.append([])
        for j, r in enumerate(routes):
            # yol: depo -> agac (s_h ya da s_u bekleme; olgunsa elmalar dusurulur) -> ... -> depo
            pts = [(dx, dy, None)] + [(X[i], Y[i], i) for i in r] + [(dx, dy, None)]
            tt = t
            path = [(tt, dx, dy)]
            for a, b in zip(pts[:-1], pts[1:]):
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                m = max(1, int(L / (0.5 * rho)))
                for s_ in range(1, m + 1):
                    x = a[0] + (b[0] - a[0]) * s_ / m
                    y = a[1] + (b[1] - a[1]) * s_ / m
                    ts = tt + L * s_ / m / speed
                    for h in T.query_ball_point((x, y), rho):
                        obs_t[h] = min(obs_t[h], ts)
                tt += L / speed
                path.append((tt, b[0], b[1]))
                if b[2] is not None:
                    dw = s_h if r_true[b[2]] >= tau else s_u
                    tt += dw
                    path.append((tt, b[0], b[1]))
                    if r_true[b[2]] >= tau:
                        done_t[b[2]] = min(done_t[b[2]], tt)
            drones[j].append(path)
        phases.append(dict(kind="fly", t0=t, t1=t + rd["flight_s"], round=rd["round"]))
        t += rd["flight_s"]
    return dict(t_end=t, phases=phases, drones=drones,
                obs_t=[None if not np.isfinite(v) else round(float(v), 3) for v in obs_t],
                done_t=[None if not np.isfinite(v) else round(float(v), 3) for v in done_t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--cols", type=int, default=50)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lkh", type=float, default=60.0)
    ap.add_argument("--lns", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=500.0)   # cm/s
    ap.add_argument("--rho", type=float, default=600.0)     # cm
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--s-h", type=float, default=6.0)
    ap.add_argument("--s-u", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(P2, "blender", "anim.json"))
    a = ap.parse_args()
    o = I.orchard(a.rows, a.cols)
    X, Y, dx, dy = o["X"], o["Y"], o["dx"], o["dy"]
    r = S.ripeness_field(X, Y, seed=a.seed)
    rh = S.prior_estimate(r, X, Y, seed=a.seed)
    # hizalama: sahayi sira yonune dondur (gorsel icin), merkezi orijine al
    th = -math.radians(o["phi"])
    cx, cy = X.mean(), Y.mean()
    def tf(x, y):
        x = np.asarray(x) - cx
        y = np.asarray(y) - cy
        return (x * math.cos(th) - y * math.sin(th)) * CM, (x * math.sin(th) + y * math.cos(th)) * CM
    scen = []
    for label, pl in ((f"PSB + LNS ({a.lns:g} s)", S.planner_psb(a.lns, workers=1)),
                      (f"LKH-3 ({a.lkh:g} s)", S.planner_lkh(a.lkh))):
        sim = S.simulate(X, Y, dx, dy, a.k, r, rh, pl, tau=a.tau, rho=a.rho, speed=a.speed,
                         s_h=a.s_h, s_u=a.s_u)
        tl = timeline(sim, X, Y, dx, dy, a.speed, a.rho, r, a.tau, a.s_h, a.s_u)
        # metreye/hizali cerceveye cevir
        dr = []
        for paths in tl["drones"]:
            dd = []
            for path in paths:
                ts, xs, ys = zip(*path)
                mx, my = tf(xs, ys)
                dd.append([[round(t, 3), round(float(x), 3), round(float(y), 3)] for t, x, y in zip(ts, mx, my)])
            dr.append(dd)
        tl["drones"] = dr
        scen.append(dict(label=label, summary=sim["summary"],
                         rounds=[{kk: v for kk, v in rd.items() if kk not in ("routes", "est_after")}
                                 for rd in sim["rounds"]], **tl))
        print(label, sim["summary"])
    tx, ty = tf(X, Y)
    ddx, ddy = tf([dx], [dy])
    data = dict(trees=dict(x=np.round(tx, 3).tolist(), y=np.round(ty, 3).tolist(),
                           r_true=np.round(r, 3).tolist(), r_hat=np.round(rh, 3).tolist()),
                depot=[float(ddx[0]), float(ddy[0])], tau=a.tau, k=a.k, speed_mps=a.speed * CM,
                s_h=a.s_h, s_u=a.s_u, rows=a.rows, cols=a.cols,
                rho_m=a.rho * CM, scenarios=scen)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(data, f)
    print("yazildi", a.out)


if __name__ == "__main__":
    main()
