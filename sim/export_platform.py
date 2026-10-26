# -*- coding: utf-8 -*-
"""Platform hasat simulasyonundan Blender verisi (v3, Tevel tipi senaryo).

JSON (metre, saniye; sahne sira yonu x eksenine hizali, agac merkezi orijinde):
  trees: x, y, cultivar (0 kirmizi, 1 sari, 2 yesil cesit), ripe[2], unripe[2]
         (yuz basina baslangic meyve sayisi; yuz 0 = +y yani ust aralik, 1 = -y)
  alleys: y, x0, x1 (sira araligi merkez cizgisi ve uclari)
  nodes: alley, x, y, trees_up (yuz 0'i bu araliga bakan agaclar), trees_lo (yuz 1)
  picks: [t_bitis, dugum, olgun, ham, platform] -- dugumde toplanan meyve (kasaya)
  platforms: platform basina sefer listesi; her sefer [[t, x, y, durum], ...]
             durum: 'move' | 'pick' (durakta drone'lar calisiyor) | 'station'
  station [x, y], k, drones_per_platform, bin_capacity, cultivars, summary
Kullanim:
  python sim/export_platform.py --orchard syn1ha --planner psb --out blender/anim3_syn1ha.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, P2)
sys.path.insert(0, os.path.join(P2, "experiments"))
import harvest_bench as HB   # noqa: E402
import platform_sim as S     # noqa: E402

CM = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orchard", default="syn1ha")
    ap.add_argument("--pass", dest="passname", default="pass1")
    ap.add_argument("--planner", default="psb")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    poly, phi = HB.geometry(a.orchard)
    p = HB.params(a.passname)
    o = S.build(poly, phi, None, seed=a.seed, p=p)
    k = HB.fleet(a.orchard)
    st = S.simulate(o, k, a.planner, seed=a.seed, p=p, log=True)
    F, XY = o["F"], o["node_xy"]
    u, w, c = o["u"], o["w"], o["c"]

    def tf(P):
        P = np.atleast_2d(np.asarray(P, float)) - c
        return np.column_stack([P @ u, P @ w]) * CM
    # dunya koordinatinda aralik uclari
    E0 = F.E0[:, :2]
    E1 = F.E1[:, :2]

    def node_xy(q):
        return XY[q] if q >= 0 else o["depot"]

    def path(i, j):
        """i -> j yolunun ara noktalari (dunya, cm)."""
        if i >= 0 and j >= 0 and F.A[i] == F.A[j]:
            return [node_xy(j)]
        best = None
        cand_i = [(None, node_xy(i))] if i < 0 else [(e, E[F.A[i]]) for e, E in ((0, E0), (1, E1))]
        cand_j = [(None, node_xy(j))] if j < 0 else [(e, E[F.A[j]]) for e, E in ((0, E0), (1, E1))]
        for ei, pi in cand_i:
            for ej, pj in cand_j:
                if ei is None:
                    hl = F.SE[2 * F.A[j] + ej]
                elif ej is None:
                    hl = F.SE[2 * F.A[i] + ei]
                else:
                    hl = F.EE[2 * F.A[i] + ei, 2 * F.A[j] + ej]
                d = np.linalg.norm(node_xy(i) - pi) + hl + np.linalg.norm(pj - node_xy(j))
                if best is None or d < best[0]:
                    best = (d, pi, pj)
        from mtsp.mtr import headland_path
        src = best[1] if i >= 0 else node_xy(i)
        dst = best[2] if j >= 0 else node_xy(j)
        pts = [best[1]] if i >= 0 else []
        pts += headland_path(F.E0, F.E1, F.dx, F.dy, src, dst)
        if j >= 0:
            pts.append(node_xy(j))
        return pts
    v = p["V_PLAT"]
    plats = []
    for trips in st["trace"]:
        tt = []
        for pth in trips:
            seq = []
            prev = -1
            prev_xy = o["depot"]
            for q, (t, node) in enumerate(pth):
                if q == 0:
                    seq.append([t, *tf(prev_xy)[0].round(3).tolist(), "station"])
                    continue
                if node == prev and node >= 0:            # durakta kalis bitti
                    seq.append([t, *tf(node_xy(node))[0].round(3).tolist(), "pick"])
                    continue
                # hareket: ara noktalar esit hizla zamanlanir
                pts = path(prev, node)
                t_end = t
                L = [np.linalg.norm(pts[0] - prev_xy)] + [np.linalg.norm(pts[m] - pts[m - 1]) for m in range(1, len(pts))]
                tot = sum(L) or 1.0
                t0 = seq[-1][0]
                acc = 0.0
                for m, pt in enumerate(pts):
                    acc += L[m]
                    seq.append([round(t0 + (t_end - t0) * acc / tot, 2), *tf(pt)[0].round(3).tolist(),
                                "move" if node < 0 or m < len(pts) - 1 else "pick"])
                prev, prev_xy = node, node_xy(node)
            tt.append(seq)
        plats.append(tt)
    T = tf(o["trees"])
    al = []
    for q in range(len(F.E0)):
        a0, a1 = tf(E0[q])[0], tf(E1[q])[0]
        al.append(dict(y=round(float(a0[1]), 3), x0=round(float(min(a0[0], a1[0])), 3), x1=round(float(max(a0[0], a1[0])), 3)))
    nodes = dict(alley=F.A.tolist(), x=tf(XY)[:, 0].round(3).tolist(), y=tf(XY)[:, 1].round(3).tolist(),
                 trees_up=[t[0] for t in o["node_trees"]], trees_lo=[t[1] for t in o["node_trees"]])
    data = dict(version=3, scenario="tethered-drone harvest platforms (fresh-market apples, fruit placed in bins)",
                trees=dict(x=T[:, 0].round(3).tolist(), y=T[:, 1].round(3).tolist(), cultivar=o["cult"].tolist(),
                           ripe=o["R_tree"].tolist(), unripe=o["U_tree"].tolist()),
                alleys=al, nodes=nodes, picks=st["picks"], platforms=plats, station=tf(o["depot"])[0].round(3).tolist(),
                k=k, drones_per_platform=p["M_DRONES"], bin_capacity=p["Q"], cultivars=[cc for cc, _ in p["CULTIVARS"]],
                params={kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in p.items()},
                summary={kk: st[kk] for kk in ("mission_s", "picked", "ripe_total", "recall", "unripe_picked", "trips",
                                                "plans", "plan_time")})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(data, open(a.out, "w"))
    print("yazildi", a.out, data["summary"], "k", k)


if __name__ == "__main__":
    main()
