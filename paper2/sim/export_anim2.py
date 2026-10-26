# -*- coding: utf-8 -*-
"""Gercekci hasat simulasyonundan Blender verisi (v2).

Cikti JSON (metre, saniye; sahne sira yonune hizali, merkez orijinde):
  trees: x, y, cultivar (0 kirmizi, 1 sari, 2 yesil), face_ripe[2], face_unripe[2]
  faces: tree, side (0/1), x, y
  cuts: [t, face, ripe_cut, unripe_cut, drone]
  drones: drone basina sorti listesi, her sorti [[t, x, y], ...]
  depot, k, params, summary
Kullanim:
  python paper2/sim/export_anim2.py --orchard syn20x25 --pass main --planner psb_mt --out paper2/blender/anim2_small.json
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
import instances as I        # noqa: E402
import harvest_sim as H      # noqa: E402
import harvest_bench as HB   # noqa: E402

CM = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orchard", default="syn20x25")
    ap.add_argument("--pass", dest="passname", default="main")
    ap.add_argument("--planner", default="psb_mt")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.orchard.startswith("syn"):
        r, c = (int(v) for v in a.orchard[3:].split("x"))
        o = I.orchard(r, c)
    else:
        o = I.osm(a.orchard)
    p = HB.params(a.passname)
    F = H.build(o["X"], o["Y"], o["phi"], o["dx"], o["dy"], seed=a.seed, p=p)
    k = a.k or max(2, int(np.ceil(np.sum(H.service_est(F["est"][F["est"] >= p["R_MIN"]], p)) / (HB.HOURS * 3600))))
    R0, U0 = F["R"].copy(), F["U"].copy()
    st = H.simulate(F, k, a.planner, seed=a.seed, p=p, log_paths=True)
    th = -math.radians(o["phi"])
    cx, cy = float(np.mean(o["X"])), float(np.mean(o["Y"]))

    def tf(x, y):
        x = np.asarray(x, float) - cx
        y = np.asarray(y, float) - cy
        return (x * math.cos(th) - y * math.sin(th)) * CM, (x * math.sin(th) + y * math.cos(th)) * CM
    n = F["n_trees"]
    tx, ty = tf(o["X"], o["Y"])
    fx, fy = tf(F["FX"], F["FY"])
    dr = []
    for sorties in st["paths"]:
        dd = []
        for pth in sorties:
            ts, xs, ys = zip(*pth)
            mx, my = tf(xs, ys)
            dd.append([[round(t, 2), round(float(x), 3), round(float(y), 3)] for t, x, y in zip(ts, mx, my)])
        dr.append(dd)
    dx, dy = tf([o["dx"]], [o["dy"]])
    data = dict(version=2, trees=dict(x=np.round(tx, 3).tolist(), y=np.round(ty, 3).tolist(),
                                      cultivar=F["cult"][:n].tolist(),
                                      face_ripe=np.column_stack([R0[:n], R0[n:]]).tolist(),
                                      face_unripe=np.column_stack([U0[:n], U0[n:]]).tolist()),
                faces=dict(tree=F["tree"].tolist(), side=([0] * n + [1] * n), x=np.round(fx, 3).tolist(),
                           y=np.round(fy, 3).tolist()),
                cuts=st["cuts"], drones=dr, depot=[float(dx[0]), float(dy[0])], k=k,
                cultivars=[c for c, _ in p["CULTIVARS"]], planner=a.planner, passname=a.passname,
                params={kk: vv for kk, vv in p.items()},
                summary={kk: st[kk] for kk in ("mission_s", "harvested", "ripe_total", "recall", "unripe_cut",
                                                "sorties", "plans", "plan_time")})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(data, open(a.out, "w"))
    print("yazildi", a.out, data["summary"], "k", k)


if __name__ == "__main__":
    main()
