# -*- coding: utf-8 -*-
"""Kare dikimli bahce hasat simulasyonundan Blender verisi (v4).

JSON (metre, saniye; sahne GERCEK bahce eksenine hizali: siralar x boyunca,
agac merkezi orijinde):
  trees: x, y, cultivar (0 kirmizi, 1 sari, 2 yesil tip), ripe, unripe (baslangic)
  region: agac basina k-d bolge (platform) numarasi (kdsteal/kd icin; yoksa -1)
  picks: [t_bitis, agac, olgun, ham, platform]
  platforms: platform basina sefer listesi; sefer [[t, x, y, durum], ...]
             durum 'station' | 'move' | 'pick'. Platform agacin capraz
             koridor kesisiminde (dort agacin arasi) durur; kesisimler arasi yol
             koridorlar boyunca L bicimlidir (once x, sonra y).
  spacing [au, aw], station [x, y], k, drones_per_platform, bin_capacity, summary
Kullanim:
  python sim/export_grid.py --orchard rect1ha --planner kdsteal --out blender/anim4_rect1ha.json
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
import grid_sim as S                 # noqa: E402
import harvest_grid_bench as HB      # noqa: E402

CM = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orchard", default="rect1ha")
    ap.add_argument("--pass", dest="passname", default="pass1")
    ap.add_argument("--planner", default="kdsteal")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = HB.params(a.passname)
    poly, phi = HB.geometry(a.orchard, a.seed)
    o = S.build(poly, phi, seed=a.seed, p=p)
    o1 = S.build(poly, phi, seed=a.seed, p=HB.params("pass1"))
    k = HB.fleet(o1, HB.params("pass1"))
    st = S.simulate(o, k, a.planner, seed=a.seed, p=p, log=True)
    Ft = o["Ftrue"]
    au, aw = o["geo"]["au"], o["geo"]["aw"]
    # gercek cercevede (U, V): Frame.theta mod 90 oldugundan dogrudan kullan
    U, V = Ft.U, Ft.V
    du, dv = au / 2.0, aw / 2.0

    def stop(q):
        if q < 0:
            return np.array([Ft.su0, Ft.sv0])
        return np.array([U[q] + du, V[q] + dv])
    plats = []
    for trips in st["trace"]:
        tt = []
        for pth in trips:
            seq = []
            prev = -1
            for qq, (t, node) in enumerate(pth):
                if qq == 0:
                    s0 = stop(-1)
                    seq.append([round(t, 2), round(s0[0] * CM, 3), round(s0[1] * CM, 3), "station"])
                    continue
                if node == prev and node >= 0:
                    sp = stop(node)
                    seq.append([round(t, 2), round(sp[0] * CM, 3), round(sp[1] * CM, 3), "pick"])
                    continue
                a0, b0 = stop(prev), stop(node)
                t0 = seq[-1][0]
                corner = np.array([b0[0], a0[1]])          # once x boyunca, sonra y boyunca
                L1, L2 = abs(corner[0] - a0[0]), abs(b0[1] - corner[1])
                tot = (L1 + L2) or 1.0
                tc = t0 + (t - t0) * L1 / tot
                seq.append([round(tc, 2), round(corner[0] * CM, 3), round(corner[1] * CM, 3), "move"])
                seq.append([round(t, 2), round(b0[0] * CM, 3), round(b0[1] * CM, 3),
                            "pick" if node >= 0 else "station"])
                prev = node
            tt.append(seq)
        plats.append(tt)
    region = [-1] * o["n_trees"]
    data = dict(version=4, scenario="square-grid orchard, tethered-drone harvest platforms, frame-aware k-d regions "
                                    "with end take-over",
                trees=dict(x=(U * CM).round(3).tolist(), y=(V * CM).round(3).tolist(), cultivar=o["cult"].tolist(),
                           ripe=o["R"].tolist(), unripe=o["U"].tolist()),
                region=region, picks=st["picks"], platforms=plats,
                spacing=[round(au * CM, 3), round(aw * CM, 3)],
                station=[round(Ft.su0 * CM, 3), round(Ft.sv0 * CM, 3)], k=k, drones_per_platform=p["M_DRONES"],
                bin_capacity=p["Q"], cultivars=[c for c, _ in p["CULTIVARS"]], planner=a.planner,
                summary={kk: st[kk] for kk in ("mission_s", "picked", "ripe_total", "recall", "unripe_picked", "trips",
                                                "steals", "plan_time", "theta_err")})
    # bolge sahipligi: her agaci ilk toplayan platform
    for t_, q, r_, u_, j in st["picks"]:
        if region[q] < 0:
            region[q] = j
    data["region"] = region
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(data, open(a.out, "w"))
    print("yazildi", a.out, data["summary"], "k", k)


if __name__ == "__main__":
    main()
