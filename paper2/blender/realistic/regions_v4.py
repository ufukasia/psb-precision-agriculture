#!/usr/bin/env python3
"""Reconstruct the planner's k-d regions for a v4 (square-grid) animation json.

`anim4_*.json` (paper2/sim/export_grid.py) stores `region[tree]` = platform that
harvested the tree first, which never differs from the picker of a pick event.
The frame-aware k-d partition that the fleet actually planned with is not
exported, so this helper rebuilds the same orchard (same seed / pass / planner
settings as export_grid.py) and calls the planner's partition
(mtsp.gridplan.KDSteal with the same weights as the first plan of grid_sim.simulate:
service estimate of every tree whose prior estimate is >= R_MIN, else 0).

Written next to the json as <name>_regions.json:
  kd_region  [tree]      planner's k-d region (platform index) per tree
  bbox       [platform]  [x0, y0, x1, y1] of that region's trees in the scene frame (m)
  takeover_picks         pick events [t_end, tree, ripe, unripe, platform, owner]
                         where the picking platform is not the k-d owner (= the
                         "end take-over" of the kdsteal planner)
  theta_est_deg          frame angle estimated by the planner (scene frame = 0)

  python3 regions_v4.py --data ../anim4_rect1ha.json [--orchard rect1ha --pass pass1 --seed 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(P2, "sim"))
sys.path.insert(0, P2)
sys.path.insert(0, os.path.join(P2, "experiments"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--orchard", default="rect1ha")
    ap.add_argument("--pass", dest="passname", default="pass1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import grid_sim as S                 # noqa: E402
    import harvest_grid_bench as HB      # noqa: E402
    from mtsp import gridplan as G       # noqa: E402

    d = json.load(open(a.data))
    p = HB.params(a.passname)
    poly, phi = HB.geometry(a.orchard, a.seed)
    o = S.build(poly, phi, seed=a.seed, p=p)
    k = int(d["k"])
    U, V = o["Ftrue"].U, o["Ftrue"].V
    x, y = np.array(d["trees"]["x"]), np.array(d["trees"]["y"])
    err = np.hypot(U * 0.01 - x, V * 0.01 - y).max()
    if len(x) != len(U) or err > 2e-3:
        raise SystemExit(f"orchard does not match the json (n {len(U)} vs {len(x)}, max position error {err:.3f} m)")
    n = len(U)
    est = o["est"]
    alive = est >= p["R_MIN"]
    w = np.where(alive, S.svc_est(est, p), 0.0)
    kd = G.KDSteal(o["Fest"], np.arange(n), w, k, steal=True)
    reg = np.full(n, -1, int)
    for j, s in enumerate(kd.region):
        reg[list(s)] = j
    bbox = []
    for j in range(k):
        m = reg == j
        bbox.append([float(x[m].min()), float(y[m].min()), float(x[m].max()), float(y[m].max())] if m.any() else None)
    take = [[t, q, r, u, j, int(reg[q])] for (t, q, r, u, j) in d["picks"] if reg[q] != j]
    out = a.out or os.path.splitext(a.data)[0] + "_regions.json"
    json.dump(dict(source=os.path.basename(a.data), kd_region=reg.tolist(), bbox=bbox, takeover_picks=take,
                   theta_est_deg=float(o["Fest"].theta), theta_true_deg=float(o["Ftrue"].theta),
                   region_sizes=[int((reg == j).sum()) for j in range(k)]), open(out, "w"))
    print("written", out)
    print("region sizes", [int((reg == j).sum()) for j in range(k)], "theta est", round(o["Fest"].theta, 3), "deg")
    print("take-over picks", len(take))
    for t in take:
        print("  t=%.0f tree %d picked by platform %d, k-d owner %d, at (%.1f, %.1f)" % (t[0], t[1], t[4], t[5], x[t[1]], y[t[1]]))


if __name__ == "__main__":
    main()
