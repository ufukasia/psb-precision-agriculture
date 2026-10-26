#!/usr/bin/env python3
"""Data preparation for the explainer video (video_v4.py / compose_v4.py).

Rebuilds the planner's inputs exactly like regions_v4.py (same orchard, pass,
seed as export_grid.py) and writes out/video/video_prep.json with everything
the video needs that is not in the animation json itself:

  cuts            the k-d partition of mtsp.gridplan.kd_regions replayed cut by
                  cut: [{"axis": 0|1, "line": [[x0,y0],[x1,y1]] (scene frame, m),
                  "depth": d, "left": [tree ids], "right": [tree ids], "regions_done": [..]}]
                  (the recursion is re-implemented here with the same rules and
                  checked against <data>_regions.json)
  axis            theta_true_deg, theta_est_deg, theta_err_deg (summary.theta_err)
  grid            estimated-frame alley lines snapped on the tree rows (scene frame, m)
  params          T_PICK, P_SUCC, REC, PREC, T_STOP, M_DRONES, Q, T_BIN, V_PLAT (grid_sim.P)
  area_ha         planted block area = occupied + closed cells x cell area (and the
                  polygon area of the simulated field for reference)

  python3 video_prep_v4.py --data ../anim4_rect1ha.json [--orchard rect1ha --pass pass1 --seed 1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(P2, "sim"))
sys.path.insert(0, P2)
sys.path.insert(0, os.path.join(P2, "experiments"))


def kd_cuts(F, idx, w, k):
    """mtsp.gridplan.kd_regions with the cuts recorded (same arithmetic)."""
    idx = np.asarray(idx, np.int64)
    w = np.asarray(w, float)
    regs, cuts = [], []

    def rec(ids, ww, j, depth):
        if j <= 1 or len(ids) <= 1:
            regs.append(ids)
            return
        j1 = j // 2
        U, V = F.U[ids], F.V[ids]
        ax = 0 if np.ptp(U) >= np.ptp(V) else 1
        key, sp = (U, F.du) if ax == 0 else (V, F.dv)
        lines = np.round((key - key.min()) / sp).astype(int)
        order = np.argsort(key, kind="stable")
        cw = np.cumsum(ww[order]) / max(ww.sum(), 1e-12)
        q = int(np.searchsorted(cw, j1 / j))
        q = min(max(q, 0), len(order) - 1)
        cut_line = lines[order[q]]
        left = lines <= cut_line
        alt = lines < cut_line
        tgt = j1 / j * ww.sum()
        if abs(ww[alt].sum() - tgt) < abs(ww[left].sum() - tgt) and alt.any():
            left = alt
            cut_line -= 1
        if left.all() or not left.any():
            left = np.zeros(len(ids), bool)
            left[order[:max(1, q)]] = True
            cut_line = None
        # cut position in the estimated frame: between line cut_line and cut_line + 1
        if cut_line is not None:
            pos = key.min() + (cut_line + 0.5) * sp
        else:
            pos = 0.5 * (key[left].max() + key[~left].min())
        other = V if ax == 0 else U
        osp = F.dv if ax == 0 else F.du
        lo, hi = other.min() - 0.5 * osp, other.max() + 0.5 * osp
        cuts.append(dict(axis=ax, pos=float(pos), lo=float(lo), hi=float(hi), depth=depth,
                         left=ids[left].tolist(), right=ids[~left].tolist(), j_left=j1, j_right=j - j1))
        rec(ids[left], ww[left], j1, depth + 1)
        rec(ids[~left], ww[~left], j - j1, depth + 1)
    rec(idx, w, k, 0)
    return regs, cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--orchard", default="rect1ha")
    ap.add_argument("--pass", dest="passname", default="pass1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "video", "video_prep.json"))
    a = ap.parse_args()
    import grid_sim as S                 # noqa: E402
    import harvest_grid_bench as HB      # noqa: E402

    d = json.load(open(a.data))
    reg_path = os.path.splitext(a.data)[0] + "_regions.json"
    REG = json.load(open(reg_path))
    p = HB.params(a.passname)
    poly, phi = HB.geometry(a.orchard, a.seed)
    o = S.build(poly, phi, seed=a.seed, p=p)
    k = int(d["k"])
    Ft, Fe = o["Ftrue"], o["Fest"]
    x, y = np.array(d["trees"]["x"]), np.array(d["trees"]["y"])
    err = np.hypot(Ft.U * 0.01 - x, Ft.V * 0.01 - y).max()
    if len(x) != len(Ft.U) or err > 2e-3:
        raise SystemExit(f"orchard does not match the json (max position error {err:.3f} m)")
    n = len(x)
    est = o["est"]
    alive = est >= p["R_MIN"]
    w = np.where(alive, S.svc_est(est, p), 0.0)
    regs, cuts = kd_cuts(Fe, np.arange(n), w, k)
    reg = np.full(n, -1, int)
    for j, s in enumerate(regs):
        reg[list(s)] = j
    if reg.tolist() != REG["kd_region"]:
        raise SystemExit("replayed k-d partition differs from the regions json")
    # estimated frame -> scene frame (true frame, metres): both frames share the centre c
    dth = math.radians(Fe.theta - Ft.theta)
    ca, sa = math.cos(dth), math.sin(dth)

    def est_to_scene(u, v):
        return [round((ca * u - sa * v) * 0.01, 3), round((sa * u + ca * v) * 0.01, 3)]

    out_cuts = []
    done_before = 0
    for c in cuts:
        if c["axis"] == 0:
            p0, p1 = est_to_scene(c["pos"], c["lo"]), est_to_scene(c["pos"], c["hi"])
        else:
            p0, p1 = est_to_scene(c["lo"], c["pos"]), est_to_scene(c["hi"], c["pos"])
        out_cuts.append(dict(axis=c["axis"], line=[p0, p1], depth=c["depth"], left=c["left"], right=c["right"],
                             j_left=c["j_left"], j_right=c["j_right"]))
    # alley centre lines of the estimated frame (every line between two rows, both directions)
    grid = []
    for key, other, sp, osp, ax in ((Fe.U, Fe.V, Fe.du, Fe.dv, 0), (Fe.V, Fe.U, Fe.dv, Fe.du, 1)):
        lines = np.round((key - key.min()) / sp).astype(int)
        for L in range(lines.min(), lines.max() + 2):
            pos = key.min() + (L - 0.5) * sp
            m = (lines == L) | (lines == L - 1)
            if not m.any():
                continue
            lo, hi = other[m].min() - 0.5 * osp, other[m].max() + 0.5 * osp
            grid.append([est_to_scene(pos, lo), est_to_scene(pos, hi)] if ax == 0 else
                        [est_to_scene(lo, pos), est_to_scene(hi, pos)])
    # planted area: occupied cells + holes of up to 3 cells (same rule as build_scene_v4.INSIDE)
    au, aw = d["spacing"]
    ix = np.rint((x - x.min()) / au).astype(int)
    iy = np.rint((y - y.min()) / aw).astype(int)
    occ = np.zeros((iy.max() + 1, ix.max() + 1), bool)
    occ[iy, ix] = True
    ins = occ.copy()
    for axis in (0, 1):
        oo = occ if axis == 1 else occ.T
        nn = oo.shape[1]
        left = np.zeros_like(oo)
        right = np.zeros_like(oo)
        for s in range(1, 5):
            left[:, s:] |= oo[:, :nn - s]
            right[:, :nn - s] |= oo[:, s:]
        m = left & right
        ins |= m if axis == 1 else m.T
    cell_area = au * aw
    area_block = float(ins.sum() * cell_area)
    px, py = poly[:, 0] * 0.01, poly[:, 1] * 0.01
    area_poly = float(0.5 * abs(np.dot(px, np.roll(py, 1)) - np.dot(py, np.roll(px, 1))))
    params = {kk: p[kk] for kk in ("T_PICK", "P_SUCC", "REC", "PREC", "T_STOP", "M_DRONES", "Q", "T_BIN", "V_PLAT", "R_MIN")}
    out = dict(source=os.path.basename(a.data), cuts=out_cuts, grid=grid,
               axis=dict(theta_true_deg=float(Ft.theta), theta_est_deg=float(Fe.theta),
                         theta_err_deg=float(d["summary"]["theta_err"])),
               params=params, area_ha=round(area_block / 1e4, 3), area_polygon_ha=round(area_poly / 1e4, 3),
               cells_inside=int(ins.sum()), cells_occupied=int(occ.sum()),
               region_sizes=REG["region_sizes"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("written", a.out)
    print(f"{len(out_cuts)} cuts, {len(grid)} grid lines, area {out['area_ha']} ha (block cells), "
          f"{out['area_polygon_ha']} ha (field polygon), axis error {out['axis']['theta_err_deg']:.3f} deg")
    for c in out_cuts:
        print(f"  depth {c['depth']} axis {'x' if c['axis'] == 0 else 'y'} line {c['line']} -> {len(c['left'])} | {len(c['right'])} trees "
              f"(k {c['j_left']} | {c['j_right']})")


if __name__ == "__main__":
    main()
