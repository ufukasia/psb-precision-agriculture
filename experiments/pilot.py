# -*- coding: utf-8 -*-
"""Faz 0 pilot: PSB (bizim hat), hibrit (PSB + rota basina LKH-3) ve
monolitik LKH-3 MINMAX mTSP -- ayni ornek, ayni depo, float Oklid makespan.

Asama 1 (sirali, makine bos): PSB ve hibrit; sure olcumu.
Asama 2 (paralel, tek cekirdekli LKH surecleri): LKH-3 anytime izi.
Cikti: out/pilot/<ad>__k<k>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import instances as I                       # noqa: E402
from mtsp import mm, hybrid, lkh3           # noqa: E402

OUT = os.path.join(HERE, "out", "pilot")


def pilot_instances():
    out = [I.orchard(20, 25), I.orchard(40, 50), I.orchard(40, 50, shape="L"), I.orchard(80, 125)]
    byn = I.vlsi_list()
    for target in (1000, 3000, 10000):
        n, nm = min(byn, key=lambda t: abs(t[0] - target))
        out.append(I.vlsi(nm))
    return out


def ours(inst, k):
    X, Y, dx, dy = inst["X"], inst["Y"], inst["dx"], inst["dy"]
    tr = []
    r, info = mm.solve(X, Y, dx, dy, k, workers=k, trace=tr)
    tr2 = []
    r2, info2 = hybrid.solve(X, Y, dx, dy, k, workers=k, time_limit=max(2.0, len(X) / k / 100), trace=tr2)
    return dict(psb=info, psb_trace=tr, hyb=info2, hyb_trace=tr2,
                psb_routes=[x.tolist() for x in r], hyb_routes=[x.tolist() for x in r2])


def lkh_job(args):
    inst, k, cap = args
    X, Y, dx, dy = inst["X"], inst["Y"], inst["dx"], inst["dy"]
    res = lkh3.mtsp_minmax(X, Y, dx, dy, k, time_limit=cap, seed=1)
    mk = None
    if res["routes"] is not None:
        mm.check_solution(res["routes"], len(X))
        mk = mm.makespan(res["routes"], X, Y, dx, dy)
    return inst["name"], k, dict(mk=mk, trace=res["trace"], wall=res["wall"], killed=res["killed"],
                                 log=res["log"],
                                 routes=[x.tolist() for x in res["routes"]] if res["routes"] else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--cap", type=float, default=600)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    os.makedirs(OUT, exist_ok=True)
    insts = pilot_instances()
    # isinma (numba derleme)
    w = I.mtsplib("eil51")
    mm.solve(w["X"], w["Y"], w["dx"], w["dy"], 3)
    recs = {}
    for inst in insts:
        for k in ks:
            t0 = time.perf_counter()
            o = ours(inst, k)
            recs[(inst["name"], k)] = dict(name=inst["name"], group=inst["group"], n=len(inst["X"]), k=k,
                                           depot=[inst["dx"], inst["dy"]], **o)
            print(f"{inst['name']:32s} n={len(inst['X']):6d} k={k:2d} PSB={o['psb']['mk']:.1f} "
                  f"({o['psb']['t_total']:.2f}s) HYB={o['hyb']['mk']:.1f} ({o['hyb']['t_total']:.2f}s)",
                  flush=True)
    jobs = [(inst, k, a.cap) for inst in insts for k in ks]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(lkh_job, j) for j in jobs]
        for f in as_completed(futs):
            name, k, r = f.result()
            rec = recs[(name, k)]
            rec["lkh"] = r
            with open(os.path.join(OUT, f"{name}__k{k}.json"), "w") as fh:
                json.dump(rec, fh)
            first = r["trace"][0] if r["trace"] else None
            print(f"LKH {name:32s} k={k:2d} mk={r['mk']} wall={r['wall']:.0f}s first={first} "
                  f"PSB={rec['psb']['mk']:.1f} HYB={rec['hyb']['mk']:.1f}", flush=True)


if __name__ == "__main__":
    main()
