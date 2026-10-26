# -*- coding: utf-8 -*-
"""Faz 3: olgunluk tabanli dinamik yeniden planlama -- gorev suresi deneyi.

Her (bahce, olgunluk tohumu, k, planlayici) icin ripeness_sim.simulate kosulur.
Hasat modeli ve gorev suresi: bkz. sim/ripeness_sim.py (hiz 5 m/s, olgun
agacta 6 s kesim, olgun cikmayan hedefte 1 s kontrol, gozlem yaricapi 6 m).
Planlayicilar tek cekirdekli surecte kosar (NUMBA_NUM_THREADS=1).

Kosum:  python experiments/sim_bench.py --workers 20
Cikti:  experiments/out/sim/<bahce>__r<tohum>__k<k>__<planlayici>.json
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse          # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, P2)
sys.path.insert(0, os.path.join(P2, "sim"))
import instances as I        # noqa: E402
import ripeness_sim as S     # noqa: E402
from mtsp import mmg as mm   # noqa: E402

OUT = os.path.join(HERE, "out", "sim")
# kucuk / orta / buyuk bahce (agac araligi 2 m, sira araligi 4 m)
ORCH = (("small", 20, 25, "rect"), ("medium", 40, 50, "rect"), ("large", 80, 125, "rect"))
RSEEDS = (1, 2, 3, 4, 5)
KS = (8, 16)
PLANNERS = ("psb", "psb_lns1", "psb_lns10", "lkh_tuned10", "lkh_tuned60", "lkh_warm10")


def make_planner(name):
    if name == "psb":
        return S.planner_psb(0.0, workers=1)
    if name.startswith("psb_lns"):
        return S.planner_psb(float(name[7:]), workers=1)
    if name.startswith("lkh_warm"):
        return S.planner_lkh(float(name[8:]), warm=True)
    return S.planner_lkh(float(name[9:]))


def run(job):
    tag, rows, cols, shape, rs, k, pname = job
    path = os.path.join(OUT, f"{tag}__r{rs}__k{k}__{pname}.json")
    if os.path.exists(path):
        return path, "hazir"
    o = I.orchard(rows, cols, shape=shape)
    X, Y, dx, dy = o["X"], o["Y"], o["dx"], o["dy"]
    r = S.ripeness_field(X, Y, seed=rs)
    rh = S.prior_estimate(r, X, Y, seed=rs)
    sim = S.simulate(X, Y, dx, dy, k, r, rh, make_planner(pname), log_routes=False)
    rec = dict(tag=tag, n=len(X), rseed=rs, k=k, planner=pname, summary=sim["summary"],
               rounds=sim["rounds"])
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)
    return path, f"ok mission(5m/s)={sim['summary']['mission_s']:.1f}s"


def _safe(job):
    try:
        return run(job)
    except Exception:
        return str(job), "HATA " + traceback.format_exc()[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(t, r, c, sh, rs, k, p) for t, r, c, sh in ORCH for rs in RSEEDS for k in KS for p in PLANNERS]
    jobs.sort(key=lambda j: ("lkh_tuned60" not in j[6], j[1] * j[2] < 5000))
    w = I.mtsplib("eil51")
    mm.improve(mm.solve(w["X"], w["Y"], w["dx"], w["dy"], 3)[0], w["X"], w["Y"], w["dx"], w["dy"], 0.2)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_safe, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            p, st = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {os.path.basename(p)} {st}", flush=True)


if __name__ == "__main__":
    main()
