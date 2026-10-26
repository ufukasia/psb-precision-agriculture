# -*- coding: utf-8 -*-
"""Hasat simulasyonu duyarlilik analizi (bir seferde bir parametre).

Taban: orta bahce (40 x 50), k = 8, olgunluk tohumu 1..5, s_h = 6 s,
hiz 5 m/s, tau = 0.7, rho = 6 m. Her parametre iki ek degerle degistirilir.
Planlayicilar tek cekirdekli surecte kosar.

Kosum:  python paper2/experiments/sim_sens.py --workers 20
Cikti:  paper2/experiments/out/sim_sens/<ayar>__r<tohum>__<planlayici>.json
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
import sim_bench as SB       # noqa: E402

OUT = os.path.join(HERE, "out", "sim_sens")
BASE = dict(s_h=6.0, speed=500.0, tau=0.7, rho=600.0)
VARY = dict(s_h=(3.0, 12.0), speed=(300.0, 800.0), tau=(0.6, 0.8), rho=(300.0, 1000.0))
PLANNERS = ("psb", "psb_lns1", "lkh_warm10", "lkh_tuned10", "lkh_tuned60")
RSEEDS = (1, 2, 3, 4, 5)
K = 8


def settings():
    out = [("base", dict(BASE))]
    for p, vals in VARY.items():
        for v in vals:
            d = dict(BASE)
            d[p] = v
            out.append((f"{p}={v:g}", d))
    return out


def run(job):
    tag, prm, rs, pname = job
    path = os.path.join(OUT, f"{tag}__r{rs}__{pname}.json")
    if os.path.exists(path):
        return path, "hazir"
    o = I.orchard(40, 50)
    X, Y, dx, dy = o["X"], o["Y"], o["dx"], o["dy"]
    r = S.ripeness_field(X, Y, seed=rs)
    rh = S.prior_estimate(r, X, Y, seed=rs)
    sim = S.simulate(X, Y, dx, dy, K, r, rh, SB.make_planner(pname), tau=prm["tau"], rho=prm["rho"],
                     speed=prm["speed"], s_h=prm["s_h"], log_routes=False)
    rec = dict(setting=tag, params=prm, rseed=rs, k=K, planner=pname, n=len(X), summary=sim["summary"],
               rounds=sim["rounds"])
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)
    return path, f"ok mission={sim['summary']['mission_s']:.1f}s"


def _safe(job):
    try:
        return run(job)
    except Exception:
        return str(job[:1] + job[2:]), "HATA " + traceback.format_exc()[-600:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(t, p, rs, pl) for t, p in settings() for rs in RSEEDS for pl in PLANNERS]
    jobs.sort(key=lambda j: "lkh_tuned60" not in j[3])
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_safe, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            p, st = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {os.path.basename(p)} {st}", flush=True)


if __name__ == "__main__":
    main()
