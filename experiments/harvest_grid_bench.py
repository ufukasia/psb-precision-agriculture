# -*- coding: utf-8 -*-
"""Kare dikimli bahcelerde secmeli hasat deneyleri (sim/grid_sim.py).

Bahceler: kosesi kesilmis 1.3 ha dikdortgen ve OSM gercek sinirli dort bahce
(~1, 2, 3.2, 5.4 ha). HER TOHUM bahce geometrisini de degistirir: aralik,
sira egriligi, konum sapmasi, eksik agac yamalari, yonelim (dikdortgende
rastgele; OSM'de gercek yon +-3 derece), meyve yuku ve olgunluk.
Gecisler: ilk toplama (%40 olgun), seyrek (%15). Filo k: ilk gecisin tahmini isi
6 saatte bitecek sekilde (bahce ve tohum basina; iki gecis icin ayni).
Planlayicilar: kdsteal (k-d + uc devralma), kd, psb (kuresel), ortools1,
greedy, rowblock; cerceve etkisi icin kdsteal@oracle, kdsteal@axis0,
psb@axis0.
Kosum:  python experiments/harvest_grid_bench.py --workers 20 [--sens]
Cikti:  experiments/out/hgrid/<bahce>__<gecis>__s<tohum>__<planlayici>.json
        experiments/out/hgrid_sens/<ayar>__s<tohum>__<planlayici>.json
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse          # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, P2)
sys.path.insert(0, os.path.join(P2, "sim"))
import instances as I        # noqa: E402
import grid_sim as S         # noqa: E402

OUT = os.path.join(HERE, "out", "hgrid")
OUT_SENS = os.path.join(HERE, "out", "hgrid_sens")
ORCH = (("rect1ha", None), ("osm1ha", "osm1463412060"), ("osm2ha", "osm1353989633"),
        ("osm3ha", "osm1348679699"), ("osm5ha", "osm1348747146"))
PASSES = {"pass1": (("red", 0.148), ("yellow", -0.202), ("green", -0.602)),
          "sparse": (("red", -0.291), ("yellow", -0.641), ("green", -1.041))}
SEEDS = (1, 2, 3, 4, 5)
PLANNERS = ("kdsteal", "kd", "psb", "ortools1", "greedy", "rowblock",
            "kdsteal@oracle", "kdsteal@axis0", "psb@axis0")
HOURS = 6.0
SENS = dict(T_PICK=(3.6, 7.3), M_DRONES=(2, 6), Q=(1500.0, 3000.0), SIG_PRIOR=(0.3, 0.9),
            V_PLAT=(50.0, 200.0), T_BIN=(60.0, 300.0), REC=(0.82, 0.97), MU_FRUIT=(150.0, 450.0),
            SIGMA=(10.0, 60.0), A_U=(350.0, 500.0))
SENS_PLANNERS = ("kdsteal", "kd", "psb", "greedy", "rowblock")
SENS_SEEDS = (1, 2, 3)


def geometry(tag, seed):
    src = dict(ORCH)[tag]
    rng = np.random.default_rng(seed * 101 + 7)
    if src is None:
        return S.rect_with_notch(130, 100, seed), float(rng.uniform(0, 90))
    o = next(d for d in I.osm_list() if d["name"] == src)
    return np.array(o["poly"], float), float(o["phi"]) + float(rng.uniform(-3, 3))


def params(passname, **over):
    p = dict(S.P)
    p["CULTIVARS"] = PASSES[passname]
    p.update(over)
    return p


def fleet(o, p):
    work = float(np.sum(S.svc_est(o["est"][o["est"] >= p["R_MIN"]], p)))
    return max(2, int(math.ceil(work / (HOURS * 3600.0))))


def _run(path, tag, seed, pl, p, extra):
    poly, phi = geometry(tag, seed)
    o = S.build(poly, phi, seed=seed, p=p)
    o1 = S.build(poly, phi, seed=seed, p=params("pass1", **{kk: vv for kk, vv in p.items() if kk != "CULTIVARS"}))
    k = fleet(o1, o1 and params("pass1", **{kk: vv for kk, vv in p.items() if kk != "CULTIVARS"}))
    t0 = time.perf_counter()
    st = S.simulate(o, k, pl, seed=seed, p=p)
    st.update(tag=tag, seed=seed, planner=pl, k=k, n_trees=o["n_trees"], wall=time.perf_counter() - t0,
              phi=phi, theta_est=o["Fest"].theta, conf=o["Fest"].conf,
              fruit_total=int(o["R"].sum() + o["U"].sum()), targets=int((o["est"] >= p["R_MIN"]).sum()),
              params={kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in p.items()}, **extra)
    with open(path + ".tmp", "w") as f:
        json.dump(st, f)
    os.replace(path + ".tmp", path)
    return st


def run(job):
    if job[0] == "main":
        _, tag, ps, seed, pl = job
        path = os.path.join(OUT, f"{tag}__{ps}__s{seed}__{pl.replace('@', '-')}.json")
        if os.path.exists(path):
            return path, "hazir"
        st = _run(path, tag, seed, pl, params(ps), dict(passname=ps))
    else:
        _, setting, prm, val, seed, pl = job
        path = os.path.join(OUT_SENS, f"{setting}__s{seed}__{pl}.json")
        if os.path.exists(path):
            return path, "hazir"
        over = {} if prm is None else {prm: val}
        st = _run(path, "rect1ha", seed, pl, params("pass1", **over), dict(passname="pass1", setting=setting))
    return path, f"ok mission={st['mission_s'] / 3600:.2f} h k={st['k']} wall={st['wall']:.0f}s"


def _safe(job):
    try:
        return run(job)
    except Exception:
        return str(job), "HATA " + traceback.format_exc()[-1200:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--sens", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_SENS, exist_ok=True)
    if a.sens:
        sets = [("base", None, None)] + [(f"{prm}={v:g}", prm, v) for prm, vals in SENS.items() for v in vals]
        jobs = [("sens", s_, prm, v, sd, pl) for s_, prm, v in sets for sd in SENS_SEEDS for pl in SENS_PLANNERS]
    else:
        jobs = [("main", tag, ps, sd, pl) for tag, _ in ORCH for ps in PASSES for sd in SEEDS for pl in PLANNERS]
    jobs.sort(key=lambda j: ("ortools" not in j[-1], j[1]))
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_safe, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            pth, st = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {os.path.basename(pth)} {st}", flush=True)


if __name__ == "__main__":
    main()
