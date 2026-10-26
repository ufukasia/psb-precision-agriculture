# -*- coding: utf-8 -*-
"""Gercekci secmeli hasat deneyleri: kablolu drone tasiyan yer platformlari
(sim/platform_sim.py; parametreler ve kaynaklar: data/harvest_params.md).

Bahceler (Tall Spindle 0.9 m x 3.3 m): sentetik 1 ha blok (30 sira x 100 m) ve
OSM gercek sinirli uc bahce (~1, 2, 3.2 ha; Egirdir-Gelendost). Gecisler: ilk
toplama gecisi (meyvenin %40'i olgun) ve seyrek gecis (%15). Filo buyuklugu k:
ilk gecisin tahmini isi 6 saatte bitecek sekilde (bahce basina sabit, iki gecis
icin ayni). Tohumlar 1..5 (meyve, on tahmin, tespit rastlantisi).
Planlayicilar: PSB-MT/R (tek atis, +LNS 1 s), OR-Tools (1 s; 10 s yalniz <= 1 ha),
acgozlu, ekip pratigi (rowblock), statik PSB, PSB + yapay gecikme
L in {60, 300, 900} s.
Ayrica --sens: 1 ha sentetik blokta varsayim parametrelerinin duyarliligi.

Kosum:  python experiments/harvest_bench.py --workers 20 [--sens]
Cikti:  experiments/out/harvest/<bahce>__<gecis>__s<tohum>__<planlayici>.json
        experiments/out/harvest_sens/<ayar>__s<tohum>__<planlayici>.json
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
import platform_sim as S     # noqa: E402

OUT = os.path.join(HERE, "out", "harvest")
OUT_SENS = os.path.join(HERE, "out", "harvest_sens")
ORCH = (("syn1ha", None), ("osm1ha", "osm1463412060"), ("osm2ha", "osm1353989633"), ("osm3ha", "osm1348679699"))
PASSES = {"pass1": (("red", 0.148), ("yellow", -0.202), ("green", -0.602)),
          "sparse": (("red", -0.291), ("yellow", -0.641), ("green", -1.041))}
SEEDS = (1, 2, 3, 4, 5)
PLANNERS = ("psb", "psb_lns1", "ortools1", "ortools10", "greedy", "rowblock", "static",
            "delay60", "delay300", "delay900")
HOURS = 6.0
# duyarlilik (bir seferde bir parametre; araliklar harvest_params.md)
SENS = dict(T_PICK=(3.6, 7.3), M_DRONES=(2, 6), Q=(1500.0, 3000.0), SIG_PRIOR=(0.3, 0.9),
            V_PLAT=(50.0, 200.0), T_BIN=(60.0, 300.0), REC=(0.82, 0.97), MU_FRUIT=(100.0, 250.0))
SENS_PLANNERS = ("psb", "ortools1", "greedy", "rowblock", "static")
SENS_SEEDS = (1, 2, 3)


def geometry(tag):
    src = dict(ORCH)[tag]
    if src is None:
        return S.rect_block(30, 100), 0.0
    o = next(d for d in I.osm_list() if d["name"] == src)
    return np.array(o["poly"], float), float(o["phi"])


def params(passname, **over):
    p = dict(S.P)
    p["CULTIVARS"] = PASSES[passname]
    p.update(over)
    return p


def fleet(tag, p=None):
    poly, phi = geometry(tag)
    p = p or params("pass1")
    o = S.build(poly, phi, None, seed=1, p=p)
    work = float(np.sum(S.svc_est(o["est"][o["est"] >= p["R_MIN"]], p)))
    return max(2, int(math.ceil(work / (HOURS * 3600.0))))


def allowed(tag, pl):
    if pl == "ortools10":
        return tag in ("syn1ha", "osm1ha")
    return True


def _run(path, tag, poly, phi, p, seed, pl, k, extra):
    o = S.build(poly, phi, None, seed=seed, p=p)
    t0 = time.perf_counter()
    st = S.simulate(o, k, pl, seed=seed, p=p)
    st.update(tag=tag, seed=seed, planner=pl, k=k, n_trees=o["n_trees"], wall=time.perf_counter() - t0,
              fruit_total=int(o["R"].sum() + o["U"].sum()), targets=int((o["est"] >= p["R_MIN"]).sum()),
              params={kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in p.items()}, **extra)
    with open(path + ".tmp", "w") as f:
        json.dump(st, f)
    os.replace(path + ".tmp", path)
    return st


def run(job):
    kind = job[0]
    if kind == "main":
        _, tag, ps, seed, pl, k = job
        path = os.path.join(OUT, f"{tag}__{ps}__s{seed}__{pl}.json")
        if os.path.exists(path):
            return path, "hazir"
        poly, phi = geometry(tag)
        st = _run(path, tag, poly, phi, params(ps), seed, pl, k, dict(passname=ps))
    else:
        _, setting, prm, val, seed, pl, k = job
        path = os.path.join(OUT_SENS, f"{setting}__s{seed}__{pl}.json")
        if os.path.exists(path):
            return path, "hazir"
        poly, phi = geometry("syn1ha")
        over = {} if prm is None else {prm: val}
        st = _run(path, "syn1ha", poly, phi, params("pass1", **over), seed, pl, k,
                  dict(passname="pass1", setting=setting))
    return path, f"ok mission={st['mission_s'] / 3600:.2f} h picked={st['picked']} wall={st['wall']:.0f}s"


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
    jobs = []
    if a.sens:
        k0 = fleet("syn1ha")
        settings = [("base", None, None)] + [(f"{prm}={v:g}", prm, v) for prm, vals in SENS.items() for v in vals]
        for setting, prm, v in settings:
            for sd in SENS_SEEDS:
                for pl in SENS_PLANNERS:
                    jobs.append(("sens", setting, prm, v, sd, pl, k0))
    else:
        for tag, _ in ORCH:
            k = fleet(tag)
            print(tag, "k =", k, flush=True)
            for ps in PASSES:
                for sd in SEEDS:
                    for pl in PLANNERS:
                        if allowed(tag, pl):
                            jobs.append(("main", tag, ps, sd, pl, k))
    slow = {"ortools10": 0, "ortools1": 1, "psb_lns1": 2}
    jobs.sort(key=lambda j: (slow.get(j[-2], 9), j[1]))
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_safe, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            pth, st = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {os.path.basename(pth)} {st}", flush=True)


if __name__ == "__main__":
    main()
