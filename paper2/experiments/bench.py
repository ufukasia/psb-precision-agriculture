# -*- coding: utf-8 -*-
"""Faz 1: tam karsilastirma -- depolu min-max mTSP.

Yontemler (her biri AYRI is, TEK cekirdek; NUMBA_NUM_THREADS=1):
  psb       : PSB tek atis + LNS anytime (butce = cap); iz (t, makespan)
  lkh_tuned : LKH-3 MINMAX, INITIAL_TOUR_ALGORITHM=MTSP, MOVE_TYPE=2 (pilotta
              alti ayar arasinda en iyisi); soguk baslangic; iz
  lkh_warm  : lkh_tuned + PSB cozumunden sicak baslangic (INITIAL_TOUR_FILE);
              PSB suresi ize eklenir
  lkh_def   : LKH-3 MINMAX varsayilan ayarlar (RUNS=1); iz
Butce (cap): n < 1000 -> 120 s, n < 5000 -> 300 s, aksi 600 s.
Ornekler: mTSPLib (16), bahce (16 x k in {4,8,16}), VLSI (8 x k in {4,8,16}).
Tum makespan'ler cozum rotalarindan float Oklid ile yeniden hesaplanir.

Tekrarlar: --seeds 1,2,3 (LKH-3 SEED ve LNS rastgele tohumu); tohum 1 dosya
adinda sonek tasimaz, digerleri "__s<tohum>" ile biter. PSB tek atisi
deterministiktir; tohum yalniz LNS'i ve LKH-3'u etkiler.
Gercek bahceler (osm): OpenStreetMap bahce sinirlarindan uretilen 16 ornek.

Kosum:  python paper2/experiments/bench.py --workers 20
Cikti:  paper2/experiments/out/bench/<grup>__<ad>__k<k>__<yontem>[__s<tohum>].json (devam modu)
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse          # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np                    # noqa: E402

import instances as I                 # noqa: E402
from mtsp import mm, lkh3, sota       # noqa: E402

OUT = os.path.join(HERE, "out", "bench")
METHODS = ("psb", "lkh_tuned", "lkh_warm", "lkh_def")
SOTA = ("hsnr", "ma", "mils", "hga")      # literaturdeki min-max mTSP kodlari (mtsp/sota.py)
ORCH_SIZES = ((20, 25), (40, 50), (60, 80), (80, 125))
ORCH_SHAPES = ("rect", "L")
ORCH_SEEDS = (1, 2)
KS = (4, 8, 16)
VLSI_TARGETS = (1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000)


# Bellek tahmini (GB): olculen RSS'ten (n ~ 9500): MA 13.9, HSNR 2.4, HGA 2.9, MILS 1.2
MEM_COEF = {"ma": (1.6e-7, 0.2), "hsnr": (2.9e-8, 0.2), "hga": (3.3e-8, 0.8), "mils": (1.4e-8, 0.2)}


def mem_est(method, n):
    a, b = MEM_COEF.get(method, (0.0, 0.5))
    return a * n * n + b


def cap_for(n):
    return 120.0 if n < 1000 else (300.0 if n < 5000 else 600.0)


def inst_specs():
    import lit_bks
    out = [("lit", nm, m) for (nm, m) in sorted(lit_bks.LIT)]
    for nm in I.MTSPLIB:
        for k in I.MTSPLIB_K:
            out.append(("mtsplib", nm, k))
    for (r, c) in ORCH_SIZES:
        for sh in ORCH_SHAPES:
            for sd in ORCH_SEEDS:
                for k in KS:
                    out.append(("orchard", f"{r}x{c}_{sh}_{sd}", k))
    for o in I.osm_list():
        for k in KS:
            out.append(("osm", o["name"], k))
    byn = I.vlsi_list()
    seen = set()
    for tg in VLSI_TARGETS:
        n, nm = min(byn, key=lambda t: abs(t[0] - tg))
        if nm in seen:
            continue
        seen.add(nm)
        for k in KS:
            out.append(("vlsi", nm, k))
    return out


def load(group, name):
    if group in ("mtsplib", "lit"):
        return I.mtsplib(name)
    if group == "vlsi":
        return I.vlsi(name)
    if group == "osm":
        return I.osm(name)
    rc, sh, sd = name.split("_")
    r, c = (int(v) for v in rc.split("x"))
    return I.orchard(r, c, shape=sh, seed=int(sd))


def job_path(group, name, k, method, seed=1):
    sfx = "" if seed == 1 else f"__s{seed}"
    return os.path.join(OUT, f"{group}__{name}__k{k}__{method}{sfx}.json")


def run(job):
    group, name, k, method, seed = job
    path = job_path(group, name, k, method, seed)
    if os.path.exists(path):
        return path, "hazir"
    o = load(group, name)
    X, Y, dx, dy = o["X"], o["Y"], o["dx"], o["dy"]
    n = len(X)
    cap = cap_for(n)
    rec = dict(group=group, name=name, n=n, k=k, method=method, cap=cap, depot=[dx, dy], seed=seed)
    t0 = time.perf_counter()
    if method == "psb":
        r, info = mm.solve(X, Y, dx, dy, k, workers=1)
        tr = [(info["t_total"], info["mk"])]
        r2, mk = mm.improve(r, X, Y, dx, dy, cap - info["t_total"], trace=tr, t_offset=info["t_total"],
                            seed=seed)
        rec.update(psb=info, trace=tr, mk=mm.makespan(r2, X, Y, dx, dy), routes=[x.tolist() for x in r2])
    elif method in SOTA:
        res = sota.SOLVERS[method](X, Y, dx, dy, k, cap, seed=seed)
        mk = None
        routes = res["routes"]
        if routes is not None:
            routes = [np.asarray(r, np.int32) for r in routes]
            mm.check_solution(routes, n)
            mk = mm.makespan(routes, X, Y, dx, dy)
        rec.update(trace=[tuple(x) for x in res["trace"]], mk=mk, killed=res.get("killed"), wall=res.get("wall"),
                   startup=res.get("startup"), log=[str(x) for x in (res.get("log") or [])][-12:],
                   routes=[r.tolist() for r in routes] if routes is not None else None)
    else:
        extra = {} if method == "lkh_def" else lkh3.TUNED
        init = None
        t_init = 0.0
        if method == "lkh_warm":
            init, info = mm.solve(X, Y, dx, dy, k, workers=1)
            t_init = time.perf_counter() - t0
        res = lkh3.mtsp_minmax(X, Y, dx, dy, k, time_limit=max(1.0, cap - t_init), extra=extra,
                               init_routes=init, seed=seed)
        tr = [(t + t_init, c) for t, c in res["trace"]]
        if init is not None:
            tr = [(t_init, mm.makespan(init, X, Y, dx, dy))] + tr
        mk = None
        if res["routes"] is not None:
            mm.check_solution(res["routes"], n)
            mk = mm.makespan(res["routes"], X, Y, dx, dy)
        rec.update(trace=tr, mk=mk, killed=res["killed"], lkh_wall=res["wall"], log=res["log"][-12:],
                   routes=[x.tolist() for x in res["routes"]] if res["routes"] else None)
    rec["lb"], rec["mst"], rec["far"] = mm.lower_bound(X, Y, dx, dy, k)
    rec["elapsed"] = time.perf_counter() - t0
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return path, f"ok mk={rec['mk']:.1f} {rec['elapsed']:.0f}s" if rec["mk"] else "ok (cozum yok)"


def _safe(job):
    try:
        return run(job)
    except Exception:
        return str(job), "HATA " + traceback.format_exc()[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--groups", default="mtsplib,orchard,vlsi")
    ap.add_argument("--lit-methods", default="psb,lkh_tuned,lkh_warm")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--mem-gb", type=float, default=None,
                    help="es zamanli islerin tahmini bellek toplami siniri (GB); asan is atlanir")
    ap.add_argument("--plan", default=None,
                    help="grup:tohumlar:yontemler;... (orn. osm:1-5:psb+lkh_tuned+lkh_warm)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meths = a.methods.split(",")
    groups = a.groups.split(",")
    lit_m = a.lit_methods.split(",")
    specs = inst_specs()
    if a.plan:
        jobs = []
        for part in a.plan.split(";"):
            g, sd, ms = part.split(":")
            lo, hi = (int(v) for v in (sd.split("-") if "-" in sd else (sd, sd)))
            jobs += [(gg, nm, k, m, s) for gg, nm, k in specs if gg == g
                     for s in range(lo, hi + 1) for m in ms.split("+")]
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
        jobs = [(g, nm, k, m, s) for g, nm, k in specs if g in groups
                for m in (lit_m if g == "lit" else meths) for s in seeds]
    jobs = [j for j in jobs if not os.path.exists(job_path(*j))]
    # once dusuk tohumlar, her tohumda uzun isler once (bos cekirdek kalmasin)
    ncache = {}

    def nof(j):
        if j[:2] not in ncache:
            ncache[j[:2]] = len(load(j[0], j[1])["X"])
        return ncache[j[:2]]
    jobs.sort(key=lambda j: (j[4], -cap_for(nof(j))))
    # numba derlemesi (onbellek) ana surecte
    w = I.mtsplib("eil51")
    mm.solve(w["X"], w["Y"], w["dx"], w["dy"], 3)
    mm.improve(mm.solve(w["X"], w["Y"], w["dx"], w["dy"], 3)[0], w["X"], w["Y"], w["dx"], w["dy"], 0.2)
    logp = os.path.join(HERE, "out", "bench.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logp, "a") as f:
            f.write(line + "\n")
    log(f"basladi: {len(jobs)} is, {a.workers} isci, bellek siniri {a.mem_gb} GB")
    done = 0
    if a.mem_gb is None:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(_safe, j) for j in jobs]
            for f in as_completed(futs):
                p, st = f.result()
                done += 1
                log(f"{done}/{len(jobs)} {os.path.basename(p)} {st}")
        log("bitti")
        return
    from concurrent.futures import FIRST_COMPLETED, wait
    pending = list(jobs)
    running = {}
    with ProcessPoolExecutor(a.workers) as ex:
        while pending or running:
            used = sum(running.values())
            i = 0
            while i < len(pending) and len(running) < a.workers:
                j = pending[i]
                est = mem_est(j[3], nof(j))
                if est > a.mem_gb:
                    pending.pop(i)
                    rec = dict(group=j[0], name=j[1], n=nof(j), k=j[2], method=j[3], seed=j[4],
                               cap=cap_for(nof(j)), skipped=f"memory: estimated {est:.1f} GB > {a.mem_gb:g} GB",
                               trace=[], mk=None, routes=None)
                    with open(job_path(*j), "w") as fh:
                        json.dump(rec, fh)
                    done += 1
                    log(f"{done}/{len(jobs)} {os.path.basename(job_path(*j))} ATLANDI bellek {est:.1f} GB")
                    continue
                if used + est <= a.mem_gb:
                    running[ex.submit(_safe, j)] = est
                    used += est
                    pending.pop(i)
                    continue
                i += 1
            fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for f in fin:
                running.pop(f)
                p, st = f.result()
                done += 1
                log(f"{done}/{len(jobs)} {os.path.basename(p)} {st}")
    log("bitti")


if __name__ == "__main__":
    main()
