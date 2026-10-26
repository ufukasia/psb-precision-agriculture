# -*- coding: utf-8 -*-
"""LKH-3 (3.0.13, kaynaktan derlenmis Linux ikilisi) sarmalayicisi.

mtsp_minmax : monolitik depolu min-max mTSP (MTSP_OBJECTIVE = MINMAX).
              Iz (TRACE_LEVEL = 1) satirlari duvar saatiyle damgalanir; tek
              kosumdan "anytime" egrisi (t, makespan) okunur.
tsp         : tek rota (depo dahil) TSP -- hibrit hatta rota basina cilalama.

LKH tamsayi mesafe kullandigi icin koordinatlar SCALE ile carpilip EUC_2D
yazilir (yuvarlama hatasi kenar basina <= 0.5/SCALE). Raporlanan tum
makespan degerleri cozum dosyasindan okunan rotalarla float Oklid olarak
YENIDEN hesaplanir; izdeki degerler yalniz egri icindir (/SCALE).
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LKH_BIN = os.environ.get("LKH3_BIN", os.path.join(ROOT, "third_party", "lkh_src", "LKH-3.0.13", "LKH"))

_TRACE = re.compile(r"^\*\s+\d+:\s+Cost\s*=\s*(\d+)(?:_(\d+))?")
_RUN = re.compile(r"^Run\s+\d+:\s+Cost\s*=\s*(\d+)(?:_(\d+))?")


def _write_problem(path, X, Y, scale):
    with open(path, "w") as f:
        f.write(f"NAME : p\nTYPE : TSP\nDIMENSION : {len(X)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(zip(X, Y), 1):
            f.write(f"{i} {x * scale:.0f} {y * scale:.0f}\n")
        f.write("EOF\n")


def _scale_for(X, Y):
    """Koordinat genisligini ~1e6-1e7 araligina getiren 10'un kuvveti."""
    span = max(np.ptp(X), np.ptp(Y), 1e-9)
    s = 10.0 ** np.floor(np.log10(5e6 / span))
    return float(max(s, 1e-6))


def _run_traced(cmd, cwd, hard_cap):
    """Sureci baslatir, stdout satirlarini duvar saatiyle damgalar."""
    t0 = time.perf_counter()
    p = subprocess.Popen(["stdbuf", "-oL"] + cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    killed = [False]

    def killer():
        if p.poll() is None:
            killed[0] = True
            p.kill()
    tm = threading.Timer(hard_cap, killer)
    tm.start()
    try:
        for line in p.stdout:
            lines.append((time.perf_counter() - t0, line.rstrip("\n")))
        p.wait()
    finally:
        tm.cancel()
    return lines, time.perf_counter() - t0, killed[0]


def write_init_tour(path, routes, n):
    """MTSP sicak baslangic: LKH ic gosterimi (Dim = n+1 dugum; depo = 1,
    depo kopyalari Dim+1 .. Dim+k-1). Nokta i -> Id i+2."""
    k = len(routes)
    seq = [1]
    for j, r in enumerate(routes):
        if j > 0:
            seq.append(n + 1 + j)
        seq.extend(int(v) + 2 for v in r)
    with open(path, "w") as f:
        f.write(f"TYPE : TOUR\nDIMENSION : {n + k}\nTOUR_SECTION\n")
        f.write("\n".join(map(str, seq)) + "\n-1\nEOF\n")


TUNED = {"INITIAL_TOUR_ALGORITHM": "MTSP", "MOVE_TYPE": 2}


def mtsp_minmax(X, Y, dx, dy, k, time_limit=60.0, seed=1, runs=1, extra=None, keep=False, init_routes=None):
    """Doner: dict(routes, mk_scaled, trace=[(t_duvar, mk)], wall, killed, log)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    PX = np.concatenate([[dx], X])
    PY = np.concatenate([[dy], Y])
    scale = _scale_for(PX, PY)
    d = tempfile.mkdtemp(prefix="lkhm_")
    try:
        _write_problem(os.path.join(d, "p.tsp"), PX, PY, scale)
        par = (f"PROBLEM_FILE = p.tsp\nSALESMEN = {k}\nMTSP_OBJECTIVE = MINMAX\nDEPOT = 1\n"
               f"RUNS = {runs}\nSEED = {seed}\nTRACE_LEVEL = 1\nMTSP_SOLUTION_FILE = p.sol\n"
               f"TOTAL_TIME_LIMIT = {time_limit}\nTIME_LIMIT = {time_limit}\n")
        for kk, vv in (extra or {}).items():
            par += f"{kk} = {vv}\n"
        if init_routes is not None:
            write_init_tour(os.path.join(d, "init.tour"), init_routes, len(X))
            par += "INITIAL_TOUR_FILE = init.tour\n"
        with open(os.path.join(d, "p.par"), "w") as f:
            f.write(par)
        lines, wall, killed = _run_traced([LKH_BIN, "p.par"], d, hard_cap=time_limit * 1.5 + 30)
        trace = []
        for t, ln in lines:
            m = _TRACE.match(ln) or _RUN.match(ln)
            if m:
                trace.append((t, int(m.group(1)) / scale))
        routes = None
        sol = os.path.join(d, "p.sol")
        if os.path.exists(sol):
            routes = []
            with open(sol) as f:
                for ln in f:
                    ln = ln.strip()
                    if ln.startswith("1 ") and "(#" in ln:
                        ids = [int(z) for z in ln.split("(#")[0].split()]
                        routes.append(np.array([i - 2 for i in ids if i != 1], np.int32))
        return dict(routes=routes, trace=trace, wall=wall, killed=killed, scale=scale,
                    log=[ln for _, ln in lines][-40:])
    finally:
        if not keep:
            shutil.rmtree(d, ignore_errors=True)


def tsp(X, Y, time_limit=10.0, seed=1, max_trials=None, runs=1, extra=None):
    """Kapali TSP turu (indeks dizisi, 0 tabanli)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    if n <= 3:
        return np.arange(n, dtype=np.int32)
    scale = _scale_for(X, Y)
    d = tempfile.mkdtemp(prefix="lkht_")
    try:
        _write_problem(os.path.join(d, "p.tsp"), X, Y, scale)
        par = (f"PROBLEM_FILE = p.tsp\nRUNS = {runs}\nSEED = {seed}\nTRACE_LEVEL = 0\n"
               f"OUTPUT_TOUR_FILE = p.tour\nTOTAL_TIME_LIMIT = {time_limit}\nTIME_LIMIT = {time_limit}\n")
        if max_trials:
            par += f"MAX_TRIALS = {int(max_trials)}\n"
        for kk, vv in (extra or {}).items():
            par += f"{kk} = {vv}\n"
        with open(os.path.join(d, "p.par"), "w") as f:
            f.write(par)
        subprocess.run([LKH_BIN, "p.par"], cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=time_limit * 1.5 + 30)
        out = []
        with open(os.path.join(d, "p.tour")) as f:
            on = False
            for ln in f:
                ln = ln.strip()
                if ln == "TOUR_SECTION":
                    on = True
                    continue
                if on:
                    v = int(ln)
                    if v == -1:
                        break
                    out.append(v - 1)
        return np.array(out, np.int32)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _write_explicit(path, PX, PY, w, scale):
    """LOWER_DIAG_ROW: d'(i,j) = Oklid + (w_i + w_j)/2 (w[0] = depo = 0)."""
    n = len(PX)
    with open(path, "w") as f:
        f.write(f"NAME : p\nTYPE : TSP\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : EXPLICIT\n"
                f"EDGE_WEIGHT_FORMAT : LOWER_DIAG_ROW\nEDGE_WEIGHT_SECTION\n")
        for i in range(n):
            d = np.hypot(PX[:i + 1] - PX[i], PY[:i + 1] - PY[i]) + 0.5 * (w[:i + 1] + w[i])
            d[i] = 0.0
            f.write(" ".join(map(str, np.rint(d * scale).astype(np.int64).tolist())) + "\n")
        f.write("EOF\n")


def _trivial(n):
    """n <= 2 hedef: LKH-3 DIMENSION >= 3 ister; her hedef ayri rotada
    (min-max icin en iyisi, k >= n oldugunda)."""
    return dict(routes=[np.array([i], np.int32) for i in range(n)], trace=[(0.0, None)], wall=0.0,
                killed=False, log=["trivial: n <= 2"])


def mtsp_minmax_weighted(X, Y, dx, dy, k, w, time_limit=60.0, seed=1, extra=None, init_routes=None):
    """Dugum agirlikli (hizmet suresi, mesafe biriminde) min-max mTSP; acik
    matris. Rota maliyeti = yol + w |rota| (mmg.Metric(w=...) ile ayni)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if len(X) <= 2 and k >= len(X):
        return _trivial(len(X))
    PX = np.concatenate([[dx], X])
    PY = np.concatenate([[dy], Y])
    W = np.concatenate([[0.0], np.full(len(X), float(w))])
    span = max(np.ptp(PX), np.ptp(PY), 1e-9) + 2 * float(w)
    scale = float(10.0 ** np.floor(np.log10(2e6 / span)))
    d = tempfile.mkdtemp(prefix="lkhw_")
    try:
        _write_explicit(os.path.join(d, "p.tsp"), PX, PY, W, scale)
        par = (f"PROBLEM_FILE = p.tsp\nSALESMEN = {k}\nMTSP_OBJECTIVE = MINMAX\nDEPOT = 1\n"
               f"RUNS = 1\nSEED = {seed}\nTRACE_LEVEL = 1\nMTSP_SOLUTION_FILE = p.sol\n"
               f"TOTAL_TIME_LIMIT = {time_limit}\nTIME_LIMIT = {time_limit}\n")
        for kk, vv in (extra or {}).items():
            par += f"{kk} = {vv}\n"
        if init_routes is not None:
            write_init_tour(os.path.join(d, "init.tour"), init_routes, len(X))
            par += "INITIAL_TOUR_FILE = init.tour\n"
        with open(os.path.join(d, "p.par"), "w") as f:
            f.write(par)
        lines, wall, killed = _run_traced([LKH_BIN, "p.par"], d, hard_cap=time_limit * 1.5 + 60)
        trace = []
        for t, ln in lines:
            m = _TRACE.match(ln) or _RUN.match(ln)
            if m:
                trace.append((t, int(m.group(1)) / scale))
        routes = None
        sol = os.path.join(d, "p.sol")
        if os.path.exists(sol):
            routes = []
            with open(sol) as f:
                for ln in f:
                    ln = ln.strip()
                    if ln.startswith("1 ") and "(#" in ln:
                        ids = [int(z) for z in ln.split("(#")[0].split()]
                        routes.append(np.array([i - 2 for i in ids if i != 1], np.int32))
        return dict(routes=routes, trace=trace, wall=wall, killed=killed, scale=scale,
                    log=[ln for _, ln in lines][-40:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def dcvrp(X, Y, dx, dy, s, v, E, time_limit=10.0, seed=1, vehicles=None, extra=None, min_size=0):
    """Dayanim kisitli sortiler: LKH-3 DCVRP (TYPE DCVRP, DISTANCE = E v,
    SERVICE_TIME_SECTION = s v; mesafe birimi koordinat birimi). Amac LKH-3'un
    DCVRP amaci (toplam). Doner: dict(sorties=[np.int32 dizileri] ya da None,
    penalty, wall, killed, log). Sorti sayisi ust siniri `vehicles`."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    s = np.asarray(s, float)
    n = len(X)
    if n <= 2:
        return dict(sorties=[np.array([i], np.int32) for i in range(n)], penalty=0, wall=0.0, killed=False,
                    log=["trivial"])
    PX = np.concatenate([[dx], X])
    PY = np.concatenate([[dy], Y])
    scale = _scale_for(PX, PY)
    if vehicles is None:
        lb = (float(np.sum(s)) + 2.0 * float(np.mean(np.hypot(X - dx, Y - dy))) / v) / E
        vehicles = int(math.ceil(1.3 * lb)) + 2
    d = tempfile.mkdtemp(prefix="lkhd_")
    try:
        with open(os.path.join(d, "p.vrp"), "w") as f:
            f.write(f"NAME : p\nTYPE : DCVRP\nDIMENSION : {n + 1}\nCAPACITY : {n + 1}\n"
                    f"DISTANCE : {int(round(E * v * scale))}\nVEHICLES : {vehicles}\n"
                    f"EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
            for i in range(n + 1):
                f.write(f"{i + 1} {PX[i] * scale:.0f} {PY[i] * scale:.0f}\n")
            f.write("DEMAND_SECTION\n1 0\n" + "".join(f"{i + 2} 1\n" for i in range(n)))
            f.write("SERVICE_TIME_SECTION\n1 0\n" + "".join(f"{i + 2} {s[i] * v * scale:.0f}\n" for i in range(n)))
            f.write("DEPOT_SECTION\n1\n-1\nEOF\n")
        with open(os.path.join(d, "p.par"), "w") as f:
            f.write(f"PROBLEM_FILE = p.vrp\nRUNS = 1\nSEED = {seed}\nTRACE_LEVEL = 1\nMTSP_MIN_SIZE = {min_size}\n"
                    f"MTSP_SOLUTION_FILE = p.sol\nTOTAL_TIME_LIMIT = {time_limit}\nTIME_LIMIT = {time_limit}\n"
                    + "".join(f"{kk} = {vv}\n" for kk, vv in (extra or {}).items()))
        lines, wall, killed = _run_traced([LKH_BIN, "p.par"], d, hard_cap=time_limit * 1.5 + 60)
        sol = os.path.join(d, "p.sol")
        sorties = None
        pen = None
        if os.path.exists(sol):
            sorties = []
            for ln in open(sol):
                if ln.startswith("1 "):
                    ids = [int(t) for t in ln.split("(")[0].split()]
                    sorties.append(np.array([i - 2 for i in ids if i != 1], np.int32))
                m = re.match(r"^\s*Cost\s*=\s*(\d+)_", ln)
            for _, ln in lines:
                m = re.search(r"Cost\s*=\s*(\d+)_(\d+)", ln)
                if m:
                    pen = int(m.group(1))
        return dict(sorties=sorties, penalty=pen, wall=wall, killed=killed, log=[x for _, x in lines[-6:]])
    finally:
        shutil.rmtree(d, ignore_errors=True)
