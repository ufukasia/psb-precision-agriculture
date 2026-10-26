# -*- coding: utf-8 -*-
"""Literaturdeki min-max mTSP sezgisellerinin (resmi kodlar) sarmalayicilari.

Her fonksiyon ayni imzayi kullanir:
    run_<ad>(X, Y, dx, dy, k, time_limit, seed=1, workdir=None)
        -> dict(routes, trace=[(t_duvar, makespan)], wall, killed, log)
routes: k adet nokta indeksi listesi (0 tabanli, depo haric) ya da None.
Depo noktalardan ayri tutulur; dosyaya ilk sehir olarak yazilir (kodlarin ic
indeksi 0, dosya id'si 1); nokta i -> ic indeks i+1, dosya id'si i+2.

Cozuculer (kaynak, lisans ve derleme notlari: third_party/sota/SOURCES.txt):
  hsnr : He & Hao (2022, C&OR 142, 105726) HSNR, kaynaktan derlendi.
  ma   : He & Hao (2023, EJOR 307) memetik arama, yazarlarin Linux ikilisi.
  mils : He, Hao & Xia (2026, C&OR 185, 107255), yazarlarin Linux ikilisi.
  hga  : Mahmoudinazlou & Kwon (2024, C&OR 162, 106455), Julia kaynagi
         (HGSmTSP), yerel Julia 1.10.5 + yerel depo ile; surucu hga_driver/.
Uc kod da EUC_2D icin GERCEK (double) Oklid mesafe kullanir; olcekleme gerekmez,
koordinatlar %.17g ile yazilir. Uc kod da tek is parcacikli calisir ve sureyi
clock() (CPU suresi) ile olcer. Izdeki degerler yalniz egri icindir; raporlanan
makespan cagiran tarafindan rotalardan float olarak yeniden hesaplanmalidir.
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
PAPER2 = os.path.dirname(HERE)
SOTA = os.path.join(PAPER2, "third_party", "sota")
HSNR_BIN = os.environ.get("HSNR_BIN", os.path.join(SOTA, "he_mTSP", "build", "hsnr"))
MA_BIN = os.environ.get("MA_BIN", os.path.join(SOTA, "he_minmax-mTSP", "x", "code", "ma"))
MILS_BIN = os.environ.get("MILS_BIN", os.path.join(SOTA, "he_mils", "binary_code", "mils"))
JULIA_DIR = os.environ.get("HGA_JULIA_DIR", os.path.join(SOTA, "julia"))
JULIA_BIN = os.path.join(JULIA_DIR, "julia-1.10.5", "bin", "julia")
HGA_DRIVER = os.path.join(SOTA, "hga_driver", "run_hga.jl")

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


# ---------------------------------------------------------------------------
#  Ortak yardimcilar
# ---------------------------------------------------------------------------
def _run_traced(cmd, cwd, hard_cap, env=None):
    """Sureci baslatir, stdout satirlarini duvar saatiyle damgalar."""
    t0 = time.perf_counter()
    p = subprocess.Popen(["stdbuf", "-oL"] + cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace", env=env)
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


def _write_he(path, X, Y, dx, dy, k, with_n):
    """He & Hao bicimi: 1. satir 'ad EUC_2D [N] k', sonra n+1 satir 'id x y'
    (ilk satir depo). HSNR N alanini okumaz (n'yi komut satirindan alir).
    MA / MILS ikilileri basliktaki N kadar satir okur (depo DAHIL): yazarlarin
    dagittigi orneklerdeki gibi N = n yazilirsa son nokta sessizce dusurulur
    (dogrulandi), bu yuzden N = n + 1 yazilir."""
    n = len(X)
    with open(path, "w") as f:
        f.write(f"p EUC_2D {n + 1} {k}\n" if with_n else f"p EUC_2D {k}\n")
        f.write(f"1 {dx:.17g} {dy:.17g}\n")
        for i in range(n):
            f.write(f"{i + 2} {X[i]:.17g} {Y[i]:.17g}\n")


def _routes_from_ids(seqs, n, k):
    """Dosya dugum indeksleri (0 = depo, j -> nokta j-1) -> 0 tabanli rotalar."""
    routes = [[int(v) - 1 for v in s if int(v) != 0] for s in seqs]
    if len(routes) != k:
        return None
    return routes


def _trace_from(lines, pat, group=1):
    """Duvar saatli satirlardan monoton azalan (t, deger) izi."""
    tr = []
    best = math.inf
    for t, ln in lines:
        m = pat.search(ln)
        if m:
            v = float(m.group(group))
            if v < best - 1e-9:
                best = v
                tr.append((t, v))
    return tr


def _finish(routes, trace, wall, killed, lines, X, Y, dx, dy):
    if not trace and routes is not None:
        from . import mm
        trace = [(wall, mm.makespan([np.asarray(r, np.int64) for r in routes], np.asarray(X, float),
                                    np.asarray(Y, float), dx, dy))]
    return dict(routes=routes, trace=trace, wall=wall, killed=killed,
                log=[ln for _, ln in lines][-40:])


class _Workdir:
    def __init__(self, workdir, prefix):
        self.keep = workdir is not None
        if self.keep:
            os.makedirs(workdir, exist_ok=True)
            self.d = workdir
        else:
            self.d = tempfile.mkdtemp(prefix=prefix)

    def __enter__(self):
        return self.d

    def __exit__(self, *a):
        if not self.keep:
            shutil.rmtree(self.d, ignore_errors=True)


def _read_he_solution(path, n, k):
    """HSNR / MA cozum dosyasi: 'the sequence of best solution is following'
    satirindan sonra k satir '0 a b c ... [0]'."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        L = f.read().splitlines()
    try:
        i = next(j for j, ln in enumerate(L) if "sequence of best solution" in ln)
    except StopIteration:
        return None
    seqs = []
    for ln in L[i + 1:]:
        tok = ln.split()
        if not tok or not all(t.lstrip("-").isdigit() for t in tok):
            break
        seqs.append(tok)
    return _routes_from_ids(seqs, n, k)


# ---------------------------------------------------------------------------
#  HSNR (He & Hao 2022) -- kaynaktan derlenmis (g++ -O3)
# ---------------------------------------------------------------------------
_HSNR_INIT = re.compile(rf"^\s*\d+\s+({_NUM})\s*$")           # 'i  fitness' (baslangic LS)
_HSNR_STEP = re.compile(rf"^\s*\d+\s+({_NUM})\s+{_NUM}\s+{_NUM}\s*$")  # 'adim  en_iyi  mevcut  x'


def run_hsnr(X, Y, dx, dy, k, time_limit, seed=1, workdir=None):
    """Komut: hsnr <dosya> <n> <sure_s> <tohum>. Kod 'ceil(cpu) >= T' ile durdugu
    icin (T-1 s sonra durur) T = floor(time_limit) + 1 verilir; tamsayi saniye."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    T = int(math.floor(time_limit)) + 1
    with _Workdir(workdir, "hsnr_") as d:
        _write_he(os.path.join(d, "p.txt"), X, Y, dx, dy, k, with_n=False)
        sol = os.path.join(d, f"p_{k}_{seed}.txt")
        if os.path.exists(sol):
            os.remove(sol)
        lines, wall, killed = _run_traced([HSNR_BIN, "p.txt", str(n), str(T), str(seed)], d,
                                          hard_cap=time_limit * 1.5 + 60)
        tr = []
        best = math.inf
        for t, ln in lines:
            m = _HSNR_STEP.match(ln) or _HSNR_INIT.match(ln)
            if m:
                v = float(m.group(1))
                if v < best - 1e-9:
                    best = v
                    tr.append((t, v))
        routes = _read_he_solution(sol, n, k)
        return _finish(routes, tr, wall, killed, lines, X, Y, dx, dy)


# ---------------------------------------------------------------------------
#  MA (He & Hao 2023) -- yazarlarin ikilisi
# ---------------------------------------------------------------------------
_MA_INIT = re.compile(rf"^\s*\d+\s+({_NUM})\s*$")
_MA_ITER = re.compile(rf"The number of iteration is\s+\d+\s+\|\|\s+({_NUM})")
_MA_END = re.compile(rf"The best solution is\s+({_NUM})")


def run_ma(X, Y, dx, dy, k, time_limit, seed=1, workdir=None):
    """Komut: ma <dosya> <sure_s> <tohum> (sure tamsayi saniye, %d ile okunur;
    round(time_limit), en az 1). Ilerleme 500 iterasyonda bir yazilir."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    T = max(1, int(round(time_limit)))
    with _Workdir(workdir, "ma_") as d:
        _write_he(os.path.join(d, "p.txt"), X, Y, dx, dy, k, with_n=True)
        sol = os.path.join(d, f"p_{k}_{seed}.txt")
        if os.path.exists(sol):
            os.remove(sol)
        lines, wall, killed = _run_traced([MA_BIN, "p.txt", str(T), str(seed)], d,
                                          hard_cap=time_limit * 1.5 + 60)
        tr = []
        best = math.inf
        for t, ln in lines:
            m = _MA_ITER.search(ln) or _MA_END.search(ln) or _MA_INIT.match(ln)
            if m:
                v = float(m.group(1))
                if v < best - 1e-9:
                    best = v
                    tr.append((t, v))
        routes = _read_he_solution(sol, n, k)
        return _finish(routes, tr, wall, killed, lines, X, Y, dx, dy)


# ---------------------------------------------------------------------------
#  MILS (He, Hao & Xia 2026) -- yazarlarin ikilisi
# ---------------------------------------------------------------------------
_MILS_GLO = re.compile(rf"GloSol:\s*({_NUM})")
_MILS_END = re.compile(rf"Write Solution With Value:\s*({_NUM})")


def run_mils(X, Y, dx, dy, k, time_limit, seed=1, workdir=None):
    """Komut: mils <dosya> <cikti> <BKS> <sure_s> <maks_iter> <tohum>.
    BKS = 0 (asla erisilmez; erken durma yok), maks_iter = 1e9, sure ondalikli.
    Ilerleme 100 iterasyonda bir 'GloSol' olarak yazilir (2 ondalik)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    with _Workdir(workdir, "mils_") as d:
        _write_he(os.path.join(d, "p.txt"), X, Y, dx, dy, k, with_n=True)
        sol = os.path.join(d, "p.sol")
        if os.path.exists(sol):
            os.remove(sol)
        lines, wall, killed = _run_traced([MILS_BIN, "p.txt", "p.sol", "0", f"{float(time_limit):g}",
                                           "1000000000", str(seed)], d, hard_cap=time_limit * 1.5 + 60)
        tr = []
        best = math.inf
        for t, ln in lines:
            m = _MILS_GLO.search(ln) or _MILS_END.search(ln)
            if m:
                v = float(m.group(1))
                if v < best - 1e-9:
                    best = v
                    tr.append((t, v))
        routes = None
        if os.path.exists(sol):
            seqs = []
            with open(sol) as f:
                for ln in f:
                    if ln.startswith("Route"):
                        seqs.append(ln.split(":", 1)[1].strip().split("-"))
            routes = _routes_from_ids(seqs, n, k)
        return _finish(routes, tr, wall, killed, lines, X, Y, dx, dy)


# ---------------------------------------------------------------------------
#  HGA (Mahmoudinazlou & Kwon 2024) -- Julia kaynagi, tek is parcacigi
# ---------------------------------------------------------------------------
_HGA_GEN = re.compile(rf"the best objective is:\s*({_NUM})")
_HGA_FIN = re.compile(rf"generations is:\s*({_NUM})")
_HGA_END = re.compile(rf"^END\s+({_NUM})")


def run_hga(X, Y, dx, dy, k, time_limit, seed=1, workdir=None):
    """solve_mTSP(k, D, C; time_limit, n_iterations=1e9, verbose=true); diger
    parametreler varsayilan (h=0.3, popsize=(10,20)). D float Oklid matris.
    Iz ve 'wall' surucunun 'START' satirindan (Julia acilisi + paket yukleme
    sonrasi) olculur; acilis suresi 'startup' alaninda. Ilerleme 1000 nesilde
    bir yazilir; ilk nokta 0. nesil (baslangic populasyonu + 1 nesil).
    n < 1200 icin baslangicta Concorde (Concorde_jll) cagrilir."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    env = dict(os.environ, JULIA_DEPOT_PATH=os.path.join(JULIA_DIR, "depot"), JULIA_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    with _Workdir(workdir, "hga_") as d:
        with open(os.path.join(d, "c.txt"), "w") as f:
            f.write(f"{dx:.17g} {dy:.17g}\n")
            for i in range(n):
                f.write(f"{X[i]:.17g} {Y[i]:.17g}\n")
        sol = os.path.join(d, "out.txt")
        if os.path.exists(sol):
            os.remove(sol)
        cmd = [JULIA_BIN, "-t", "1", f"--project={os.path.join(JULIA_DIR, 'env_hga')}", HGA_DRIVER,
               "c.txt", str(k), f"{float(time_limit):g}", str(seed), "out.txt"]
        lines, wall_total, killed = _run_traced(cmd, d, hard_cap=time_limit * 1.5 + 120, env=env)
        t_start = next((t for t, ln in lines if ln.strip() == "START"), 0.0)
        tr = []
        best = math.inf
        for t, ln in lines:
            m = _HGA_GEN.search(ln) or _HGA_FIN.search(ln) or _HGA_END.match(ln)
            if m:
                v = float(m.group(1))
                if v < best - 1e-9:
                    best = v
                    tr.append((t - t_start, v))
        routes = None
        if os.path.exists(sol):
            with open(sol) as f:
                seqs = [ln.split() for ln in f.read().splitlines()]
            routes = [[int(v) - 1 for v in s] for s in seqs]
            routes += [[] for _ in range(k - len(routes))]
        res = _finish(routes, tr, wall_total - t_start, killed, lines, X, Y, dx, dy)
        res["startup"] = t_start
        return res


SOLVERS = {"hsnr": run_hsnr, "ma": run_ma, "mils": run_mils, "hga": run_hga}
