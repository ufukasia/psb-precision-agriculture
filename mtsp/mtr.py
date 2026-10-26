# -*- coding: utf-8 -*-
"""Sira araligi kisitli, kasa kapasiteli cok seferli min-max planlama (PSB-MT/R).

Hedefler sira araliklarindaki duraklardir: dugum i = (aralik a_i, aralik
boyunca konum s_i). Platform yalniz sira araliklari boyunca ve bas tarlada
(araligin iki ucu arasinda) ilerler:
  d(i, j) = |s_i - s_j|                                  ayni aralikta
          = min_{u, w in uclar} |s_i - u| + |e_u - e_w| + |w - s_j| + c_turn   farkli aralikta
(|e_u - e_w|: iki uc noktasi arasi Oklid -- bas tarla yolu). Depo bas tarladaki
istasyondur: d(D, i) = min_u |D - e_u| + |u - s_i|.
Sefer (kasa turu): istasyondan cikar, duraklara ugrar, kasa dolunca (toplam
talep <= Q) istasyona doner; istasyonda kasa degisimi t_bin. Amac: son
platformun bitis zamani.

PSB-MT/R: (1) dev tur = sira araliklarinin yilan (serpantin) sirasi ve
varyantlari; (2) kapasiteye uygun seferlere toplam sureyi en kucukleyen optimal
bolme (dinamik programlama); (3) seferlerin platformlara atanmasi (mt.assign:
musaitlik zamanli LPT + tasima/degistirme).
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from . import mt


@njit(cache=True, inline="always")
def _dist(i, j, A, S, E0, E1, turn, EE):
    """Ayni aralikta |s_i - s_j|; farkli aralikta uca git, bas tarla yolu
    (EE: uc-uc en kisa yol), ucdan hedefe; + donus cezasi."""
    a, b = A[i], A[j]
    if a == b:
        return abs(S[i] - S[j])
    best = 1e300
    for u in range(2):
        su = E0[a, 2] if u == 0 else E1[a, 2]
        for w in range(2):
            sw = E0[b, 2] if w == 0 else E1[b, 2]
            d = abs(S[i] - su) + EE[2 * a + u, 2 * b + w] + abs(sw - S[j])
            if d < best:
                best = d
    return best + turn


@njit(cache=True, inline="always")
def _dist_depot(i, A, S, E0, E1, SE):
    a = A[i]
    d0 = SE[2 * a] + abs(S[i] - E0[a, 2])
    d1 = SE[2 * a + 1] + abs(S[i] - E1[a, 2])
    return min(d0, d1)


@njit(cache=True, nogil=True)
def route_time(r, A, S, E0, E1, turn, EE, SE, v, svc):
    if len(r) == 0:
        return 0.0
    d = _dist_depot(r[0], A, S, E0, E1, SE) + _dist_depot(r[-1], A, S, E0, E1, SE)
    for q in range(len(r) - 1):
        d += _dist(r[q], r[q + 1], A, S, E0, E1, turn, EE)
    t = d / v
    for q in range(len(r)):
        t += svc[r[q]]
    return t


@njit(cache=True, nogil=True)
def _split_cap(g, A, S, E0, E1, turn, EE, SE, v, svc, dem, Q, t_bin):
    n = len(g)
    f = np.full(n + 1, np.inf)
    prv = np.full(n + 1, -1, np.int64)
    f[0] = 0.0
    for i in range(n):
        if not np.isfinite(f[i]):
            continue
        path = 0.0
        load = 0.0
        sv = 0.0
        first = _dist_depot(g[i], A, S, E0, E1, SE)
        for j in range(i, n):
            if j > i:
                path += _dist(g[j - 1], g[j], A, S, E0, E1, turn, EE)
            load += dem[g[j]]
            if load > Q and j > i:
                break
            sv += svc[g[j]]
            dur = (first + path + _dist_depot(g[j], A, S, E0, E1, SE)) / v + sv
            tot = f[i] + dur + t_bin
            if tot < f[j + 1]:
                f[j + 1] = tot
                prv[j + 1] = i
            if load > Q:
                break
    cuts = []
    j = n
    while j > 0:
        cuts.append(j)
        j = prv[j]
    cuts.append(0)
    return f[n], np.array(cuts[::-1], np.int64)


def headland(E0, E1, dx, dy):
    """Bas tarla yol agi: tum aralik uclari ve istasyonun dis bukey zarfinin
    cevresi. Bir noktadan digerine: zarfa dik izdusum + cevre boyunca kisa yon
    + izdusumden noktaya. Doner: EE (2A x 2A, uc-uc), SE (2A, istasyon-uc)."""
    P = np.vstack([E0[:, :2], E1[:, :2], [[dx, dy]]])
    nA = len(E0)
    pts = sorted(map(tuple, P))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    H = np.array(lo[:-1] + up[:-1])
    seg = np.roll(H, -1, axis=0) - H
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    per = cum[-1]

    def proj(x):
        best = None
        for q in range(len(H)):
            ab = seg[q]
            t = 0.0 if L[q] == 0 else float(np.clip(((x - H[q]) @ ab) / (L[q] ** 2), 0, 1))
            pt = H[q] + t * ab
            d = float(np.hypot(*(x - pt)))
            if best is None or d < best[0]:
                best = (d, cum[q] + t * L[q])
        return best
    ends = [E0[a, :2] if u == 0 else E1[a, :2] for a in range(nA) for u in range(2)]
    pr = [proj(np.asarray(e, float)) for e in ends]
    st = proj(np.array([dx, dy], float))
    m = len(ends)
    EE = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            arc = abs(pr[i][1] - pr[j][1])
            d = pr[i][0] + min(arc, per - arc) + pr[j][0]
            EE[i, j] = EE[j, i] = min(d, d)
    SE = np.array([st[0] + min(abs(st[1] - q[1]), per - abs(st[1] - q[1])) + q[0] for q in pr])
    return EE, SE


def headland_path(E0, E1, dx, dy, a, b):
    """Gorsellestirme icin: a ve b noktalari arasi bas tarla yolu ara noktalari
    (dis bukey zarfa izdusum, zarf koseleri boyunca kisa yon)."""
    P = np.vstack([E0[:, :2], E1[:, :2], [[dx, dy]]])
    from scipy.spatial import ConvexHull
    H = P[ConvexHull(P).vertices]
    seg = np.roll(H, -1, axis=0) - H
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    per = cum[-1]

    def proj(x):
        best = None
        for q in range(len(H)):
            t = 0.0 if L[q] == 0 else float(np.clip(((x - H[q]) @ seg[q]) / (L[q] ** 2), 0, 1))
            pt = H[q] + t * seg[q]
            d = float(np.hypot(*(x - pt)))
            if best is None or d < best[0]:
                best = (d, cum[q] + t * L[q], pt)
        return best
    pa, pb = proj(np.asarray(a, float)), proj(np.asarray(b, float))
    fwd = (pb[1] - pa[1]) % per
    step = 1 if fwd <= per - fwd else -1
    pts = [pa[2]]
    sa, sb = pa[1], pb[1]
    verts = [(cum[q], H[q]) for q in range(len(H))]
    if step == 1:
        arc = [(c_ - sa) % per for c_, _ in verts]
        inside = sorted([(x, v) for x, (c_, v) in zip(arc, verts) if 0 < x < fwd], key=lambda t: t[0])
    else:
        arc = [(sa - c_) % per for c_, _ in verts]
        inside = sorted([(x, v) for x, (c_, v) in zip(arc, verts) if 0 < x < per - fwd], key=lambda t: t[0])
    pts += [v for _, v in inside]
    pts.append(pb[2])
    return [np.asarray(q, float) for q in pts] + [np.asarray(b, float)]


class Field:
    """Sira araligi yapisi: A (dugum araligi), S (aralik boyunca konum),
    E0/E1 (aralik uclari: x, y, s), depo, donus cezasi (mesafe birimi),
    bas tarla uc-uc (EE) ve istasyon-uc (SE) yol uzunluklari."""

    def __init__(self, A, S, E0, E1, dx, dy, turn, EE=None, SE=None):
        self.A = np.ascontiguousarray(A, np.int64)
        self.S = np.ascontiguousarray(S, float)
        self.E0 = np.ascontiguousarray(E0, float)
        self.E1 = np.ascontiguousarray(E1, float)
        self.dx, self.dy, self.turn = float(dx), float(dy), float(turn)
        if EE is None:
            EE, SE = headland(self.E0, self.E1, self.dx, self.dy)
        self.EE = np.ascontiguousarray(EE, float)
        self.SE = np.ascontiguousarray(SE, float)

    def sub(self, idx):
        return Field(self.A[idx], self.S[idx], self.E0, self.E1, self.dx, self.dy, self.turn, self.EE, self.SE)

    def rtime(self, r, v, svc):
        return route_time(np.asarray(r, np.int64), self.A, self.S, self.E0, self.E1, self.turn, self.EE, self.SE,
                          v, svc)

    def matrix(self, v):
        """Tam sure matrisi (OR-Tools icin), 0 = depo."""
        n = len(self.A)
        M = np.zeros((n + 1, n + 1))
        for i in range(n):
            M[0, i + 1] = M[i + 1, 0] = _dist_depot(i, self.A, self.S, self.E0, self.E1, self.SE) / v
        M[1:, 1:] = _full(self.A, self.S, self.E0, self.E1, self.turn, self.EE) / v
        return M


@njit(cache=True, nogil=True)
def _full(A, S, E0, E1, turn, EE):
    n = len(A)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = _dist(i, j, A, S, E0, E1, turn, EE)
    return M


def serpentine(F, order=None, start_dir=0):
    """Araliklari `order` sirasinda dolasan yilan dev tur."""
    alleys = np.unique(F.A) if order is None else order
    out = []
    for q, a in enumerate(alleys):
        ids = np.where(F.A == a)[0]
        ids = ids[np.argsort(F.S[ids])]
        if (q + start_dir) % 2:
            ids = ids[::-1]
        out.append(ids)
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)


def split(F, g, v, svc, dem, Q, t_bin):
    tot, cuts = _split_cap(g, F.A, F.S, F.E0, F.E1, F.turn, F.EE, F.SE, float(v), svc, dem, float(Q), float(t_bin))
    return tot, [g[cuts[q]:cuts[q + 1]] for q in range(len(cuts) - 1)]


def schedule(F, trips, v, svc, k, avail, t_bin):
    trips = [np.asarray(t, np.int64) for t in trips if len(t)]
    durs = [F.rtime(t, v, svc) for t in trips]
    own, mk = mt.assign(durs, k, avail, t_bin)
    return [[trips[q] for q in sorted(own[j], key=lambda q: F.rtime(trips[q][:1], v, svc))] for j in range(k)], mk


def solve(F, svc, dem, v, Q, k, avail=None, t_bin=120.0, budget=0.0, seed=1):
    """Doner: (platform basina sefer listesi, makespan, bilgi)."""
    t0 = time.perf_counter()
    svc = np.ascontiguousarray(svc, float)
    dem = np.ascontiguousarray(dem, float)
    avail = np.zeros(k) if avail is None else np.asarray(avail, float)
    n = len(F.A)
    if n == 0:
        return [[] for _ in range(k)], float(np.max(avail)), dict(t=0.0)
    alleys = np.unique(F.A)
    # depoya yakin araliktan baslayan iki yon, iki baslangic yonu
    near = int(np.argmin([min(F.SE[2 * a], F.SE[2 * a + 1]) for a in alleys]))
    cands = [alleys, alleys[::-1], np.roll(alleys, -near), np.roll(alleys[::-1], -(len(alleys) - 1 - near))]
    best = None
    rng = np.random.default_rng(seed)
    it = 0
    while True:
        for order in cands:
            for sd in (0, 1):
                g = serpentine(F, order, sd)
                _, trips = split(F, g, v, svc, dem, Q, t_bin)
                sc, mk = schedule(F, trips, v, svc, k, avail, t_bin)
                if best is None or mk < best[0] - 1e-9:
                    best = (mk, sc)
        it += 1
        if budget <= 0 or time.perf_counter() - t0 > budget:
            break
        # LNS: aralik sirasinda rastgele blok ters cevirme
        base = cands[int(rng.integers(len(cands)))].copy()
        if len(base) < 2:
            break
        a, b = np.sort(rng.choice(len(base), 2, replace=False))
        base[a:b + 1] = base[a:b + 1][::-1]
        cands = [base]
    return best[1], best[0], dict(t=time.perf_counter() - t0, iters=it)
