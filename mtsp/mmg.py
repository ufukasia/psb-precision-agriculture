# -*- coding: utf-8 -*-
"""mm.py'nin genel-metrik surumu (simulasyon icin).

Algoritma mm.py ile AYNIDIR (Oklid ve w=0'da rota cikti ozdes -- test:
experiments/test_mmg_equiv.py). Fark: mesafe _dist(H) uzerinden;
  * dugum basina hizmet suresi w (mesafe biriminde = sure x hiz):
    rota maliyeti = yol + w |rota|  (kenara (w_a + w_b)/2 eklenir; depo 0)
  * istege bagli sira kisitli metrik (H[0] = 1).
Genellik ~%15-20 hiz kaybina yol actigi icin karsilastirma deneyleri
(bench.py) Oklid'e ozel mm.py'yi kullanir; olgunluk simulasyonu (hasat
suresi = dugum agirligi) bu modulu kullanir.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

EPS = 1e-9


# ---------------------------------------------------------------------------
#  Temel yardimcilar
# ---------------------------------------------------------------------------
def knn(X, Y, k):
    """Her nokta icin k en yakin komsu (kendisi haric), int32 (n, k')."""
    n = len(X)
    k = int(min(k, n - 1))
    if k <= 0:
        return np.zeros((n, 0), np.int32)
    P = np.column_stack([X, Y])
    _, idx = cKDTree(P).query(P, k=k + 1)
    out = np.empty((n, k), np.int32)
    for i in range(n):
        row = idx[i]
        row = row[row != i][:k]
        if len(row) < k:                       # kopya noktalar: kendisi listede olmayabilir
            row = idx[i][1:k + 1]
        out[i] = row
    return out


@njit(cache=True, inline="always")
def _dist(xa, ya, ra, xb, yb, rb, H):
    """Metrik. H = [mod, L, R, w]; mod 0: Oklid; mod 1: sira kisitli (siralar
    x ekseni boyunca, bas tarlalar x=L ve x=R; ayni siradaysa dogrudan,
    degilse yakin bas tarladan dolasma). ra/rb: sira kimligi (depo: -1).
    w: dugum basina hizmet (mesafe biriminde); kenara (w_a + w_b)/2 eklenir,
    boylece rota uzunlugu = yol + w * |rota| (depo agirliksiz)."""
    if H[0] == 0.0 or (ra == rb and ra >= 0.0):
        dx = xa - xb
        dy = ya - yb
        d = math.sqrt(dx * dx + dy * dy)
    else:
        d = min(xa + xb - 2.0 * H[1], 2.0 * H[2] - xa - xb) + abs(ya - yb)
    if H[3] != 0.0:
        if ra >= 0.0:
            d += 0.5 * H[3]
        if rb >= 0.0:
            d += 0.5 * H[3]
    return d


@njit(cache=True, inline="always")
def _d(X, Y, Rw, H, i, j):
    return _dist(X[i], Y[i], Rw[i], X[j], Y[j], Rw[j], H)


@njit(cache=True, nogil=True)
def tour_len(tour, X, Y, Rw, H):
    n = len(tour)
    s = 0.0
    for i in range(n):
        s += _d(X, Y, Rw, H, tour[i], tour[(i + 1) % n])
    return s


class Metric:
    """Mesafe tanimi. rows=None: Oklid. rows verilirse sira kisitli metrik
    (bkz. _dist); L, R bas tarla x konumlari; w dugum basina hizmet
    (mesafe biriminde = hizmet suresi x hiz); ys: aday komsu listesi icin
    y olcegi (sira ici komsulari one cikarir)."""

    def __init__(self, n, rows=None, L=0.0, R=0.0, w=0.0, ys=1.0):
        self.Rw = np.zeros(n) if rows is None else np.asarray(rows, float).copy()
        self.H = np.array([0.0 if rows is None else 1.0, L, R, w], float)
        self.ys = float(ys)

    @property
    def euclid(self):
        return self.H[0] == 0.0 and self.H[3] == 0.0

    def sub(self, idx):
        m = Metric.__new__(Metric)
        m.Rw = self.Rw[np.asarray(idx, np.int64)]
        m.H = self.H
        m.ys = self.ys
        return m

    def with_depot_first(self, idx):
        """[depo] + idx yerel dizisi icin metrik."""
        m = self.sub(idx)
        m.Rw = np.concatenate([[-1.0], m.Rw])
        return m


def _M(M, n):
    return M if M is not None else Metric(n)


def dist_np(xa, ya, ra, xb, yb, rb, H):
    """_dist'in numpy (vektor) karsiligi."""
    xa, ya, ra, xb, yb, rb = (np.asarray(v, float) for v in (xa, ya, ra, xb, yb, rb))
    e = np.hypot(xa - xb, ya - yb)
    if H[0] != 0.0:
        cross = np.minimum(xa + xb - 2 * H[1], 2 * H[2] - xa - xb) + np.abs(ya - yb)
        e = np.where((ra == rb) & (ra >= 0), e, cross)
    if H[3] != 0.0:
        e = e + 0.5 * H[3] * ((ra >= 0).astype(float) + (rb >= 0).astype(float))
    return e


def route_cost(route, X, Y, dx, dy, M=None):
    """Depodan cikip depoya donen rota uzunlugu (route: nokta indeksleri)."""
    if len(route) == 0:
        return 0.0
    r = np.asarray(route)
    px = X[r]
    py = Y[r]
    if M is None or M.euclid:
        s = float(np.hypot(np.diff(px), np.diff(py)).sum())
        return s + math.hypot(px[0] - dx, py[0] - dy) + math.hypot(px[-1] - dx, py[-1] - dy)
    pr = M.Rw[r]
    ex = np.concatenate([[dx], px, [dx]])
    ey = np.concatenate([[dy], py, [dy]])
    er = np.concatenate([[-1.0], pr, [-1.0]])
    return float(dist_np(ex[:-1], ey[:-1], er[:-1], ex[1:], ey[1:], er[1:], M.H).sum())


def makespan(routes, X, Y, dx, dy, M=None):
    return max(route_cost(r, X, Y, dx, dy, M) for r in routes)


def knn_m(X, Y, k, M=None):
    """Aday listeleri: Oklid KNN; sira kisitli modda y olcekli."""
    if M is None or M.H[0] == 0.0:
        return knn(X, Y, k)
    return knn(X, np.asarray(Y) * M.ys, k)


def check_solution(routes, n):
    """Her nokta tam bir kez: degilse AssertionError."""
    seen = np.zeros(n, np.int64)
    for r in routes:
        for v in r:
            seen[v] += 1
    assert np.all(seen == 1), "gecersiz cozum: eksik/kopya nokta"


# ---------------------------------------------------------------------------
#  1. Greedy-Edge dev tur
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


@njit(cache=True, nogil=True)
def greedy_edge(X, Y, Rw, H, nbr):
    n = len(X)
    if n <= 3:
        return np.arange(n).astype(np.int32)
    K = nbr.shape[1]
    m = n * K
    ei = np.empty(m, np.int32)
    ej = np.empty(m, np.int32)
    w = np.empty(m)
    c = 0
    for i in range(n):
        for t in range(K):
            j = nbr[i, t]
            if i < j:
                ei[c] = i
                ej[c] = j
                w[c] = _d(X, Y, Rw, H, i, j)
                c += 1
            elif j < i:
                # (j, i) j'nin listesinde yoksa yine aday olsun
                found = False
                for u in range(K):
                    if nbr[j, u] == i:
                        found = True
                        break
                if not found:
                    ei[c] = j
                    ej[c] = i
                    w[c] = _d(X, Y, Rw, H, i, j)
                    c += 1
    order = np.argsort(w[:c], kind="mergesort")
    deg = np.zeros(n, np.int32)
    par = np.arange(n).astype(np.int32)
    adj = -np.ones((n, 2), np.int32)
    links = 0
    for o in order:
        i = ei[o]
        j = ej[o]
        if deg[i] >= 2 or deg[j] >= 2:
            continue
        ri = _find(par, i)
        rj = _find(par, j)
        if ri == rj:
            continue
        par[ri] = rj
        adj[i, deg[i]] = j
        deg[i] += 1
        adj[j, deg[j]] = i
        deg[j] += 1
        links += 1
        if links == n - 1:
            break
    # parcalari birlestir: uc noktadan yuru, en yakin bos parca ucuna atla
    ends = np.empty(n, np.int32)
    ne = 0
    for i in range(n):
        if deg[i] < 2:
            ends[ne] = i
            ne += 1
    ends = ends[:ne]
    used = np.zeros(n, np.bool_)
    tour = np.empty(n, np.int32)
    pos = 0
    cur_end = ends[0]
    while True:
        # cur_end'den parcayi yuru
        prev = -1
        cur = cur_end
        while True:
            tour[pos] = cur
            pos += 1
            used[cur] = True
            nxt = -1
            for t in range(deg[cur]):
                if adj[cur, t] != prev and not used[adj[cur, t]]:
                    nxt = adj[cur, t]
                    break
            if nxt == -1:
                break
            prev = cur
            cur = nxt
        if pos >= n:
            break
        best = -1
        bd = 1e300
        for e in range(ne):
            v = ends[e]
            if not used[v]:
                dd = _d(X, Y, Rw, H, cur, v)
                if dd < bd:
                    bd = dd
                    best = v
        cur_end = best
    return tour


# ---------------------------------------------------------------------------
#  2-opt + Or-opt (dizi gosterimi, komsu listeli, ilk iyilesme)
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _reverse(tour, pos, i, j):
    """tour[i..j] (dairesel) ters cevir; kisa tarafi cevirir."""
    n = len(tour)
    L = (j - i) % n + 1
    if 2 * L > n:
        i, j = (j + 1) % n, (i - 1) % n
        L = n - L
    for _ in range(L // 2):
        a = tour[i]
        b = tour[j]
        tour[i] = b
        pos[b] = i
        tour[j] = a
        pos[a] = j
        i = (i + 1) % n
        j = (j - 1) % n


@njit(cache=True, nogil=True)
def ls_tour(tour, X, Y, Rw, H, nbr, max_passes, use_oropt):
    """Kapali tur uzerinde 2-opt + Or-opt(1..3). tour yerinde degisir.
    Doner: uygulanan hamle sayisi."""
    n = len(tour)
    if n < 5:
        return 0
    K = nbr.shape[1]
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[tour[i]] = i
    moves = 0
    buf = np.empty(n, np.int32)
    for _pass in range(max_passes):
        improved = False
        for a in range(n):
            # ---- 2-opt (ardil yonu) ----
            i = pos[a]
            b = tour[(i + 1) % n]
            dab = _d(X, Y, Rw, H, a, b)
            for t in range(K):
                c = nbr[a, t]
                g1 = dab - _d(X, Y, Rw, H, a, c)
                if g1 <= EPS:
                    break
                j = pos[c]
                d = tour[(j + 1) % n]
                if c == b or d == a:
                    continue
                delta = g1 + _d(X, Y, Rw, H, c, d) - _d(X, Y, Rw, H, b, d)
                if delta > EPS:
                    _reverse(tour, pos, (i + 1) % n, j)
                    moves += 1
                    improved = True
                    i = pos[a]
                    b = tour[(i + 1) % n]
                    dab = _d(X, Y, Rw, H, a, b)
            # ---- 2-opt (oncul yonu) ----
            i = pos[a]
            b = tour[(i - 1) % n]
            dab = _d(X, Y, Rw, H, a, b)
            for t in range(K):
                c = nbr[a, t]
                g1 = dab - _d(X, Y, Rw, H, a, c)
                if g1 <= EPS:
                    break
                j = pos[c]
                d = tour[(j - 1) % n]
                if c == b or d == a:
                    continue
                delta = g1 + _d(X, Y, Rw, H, c, d) - _d(X, Y, Rw, H, b, d)
                if delta > EPS:
                    _reverse(tour, pos, j, (i - 1) % n)
                    moves += 1
                    improved = True
                    i = pos[a]
                    b = tour[(i - 1) % n]
                    dab = _d(X, Y, Rw, H, a, b)
            if not use_oropt:
                continue
            # ---- Or-opt: a'dan baslayan 1..3 uzunlugunda segment ----
            for L in range(1, 4):
                if L >= n - 2:
                    break
                i = pos[a]
                sf = a
                sl = tour[(i + L - 1) % n]
                p = tour[(i - 1) % n]
                nx = tour[(i + L) % n]
                rem = _d(X, Y, Rw, H, p, sf) + _d(X, Y, Rw, H, sl, nx) - _d(X, Y, Rw, H, p, nx)
                if rem <= EPS:
                    continue
                bestg = EPS
                bu = -1
                brev = False
                for side in range(2):
                    end = sf if side == 0 else sl
                    for t in range(K):
                        c = nbr[end, t]
                        if _d(X, Y, Rw, H, end, c) >= rem:
                            break
                        for which in range(2):
                            if which == 0:
                                u = c
                            else:
                                u = tour[(pos[c] - 1) % n]
                            # u segmentte mi ya da u == p mi
                            off = (pos[u] - i) % n
                            if off < L or u == p:
                                continue
                            v = tour[(pos[u] + 1) % n]
                            duv = _d(X, Y, Rw, H, u, v)
                            add1 = _d(X, Y, Rw, H, u, sf) + _d(X, Y, Rw, H, sl, v) - duv
                            add2 = _d(X, Y, Rw, H, u, sl) + _d(X, Y, Rw, H, sf, v) - duv
                            if rem - add1 > bestg:
                                bestg = rem - add1
                                bu = u
                                brev = False
                            if rem - add2 > bestg:
                                bestg = rem - add2
                                bu = u
                                brev = True
                if bu >= 0:
                    # yeniden kur: nx'ten baslayip p'ye dek, bu'dan sonra segmenti ekle
                    w = 0
                    k0 = (i + L) % n
                    for s in range(n - L):
                        x = tour[(k0 + s) % n]
                        buf[w] = x
                        w += 1
                        if x == bu:
                            if brev:
                                for q in range(L):
                                    buf[w] = tour[(i + L - 1 - q) % n]
                                    w += 1
                            else:
                                for q in range(L):
                                    buf[w] = tour[(i + q) % n]
                                    w += 1
                    for s in range(n):
                        tour[s] = buf[s]
                        pos[buf[s]] = s
                    moves += 1
                    improved = True
                    break
        if not improved:
            break
    return moves


def giant_tour(X, Y, nbr, ls=True, max_passes=50, M=None):
    M = _M(M, len(X))
    t = greedy_edge(X, Y, M.Rw, M.H, nbr)
    if ls:
        ls_tour(t, X, Y, M.Rw, M.H, nbr, max_passes, True)
    return t


# ---------------------------------------------------------------------------
#  2. Depolu min-max en iyi bolme (dairesel dev tur)
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _split_feasible(g, X, Y, dx, dy, k, T, cum, d0, nxt):
    """Uygunsa (baslangic) indeksini, degilse -1 doner."""
    n = len(g)
    m = 2 * n
    # nxt[s] = s'den baslayan segmentin bitisinden sonraki indeks (iki gostergeli)
    e = 0
    for s in range(m):
        if e < s:
            e = s
        # cost(s..e) = d0[s] + cum[e]-cum[s] + d0[e]
        if d0[s % n] * 2.0 > T:
            nxt[s] = s          # tek nokta bile sigmaz
            continue
        while e + 1 < m and e + 1 < s + n and \
                d0[s % n] + cum[e + 1] - cum[s] + d0[(e + 1) % n] <= T:
            e += 1
        nxt[s] = e + 1
    for s in range(n):
        p = s
        for _ in range(k):
            if nxt[p] == p:
                break
            p = nxt[p]
            if p >= s + n:
                break
        if p >= s + n:
            return s
    return -1


@njit(cache=True, nogil=True)
def _split_prep(g, X, Y, Rw, H, dx, dy):
    n = len(g)
    cum = np.zeros(2 * n)
    for t in range(1, 2 * n):
        a = g[(t - 1) % n]
        b = g[t % n]
        cum[t] = cum[t - 1] + _d(X, Y, Rw, H, a, b)
    d0 = np.empty(n)
    for t in range(n):
        v = g[t]
        d0[t] = _dist(X[v], Y[v], Rw[v], dx, dy, -1.0, H)
    return cum, d0


def split_minmax(g, X, Y, dx, dy, k, rel_tol=1e-7, M=None):
    """Dev tur g'yi en fazla k ardisik (dairesel) rotaya, makespan'i en kucuk
    olacak sekilde boler. Doner: rota listesi (np.int32 dizileri)."""
    n = len(g)
    g = np.asarray(g, np.int32)
    M = _M(M, len(X))
    cum, d0 = _split_prep(g, X, Y, M.Rw, M.H, dx, dy)
    nxt = np.empty(2 * n, np.int64)
    lo = 2.0 * float(d0.max())
    hi = 2.0 * float(d0.max()) + float(cum[n])       # tek rota her zaman uygun
    s_hi = _split_feasible(g, X, Y, dx, dy, k, hi, cum, d0, nxt)
    best_s, best_T = s_hi, hi
    if _split_feasible(g, X, Y, dx, dy, k, lo, cum, d0, nxt) >= 0:
        best_T = lo
        best_s = _split_feasible(g, X, Y, dx, dy, k, lo, cum, d0, nxt)
    else:
        while hi - lo > rel_tol * hi:
            mid = 0.5 * (lo + hi)
            s = _split_feasible(g, X, Y, dx, dy, k, mid, cum, d0, nxt)
            if s >= 0:
                hi, best_s, best_T = mid, s, mid
            else:
                lo = mid
    _split_feasible(g, X, Y, dx, dy, k, best_T, cum, d0, nxt)
    routes = []
    p = best_s
    while p < best_s + n:
        q = min(int(nxt[p]), best_s + n)        # son segment baslangica tasmasin
        routes.append(np.array([g[t % n] for t in range(p, q)], np.int32))
        p = q
    return routes


# ---------------------------------------------------------------------------
#  3. Rota cilalama (depo dahil kapali tur) -- paralel
# ---------------------------------------------------------------------------
def _polish_one(route, X, Y, dx, dy, knn_k, passes, M=None):
    m = len(route)
    if m < 3:
        return route
    lx = np.empty(m + 1)
    ly = np.empty(m + 1)
    lx[0], ly[0] = dx, dy
    lx[1:] = X[route]
    ly[1:] = Y[route]
    LM = _M(M, len(X)).with_depot_first(route)
    nb = knn_m(lx, ly, knn_k, LM)
    t = np.arange(m + 1, dtype=np.int32)
    ls_tour(t, lx, ly, LM.Rw, LM.H, nb, passes, True)
    z = int(np.where(t == 0)[0][0])
    t = np.roll(t, -z)[1:] - 1
    return route[t]


_POOL = None


def _pool(workers):
    global _POOL
    if _POOL is None or _POOL._max_workers != workers:
        _POOL = ThreadPoolExecutor(workers)
    return _POOL


def polish_routes(routes, X, Y, dx, dy, knn_k=10, passes=50, workers=8, which=None, M=None):
    idx = range(len(routes)) if which is None else which
    idx = list(idx)
    if workers <= 1 or len(idx) <= 1:
        res = [_polish_one(routes[i], X, Y, dx, dy, knn_k, passes, M) for i in idx]
    else:
        res = list(_pool(workers).map(lambda i: _polish_one(routes[i], X, Y, dx, dy, knn_k, passes, M), idx))
    out = list(routes)
    for i, r in zip(idx, res):
        out[i] = r
    return out


# ---------------------------------------------------------------------------
#  4. Min-max dengeleme: en uzun rotadan relocate / swap
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _balance(X, Y, Rw, H, nbr, succ, pred, rid, rcost, n, k, max_moves):
    """Dugumler 0..n-1 noktalar, n..n+k-1 depo kopyalari (X,Y'de depoda).
    Bagli liste uzerinde en uzun rotanin makespan'ini dusuren en iyi
    relocate/swap hamlesi uygulanir. Doner: hamle sayisi; changed[r]
    isaretleri rcost'un yaninda dondurulmez -- cagiran yeniden hesaplar."""
    K = nbr.shape[1]
    moves = 0
    touched = np.zeros(k, np.bool_)
    while moves < max_moves:
        rs = 0
        for r in range(1, k):
            if rcost[r] > rcost[rs]:
                rs = r
        cur = rcost[rs]
        best = cur - 1e-7
        btype = 0
        bv = -1
        bu = -1
        bc = -1
        # rs rotasindaki her v
        v = succ[n + rs]
        while v != n + rs:
            p = pred[v]
            q = succ[v]
            dpv = _d(X, Y, Rw, H, p, v)
            dvq = _d(X, Y, Rw, H, v, q)
            save = dpv + dvq - _d(X, Y, Rw, H, p, q)
            for t in range(K):
                c = nbr[v, t]
                r2 = rid[c]
                if r2 == rs:
                    continue
                # relocate v: c'nin ardina ya da onune
                for which in range(2):
                    if which == 0:
                        u = c
                        w = succ[c]
                    else:
                        u = pred[c]
                        w = c
                    ins = _d(X, Y, Rw, H, u, v) + _d(X, Y, Rw, H, v, w) - _d(X, Y, Rw, H, u, w)
                    nm = max(cur - save, rcost[r2] + ins)
                    if nm < best:
                        best = nm
                        btype = 1
                        bv = v
                        bu = u
                        bc = r2
                # swap v <-> c
                pc = pred[c]
                qc = succ[c]
                if p == c or q == c:
                    continue
                d1 = _d(X, Y, Rw, H, p, c) + _d(X, Y, Rw, H, c, q) - dpv - dvq
                d2 = _d(X, Y, Rw, H, pc, v) + _d(X, Y, Rw, H, v, qc) - _d(X, Y, Rw, H, pc, c) - _d(X, Y, Rw, H, c, qc)
                nm = max(cur + d1, rcost[r2] + d2)
                if nm < best:
                    best = nm
                    btype = 2
                    bv = v
                    bu = c
                    bc = r2
            v = q
        if btype == 0:
            break
        if btype == 1:
            v = bv
            p = pred[v]
            q = succ[v]
            save = _d(X, Y, Rw, H, p, v) + _d(X, Y, Rw, H, v, q) - _d(X, Y, Rw, H, p, q)
            succ[p] = q
            pred[q] = p
            u = bu
            w = succ[u]
            ins = _d(X, Y, Rw, H, u, v) + _d(X, Y, Rw, H, v, w) - _d(X, Y, Rw, H, u, w)
            succ[u] = v
            pred[v] = u
            succ[v] = w
            pred[w] = v
            rcost[rs] -= save
            rcost[bc] += ins
            rid[v] = bc
        else:
            v = bv
            c = bu
            p = pred[v]
            q = succ[v]
            pc = pred[c]
            qc = succ[c]
            d1 = _d(X, Y, Rw, H, p, c) + _d(X, Y, Rw, H, c, q) - _d(X, Y, Rw, H, p, v) - _d(X, Y, Rw, H, v, q)
            d2 = _d(X, Y, Rw, H, pc, v) + _d(X, Y, Rw, H, v, qc) - _d(X, Y, Rw, H, pc, c) - _d(X, Y, Rw, H, c, qc)
            succ[p] = c
            pred[c] = p
            succ[c] = q
            pred[q] = c
            succ[pc] = v
            pred[v] = pc
            succ[v] = qc
            pred[qc] = v
            rcost[rs] += d1
            rcost[bc] += d2
            rid[v] = bc
            rid[c] = rs
        touched[rs] = True
        touched[bc] = True
        moves += 1
    return moves, touched


def _to_linked(routes, n, k):
    succ = np.empty(n + k, np.int64)
    pred = np.empty(n + k, np.int64)
    rid = np.empty(n + k, np.int64)
    for r in range(k):
        dep = n + r
        rid[dep] = r
        seq = [dep] + [int(v) for v in (routes[r] if r < len(routes) else [])]
        for a, b in zip(seq, seq[1:] + seq[:1]):
            succ[a] = b
            pred[b] = a
        for v in seq[1:]:
            rid[v] = r
    return succ, pred, rid


def _from_linked(succ, n, k):
    out = []
    for r in range(k):
        dep = n + r
        seq = []
        v = succ[dep]
        while v != dep:
            seq.append(v)
            v = succ[v]
        out.append(np.array(seq, np.int32))
    return out


def balance_routes(routes, X, Y, dx, dy, nbr, max_moves=100000, M=None):
    n = len(X)
    k = len(routes)
    M = _M(M, n)
    XX = np.concatenate([X, np.full(k, dx)])
    YY = np.concatenate([Y, np.full(k, dy)])
    RR = np.concatenate([M.Rw, np.full(k, -1.0)])
    succ, pred, rid = _to_linked(routes, n, k)
    rc = np.array([route_cost(r, X, Y, dx, dy, M) for r in routes], float)
    mv, touched = _balance(XX, YY, RR, M.H, nbr, succ, pred, rid, rc, n, k, max_moves)
    return _from_linked(succ, n, k), int(mv), [i for i in range(k) if touched[i]]


# ---------------------------------------------------------------------------
#  Tam hat
# ---------------------------------------------------------------------------
def solve(X, Y, dx, dy, k, knn_k=10, workers=8, rounds=20, polish_fn=None, trace=None, M=None):
    """PSB hatti. Doner: (rotalar, bilgi sozlugu). trace: (t, makespan)
    listesi (anytime egrisi icin; t0 = cagri basi)."""
    X = np.ascontiguousarray(X, float)
    Y = np.ascontiguousarray(Y, float)
    t0 = time.perf_counter()
    nbr = knn_m(X, Y, knn_k, M)
    g = giant_tour(X, Y, nbr, M=M)
    t_giant = time.perf_counter() - t0
    routes = split_minmax(g, X, Y, dx, dy, k, M=M)
    while len(routes) < k:                     # k'dan az rota cikarsa bos rota ekle
        routes.append(np.zeros(0, np.int32))
    t_split = time.perf_counter() - t0
    mk_split = makespan(routes, X, Y, dx, dy, M)
    pf = polish_fn or (lambda rs, which=None: polish_routes(rs, X, Y, dx, dy, knn_k, 50, workers, which, M))
    routes = pf(routes)
    mk_pol = makespan(routes, X, Y, dx, dy, M)
    if trace is not None:
        trace.append((time.perf_counter() - t0, mk_pol))
    t_polish = time.perf_counter() - t0
    best = mk_pol
    best_routes = routes
    total_moves = 0
    for it in range(rounds):
        routes, mv, touched = balance_routes(routes, X, Y, dx, dy, nbr, M=M)
        total_moves += mv
        if mv == 0:
            break
        routes = pf(routes, which=touched)
        mk = makespan(routes, X, Y, dx, dy, M)
        if trace is not None:
            trace.append((time.perf_counter() - t0, mk))
        if mk < best - 1e-9:
            best, best_routes = mk, routes
        else:
            break
    t_total = time.perf_counter() - t0
    check_solution(best_routes, len(X))
    return best_routes, dict(mk_split=mk_split, mk_polish=mk_pol, mk=best,
                             t_giant=t_giant, t_split=t_split, t_polish=t_polish,
                             t_total=t_total, moves=total_moves)


# ---------------------------------------------------------------------------
#  5. Anytime iyilestirme: alt-problem yeniden cozumu (LNS)
# ---------------------------------------------------------------------------
def _centroid(r, X, Y, dx, dy):
    if len(r) == 0:
        return dx, dy
    return float(X[r].mean()), float(Y[r].mean())


def _subsolve(nodes, m, X, Y, dx, dy, knn_k, rng, order_hint=None, M=None):
    """nodes (global indeksler) uzerinde m aracli min-max: dev tur + bolme +
    cilalama + dengeleme. order_hint verilirse dev tur o siradan baslar
    (Or-opt ile duzeltilir) -- cesitlendirme icin."""
    lx = X[nodes]
    ly = Y[nodes]
    LM = _M(M, len(X)).sub(nodes)
    nb = knn_m(lx, ly, knn_k, LM)
    if order_hint is None:
        g = greedy_edge(lx, ly, LM.Rw, LM.H, nb)
    else:
        g = np.asarray(order_hint, np.int32).copy()
    ls_tour(g, lx, ly, LM.Rw, LM.H, nb, 50, True)
    rs = split_minmax(g, lx, ly, dx, dy, m, M=LM)
    while len(rs) < m:
        rs.append(np.zeros(0, np.int32))
    rs = [_polish_one(r, lx, ly, dx, dy, knn_k, 50, LM) for r in rs]
    for _ in range(10):
        rs2, mv, touched = balance_routes(rs, lx, ly, dx, dy, nb, M=LM)
        if mv == 0:
            break
        rs2 = [(_polish_one(r, lx, ly, dx, dy, knn_k, 50, LM) if i in touched else r) for i, r in enumerate(rs2)]
        if max(route_cost(r, lx, ly, dx, dy, LM) for r in rs2) < max(route_cost(r, lx, ly, dx, dy, LM) for r in rs) - 1e-9:
            rs = rs2
        else:
            break
    return [nodes[r] for r in rs]


def _angle_order(routes, X, Y, dx, dy, rng, jitter=0.0):
    ang = []
    for r in routes:
        cx, cy = _centroid(r, X, Y, dx, dy)
        ang.append(math.atan2(cy - dy, cx - dx) + rng.normal(0, jitter))
    return list(np.argsort(ang))


def improve(routes, X, Y, dx, dy, budget_s, knn_k=10, group=(2, 3, 4), seed=1, trace=None,
            t_offset=0.0, max_fail=None, M=None):
    """Zaman butcesi boyunca LNS. Her adimda en uzun rota ve merkezlerine en
    yakin (m-1) rota birlestirilir, alt problem yeniden cozulur; grubun
    makespan'i dusuyorsa kabul. Cesitlendirme: dev tur ya GE ya da mevcut
    rotalarin rastgele aci sirasiyla birlesimi. Doner: (rotalar, makespan)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    routes = [np.asarray(r, np.int32) for r in routes]
    costs = np.array([route_cost(r, X, Y, dx, dy, M) for r in routes])
    k = len(routes)
    fails = 0
    max_fail = max_fail or 50 * k
    group = tuple(group)
    while time.perf_counter() - t0 < budget_s:
        if fails >= max_fail:
            # tikandi: grubu buyut (komsulugu genislet); k'ya ulasinca dur
            if max(group) >= k:
                break
            group = tuple(g + 1 for g in group)
            fails = 0
        rs = int(np.argmax(costs))
        m = int(min(k, rng.choice(group)))
        cs = np.array([_centroid(r, X, Y, dx, dy) for r in routes])
        d = np.hypot(cs[:, 0] - cs[rs, 0], cs[:, 1] - cs[rs, 1])
        d[rs] = -1
        # yakin rotalardan biraz rastgele secim
        cand = list(np.argsort(d)[1:min(k, 2 * m + 1)])
        pick = [rs] + list(rng.choice(cand, size=min(m - 1, len(cand)), replace=False)) if m > 1 else [rs]
        nodes = np.concatenate([routes[i] for i in pick]).astype(np.int64)
        if len(nodes) < 2:
            break
        old = float(costs[pick].max())
        if rng.random() < 0.5:
            hint = None
        else:
            sub = [routes[i] for i in pick]
            order = _angle_order(sub, X, Y, dx, dy, rng, jitter=0.3)
            loc = {int(v): j for j, v in enumerate(nodes)}
            hint = [loc[int(v)] for i in order for v in (sub[i] if rng.random() < 0.5 else sub[i][::-1])]
        new = _subsolve(nodes, len(pick), X, Y, dx, dy, knn_k, rng, hint, M)
        nc = np.array([route_cost(r, X, Y, dx, dy, M) for r in new])
        if nc.max() < old - 1e-9:
            for i, r, c in zip(pick, new, nc):
                routes[i] = np.asarray(r, np.int32)
                costs[i] = c
            fails = 0
            if trace is not None:
                trace.append((t_offset + time.perf_counter() - t0, float(costs.max())))
        else:
            fails += 1
    check_solution(routes, len(X))
    return routes, float(costs.max())


# ---------------------------------------------------------------------------
#  Alt sinir (depolu min-max)
# ---------------------------------------------------------------------------
def lower_bound(X, Y, dx, dy, k, M=None):
    """max( 2 max_v d(D,v),  MST(P u {D}) / k ).
    Her rota depodan gecen kapali tur: L(T_j) >= MST(P_j u D) ve k agacin
    birlesimi (D'de birlesik) P u D'yi geren bagli bir alt grafiktir; ayrica
    her tur en uzak noktasina gidip donmek zorundadir."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import Delaunay
    P = np.unique(np.column_stack([np.concatenate([[dx], X]), np.concatenate([[dy], Y])]), axis=0)
    tri = Delaunay(P)
    s = tri.simplices
    E = np.unique(np.sort(np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]]), 1), axis=0)
    w = np.hypot(*(P[E[:, 0]] - P[E[:, 1]]).T)
    mst = float(minimum_spanning_tree(coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(P), len(P)))).sum())
    if M is None or M.euclid:
        far = 2.0 * float(np.hypot(np.asarray(X) - dx, np.asarray(Y) - dy).max())
        return max(far, mst / k), mst, far
    # genel metrik: Oklid MST alt sinir olarak gecerli (metrik >= Oklid);
    # hizmet agirligi her noktada bir kez odenir
    n = len(X)
    dd = dist_np(X, Y, M.Rw, np.full(n, dx), np.full(n, dy), np.full(n, -1.0), M.H)
    far = 2.0 * float(dd.max())
    tot = mst + M.H[3] * n
    return max(far, tot / k), tot, far


def improve_parallel(routes, X, Y, dx, dy, budget_s, workers=8, knn_k=10, group=(2, 3, 4), seed=1,
                     trace=None, t_offset=0.0, max_fail=None, M=None):
    """Paralel LNS: her adimda en uzun rotalardan baslayarak AYRIK gruplar
    secilir ve es zamanli (numba nogil, is parcacigi) yeniden cozulur. Her
    grup kendi grup-makespan'i dusuyorsa kabul edilir; global makespan
    asla artmaz."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    routes = [np.asarray(r, np.int32) for r in routes]
    costs = np.array([route_cost(r, X, Y, dx, dy, M) for r in routes])
    k = len(routes)
    fails = 0
    max_fail = max_fail or 50 * k
    pool = _pool(workers)
    while time.perf_counter() - t0 < budget_s and fails < max_fail:
        cs = np.array([_centroid(r, X, Y, dx, dy) for r in routes])
        order = list(np.argsort(-costs))
        used = np.zeros(k, bool)
        jobs = []
        for rs in order:
            if len(jobs) >= workers:
                break
            if used[rs]:
                continue
            m = int(min(k, rng.choice(group)))
            d = np.hypot(cs[:, 0] - cs[rs, 0], cs[:, 1] - cs[rs, 1])
            d[rs] = -1
            cand = [i for i in np.argsort(d)[1:] if not used[i]][:2 * m]
            if m > 1 and not cand:
                continue
            pick = [rs] + (list(rng.choice(cand, size=min(m - 1, len(cand)), replace=False)) if m > 1 else [])
            for i in pick:
                used[i] = True
            nodes = np.concatenate([routes[i] for i in pick]).astype(np.int64)
            if len(nodes) < 2:
                continue
            if rng.random() < 0.5:
                hint = None
            else:
                sub = [routes[i] for i in pick]
                od = _angle_order(sub, X, Y, dx, dy, rng, jitter=0.3)
                loc = {int(v): j for j, v in enumerate(nodes)}
                hint = [loc[int(v)] for i in od for v in (sub[i] if rng.random() < 0.5 else sub[i][::-1])]
            sub_rng = np.random.default_rng(rng.integers(1 << 31))
            jobs.append((pick, nodes, hint, sub_rng))
        if not jobs:
            break
        res = list(pool.map(lambda j: _subsolve(j[1], len(j[0]), X, Y, dx, dy, knn_k, j[3], j[2], M), jobs))
        top_before = float(costs.max())
        for (pick, _n, _h, _r), new in zip(jobs, res):
            nc = np.array([route_cost(r, X, Y, dx, dy, M) for r in new])
            if nc.max() < costs[pick].max() - 1e-9:
                for i, r, c in zip(pick, new, nc):
                    routes[i] = np.asarray(r, np.int32)
                    costs[i] = c
        if costs.max() < top_before - 1e-9:
            fails = 0
            if trace is not None:
                trace.append((t_offset + time.perf_counter() - t0, float(costs.max())))
        else:
            fails += 1
    check_solution(routes, len(X))
    return routes, float(costs.max())
