# -*- coding: utf-8 -*-
"""Kare (izgara) dikimli bahcede hasat platformlari icin planlama.

Platform agac siralari arasindaki koridorlarda IKI yonde ilerler; yol uzunlugu
bahcenin eksen cercevesinde Manhattan (L1) mesafesidir, yon degisiminde donus
cezasi eklenir. Bahcenin ekseni planlayiciya verilmez: agac konumlarindan
izgara acisi kestirilir (tarak skoru; esinti: ilk makale, asil2026frame).

Icerik
  grid_angle      : izgara acisi ve guven (90 derece periyotlu, [-45, 45))
  Frame           : kestirilen cerceve, aralik, L1 metrigi, istasyon
  serpentine      : bir dugum kumesinin uzun eksen boyunca yilan sirasi
  split_trips     : kasa kapasitesine uygun seferlere optimal bolme (DP)
  kd_regions      : agirlikli k-d bolme; kesimler koridorlara oturtulur
  psb_global      : tum bahce icin PSB (yilan dev tur + bolme + atama)
  KDSteal         : k-d bolge sahipligi + uc devralma (is calma) -- bir platform
                    kendi bolgesini bitirince en cok isi kalan platformun
                    kuyrugunu (onun konumundan en uzak uc) dengeleyecek
                    noktadan boler ve devralir.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import mt


# --------------------------------------------------------------------------
# Izgara acisi (tarak skoru)
# --------------------------------------------------------------------------
def _comb(v, nbins):
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return float(nbins)
    idx = np.minimum(((v - lo) / (hi - lo) * nbins).astype(np.int64), nbins - 1)
    p = np.bincount(idx, minlength=nbins).astype(float)
    p /= p.sum()
    return float((p * p).sum() * nbins)


def nn_angle(X, Y, kk=4):
    """En yakin komsu yonlerinin 90 derece periyotlu dairesel ortalamasi.
    Kare/dikdortgen kafeste en yakin komsular eksenler boyuncadir (kosegen
    sqrt(2) kat uzakta), bu yuzden kosegen hatlara kilitlenmez."""
    from scipy.spatial import cKDTree
    P = np.column_stack([X, Y])
    _, idx = cKDTree(P).query(P, kk + 1)
    D = P[idx[:, 1:]] - P[:, None, :]
    ang = np.arctan2(D[..., 1], D[..., 0]).ravel()
    z = np.exp(4j * ang)
    return float(np.degrees(np.angle(z.mean())) / 4.0), float(abs(z.mean()))


def grid_angle(X, Y, step=1.0):
    """(theta_deg in [-45, 45), guven). Kaba aci en yakin komsu yonlerinden,
    ince ayar tarak skorundan (noktalar theta kadar geri dondurulunce izgara
    satirlari bir eksende tarak gibi coker), +-5 derece icinde."""
    P = np.column_stack([X, Y]) - np.mean(np.column_stack([X, Y]), 0)
    nb = max(32, min(2048, int(math.sqrt(len(P)) * 12)))

    def score(t):
        a = math.radians(t)
        c, s = math.cos(a), math.sin(a)
        return max(_comb(c * P[:, 0] + s * P[:, 1], nb), _comb(-s * P[:, 0] + c * P[:, 1], nb))
    th0, r = nn_angle(P[:, 0], P[:, 1])
    fine = np.arange(th0 - 5.0, th0 + 5.0 + 1e-9, 0.05)
    sf = np.array([score(t) for t in fine])
    th = float(fine[int(np.argmax(sf))])
    th = (th + 45.0) % 90.0 - 45.0
    return th, r


# --------------------------------------------------------------------------
# Cerceve ve metrik
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _d(ua, va, ub, vb, turn, tol):
    du = abs(ua - ub)
    dv = abs(va - vb)
    t = turn if (du > tol and dv > tol) else 0.0
    return du + dv + t


class Frame:
    """Kestirilen cerceve: theta (derece), merkez, aralik (su, sv); U, V
    noktalarin cerceve koordinatlari; istasyon (su0, sv0); donus cezasi
    (mesafe birimi)."""

    def __init__(self, X, Y, station, turn, theta=None):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        if theta is None:
            theta, self.conf = grid_angle(X, Y)
        else:
            self.conf = float("nan")
        self.theta = float(theta)
        a = math.radians(self.theta)
        self.c = np.array([X.mean(), Y.mean()])
        self.ca, self.sa = math.cos(a), math.sin(a)
        self.U, self.V = self.to_frame(X, Y)
        su, sv = self.to_frame(np.array([station[0]]), np.array([station[1]]))
        self.su0, self.sv0 = float(su[0]), float(sv[0])
        self.turn = float(turn)
        self.du = self._spacing(self.U, self.V)
        self.dv = self._spacing(self.V, self.U)
        self.tol = 0.5 * min(self.du, self.dv)

    def to_frame(self, X, Y):
        x = np.asarray(X, float) - self.c[0]
        y = np.asarray(Y, float) - self.c[1]
        return self.ca * x + self.sa * y, -self.sa * x + self.ca * y

    @staticmethod
    def _spacing(A, B):
        """A ekseni boyunca ardisik agac araligi (ayni hattaki komsular)."""
        order = np.lexsort((A, np.round(B / max(np.ptp(B) / max(1, int(math.sqrt(len(B)))), 1.0))))
        d = np.abs(np.diff(A[order]))
        d = d[d > 1e-6]
        return float(np.median(d[d < np.percentile(d, 75) * 1.5])) if len(d) else 1.0


@njit(cache=True, nogil=True)
def route_time(r, U, V, su, sv, turn, tol, v, svc):
    if len(r) == 0:
        return 0.0
    d = _d(su, sv, U[r[0]], V[r[0]], turn, tol) + _d(U[r[-1]], V[r[-1]], su, sv, turn, tol)
    for q in range(len(r) - 1):
        d += _d(U[r[q]], V[r[q]], U[r[q + 1]], V[r[q + 1]], turn, tol)
    t = d / v
    for q in range(len(r)):
        t += svc[r[q]]
    return t


@njit(cache=True, nogil=True)
def _split(g, U, V, su, sv, turn, tol, v, svc, dem, Q, t_bin):
    n = len(g)
    f = np.full(n + 1, np.inf)
    prv = np.full(n + 1, -1, np.int64)
    f[0] = 0.0
    for i in range(n):
        if not np.isfinite(f[i]):
            continue
        first = _d(su, sv, U[g[i]], V[g[i]], turn, tol)
        path = 0.0
        load = 0.0
        sv_ = 0.0
        for j in range(i, n):
            if j > i:
                path += _d(U[g[j - 1]], V[g[j - 1]], U[g[j]], V[g[j]], turn, tol)
            load += dem[g[j]]
            if load > Q and j > i:
                break
            sv_ += svc[g[j]]
            dur = (first + path + _d(U[g[j]], V[g[j]], su, sv, turn, tol)) / v + sv_
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


def rtime(F, r, v, svc):
    return route_time(np.asarray(r, np.int64), F.U, F.V, F.su0, F.sv0, F.turn, F.tol, float(v), svc)


def split_trips(F, g, v, svc, dem, Q, t_bin):
    g = np.asarray(g, np.int64)
    if len(g) == 0:
        return []
    _, cuts = _split(g, F.U, F.V, F.su0, F.sv0, F.turn, F.tol, float(v), svc, dem, float(Q), float(t_bin))
    return [g[cuts[q]:cuts[q + 1]] for q in range(len(cuts) - 1)]


def serpentine(F, idx, axis=None, start=None):
    """idx dugumlerinin yilan sirasi. axis 0: hatlar U boyunca (V'ye gore
    gruplu), 1: tersi; None: uzun eksen. start: (u, v) yakin kose."""
    idx = np.asarray(idx, np.int64)
    if len(idx) == 0:
        return idx
    U, V = F.U[idx], F.V[idx]
    if axis is None:
        axis = 0 if np.ptp(U) >= np.ptp(V) else 1
    along, across, sp = (U, V, F.dv) if axis == 0 else (V, U, F.du)
    line = np.round((across - across.min()) / sp).astype(int)
    lines = np.unique(line)
    if start is not None:
        s_al, s_ac = (start[0], start[1]) if axis == 0 else (start[1], start[0])
        if abs(s_ac - across.max()) < abs(s_ac - across.min()):
            lines = lines[::-1]
        first_fwd = abs(s_al - along.min()) <= abs(s_al - along.max())
    else:
        first_fwd = True
    out = []
    for q, ln in enumerate(lines):
        m = np.where(line == ln)[0]
        m = m[np.argsort(along[m])]
        if (q % 2 == 0) != first_fwd:
            m = m[::-1]
        out.append(idx[m])
    return np.concatenate(out)


# --------------------------------------------------------------------------
# k-d bolge bolme (agirlikli, kesim koridorlara oturtulur)
# --------------------------------------------------------------------------
def kd_regions(F, idx, w, k):
    idx = np.asarray(idx, np.int64)
    w = np.asarray(w, float)
    out = []

    def rec(ids, ww, j):
        if j <= 1 or len(ids) <= 1:
            out.append(ids)
            return
        j1 = j // 2
        U, V = F.U[ids], F.V[ids]
        ax = 0 if np.ptp(U) >= np.ptp(V) else 1
        key, sp = (U, F.du) if ax == 0 else (V, F.dv)
        lines = np.round((key - key.min()) / sp).astype(int)
        order = np.argsort(key, kind="stable")
        cw = np.cumsum(ww[order]) / max(ww.sum(), 1e-12)
        q = int(np.searchsorted(cw, j1 / j))
        q = min(max(q, 0), len(order) - 1)
        cut_line = lines[order[q]]
        # kesimi koridora oturt: hat cut_line'in sonuna (agirlik dengesine en yakin taraf)
        left = lines <= cut_line
        alt = lines < cut_line
        tgt = j1 / j * ww.sum()
        if abs(ww[alt].sum() - tgt) < abs(ww[left].sum() - tgt) and alt.any():
            left = alt
        if left.all() or not left.any():
            left = np.zeros(len(ids), bool)
            left[order[:max(1, q)]] = True
        rec(ids[left], ww[left], j1)
        rec(ids[~left], ww[~left], j - j1)
    rec(idx, w, k)
    return [r for r in out]


# --------------------------------------------------------------------------
# Kuresel PSB (yilan dev tur + optimal kasa bolme + atama)
# --------------------------------------------------------------------------
def psb_global(F, idx, svc, dem, v, Q, k, avail, t_bin):
    """Doner: platform basina sefer listesi (global dugum indeksleri)."""
    idx = np.asarray(idx, np.int64)
    best = None
    st = (F.su0, F.sv0)
    for axis in (0, 1):
        g = serpentine(F, idx, axis, st)
        trips = split_trips(F, g, v, svc, dem, Q, t_bin)
        durs = [rtime(F, t, v, svc) for t in trips]
        own, mk = mt.assign(durs, k, avail, t_bin)
        if best is None or mk < best[0]:
            best = (mk, [[trips[q] for q in sorted(own[j])] for j in range(k)])
    return best[1]


# --------------------------------------------------------------------------
# k-d sahiplik + uc devralma
# --------------------------------------------------------------------------
class KDSteal:
    """Her platformun kendi k-d bolgesinde yilan sirasi (kuyruk). Platform
    istasyona her donuste kuyrugunun basindan kasaya sigacak kadar dugum alir
    (bilinen tahminlerle). Kuyrugu bosalan platform, kalan isi en buyuk olan
    platformun kuyrugunu, iki tarafin bitis zamanlari esitlenecek noktadan
    keser ve kuyrugun SONUNU (sahibinin konumundan en uzak uc) ters yonde
    devralir."""

    def __init__(self, F, idx, w, k, steal=True, axis=None, split=None):
        """axis: bolge ici yilan hatlarinin ekseni (None: bolgenin uzun ekseni;
        'field': tum bahcenin uzun ekseni -- daha uzun hat, daha az donus).
        split: (v, Q, t_bin) verilirse seferler kuyruk basindan optimal bolme ile."""
        self.F = F
        self.k = k
        self.steal = steal
        self.split = split
        regs = kd_regions(F, idx, w, k)
        while len(regs) < k:
            regs.append(np.zeros(0, np.int64))
        ax = None
        if axis == "field":
            ax = 0 if np.ptp(F.U[idx]) >= np.ptp(F.V[idx]) else 1
        self.queue = [list(serpentine(F, r, ax, (F.su0, F.sv0))) for r in regs]
        self.region = [set(map(int, r)) for r in regs]
        self.steals = 0

    def remaining_work(self, j, svc, alive):
        return float(sum(svc[q] for q in self.queue[j] if alive[q]))

    def next_trip(self, j, svc, dem, Q, alive, pending, now_rem=None):
        """j platformu icin bir sonraki sefer (dugum listesi) ya da [].
        alive: ziyarete deger (tahmini olgun, islenmemis, rezervsiz);
        pending: islenmemis ve rezervsiz (tahmin sonra degisebilir, kuyrukta kalir)."""
        self.queue[j] = [q for q in self.queue[j] if pending[q]]
        if not any(alive[q] for q in self.queue[j]) and self.steal:
            self._steal(j, svc, alive, pending, now_rem)
        trip, load = [], 0.0
        if self.split is not None:
            v, Qs, tb = self.split
            seq = np.array([q for q in self.queue[j] if alive[q]], np.int64)
            if len(seq):
                trips = split_trips(self.F, seq, v, svc, dem, Qs, tb)
                trip = [int(q) for q in trips[0]]
        else:
            for q in self.queue[j]:
                if not alive[q]:
                    continue
                if trip and load + dem[q] > Q:
                    break
                trip.append(q)
                load += dem[q]
        taken = set(trip)
        self.queue[j] = [q for q in self.queue[j] if q not in taken]
        return trip

    def _steal(self, j, svc, alive, pending, now_rem):
        rem = [self.remaining_work(i, svc, alive) + (now_rem[i] if now_rem is not None else 0.0)
               for i in range(self.k)]
        rem[j] = -1.0
        d = int(np.argmax(rem))
        qd = [q for q in self.queue[d] if pending[q]]
        if sum(1 for q in qd if alive[q]) < 2:
            return
        # donor'un kalan isi (ucustaki seferi dahil) ile yardimcinin isi esitlensin
        base = now_rem[d] if now_rem is not None else 0.0
        acc, cut = base, len(qd)
        tot = base + sum(svc[q] for q in qd if alive[q])
        for p, q in enumerate(qd):
            if acc >= tot / 2.0:
                cut = p
                break
            if alive[q]:
                acc += svc[q]
        cut = min(max(cut, 1), len(qd) - 1)
        self.queue[d] = qd[:cut]
        self.queue[j] = qd[cut:][::-1]            # kuyrugun sonundan basla (uc devralma)
        self.steals += 1
