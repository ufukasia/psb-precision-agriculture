# -*- coding: utf-8 -*-
"""Gercek bahce sinirlarindan (OpenStreetMap, landuse=orchard) ornek uretimi.

Kaynak: isparta_orchards.json -- Overpass API, way["landuse"="orchard"] in
bbox (37.70, 30.50, 38.30, 31.30): Egirdir--Gelendost (Isparta) elma bolgesi.
Veri (c) OpenStreetMap katkicilari, ODbL.

Her poligon icin:
  * yerel metrik izdusum (esdikdortgen, enlem ortalamasi), birim cm;
  * sira yonu = poligonun en kucuk alanli dikdortgeninin uzun kenari;
  * agac kafesi: sira araligi 400 cm, agac araligi 200 cm, sinirdan 200 cm iceride;
  * %5 eksik agac, sigma = 20 cm konum gurultusu (sentetik bahcelerle ayni);
  * depo: bas tarla ortasi, bir agac araligi disarida (sentetik ile ayni kural).
Secim: agac sayisina gore dort aralik ([300,1000), [1000,3000), [3000,6000),
>= 6000), her araliktan alana gore esit aralikli 4 poligon -> 16 bahce.

Kosum:  python data/osm/make_osm_orchards.py
Cikti:  data/osm/osm_orchards.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "isparta_orchards.json")
OUT = os.path.join(HERE, "osm_orchards.json")
ROW, TREE, INSET = 400.0, 200.0, 200.0
BINS = ((300, 1000), (1000, 3000), (3000, 6000), (6000, 10 ** 9))
PER_BIN = 4


def hull(P):
    P = sorted(map(tuple, P))
    if len(P) < 3:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def row_angle(P):
    """En kucuk alanli kapsayan dikdortgenin uzun kenar acisi (radyan)."""
    H = hull(P)
    best = None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        th = math.atan2(e[1], e[0])
        u = np.array([math.cos(th), math.sin(th)])
        v = np.array([-u[1], u[0]])
        a, b = H @ u, H @ v
        w, h = np.ptp(a), np.ptp(b)
        if best is None or w * h < best[0]:
            best = (w * h, th if w >= h else th + math.pi / 2)
    return best[1]


def seg_dist(Q, A, B):
    """Q noktalarinin A->B kenarlarina en kucuk uzakligi."""
    d = np.full(len(Q), np.inf)
    for a, b in zip(A, B):
        ab = b - a
        L2 = ab @ ab
        t = np.clip(((Q - a) @ ab) / (L2 if L2 > 0 else 1.0), 0, 1)
        proj = a + t[:, None] * ab
        d = np.minimum(d, np.hypot(*(Q - proj).T))
    return d


def lattice(poly, seed):
    rng = np.random.default_rng(seed)
    th = row_angle(poly)
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-u[1], u[0]])
    c = poly.mean(0)
    pu, pv = (poly - c) @ u, (poly - c) @ v
    us = np.arange(pu.min() + INSET, pu.max() - INSET + 1e-9, TREE)
    vs = np.arange(pv.min() + INSET, pv.max() - INSET + 1e-9, ROW)
    G = np.array([(c + a * u + b * v) for b in vs for a in us]) if len(us) and len(vs) else np.zeros((0, 2))
    if not len(G):
        return None
    inside = Path(poly).contains_points(G)
    G = G[inside]
    if not len(G):
        return None
    G = G[seg_dist(G, poly, np.roll(poly, -1, axis=0)) >= INSET - 1e-6]
    keep = rng.random(len(G)) >= 0.05
    G = G[keep] + rng.normal(0, 20.0, (keep.sum(), 2))
    gu = (G - c) @ u
    gv = (G - c) @ v
    d = c + u * (gu.min() - TREE) + v * 0.5 * (gv.min() + gv.max())
    return G, d, math.degrees(th)


def main():
    els = json.load(open(SRC))["elements"]
    lat0 = float(np.mean([p["lat"] for e in els for p in e["geometry"]]))
    kx = 111320.0 * math.cos(math.radians(lat0)) * 100.0
    ky = 110540.0 * 100.0
    cands = []
    for e in els:
        g = e["geometry"]
        if len(g) < 4:
            continue
        P = np.array([(p["lon"] * kx, p["lat"] * ky) for p in g])
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        P = P - P.mean(0)
        area = 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1)))
        res = lattice(P, seed=int(e["id"]) % (1 << 30))
        if res is None:
            continue
        G, d, phi = res
        cands.append(dict(id=int(e["id"]), area_m2=area / 1e4, n=len(G), G=G, d=d, phi=phi, poly=P))
    sel = []
    for lo, hi in BINS:
        b = sorted([c for c in cands if lo <= c["n"] < hi], key=lambda c: c["n"])
        if not b:
            continue
        idx = np.unique(np.linspace(0, len(b) - 1, min(PER_BIN, len(b))).round().astype(int))
        sel += [b[i] for i in idx]
    out = []
    for c in sel:
        out.append(dict(name=f"osm{c['id']}", way_id=c["id"], n=c["n"], area_ha=round(c["area_m2"] / 1e4, 3),
                        phi=round(c["phi"], 2), X=np.round(c["G"][:, 0], 1).tolist(), Y=np.round(c["G"][:, 1], 1).tolist(),
                        dx=round(float(c["d"][0]), 1), dy=round(float(c["d"][1]), 1),
                        poly=np.round(c["poly"], 1).tolist()))
        print(f"osm{c['id']}: n={c['n']}, {c['area_m2'] / 1e4:.2f} ha, rows at {c['phi']:.0f} deg")
    json.dump(dict(source="OpenStreetMap contributors (ODbL), Overpass API, landuse=orchard, "
                          "bbox 37.70,30.50,38.30,31.30 (Egirdir-Gelendost, Isparta, Turkiye)",
                   lat0=lat0, instances=out), open(OUT, "w"))
    print(len(out), "bahce ->", OUT, "| aday:", len(cands))


if __name__ == "__main__":
    main()
