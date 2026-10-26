# -*- coding: utf-8 -*-
"""Makale 2 ornek kumesi: bahce (sentetik, makale 1 ureteci), VLSI, mTSPLib.

Her ornek: dict(name, group, X, Y, dx, dy) -- (dx, dy) depo.
Bahce deposu: sahanin uzun eksenine gore kisa kenarin (bas tarlanin)
ortasi, bir agac araligi disarida (robotlar siralara bas tarladan girer).
VLSI deposu: sinir kutusunun sol-alt kosesi (makine baslangic konumu).
mTSPLib: 1 numarali dugum depo (Necula vd. 2015 tanimi).
"""
from __future__ import annotations

import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
ROOT = os.path.dirname(P2)
sys.path.insert(0, os.path.join(ROOT, "paper_experiments"))
sys.path.insert(0, ROOT)


def _read_tsp(path):
    src = open(path, encoding="utf-8", errors="replace").read()
    body = src.split("NODE_COORD_SECTION", 1)[1]
    pts = re.findall(r"^\s*\d+\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$", body, re.M)
    P = np.array([(float(x), float(y)) for x, y in pts])
    return P[:, 0].copy(), P[:, 1].copy()


def mtsplib(name):
    X, Y = _read_tsp(os.path.join(ROOT, "data_tsplib", f"{name}.tsp"))
    return dict(name=name, group="mtsplib", X=X[1:].copy(), Y=Y[1:].copy(), dx=float(X[0]), dy=float(Y[0]))


def vlsi(name):
    X, Y = _read_tsp(os.path.join(ROOT, "data_vlsi", f"{name}.tsp"))
    return dict(name=name, group="vlsi", X=X, Y=Y, dx=float(X.min()), dy=float(Y.min()))


def headland_depot(X, Y, spacing, phi):
    """Sira yonu u = (cos phi, sin phi); siralarin bas ucunun (bas tarla)
    ortasi, `spacing` disarida."""
    P = np.column_stack([X, Y])
    c = P.mean(0)
    u = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    v = np.array([-u[1], u[0]])
    pu = (P - c) @ u
    pv = (P - c) @ v
    d = c + u * (pu.min() - spacing) + v * 0.5 * (pv.min() + pv.max())
    return float(d[0]), float(d[1])


def orchard(rows, cols, a=400, b=200, missing=0.05, sigma=20.0, shape="rect", seed=1):
    import orchard as O
    rng = np.random.default_rng(seed * 7919 + rows * 31 + cols)
    phi = float(rng.uniform(-45, 45))
    o = O.make_orchard(rows, cols, a, b, missing, sigma, shape, phi, seed=int(rng.integers(1 << 30)))
    X = np.array(o["x"])
    Y = np.array(o["y"])
    dx, dy = headland_depot(X, Y, b, phi)
    nm = f"orch{rows}x{cols}_a{a}_m{int(100 * missing)}_s{int(sigma)}_{shape}_{seed}"
    return dict(name=nm, group="orchard", X=X, Y=Y, dx=dx, dy=dy, rows=o["row"], cols=o["col"], phi=phi)


def osm_list():
    """Gercek bahce sinirlarindan uretilmis ornekler (bkz. paper2/data/osm)."""
    import json
    return json.load(open(os.path.join(P2, "data", "osm", "osm_orchards.json")))["instances"]


def osm(name):
    o = next(d for d in osm_list() if d["name"] == name)
    return dict(name=name, group="osm", X=np.array(o["X"], float), Y=np.array(o["Y"], float),
                dx=float(o["dx"]), dy=float(o["dy"]), phi=o["phi"], area_ha=o["area_ha"])


MTSPLIB = ("eil51", "berlin52", "eil76", "rat99")
MTSPLIB_K = (2, 3, 5, 7)


def vlsi_list(max_n=10 ** 9):
    instances = [(10150, "xmc10150"), (8197, "ida8197"), (5557, "fea5557"),
                 (3954, "dkf3954"), (2993, "xva2993"), (1973, "dkd1973"),
                 (1488, "fra1488"), (984, "pbd984")]
    return [(n, nm) for n, nm in instances if n <= max_n]
