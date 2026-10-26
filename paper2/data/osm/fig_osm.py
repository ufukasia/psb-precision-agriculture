# -*- coding: utf-8 -*-
"""Gercek bahce ornekleri: OSM sinirlari, agac kafesi ve depo (makale sekli)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "makale", "figs", "fig_osm.pdf")
d = json.load(open(os.path.join(HERE, "osm_orchards.json")))["instances"]
pick = [d[i] for i in (0, 4, 6, 9, 10, 12, 14, 15)]
fig, axs = plt.subplots(2, 4, figsize=(11, 5.6))
for ax, o in zip(axs.flat, pick):
    P = np.array(o["poly"]) / 100
    ax.fill(P[:, 0], P[:, 1], color="#d9ead3", ec="#6aa84f", lw=0.8)
    ax.plot(np.array(o["X"]) / 100, np.array(o["Y"]) / 100, ".", ms=0.6 if o["n"] > 3000 else 1.2, color="#274e13")
    ax.plot(o["dx"] / 100, o["dy"] / 100, "*", ms=9, color="#cc0000")
    ax.set_aspect("equal")
    ax.set_title(f"n = {o['n']}, {o['area_ha']:.1f} ha", fontsize=9)
    ax.tick_params(labelsize=7)
for ax in axs[1]:
    ax.set_xlabel("m", fontsize=8)
fig.tight_layout()
fig.savefig(OUT)
print(OUT)
