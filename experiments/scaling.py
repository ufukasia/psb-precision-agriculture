# -*- coding: utf-8 -*-
"""PSB tek atis: cekirdek sayisina gore hizlanma ve n'ye gore olcekleme.

Kosum:  python experiments/scaling.py
Cikti:  experiments/out/extra/scaling.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import instances as I        # noqa: E402
from mtsp import mm          # noqa: E402

OUT = os.path.join(HERE, "out", "extra", "scaling.json")
WORKERS = (1, 2, 4, 8, 16)
REPS = 5


def main():
    w = I.mtsplib("eil51")
    mm.solve(w["X"], w["Y"], w["dx"], w["dy"], 3, workers=2)
    cases = [("orchard", f"{r}x{c}", I.orchard(r, c)) for r, c in ((20, 25), (40, 50), (60, 80), (80, 125))]
    byn = I.vlsi_list()
    for tg in (1000, 3000, 10000):
        n, nm = min(byn, key=lambda t: abs(t[0] - tg))
        cases.append(("vlsi", nm, I.vlsi(nm)))
    res = []
    for grp, nm, o in cases:
        X, Y, dx, dy = o["X"], o["Y"], o["dx"], o["dy"]
        for k in (8, 16):
            for wk in WORKERS:
                ts, mks = [], []
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    r, info = mm.solve(X, Y, dx, dy, k, workers=wk)
                    ts.append(time.perf_counter() - t0)
                    mks.append(info["mk"])
                rec = dict(group=grp, name=nm, n=len(X), k=k, workers=wk, t_med=statistics.median(ts),
                           t_all=ts, mk=mks[0], mk_same=len(set(round(m, 6) for m in mks)) == 1)
                res.append(rec)
                print(grp, nm, len(X), k, wk, f"{rec['t_med']*1000:.1f} ms", f"mk={rec['mk']:.1f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
