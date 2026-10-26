# -*- coding: utf-8 -*-
"""mmg (genel metrik) ile mm (Oklid'e ozel) Oklid ve w=0'da OZDES cikti verir mi?
Ayrica w > 0'da rota maliyeti = yol + w |rota| tutarliligi."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import instances as I            # noqa: E402
from mtsp import mm, mmg         # noqa: E402


def main():
    ok = True
    for o in (I.mtsplib("rat99"), I.mtsplib("pcb1173"), I.orchard(40, 50)):
        X, Y, dx, dy = o["X"], o["Y"], o["dx"], o["dy"]
        for k in (3, 8):
            a, ia = mm.solve(X, Y, dx, dy, k, workers=1)
            b, ib = mmg.solve(X, Y, dx, dy, k, workers=1)
            same = all(np.array_equal(p, q) for p, q in zip(a, b)) and ia["mk"] == ib["mk"]
            ok &= same
            w = 250.0
            M = mmg.Metric(len(X), w=w)
            c, ic = mmg.solve(X, Y, dx, dy, k, workers=1, M=M)
            mmg.check_solution(c, len(X))
            cons = all(abs(mmg.route_cost(r, X, Y, dx, dy, M) - (mm.route_cost(r, X, Y, dx, dy) + w * len(r))) < 1e-6
                       for r in c)
            ok &= cons
            print(o["name"][:20], k, "ozdes" if same else "FARKLI", "agirlik-tutarli" if cons else "TUTARSIZ")
    print("TAMAM" if ok else "HATA")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
