from __future__ import annotations

import math
import numpy as np


def make_orchard(rows, cols, a, b, missing=0.0, sigma=0.0, shape="rect", phi=0.0, seed=0):
    rng = np.random.default_rng(seed)
    R, Cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    R = R.ravel()
    Cc = Cc.ravel()
    keep = np.ones(R.size, bool)
    if shape == "L":
        keep &= ~((R >= rows // 2) & (Cc >= cols // 2))
    if missing > 0:
        keep &= rng.random(R.size) >= missing
    R, Cc = R[keep], Cc[keep]
    x = Cc * float(b) + rng.normal(0.0, sigma, R.size)
    y = R * float(a) + rng.normal(0.0, sigma, R.size)
    x -= x.mean()
    y -= y.mean()
    t = math.radians(phi)
    ox, oy = rng.uniform(1e4, 5e4, 2)
    X = ox + x * math.cos(t) - y * math.sin(t)
    Y = oy + x * math.sin(t) + y * math.cos(t)
    return dict(x=X.tolist(), y=Y.tolist(), row=R.tolist(), col=Cc.tolist(), phi=float(phi))
