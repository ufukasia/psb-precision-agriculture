#!/usr/bin/env python3
"""Download the CC0 Poly Haven assets used by build_scene.py.

Usage:  python3 fetch_assets.py            (downloads into ./assets/polyhaven/)
All assets are CC0 (https://polyhaven.com/license); see assets/SOURCES.md.
Only the public Poly Haven API is used (no login).
"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "assets", "polyhaven")
API = "https://api.polyhaven.com"

# (asset id, kind, resolution)
MODELS = [
    ("tree_small_02", "2k"),     # broadleaf tree, used as orchard tree base
    ("food_apple_01", "2k"),     # photoscanned apple
    ("plastic_crate_02", "1k"),  # harvest bins at the depot
]
TEXTURES = [
    ("leafy_grass", "2k"),       # grass alleys
    ("brown_mud_dry", "2k"),     # herbicide strip under the tree row
    ("bark_brown_02", "1k"),     # trellis posts / extra bark
]
HDRIS = [
    ("kloofendal_48d_partly_cloudy_puresky", "4k"),
]


UA = {"User-Agent": "orchard-figure-fetch/1.0 (python-urllib)"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("  ->", os.path.relpath(dst, HERE))
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, \
            open(dst + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(dst + ".part", dst)


def fetch_model(aid, res):
    d = get_json(f"{API}/files/{aid}")
    b = d["blend"][res]["blend"]
    base = os.path.join(ROOT, "models", aid)
    fetch(b["url"], os.path.join(base, f"{aid}_{res}.blend"))
    for rel, f in b.get("include", {}).items():
        fetch(f["url"], os.path.join(base, rel))


def fetch_texture(aid, res):
    d = get_json(f"{API}/files/{aid}")
    base = os.path.join(ROOT, "textures", aid)
    for m in ("Diffuse", "nor_gl", "Rough", "Displacement", "AO"):
        if m not in d:
            continue
        fmts = d[m].get(res, {})
        fmt = "jpg" if "jpg" in fmts else ("png" if "png" in fmts else "exr")
        if fmt not in fmts:
            continue
        fetch(fmts[fmt]["url"], os.path.join(base, f"{aid}_{m.lower()}_{res}.{fmt}"))


def fetch_hdri(aid, res):
    d = get_json(f"{API}/files/{aid}")
    f = d["hdri"][res]["hdr"]
    fetch(f["url"], os.path.join(ROOT, "hdris", f"{aid}_{res}.hdr"))


def main():
    for aid, res in MODELS:
        print("model", aid, res)
        fetch_model(aid, res)
    for aid, res in TEXTURES:
        print("texture", aid, res)
        fetch_texture(aid, res)
    for aid, res in HDRIS:
        print("hdri", aid, res)
        fetch_hdri(aid, res)
    print("done")


if __name__ == "__main__":
    main()
