#!/usr/bin/env python3
"""Render all paper stills (Cycles/OptiX) by calling Blender once per figure.

  python3 render_stills.py                 # all five figures, 2400x1350
  python3 render_stills.py --only overview_small closeup_harvest
  python3 render_stills.py --preview       # 960x540 / 32 samples quick check
  python3 render_stills.py --samples 256

Figures (written to out/):
  fig_overview_small.png   anim_small.json,  oblique aerial, round-0 routes, t = 30 % of round-0 flight
  fig_overview_medium.png  anim_medium.json, same style
  fig_overview_large.png   anim_large.json,  same style
  fig_closeup_harvest.png  drone cutting an apple beside a ripe tree (anim_small.json)
  fig_closeup_detail.png   tree/apple detail with the drone in the background
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "out")

FIGS = {
    "overview_small": ("anim_small.json", "overview", "fig_overview_small.png", []),
    "overview_medium": ("anim_medium.json", "overview", "fig_overview_medium.png", []),
    "overview_large": ("anim_large.json", "overview", "fig_overview_large.png", []),
    "closeup_harvest": ("anim_small.json", "closeup", "fig_closeup_harvest.png", []),
    "closeup_detail": ("anim_small.json", "detail", "fig_closeup_detail.png", []),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(FIGS))
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--res", default="2400x1350")
    ap.add_argument("--blender", default="blender")
    ap.add_argument("--save-blend", action="store_true", help="also write out/<fig>.blend")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in a.only:
        data, shot, png, extra = FIGS[name]
        out = os.path.join(OUT, png if not a.preview else "preview_" + png)
        cmd = [a.blender, "-b", "--python", os.path.join(HERE, "build_scene.py"), "--",
               "--data", os.path.join(DATA, data), "--shot", shot, "--out", out,
               "--res", a.res, "--samples", str(a.samples)] + extra
        if a.preview:
            cmd.append("--preview")
        if a.save_blend:
            cmd += ["--save", os.path.join(OUT, png.replace(".png", ".blend"))]
        print(">>", name, " ".join(cmd), flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in r.stdout.splitlines():
            if "[build" in line or "Error" in line or "Traceback" in line:
                print("  ", line)
        print(f"   -> {out}  ({time.time() - t0:.0f} s, exit {r.returncode})", flush=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
