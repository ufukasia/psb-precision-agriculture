#!/usr/bin/env python3
"""Render the v3 (tethered-drone harvest platform) figures by calling Blender
once per figure with build_scene_v3.py.

  python3 render_v3.py                       # all four figures, 2400x1350, 160 spp -> out/
  python3 render_v3.py --preview             # 960x540 / 32 spp -> out/preview_v3/
  python3 render_v3.py --only platform closeup --samples 256
  python3 render_v3.py --extra --t 9000      # pass extra options through to build_scene_v3.py

Figures:
  fig_v3_overview.png   oblique aerial view of the block, several platforms working
  fig_v3_platform.png   one platform at a stop, four tethered drones picking on both faces
  fig_v3_closeup.png    drone suction gripper at a mixed ripe/unripe canopy face
  fig_v3_station.png    bin station on the headland with a platform returning
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "anim3_syn1ha.json"))
OUT = os.path.join(HERE, "out")
PREVIEW_OUT = os.path.join(OUT, "preview_v3")

FIGS = {
    "overview": ("overview", "fig_v3_overview.png"),
    "platform": ("platform", "fig_v3_platform.png"),
    "closeup": ("closeup", "fig_v3_closeup.png"),
    "station": ("station", "fig_v3_station.png"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(FIGS))
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--res", default="2400x1350")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--blender", default="blender")
    ap.add_argument("--save-blend", action="store_true", help="also write out/<fig>.blend")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra options for build_scene_v3.py")
    a = ap.parse_args()
    outdir = PREVIEW_OUT if a.preview else OUT
    os.makedirs(outdir, exist_ok=True)
    for name in a.only:
        shot, png = FIGS[name]
        out = os.path.join(outdir, png)
        cmd = [a.blender, "-b", "--python", os.path.join(HERE, "build_scene_v3.py"), "--",
               "--data", a.data, "--shot", shot, "--out", out, "--res", a.res, "--samples", str(a.samples)]
        if a.preview:
            cmd.append("--preview")
        if a.save_blend:
            cmd += ["--save", os.path.join(outdir, png.replace(".png", ".blend"))]
        cmd += a.extra
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
