#!/usr/bin/env python3
"""Render the harvesting animation (Cycles by default; EEVEE optional) with build_scene.py --shot anim.

  python3 render_anim.py --data ../anim_small.json --frames 1-5            # quick test frames
  python3 render_anim.py --data ../anim_small.json --test                   # 5 spread test frames
  python3 render_anim.py --data ../anim_small.json                          # full animation
  python3 render_anim.py --data ../anim_small.json --engine cycles --samples 64

Timeline: mission seconds are mapped to output frames with --speed mission
seconds per output second (default 4 -> a 182 s mission becomes ~46 s at 24 fps).
Drones fly their recorded routes, descend beside a tree while cutting, apples
disappear from a tree and appear on the ground at its recorded harvest time
(done_t).  Frames are written to out/anim/<name>_####.png; assemble with e.g.
  ffmpeg -framerate 24 -i out/anim/small_%04d.png -c:v libx264 -pix_fmt yuv420p out/anim_small.mp4
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", default="cycles", choices=["eevee", "cycles"])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--speed", type=float, default=4.0, help="mission seconds per output second")
    ap.add_argument("--frames", default=None, help="A-B frame range (default: all)")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--test", action="store_true", help="render 5 frames spread over the mission")
    ap.add_argument("--blender", default="blender")
    a = ap.parse_args()
    name = os.path.basename(a.data).replace("anim_", "").replace(".json", "")
    out_dir = os.path.join(HERE, "out", "anim")
    os.makedirs(out_dir, exist_ok=True)
    base = [a.blender, "-b", "--python", os.path.join(HERE, "build_scene.py"), "--",
            "--data", os.path.abspath(a.data), "--shot", "anim", "--engine", a.engine,
            "--samples", str(a.samples), "--res", a.res, "--fps", str(a.fps), "--speed", str(a.speed),
            "--out", os.path.join(out_dir, name + "_")]
    runs = []
    if a.test:
        import json
        with open(a.data) as f:
            t_end = json.load(f)["scenarios"][0]["t_end"]
        n = int(t_end / a.speed * a.fps) + 1
        for q in range(5):
            fr = 1 + int(q * (n - 1) / 4)
            runs.append(base + ["--frames", f"{fr}-{fr}"])
    else:
        cmd = list(base)
        if a.frames:
            cmd += ["--frames", a.frames]
        cmd += ["--step", str(a.step)]
        runs.append(cmd)
    for cmd in runs:
        print(">>", " ".join(cmd), flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in r.stdout.splitlines():
            if "[build" in line or "Error" in line or "Traceback" in line or "Saved:" in line:
                print("  ", line)
        print(f"   ({time.time() - t0:.0f} s, exit {r.returncode})", flush=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
