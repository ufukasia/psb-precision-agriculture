# Path–Split–Balance

Clean source code and reproducibility materials for **“Path–Split–Balance: sub-second min–max fleet planning for selective apple harvesting with tethered-drone platforms”** by Ufuk Asil, prepared for *Precision Agriculture*.

## Repository layout

- `mtsp/`: PSB, PSB+LNS, PSB-MT and PSB-MT/R; external-solver wrappers.
- `sim/`: selective-harvest and event-based platform simulations.
- `experiments/`: benchmark drivers, instance loaders and analysis scripts.
- `data/`: benchmark instances and orchard data used by the experiments.
- `blender/`: scene and video-generation scripts; no render caches or large assets.
- `third_party/sota/SOURCES.txt`: comparison-solver sources and revisions.

The main branch intentionally contains no manuscript build files, raw-result clutter, virtual environments, solver binaries or rendered frame caches.

## Quick start

Python 3.12 was used for the reported experiments.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python - <<'PY'
import sys
sys.path[:0] = ['.', 'experiments']
import instances
from mtsp import mm
o = instances.orchard(10, 12, shape='rect', seed=1)
routes, info = mm.solve(o['X'], o['Y'], o['dx'], o['dy'], k=4, workers=1)
print('Trees:', len(o['X']))
print('Initial makespan:', info['mk'])
print('Route sizes:', [len(r) for r in routes])
PY
```

The full experiment drivers are `experiments/bench.py`, `experiments/ablation.py`, `experiments/harvest_bench.py` and `experiments/harvest_grid_bench.py`. They resume from existing JSON records. Repeating all experiments is computationally expensive; non-PSB methods require their external solvers.

## Archived results and video

- [Run-level records and generated reports — ZIP](https://github.com/ufukasia/psb-precision-agriculture/releases/download/v1.0.1/psb-reproducibility-data-v1.0.0.zip)
- [Online Resource 1 — English explainer video, 1080p, 95 s](https://github.com/ufukasia/psb-precision-agriculture/releases/download/v1.0.0/ESM_1.mp4)

To regenerate all reported tables and numerical macros without repeating the experiments, extract the ZIP in the repository root. It creates `experiments/out/`. Then run:

```bash
python experiments/report.py
python experiments/make_numbers.py
python experiments/make_numbers.py --prag-tr-quality
```

Generated LaTeX tables and macros are written to `generated/`. The video is an illustrative Blender rendering of a simulation, not a physical harvesting trial.

## External solvers

Third-party programs with unclear redistribution terms are not bundled. Their upstream URLs, revisions and build/run notes are in `third_party/sota/SOURCES.txt`.

- **LKH-3 3.0.13:** install from Keld Helsgaun’s official distribution and set `LKH3_BIN`.
- **HSNR, MA, MILS and HGA:** obtain from the listed author repositories. Defaults follow the documented directory layout; `HSNR_BIN`, `MA_BIN`, `MILS_BIN` and `HGA_JULIA_DIR` override local paths.
- **OR-Tools:** installed through `requirements.txt`.

## Attribution and license

Orchard boundaries are derived from **OpenStreetMap contributors** through Overpass API and remain under **ODbL 1.0**. Source bounding boxes and way identifiers are preserved in `data/osm/`. TSPLIB/mTSPLib and Waterloo VLSI instances retain their upstream terms. Blender asset authors and CC0 sources are listed in `blender/realistic/assets/SOURCES.md`.

Original code is MIT licensed (`LICENSE`); the video is CC BY 4.0. Third-party datasets, solvers and assets retain their own licenses. Citation metadata is provided in `CITATION.cff`.
