# Path–Split–Balance for selective apple harvesting

Reproducibility materials for the manuscript **“Path–Split–Balance: sub-second min–max fleet planning for selective apple harvesting with tethered-drone platforms”** by Ufuk Asil (OSTİM Technical University). The manuscript is under preparation for submission to *Precision Agriculture*.

## Materials

- `paper2/mtsp/`: PSB, PSB+LNS, PSB-MT and PSB-MT/R implementations and external-solver wrappers.
- `paper2/sim/`: selective-harvest and event-based platform simulations.
- `paper2/experiments/`: experiment drivers and analysis scripts.
- `paper2/experiments/out/`: run-level JSON records and generated reports (about 53 MB).
- `paper2/makale_prag_tr/`: manuscript source, figures and generated numerical macros.
- `data_tsplib/`, `data_vlsi/`, `paper2/data/osm/`: benchmark instances used in the reported experiments.
- `paper2/blender/`: simulation export and Blender scene/video-generation scripts.
- `paper2/third_party/sota/SOURCES.txt`: exact sources and revisions of comparison solvers.

**Online Resource 1:** [English explainer video (1080p, 95 s)](https://github.com/ufukasia/psb-precision-agriculture/releases/download/v1.0.0/ESM_1.mp4)

The video is also attached to release **v1.0.0**. Its Blender scene is an illustrative rendering of the simulation, not a physical harvesting trial.

## Environment

Python 3.12 was used for the reported runs. Install the Python dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The core PSB experiments require NumPy, SciPy and Numba only. OR-Tools is needed for the `ortools1` baseline. Matplotlib is needed for figures. Blender 5.2, FFmpeg and Pillow are needed only to regenerate the video. External comparison solvers are not redistributed here; see below.

## Reproduce the reported tables and numerical macros

All existing records are included, so these analysis commands do not repeat the computational experiments:

```bash
python paper2/experiments/report.py
python paper2/experiments/make_numbers.py
python paper2/experiments/make_numbers.py --prag-tr-quality
```

Outputs are written under `paper2/experiments/out/report/` and `paper2/makale_prag_tr/`.

To rebuild the manuscript (XeLaTeX and latexmk required):

```bash
cd paper2/makale_prag_tr
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## Run a small PSB example

```bash
python - <<'PY'
import sys
sys.path[:0] = ['paper2', 'paper2/experiments']
import instances
from mtsp import mm
o = instances.orchard(10, 12, shape='rect', seed=1)
routes, info = mm.solve(o['X'], o['Y'], o['dx'], o['dy'], k=4, workers=1)
print('Trees:', len(o['X']))
print('Initial makespan:', info['mk'])
print('Initial planning time (s):', info['t_total'])
print('Route sizes:', [len(r) for r in routes])
PY
```

The full drivers are `paper2/experiments/bench.py`, `ablation.py`, `harvest_bench.py` and `harvest_grid_bench.py`. They resume from existing JSON records. Running the complete benchmark is computationally expensive and requires the external solvers for non-PSB methods.

## External solvers

The wrappers are included, but third-party programs whose redistribution terms are unclear are excluded. Their upstream URLs, revisions and build/run notes are recorded in `paper2/third_party/sota/SOURCES.txt`.

- **LKH-3 3.0.13:** obtain from Keld Helsgaun’s official LKH-3 distribution and compile it. Set `LKH3_BIN` to the executable.
- **HSNR, MA, MILS and HGA:** obtain them from the authors’ repositories listed in `SOURCES.txt`. The wrapper defaults match that directory layout; `HSNR_BIN`, `MA_BIN`, `MILS_BIN` and `HGA_JULIA_DIR` can override the local paths.
- **OR-Tools:** installed from `requirements.txt`; no bundled copy is needed.

## Data and media attributions

Orchard boundaries in `paper2/data/osm/` are derived from **OpenStreetMap contributors**, obtained through Overpass API, and are subject to **ODbL 1.0**. The source bounding box and OSM way identifiers are preserved in the JSON records.

TSPLIB/mTSPLib and Waterloo VLSI instances retain their upstream terms. TSPLIB: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ ; VLSI: https://www.math.uwaterloo.ca/tsp/vlsi/index.html .

Blender asset authors and CC0 sources are listed in `paper2/blender/realistic/assets/SOURCES.md`. Large downloadable assets and rendered frame caches are not stored in this repository.

## License and citation

Original source code is released under the MIT License (`LICENSE`). This does not replace third-party dataset or solver licenses. The explainer video is released under **CC BY 4.0**. Please cite the manuscript and the software using `CITATION.cff`; archival DOI metadata can be added after a Zenodo release.
