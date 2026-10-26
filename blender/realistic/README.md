# Photorealistic orchard / harvesting-drone figures

Blender (5.2 LTS, Cycles + OptiX) visualisations of the drone-harvesting
simulation in `../anim_small.json`, `../anim_medium.json`, `../anim_large.json`.

## Files

| File | Purpose |
|---|---|
| `fetch_assets.py` | downloads the CC0 Poly Haven assets into `assets/polyhaven/` (public API, no login) |
| `build_scene.py` | builds the whole scene from a json (trees, apples, trellis, depot, drones, route overlay, camera, lighting) and renders one still or an animation |
| `render_stills.py` | wrapper that renders the five paper figures into `out/` |
| `render_anim.py` | wrapper for the keyframed animation (EEVEE by default; Cycles optional) |
| `assets/SOURCES.md` | asset credits and licenses |
| `out/` | rendered figures (`fig_*.png`); `out/preview/` low-res iterations; `out/anim/` test frames |

## How to run

```bash
cd blender/realistic
python3 fetch_assets.py                       # once (about 190 MB); Grass004 from ambientCG is already in assets/ambientcg/
python3 render_stills.py                      # all five figures, 2400x1350, 160 samples (~15 min on an RTX 3090)
python3 render_stills.py --only closeup_harvest --samples 256
python3 render_stills.py --preview            # 960x540, 32 samples, fast iteration
python3 render_anim.py --data ../anim_small.json --test --engine cycles --samples 32   # 5 spread test frames
python3 render_anim.py --data ../anim_small.json --engine cycles --samples 48 --res 1920x1080   # full animation to out/anim/
```

Direct Blender call (what the wrappers run):

```bash
blender -b --python build_scene.py -- --data ../anim_medium.json --shot overview \
        --out out/fig_overview_medium.png --res 2400x1350 --samples 160
# shots: overview | closeup | detail | anim ; other options: --t SECONDS --scenario 0|1 --round R
#        --preview --engine eevee --gpu all --save scene.blend --frames A-B --step S --fps F --speed X
```

Figures:

* `fig_overview_small.png`, `fig_overview_medium.png`, `fig_overview_large.png`
  — oblique aerial view (about 38° down) from the depot side of the whole
  field, scenario 0 (PSB + LNS), round-0 routes of all drones drawn as thin
  coloured tubes at flight height, drone positions at *t* = 30 % of the
  round-0 flight phase marked with a vertical stalk + ring in the drone's colour.
* `fig_closeup_harvest.png` — one drone hovering beside a ripe tree at a recorded
  cutting dwell, cutter touching an apple, two apples mid-fall, fallen apples on
  the herbicide strip, ripe (red) and unripe (green) neighbouring trees.
* `fig_closeup_detail.png` — canopy / apple detail with the same drone in the background.

## What is real data and what is illustration

Taken directly from the json (not invented):

* tree positions (`trees.x/y`), ripeness (`r_true >= tau` → red apples, otherwise green);
* which trees have been harvested at the depicted time (`done_t`, apples removed and shown on the ground);
* drone routes (`drones[j][round]`), drone positions at time *t* (linear interpolation along the recorded path), which drone is cutting at which tree (dwell ≥ 2 s at a fixed point), depot position, number of drones `k`.

Illustration / artistic choices:

* the tree model (Poly Haven `tree_small_02`, two or three interleaved crowns per tree position, rescaled to ~3.3 m high, ~2 m along the row) stands in for a high-density orchard tree; the apple positions on the canopy are random (seeded) placements of a scanned apple on the leaf clusters, the fruit counts (56 ripe / 40 unripe per tree) are illustrative;
* the drone (0.9 m motor-to-motor quadcopter, 0.26 m body, forward cutter arm, LED stripe in the route colour), its hover offset beside the tree (cutter touching the alley-side apple at ~2.3 m), cruise altitude 3.6 m and pitch while flying;
* trellis posts/wires, grass alleys, herbicide strips, headland track, trailer, harvest bins, landing pads, lighting (HDRI + sun), cameras, depth of field;
* the route tubes and position markers are deliberately diagrammatic overlays (emissive, no shadows) because real-scale drones are a few pixels in the overview.

The procedural parts use seeded random numbers, so a given command reproduces the same image.

## Asset credits

All external assets are CC0 (Poly Haven, ambientCG); see `assets/SOURCES.md`
for the author, URL and license of each item. No login-gated or non-commercial
assets are used.

## Measured timings (RTX 3090, OptiX, 2400x1350)

| figure | scene build | render |
|---|---|---|
| fig_overview_small (467 trees, 160 spp) | 7 s | 39 s |
| fig_overview_medium (1,885 trees, 160 spp) | 9 s | 19 s |
| fig_overview_large (9,453 trees, 16 drones, 160 spp) | 29 s | 27 s |
| fig_closeup_harvest / fig_closeup_detail (192 spp, LOD0 trees, DoF) | ~8 s | ~40 s each |
| animation frame, 960x540, Cycles 32 spp | 20 s (once) | 6 s / frame |
| animation frame, 960x540, EEVEE | 20 s (once) | ~32 s / frame (slower than Cycles here because of the alpha-hashed leaf cards) |

Low-res previews (`--preview`: 960x540, 32 spp) take about 5 s to render.

## Notes / limitations

* The orchard tree is a scanned generic small-leaved broadleaf tree, not a true tall-spindle apple tree; interleaving crowns gives the dense "fruiting wall" look but branch architecture (central leader, feathered laterals) and the larger oval apple leaves are not modelled. No CC0/CC-BY apple-tree-with-fruit model downloadable without an account was found (see `assets/SOURCES.md`).
* The drone is procedural (no suitable CC0/CC-BY agricultural drone model without login was found); its proportions follow a ~0.9 m agricultural quadcopter.
* The animation path (`--shot anim`) keyframes drone motion and apple removal per tree; it was only validated on test frames, not rendered in full.
* Memory: trees are linked-data instances (one LOD1 mesh of 376 k polygons, LOD0 1.15 M polygons for near trees in close-ups), so the 9,453-tree field fits comfortably on a 24 GB GPU.

---

# v3: tethered-drone harvest platforms (fresh-market apples, Tall Spindle orchard)

The application scenario changed after the agricultural-literature review:
small electric ground platforms, each carrying one apple bin and a tether hub,
drive along the alleys of a high-density Tall Spindle block; at every stop the
four tethered drones of a platform pick ripe apples on the two canopy faces
bounding the alley and place them into the bin (no fruit on the ground); a full
bin is swapped at the bin station on the headland. The v3 scene is built by
`build_scene_v3.py` from `../anim3_syn1ha.json` (written by
`sim/export_platform.py`, see the docstring of `sim/platform_sim.py`).
The old scripts (`build_scene.py`, `render_stills.py`, `render_anim.py`) are
untouched and still render the old figures.

## Files

| File | Purpose |
|---|---|
| `build_scene_v3.py` | builds the v3 scene from the json and renders one still (`--shot overview | platform | closeup | station | swatch`) |
| `render_v3.py` | wrapper: the four paper figures into `out/` (`--preview` -> `out/preview_v3/`, 960x540, 32 spp) |
| `out/fig_v3_overview.png` | oblique aerial view of the 1 ha block from the south-west corner at a mid-mission time: 12 platforms at pick stops in different alleys, 2 driving, 1 at the station; cultivar blocks (6 rows each) visible by fruit colour on the south canopy faces; every platform marked with a thin ring in its colour (true-scale platforms) |
| `out/fig_v3_platform.png` | one platform at a stop seen from behind/above in the alley: four tethered drones, two per canopy face, suction cups on ripe apples, bin 57 % full |
| `out/fig_v3_closeup.png` | drone suction gripper on a red apple at a mixed ripe / unripe (red-type) canopy face |
| `out/fig_v3_station.png` | bin station on the headland: full and empty bins, tractor with front forks and bin trailer, two platforms waiting for a bin swap, one platform returning with a full bin |
| `out/preview_v3/` | low-res iterations, `fig_v3_swatch.png` = the six apple colour classes |

```bash
cd blender/realistic
python3 render_v3.py --preview                 # quick check of all four figures (about 25 s each)
python3 render_v3.py                           # finals, 2400x1350, 160 spp
python3 render_v3.py --only platform --samples 256 --extra --t 18437 --platform 7
blender -b --python build_scene_v3.py -- --data ../anim3_syn1ha.json --shot overview \
        --out out/fig_v3_overview.png --res 2400x1350 --samples 160 [--t S] [--platform P] [--apple-div N] [--save x.blend]
```

## What is data and what is illustration

Taken from the json (nothing of this is invented):

* tree positions (`trees.x/y`, 0.9 m in-row spacing with the simulator's 10 cm
  jitter and 5 % missing trees, rows 3.3 m apart), cultivar per tree
  (blocks of 6 rows: 0 red-type, 1 yellow-type, 2 green-type), initial ripe and
  unripe fruit counts per tree and per canopy face (`trees.ripe`, `trees.unripe`;
  face 0 = +y, face 1 = -y);
* alleys (`alleys`), the 3.6 m stop cells (`nodes`) and which tree faces border
  each cell (`trees_up`, `trees_lo`);
* pick events (`picks`): the fruit removed at a cell by a platform at a given
  time. The removed ripe (and mistakenly removed unripe) fruit of an event is
  distributed over the faces of that cell in proportion to the fruit still on
  each face (largest-remainder rounding); for the stop in progress at the
  depicted time the elapsed fraction of the event is removed. Removed apples
  disappear from the canopy; the outermost / highest apples of a face are
  removed first;
* platform traces (`platforms`): position by linear interpolation along the
  recorded trace, state `pick` between two consecutive `pick` points at the same
  place, `move` while driving, `station` between trips; heading from the trace;
* bin fill: fruit picked in the current trip / `bin_capacity` (2300 fruit),
  the cultivar mix of the picked fruit sets the colours of the top layer in
  the bin; number of full bins at the station = completed trips (capped at 8
  shown on the ground); `k` = 15 platforms, `drones_per_platform` = 4,
  `station` position.

Illustration / modelling choices:

* **Trees.** The scanned Poly Haven `tree_small_02` (a generic small-leaved
  broadleaf tree) stands in for a Tall Spindle apple tree: two interleaved
  crowns per tree position, rescaled to 3.1 m height and about 1.1 m across the
  row (upper crown 1.0-3.1 m, lower crown 0.5-1.9 m), giving the continuous
  fruiting wall of a 0.9 m planting; a conduit stake per tree and a 3-wire
  trellis (wooden posts every 9 m, wires at 0.9 / 1.8 / 2.7 m). Central-leader
  branch architecture and apple-sized leaves are not modelled. Foliage gets a
  per-tree brightness variation and a modest cultivar tint (green-type lighter
  yellow-green, red-type slightly darker) so that cultivar blocks read from the
  air.
* **Apples.** One rendered apple per fruit of the data (`--apple-div 1`; the
  scanned `food_apple_01`, 7.6 cm). Where an apple hangs on the canopy face is a
  seeded random choice among the leaf clusters of that face, biased to the
  outer canopy; within a face ripe fruit preferentially occupies the outer/upper
  positions but is mixed with unripe fruit (score = 0.5 outer + 0.3 upper +
  0.3 noise). Colours per cultivar type (ripe / unripe): red-type red-striped /
  yellow-green with a slight blush, yellow-type golden yellow / light green,
  green-type bright grass green / darker green (`fig_v3_swatch.png`), i.e. colour
  alone is not ripeness.
* **Platform.** Procedural: 2.5 x 1.2 m, deck 0.45 m, four 0.6 m wheels,
  battery pack, safety-orange rails, an ID stripe in the platform colour, a
  mast with the tether hub at 2.75 m, two fruit-detection camera pods per
  side; a standard 1.2 x 1.2 x 0.75 m wooden bin on the rear deck.
* **Drones.** Procedural quadcopter, 0.70 m motor-to-motor with prop guards,
  stereo camera, forward suction-cup gripper (0.52 m reach, bellows), vacuum
  hose, tether anchor underneath; a 4.5 mm cable with catenary sag from the hub
  to each drone. At a `pick` stop two drones per canopy face hover at 1.55 /
  2.45 m with the cup on the next ripe apple to be picked (nearest to the
  planned position; a face without ripe fruit left gets a scanning hover);
  while driving or at the station the four drones hover in a 2 x 2 formation
  1.95 m above the deck (Tevel-style escort; docking cradles are not modelled).
* **Bin station.** Bins on the ground (full ones with the dominant red-type
  mix, four empties stacked), a compact tractor with front forks and a flat bin
  trailer carrying two full bins, a small cabinet; platforms queued at the
  station are spread 3.4 m apart along the headland lane. All of this is
  illustration placed around `station[x,y]`.
* **Ground.** Mown grass alleys with two wheel tracks per alley (1.0 m track
  width), 1 m herbicide strip under each row, a dirt headland lane 0.5-4.5 m
  around the block and a compacted pad at the station (procedural masks on
  CC0 textures); grass clumps scattered near the camera in the close shots.
* Lighting (HDRI + sun), cameras, depth of field, the platform rings in the
  overview (emissive, no shadows, deliberately diagrammatic).

## Time / viewpoint rules (chosen automatically from the data)

* `overview`: t sampled every 60 s in 25-65 % of the mission; maximise
  10 x min(12, alleys with a platform at a pick stop) + 5 x min(2, platforms
  driving) + 2 if a platform is at the station; earliest wins ties
  (-> t = 11528 s: 12 picking in 12 alleys, 2 driving, 1 at the station).
  Camera: azimuth 42 deg from the south-west corner, 34 deg down, fitted to the
  block + station; sun from behind the camera.
* `platform`: among all pick stops with both canopy faces present and a bin
  fill of 35-75 % at mid-stop, maximise min(initial ripe fruit on the two faces)
  (-> platform 7, cell 711, t = 18437 s, fill 0.57). Camera 10.5 m behind the
  platform in the alley, 4.6 m high.
* `closeup`: among all pick stops, the canopy face whose remaining ripe
  fraction at mid-stop is closest to 0.5 with at least 40 fruit left, red-type
  preferred (-> platform 4, cell 116, south face, t = 3166 s, 76 ripe / 102
  unripe left); the lower drone on that face; camera 1.6 m from the cup, f/4.
* `station`: for every trip end take t = arrival - 9 s (platform about 9 m
  from the station); maximise 2 x min(2, other platforms at the station) +
  min(4, completed trips) (-> platform 0 returning at t = 3970 s with a 81 %
  full bin, 2 platforms at the station, 10 trips completed).

`--t`, `--platform` override the rules; everything procedural is seeded, so a
command reproduces the same image.

## Timings (RTX 3090, OptiX, 2400x1350, 160 spp)

See `out/final_render_v3.log`. Scene build is 10-12 s for every shot (3069
trees x 2 crowns, 3069 stakes, about 280-375 k apple instances via geometry
nodes, 15 platforms with 60 drones); previews (960x540, 32 spp) render in
10-13 s.

| figure | build | render |
|---|---|---|
| fig_v3_overview (323 k apples, 15 platform rings) | 12 s | 25 s |
| fig_v3_platform (LOD0 trees within 14 m, DoF) | 15 s | 31 s |
| fig_v3_closeup (LOD0 trees within 14 m, DoF) | 22 s | 35 s |
| fig_v3_station (LOD0 trees within 16 m) | 9 s | 25 s |

## Remaining realism limitations (v3)

* The tree is a rescaled generic scanned tree: no central leader / feathered
  laterals, small leaves, and the leaf cards are squashed anisotropically
  (0.40 in plan vs 0.68 in height), noticeable in the close-up.
* Apples hang at leaf-cluster centres, not on spurs/branches; no stems tied to
  twigs, no fruit clusters, no bruising or sun-burn; the same scan for all fruit.
* The platform, drones, tractor and trailer are simplified procedural models
  (no suitable CC0 models without a login were found); docking cradles, cable
  winches and the fruit-transfer path from the drone to the bin are not shown;
  while driving the drones simply hover above the deck.
* The bin fill is a solid pile with one instanced top layer; the full bins at
  the station all show the red-type mix.
* No end-post anchors, irrigation lines, wind-break trees, surrounding
  landscape or workers; the ground outside the block is a flat grass plane.
* All fleet members are drawn at the same depicted instant with no motion
  blur; the platform traces are polylines, so headland turns are sharp.

---

# v4: square-grid orchard (free-standing trees, alleys in both directions, k-d regions with end take-over)

Second orchard layout requested for the paper: a chessboard planting of
free-standing central-leader trees at about 4.0 x 4.1 m, no trellis, with the
imperfections of a real block (per-tree position noise, slightly bent rows,
missing trees and dead patches, a notched boundary). The platforms drive
through the alleys in both directions, stop at alley crossings (the gap
between four trees) and the four tethered drones pick the adjacent tree; the
path between two stops is L-shaped along the alleys. The fleet uses
frame-aware k-d regions: each platform owns a compact rectangle of the block
and, when it has finished its own region, takes over the far end of the
most-behind neighbour's region ("end take-over"). The v4 scene is built by
`build_scene_v4.py` from `../anim4_rect1ha.json` (written by
`sim/export_grid.py`, see the docstring of `sim/grid_sim.py`).
The v3 scripts and figures are untouched; v4 reuses the v3 platform, drone,
bin and station models verbatim.

## Files

| File | Purpose |
|---|---|
| `build_scene_v4.py` | builds the v4 scene from the json and renders one still (`--shot overview | topdown | takeover | platform | swatch`) |
| `regions_v4.py` | reconstructs the planner's k-d partition and the take-over picks (see below) and writes `../anim4_rect1ha_regions.json` |
| `render_v4.py` | wrapper: the four paper figures into `out/` (`--preview` -> `out/preview_v4/`, 960x540, 32 spp) |
| `out/fig_v4_overview.png` | oblique aerial view of the whole irregular block from the south-west, mid-mission: all 7 platforms at pick stops inside their k-d regions, thin region outlines on the ground plus a very light ground tint per region, true-scale platform rings |
| `out/fig_v4_topdown.png` | near-top-down view (78 deg) of the square grid with its bent rows, gaps and dead patches, notched boundary, alleys in both directions with wheel tracks, 7 platforms at alley crossings |
| `out/fig_v4_takeover.png` | end take-over: platform 5 (finished region 5) picks at the far north-west end of region 1 while the owner, platform 1, works towards it two rows away in the same column alley (8.7 m apart); the owner's region outline runs along the headland in the foreground |
| `out/fig_v4_platform.png` | one platform at an alley crossing seen from the east-west alley: four tethered drones with suction cups on the ripe apples of the tree south-west of the crossing, bin 55 % full |
| `out/preview_v4/` | low-res iterations (`preview_round*.log` = build logs of the iterations) |
| `out/final_render_v4.log` | build / render log of the final figures |

```bash
cd blender/realistic
python3 regions_v4.py --data ../anim4_rect1ha.json   # once: k-d regions + take-over picks (needs the sim modules: numpy, scipy, numba)
python3 render_v4.py --preview                       # quick check of all four figures (about 20 s each)
python3 render_v4.py                                 # finals, 2400x1350, 160 spp
python3 render_v4.py --only platform --samples 256 --extra --t 14581 --platform 6
blender -b --python build_scene_v4.py -- --data ../anim4_rect1ha.json --shot overview \
        --out out/fig_v4_overview.png --res 2400x1350 --samples 160 [--t S] [--platform P] [--apple-div N] [--region-tint F] [--save x.blend]
```

## What is data and what is illustration

Taken from the json (nothing of this is invented):

* tree positions (`trees.x/y`; the simulator's 4.02 x 4.07 m lattice with
  25 cm per-tree noise, low-frequency row bending, 6 % random missing trees,
  1-3 dead patches with 50 % missing trees, 2 m inset from a notched
  130 x 100 m rectangle; 662 trees on a 35 x 28 cell grid, 42 empty cells
  inside the block), cultivar per tree (blocks of 5 rows: 0 red-type,
  1 yellow-type, 2 green-type), initial ripe and unripe fruit per tree
  (`trees.ripe`, `trees.unripe`, 68,721 / 129,392 in total);
* pick events (`picks`): the ripe (and mistakenly picked unripe) fruit removed
  from a tree by a platform at a given time. Removed apples disappear from the
  crown, outer / upper apples first; for the stop in progress at the depicted
  time the elapsed fraction of the event is removed;
* platform traces (`platforms`): position by linear interpolation along the
  recorded L-shaped trace (station -> along the headland -> up the column
  alley -> crossing), state `pick` between two consecutive `pick` points at
  the same place, `move` while driving, `station` between trips; heading from
  the trace. A pick stop is the alley crossing north-east of the picked tree
  (tree + half a spacing in x and y, as in `export_grid.py`);
* wheel tracks: every trace segment driven by any platform before the depicted
  time is drawn as a pair of tyre tracks (data); the faint tracks along all
  other alleys in both directions are illustration (orchard traffic);
* bin fill: fruit picked in the current trip / `bin_capacity` (2300 fruit),
  the cultivar mix of the picked fruit sets the colours of the top layer in
  the bin; full bins at the station = completed trips (capped at 8 shown);
  `k` = 7 platforms, `drones_per_platform` = 4, `station` position;
* k-d regions and take-over picks: see the next section.

Illustration / modelling choices:

* **Trees.** The scanned Poly Haven `tree_small_02` stands in for a
  free-standing central-leader apple tree: a main crown (0.70 x 0.56 in plan,
  0.84 in height -> 3.8 m) plus a smaller second crown rotated by 165 deg that
  fills the other side (3.1 m), giving a roughly 2.0 x 2.9 m canopy; per tree
  a random yaw (free-standing trees have no row orientation), a scale of
  0.90-1.10 (3.4-4.2 m tall) and a lean of up to 2.5 deg; a procedural bark
  trunk under the crown. With the 4.0 m spacing this leaves alleys of about
  1.1-2.0 m at canopy height. No trellis, no stakes. Foliage gets a per-tree
  brightness variation and a modest cultivar tint as in v3.
* **Apples.** One rendered apple per fruit of the data (`--apple-div 1`,
  N = 1; 198,113 fruit initially, about 150-185 k apple instances at the
  depicted times). The apple slots of a tree are a seeded random choice among
  the leaf clusters of the whole crown, biased to the outer canopy; ripe fruit
  preferentially occupies the outer / upper slots but is mixed with unripe
  fruit (score = 0.5 outer + 0.3 upper + 0.3 noise). Same six colour classes
  as v3 (red-type ripe red-striped / unripe yellow-green; yellow-type golden /
  light green; green-type bright green / darker green).
* **Platform, drones, bin, bin station.** The v3 models unchanged (2.5 x 1.2 m
  platform with mast and tether hub at 2.75 m, 0.70 m quadcopters with suction
  gripper, 1.2 m wooden bin, tractor with forks and bin trailer). At a `pick`
  stop the four drones surround the crown of the tree being picked: four
  planned azimuths (-72, -24, +24, +72 deg from the tree -> platform direction,
  i.e. the half of the crown facing the crossing) at 2.55 / 1.65 / 2.45 /
  1.55 m, each with the cup on the apple nearest to its planned position among
  the next 60 ripe apples in picking order (a tree without ripe fruit left gets
  a scanning hover); a catenary tether
  from the hub to each drone. While driving / at the station the drones hover
  in the 2 x 2 escort formation above the deck. The station layout of v3 is
  rotated by 90 deg so that the headland lane runs along x and the block lies
  north of `station[]`; platforms queued at the station are spread 3.4 m apart
  along the lane.
* **Ground.** Mown grass sward everywhere, a 0.8 m bare-soil disc under every
  tree, wheel tracks (1.0 m track width) along both alley directions following
  the bent rows (midpoints of neighbouring tree pairs), a dirt headland lane
  0.3-5.2 m outside the irregular block boundary (occupied cells, holes of up
  to three cells closed) and a compacted pad at the station. These masks are
  rasterised from the data into two 0.1 m/px images sampled by the ground
  shader (CC0 grass / mud textures), so the lane follows the notch and the
  tracks follow the traces; grass clumps are scattered near the camera in the
  platform shot.
* **Overlay.** Thin emissive rectangles on the ground (z = 0.12 m, 9 cm
  radius) around every k-d region: the cell rectangle of the region's trees,
  i.e. the alley centre lines around it, inset by 0.3 m so that neighbouring
  outlines do not overlap; in the aerial shots the grass inside each region
  is additionally tinted very lightly (12 %) with the region colour
  (`--region-tint`); platform rings (2.3 m radius, true scale) 4.5 m above
  the platforms in the aerial shots and 1.9 m rings at 3.4 m in the take-over
  shot. All overlays are emissive, cast no shadows and use the platform
  colours of v3.
* Lighting (HDRI + sun), cameras, depth of field.

## k-d regions and the take-over rule

`anim4_*.json` stores `region[tree]` = the platform that harvested the tree
first, which by construction never differs from the picker of a pick event,
so take-overs cannot be detected from it. `regions_v4.py` therefore rebuilds
the same orchard (same `--orchard rect1ha --pass pass1 --seed 1` as
`export_grid.py`, checked against the json positions to 2 mm) and calls the
planner's partition itself (`mtsp.gridplan.KDSteal` on the estimated frame,
weights = service-time estimate of every tree whose prior estimate is
>= R_MIN, exactly the first plan of `grid_sim.simulate`). The result,
`../anim4_rect1ha_regions.json`, holds the k-d owner of every tree (region
sizes 140 / 181 / 59 / 75 / 61 / 83 / 63; the regions are disjoint rectangles
in cell space) and the **take-over picks = pick events whose platform is not
the k-d owner of the tree** (26 of 658; `summary.steals` = 2 take-overs:
platform 5 takes 13 trees at the north-west end of region 1 from t = 14,616 s,
platform 3 takes 13 trees at the east end of region 6 from t = 17,102 s).
Without the regions file the script falls back to the first picker and the
take-over shot degrades to an overview.

## Time / viewpoint rules (chosen automatically from the data)

* `overview` and `topdown`: t sampled every 60 s in 25-65 % of the mission;
  maximise 10 x (platforms at a pick stop) + 5 x min(2, platforms driving)
  (+ 2 if one is at the station; weights 3 / 0 for `topdown`); earliest wins
  ties (-> t = 5070 s, all 7 platforms picking). Overview camera: azimuth
  48 deg from the south-west, 36 deg down, fitted to the block + station;
  top-down camera: 78 deg down from the south, fitted to the block.
* `takeover`: for every take-over pick take t_mid of its stop; keep those
  where the owner is itself at a pick stop inside its own region at t_mid;
  minimise the distance between the two platforms (ties: more take-over
  picks already done by the helper) (-> t = 15,263 s: platform 5 picks tree
  538 (red-type) inside region 1, 8.7 m from platform 1). Camera outside the
  block looking across the pair towards the block centre, 30 deg down.
* `platform`: among all pick stops of at least 60 s with a bin fill of
  0.30-0.80 at mid-stop, maximise the ripe fruit still on the tree at
  mid-stop (+15 % for the red-type cultivar) (-> platform 6 at tree 70,
  t = 14,581 s, fill 0.55, 381 ripe fruit left). Camera 8 m east of the
  crossing in the east-west alley, 4.1 m high, 35 mm, f/5.6, looking past the
  platform at the picked tree.

`--t`, `--platform` override the rules; everything procedural is seeded, so a
command reproduces the same image.

## Timings (RTX 3090, OptiX, 2400x1350, 160 spp)

See `out/final_render_v4.log`. Scene build is about 9-13 s for every shot
(2 s tree kit, 6-8 s ground masks, 662 trees x 2 crowns, 150-185 k apple
instances via geometry nodes, 7 platforms with 28 drones); previews (960x540,
32 spp) render in 6-7 s.

| figure | build | render |
|---|---|---|
| fig_v4_overview (186 k apples, 7 rings, 7 outlines, ground tint) | 12 s | 17 s |
| fig_v4_topdown (same scene, 78 deg camera) | 13 s | 18 s |
| fig_v4_takeover (LOD0 trees within 18 m, tethers) | 15 s | 23 s |
| fig_v4_platform (LOD0 trees within 14 m, grass clumps, DoF) | 14 s | 25 s |

## Remaining realism limitations (v4)

* The tree is a rescaled generic scanned tree (small leaves, no true
  central-leader architecture, no feathered laterals); from the air the
  crowns look slightly smaller and darker than real 2.5 m apple canopies, and
  in the close shot the apples look large next to the small leaves.
* Apples hang at leaf-cluster centres, not on spurs/branches; no stems, no
  clusters, no sun-burn; the same scan for all fruit.
* The platform, drones, tractor and trailer are the simplified procedural v3
  models; the fruit-transfer path from drone to bin and docking cradles are
  not shown; the four drones only use the crossing-facing half of the crown,
  the tethers may clip outer branches.
* Dead patches and missing trees are simply absent trees on the sward (no
  stumps, no replants); the lattice offset of bent rows is only approximated
  for the alley centre lines and the region outlines (cell rectangles), so
  an outline can pass through the canopy edge of a strongly bent row.
* No irrigation lines, wind-break trees, surrounding landscape or workers;
  the ground outside the block is a flat grass plane; the wheel tracks of
  the alleys that no platform has driven yet are illustrative.
* All fleet members are drawn at the same instant with no motion blur; the
  platform traces are polylines, so the L-shaped turns are sharp.

# video: explainer / promotional video of the v4 harvest simulation

`out/video/harvest_explainer_tr.mp4` (Turkish, primary) and
`out/video/harvest_explainer_en.mp4` (English): 1920x1080, 30 fps, H.264,
about 95 s. The 3D frames are rendered once without any text
(`video_v4.py`, Cycles / OptiX, two GPUs in parallel); captions, title and
summary cards, the mission clock, the progress bar and the animated map
overlay are added afterwards in 2D (`compose_v4.py`, PIL -> ffmpeg), so both
language versions reuse the same frames. `out/video/draft_tr.mp4` /
`draft_en.mp4` are the quick low-resolution drafts (960x540, 16 spp, every
other frame doubled) used to check story and realism.

## Files

| File | Purpose |
|---|---|
| `make_video.sh` | the whole pipeline: `./make_video.sh draft` (drafts), `./make_video.sh final` (finals), `./make_video.sh compose` (captions only, frames already rendered); two render queues (RTX 3090 / RTX 4060 Ti), Blender with `-t 2` |
| `video_shots_v4.py` | shot table shared by the two scripts below (frame counts, mission windows, speed factors, 2D section lengths) |
| `video_prep_v4.py` | replays the planner's k-d partition cut by cut (same arithmetic as `mtsp.gridplan.kd_regions`, checked against `../anim4_rect1ha_regions.json`), the estimated-frame alley grid, the axis angles, the simulation parameters and the planted area -> `out/video/video_prep.json` (needs the sim modules like `regions_v4.py`) |
| `video_v4.py` | Blender script, one call per shot (or per frame range): builds the scene with `build_scene_v4.py` in library mode (`--shot none`, the only change to that script) and animates everything per frame |
| `compose_v4.py` | 2D compositing + encoding, `--lang tr|en` |
| `out/video/frames/<shot>/frame_NNNN.png`, `meta.json` | 3D frames and the mission time of every frame (plus the map camera mapping, the close-up choreography events, the take-over pair) |
| `out/video/log/` | render logs per shot / frame range |

```bash
cd blender/realistic
python3 video_prep_v4.py --data ../anim4_rect1ha.json          # once (also done by make_video.sh)
./make_video.sh draft                                          # ~40 min on both GPUs -> out/video/draft_tr.mp4, draft_en.mp4
./make_video.sh final                                          # finals -> out/video/harvest_explainer_{tr,en}.mp4
blender -b -t 2 --python video_v4.py -- --shot closeup --frames 100-130 --gpu 3090 --res 1920x1080 --samples 64   # one shot / range
python3 compose_v4.py --lang en --out out/video/harvest_explainer_en.mp4                                        # captions only
```

## Storyboard (95 s)

| # | section | length | 3D frames | mission time | what moves |
|---|---|---|---|---|---|
| 1 | title card + fleet fly-in | 3 + 9 s | intro, 270 | before t = 0 | camera flies from high above the north-east of the block to the bin station; the 7 platforms wait in a line, their 4 drones each lift off (illustration: the data starts at t = 0 with the platforms leaving) |
| 2 | map sharing | 16 s | map, 1 still | t = 0 | 2D overlay on a straight-down still: (a) the estimated-frame alley grid rotates from an 8 deg offset onto the rows and the axis error (`summary.theta_err` = 0.14 deg) is shown; (b) the 6 k-d cuts appear one after another along the alleys in the planner's recursion order, every finished region gets its colour, the platform rings at the station take their region colour, legend with the region sizes |
| 3 | mission time-lapse | 24 s | mission, 720 | 0 -> 2 h, 300x | oblique aerial with a slow orbit; mission clock, progress bar (ripe fruit picked / 68 721 from `picks`); platforms fan out, drive their L-shaped alley paths, stop at crossings (drones move to the crown), apples vanish from the trees, bins fill, full bins are swapped at the station and appear on the ground |
| 4 | close-up | 18 s | closeup, 540 | 14 191 -> 14 209 s, real time | platform 7 at tree 70 (same stop as `fig_v4_platform`): the drones leave the deck when the 20 s setup time (`T_STOP`) of the stop is over, each approaches a ripe apple, the cup touches it, the apple detaches and follows the cup, is carried over the bin and dropped (free fall), the drone returns; one grasp fails and is retried; unripe apples stay |
| 5 | end take-over | 11 s | takeover, 330 | 14 300 -> 17 400 s, 282x | view from the north on the northern halves of regions 2 and 6 (owner platform 2 / helper platform 6, 1-based as in the captions): platform 6 enters the far (north-west) end of region 2 and works east towards platform 2; both regions outlined, both platforms ringed |
| 6 | finish + summary card | 8 + 6 s | finish, 240 | 17 400 s -> end (160x), then hold | last trees, all platforms return to the station, aerial pull-back; summary card from `summary` |

Platforms are numbered 1-7 in the captions (index + 1 of the json).

## What is data and what is illustration (video)

Data (from `../anim4_rect1ha.json`, `../anim4_rect1ha_regions.json`,
`out/video/video_prep.json`):

* tree positions, cultivars, initial fruit; pick events; platform traces
  (position by linear interpolation along the recorded L-shaped traces, as in
  the figures), stop times, trips, bin fill, bins completed at the station;
  k-d regions, take-over picks; `k`, `drones_per_platform`, `bin_capacity`,
  `station`; the summary card (`summary`: mission time, picked / ripe total,
  unripe picked, trips, steals) and the axis error (`summary.theta_err`);
* **apples vanish exactly as the pick events say**: every apple of a tree
  carries the time at which the simulation removed it (the ripe fruit of a
  stop disappears outer / upper first, spread evenly over the stop), and a
  geometry-nodes instancer deletes the apples whose time is before the
  frame's mission time (a driver on the frame);
* the progress bar is the cumulative ripe fruit of the pick events divided by
  the ripe total (so it ends at the recall, 67.7 %), the clock is mission time;
* the k-d cut sequence of section 2 is the planner's own recursion
  (`kd_regions` replayed with the same weights, verified to reproduce the
  regions file), the cut positions lie between the two rows the planner
  split, the grid lines are the alley centre lines of the *estimated* frame
  (0.14 deg off the true rows);
* close-up timing: the drones start picking when the stop's setup time
  (`T_STOP` = 20 s) is over, each drone takes `T_PICK` = 6 s per fruit, the
  caption quotes `T_PICK`, `P_SUCC`, `REC`, `PREC` from `grid_sim.P`; the
  planted area in the captions (1.2 ha) is the number of occupied cells plus
  closed holes (<= 3 cells) times the cell area (1.169 ha; the simulated
  field polygon is 1.244 ha).

Illustration:

* the fly-in before t = 0 (platforms queued 3.4 m apart on the lane, drones
  landed on the front deck and lifting off), all camera paths, the 8 deg
  initial offset of the grid in section 2 (the planner starts from the raw
  tree positions, not from a wrong axis);
* in the time-lapses the four drones of a platform take their planned
  positions around the crown (same rule as the figures, fruit state at the
  stop's start) during the first 20 s of a stop and return to the escort
  formation in its last 5 s; between frames of a 300x time-lapse they do not
  move; the wheel tracks are those of the whole shot window;
* the bin on a platform is a pile whose height follows the fill fraction
  (top layer colours = the cultivar mix of everything that platform picked
  in the mission), it is empty while the platform is at the station;
* the whole close-up choreography (approach, 0.6 s suction grasp with a
  1 cm push, detach with a 0.45 m back-off, carry 0.45 m above the straight
  line, release over the bin with free fall at g, return; the failed grasp of
  drone 2 in its second cycle; the hover bob); the picked apples are the
  next ripe apples of the tree in the simulation's pick order, chosen per
  drone nearest to its planned azimuth / height;
* motion blur (shutter 0.5) in the time-lapses, depth of field in the
  close-up, lighting, the fades, the title / card design.

## Render settings and timings

Finals: 1920x1080, Cycles / OptiX, 64 samples with adaptive sampling
(threshold 0.03) and OptiX denoising, persistent data, motion blur in the
time-lapses, Blender `-t 2`. Drafts: 960x540, 16 samples, every other frame.
Measured (final run, both GPUs in parallel, Blender `-t 2`, CPU shared with
benchmark jobs); the per-frame time includes the per-frame scene sync and the
BVH rebuild of the 150-200 k apple instances (the motion-blur shots evaluate
the scene twice per frame), scene build is 10-25 s per call:

| shot | frames | GPU | s/frame | total |
|---|---|---|---|---|
| mission (motion blur, 200 k apples, rings, outlines) | 720 | RTX 3090 | 9.7 | 116 min |
| closeup 1-470 (LOD0 trees within 14 m, 28 tethers, DoF) | 470 | RTX 3090 | 7.0 | 55 min |
| map (one still) | 1 | RTX 4060 Ti | 12 | 0.4 min |
| intro (no motion blur) | 270 | RTX 4060 Ti | 6.6 | 30 min |
| takeover (motion blur) | 330 | RTX 4060 Ti | 11.4 | 63 min |
| finish (motion blur) | 240 | RTX 4060 Ti | 8.8 | 35 min |
| closeup 471-540 | 70 | RTX 4060 Ti | 10.2 | 12 min |

3D frames: 2 031 frames in 172 min wall (3090 queue 171 min, 4060 Ti queue
141 min); compositing + encoding: about 5 min per language (CPU only,
libx264 crf 18). Draft (`./make_video.sh draft`, 960x540, 16 spp, every
other frame): 37 min for the frames (mission 5.0 s/frame on the 3090, the
per-frame overhead dominates at that size) + 2 x 2 min compositing.

## Limitations (video)

* The time-lapse is a true sampling of the simulation every 10 s of mission
  time (300x): platforms jump between stops (a 4 m move takes 4 s = 0.4
  frame), motion blur turns the longer drives into streaks; drones are a few
  pixels from the air.
* The close-up is an illustration of one stop: the simulation only knows the
  stop duration and the fruit counts, not where each apple hangs or how the
  drones fly; the tethers are straight catenaries and may clip branches; the
  apple is carried on the cup without a visible vacuum effect; the bin
  content does not grow with the 11 apples dropped (it follows the data's
  fill fraction, which changes by < 1 % in 18 s).
* The map section uses a straight-down still: the overlay is drawn in 2D with
  the camera's ground mapping, so lines at the canopy edge are offset by up
  to 1.5 % (tree height / camera height).
* The same scanned tree / apple assets and their limitations as the figures
  (small leaves next to 8 cm apples in the close-up, no spurs / stems).
