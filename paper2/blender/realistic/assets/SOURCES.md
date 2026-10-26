# Asset sources and licenses

All assets below are released under **CC0 1.0 (public domain dedication)** and were
downloaded without any account/login via the public download endpoints.
Nothing in this folder requires attribution, but the sources are listed here for
transparency and reproducibility (`fetch_assets.py` re-downloads the Poly Haven
items; the ambientCG set was fetched with the URL given below).

| Asset | Type | Author | Source URL | License | Used for |
|---|---|---|---|---|---|
| Tree Small 02 (`tree_small_02`) | 3D model (.blend, 2k textures, baked LOD0/LOD1 meshes) | Rico Cilliers (Poly Haven) | https://polyhaven.com/a/tree_small_02 | CC0 | orchard tree (two rotated instances per tree position, rescaled to ~3.3 m) |
| Food Apple 01 (`food_apple_01`) | 3D model (photogrammetry, .blend, 2k textures) | Oliver Harries (Poly Haven) | https://polyhaven.com/a/food_apple_01 | CC0 | apples on the canopy, fallen/falling apples (decimated, rescaled to 7.6 cm; unripe = hue-shifted material) |
| Plastic Crate 02 (`plastic_crate_02`) | 3D model (.blend, 1k textures) | Poly Haven | https://polyhaven.com/a/plastic_crate_02 | CC0 | harvest bins on the depot trailer |
| Brown Mud Dry (`brown_mud_dry`) | PBR texture set, 2k | Poly Haven | https://polyhaven.com/a/brown_mud_dry | CC0 | herbicide strip under the tree rows, headland track |
| Bark Brown 02 (`bark_brown_02`) | PBR texture set, 1k | Poly Haven | https://polyhaven.com/a/bark_brown_02 | CC0 | wooden trellis posts |
| Leafy Grass (`leafy_grass`) | PBR texture set, 2k | Poly Haven | https://polyhaven.com/a/leafy_grass | CC0 | downloaded, currently unused (too dry/tan for mown alleys) |
| Kloofendal 48d Partly Cloudy (Pure Sky) | HDRI, 4k | Greg Zaal (Poly Haven) | https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky | CC0 | sky lighting / background |
| Grass004 | PBR texture set, 2K JPG | ambientCG (Lennart Demes) | https://ambientcg.com/view?id=Grass004 (file: https://ambientcg.com/get?file=Grass004_2K-JPG.zip) | CC0 | mown grass alleys |

Poly Haven license statement: https://polyhaven.com/license  
ambientCG license statement: https://docs.ambientcg.com/license/

Everything else in the figures (drone model, cutter arm, trellis wires, trailer,
landing pads, route overlays) is generated procedurally by `build_scene.py`.

## Options considered but not used

* Sketchfab "Drone For agriculture (Low Poly)" by Bharat Nagar, CC BY 4.0,
  https://sketchfab.com/3d-models/drone-for-agriculturelow-poly-b6220a9a99d349d5aa64c8e7edef3758
  — downloading requires a Sketchfab account, so it was not used.
* Venturon "Apple Tree GLB FBX" (itch.io, CC0) — flat-colour low-poly mesh without
  textures, not realistic enough.
