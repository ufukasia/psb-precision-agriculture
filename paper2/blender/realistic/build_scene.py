"""Build a photorealistic apple-orchard harvesting scene from an anim_*.json file.

Run with Blender (5.x) in background mode, e.g.

  blender -b --python build_scene.py -- --data ../anim_small.json --shot overview \
          --out out/fig_overview_small.png --res 2400x1350 --samples 160

Everything that is *data* (tree positions, ripeness, routes, drone positions,
harvest times) comes from the json.  Everything that is *illustration*
(tree model, apple placement on the canopy, drone model, trellis, depot) is
generated here from CC0 Poly Haven assets (see assets/SOURCES.md).

Options (after "--"):
  --data PATH          anim json (required)
  --scenario IDX       scenario index in the json (default 0 = PSB+LNS)
  --shot NAME          overview | closeup | detail | anim
  --t SECONDS          mission time to depict (default: 30 % into round-0 flight)
  --round R            which route round to overlay in the overview (default 0)
  --out PATH           output png (stills) or frame prefix (anim)
  --res WxH            resolution (default 2400x1350)
  --samples N          Cycles samples (default 160)
  --preview            960x540, 32 samples, LOD1 trees everywhere
  --engine cycles|eevee
  --gpu 3090|all       which OptiX device(s) to use (default 3090 only)
  --save PATH          also save the .blend
  --frames A-B --step S  (anim shot) frame range / step
  --fps F              (anim shot) frames per second of the output (default 24)
  --speed X            (anim shot) mission seconds per output second (default 4)
"""
import json
import math
import os
import random
import sys
import time

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets", "polyhaven")

# --------------------------------------------------------------------------- args
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opt = {
        "data": None, "scenario": 0, "shot": "overview", "t": None, "round": 0,
        "out": None, "res": "2400x1350", "samples": 160, "preview": False,
        "engine": "cycles", "gpu": "3090", "save": None, "frames": None,
        "step": 1, "fps": 24, "speed": 4.0, "seed": 7, "drone": None,
        "pitch": 90.0, "height": 17.0, "lens": 35.0,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        key = a[2:]
        if key in ("preview",):
            opt[key] = True
            i += 1
            continue
        opt[key] = argv[i + 1]
        i += 2
    opt["scenario"] = int(opt["scenario"])
    opt["round"] = int(opt["round"])
    opt["samples"] = int(opt["samples"])
    opt["step"] = int(opt["step"])
    opt["fps"] = int(opt["fps"])
    opt["speed"] = float(opt["speed"])
    opt["seed"] = int(opt["seed"])
    if opt["t"] is not None:
        opt["t"] = float(opt["t"])
    if opt["drone"] is not None:
        opt["drone"] = int(opt["drone"])
    for key in ("pitch", "height", "lens"):
        opt[key] = float(opt[key])
    if opt["preview"]:
        opt["res"] = "960x540"
        opt["samples"] = 32
    return opt


OPT = parse_args()
random.seed(OPT["seed"])
np.random.seed(OPT["seed"])
T0 = time.time()


def log(*a):
    print(f"[build {time.time() - T0:6.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------- data
with open(OPT["data"]) as f:
    DATA = json.load(f)
TREES = DATA["trees"]
NT = len(TREES["x"])
TAU = DATA["tau"]
K = DATA["k"]
DEPOT = DATA["depot"]
S = DATA["scenarios"][OPT["scenario"]]
XS = np.array(TREES["x"])
YS = np.array(TREES["y"])
RT = np.array(TREES["r_true"])
RIPE = RT >= TAU
FIELD = (XS.min(), XS.max(), YS.min(), YS.max())
FIELD_W = FIELD[1] - FIELD[0]
FIELD_H = FIELD[3] - FIELD[2]
FIELD_SIZE = max(FIELD_W, FIELD_H)

# mission time to depict
fly0 = [p for p in S["phases"] if p["kind"] == "fly" and p["round"] == 0][0]
if OPT["t"] is None:
    OPT["t"] = fly0["t0"] + 0.30 * (fly0["t1"] - fly0["t0"])
T_NOW = OPT["t"]
log(f"data {os.path.basename(OPT['data'])}: {NT} trees, k={K}, ripe={int(RIPE.sum())}, "
    f"scenario '{S['label']}', t={T_NOW:.1f}s of {S['t_end']:.1f}s")

# row structure (rows run along x, 4 m spacing in y)
ROW_PITCH = 4.0
row_ids = np.round(YS / ROW_PITCH).astype(int)
ROW_OFFSET = float(np.mean(YS - row_ids * ROW_PITCH))
ROWS = sorted(set(row_ids.tolist()))

# geometry constants (metres)
FLY_Z = 3.6          # cruise altitude (just above canopy)
CUT_Z = 2.3          # hover height while cutting
CUT_MIN = 2.0        # dwell >= this -> cutting stop
ARM_REACH = 0.62     # drone centre -> cutter tip
CANOPY_OUT = 0.18    # extra stand-off: interleaved crowns extend beyond the apple cloud
TREE_H = 3.3
CANOPY_HALF_Y = 0.85  # across-row canopy half width after scaling

DRONE_COLORS = [
    (0.90, 0.20, 0.15), (0.15, 0.45, 0.95), (0.95, 0.65, 0.10), (0.20, 0.75, 0.35),
    (0.80, 0.25, 0.85), (0.10, 0.80, 0.85), (0.95, 0.40, 0.55), (0.55, 0.55, 0.20),
    (0.95, 0.30, 0.00), (0.30, 0.30, 0.95), (0.60, 0.85, 0.10), (0.85, 0.10, 0.45),
    (0.15, 0.65, 0.60), (0.75, 0.50, 0.25), (0.45, 0.20, 0.80), (0.90, 0.80, 0.30),
]


# ------------------------------------------------------------------- route helpers
_CUT_CACHE = {}


def cut_events(paths):
    """[(t_arrive, t_leave, x, y)] cutting dwells (>= CUT_MIN s at a fixed point)."""
    key = id(paths)
    if key in _CUT_CACHE:
        return _CUT_CACHE[key]
    ev = []
    for path in paths:
        for (t0, x0, y0), (t1, x1, y1) in zip(path[:-1], path[1:]):
            if abs(x1 - x0) + abs(y1 - y0) < 1e-9 and t1 - t0 >= CUT_MIN:
                ev.append((t0, t1, x0, y0))
    _CUT_CACHE[key] = ev
    return ev


def pos_at(paths, t):
    for path in paths:
        if path[0][0] <= t <= path[-1][0]:
            for (t0, x0, y0), (t1, x1, y1) in zip(path[:-1], path[1:]):
                if t0 <= t <= t1:
                    u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                    return x0 + u * (x1 - x0), y0 + u * (y1 - y0), True
    return None, None, False


def heading(paths, t):
    x0, y0, f0 = pos_at(paths, t - 0.2)
    x1, y1, f1 = pos_at(paths, t + 0.4)
    if not (f0 and f1) or (abs(x1 - x0) + abs(y1 - y0)) < 1e-6:
        return None
    return math.atan2(y1 - y0, x1 - x0)


def nearest_tree(x, y):
    d = (XS - x) ** 2 + (YS - y) ** 2
    return int(np.argmin(d))


def drone_pose(j, t):
    """Return dict(x,y,z,yaw,pitch,state,tree) for drone j at mission time t.
    state: 'pad' (on landing pad), 'fly', 'cut' (hovering beside a tree)."""
    paths = S["drones"][j]
    for (ta, tl, x, y) in cut_events(paths):
        if ta <= t <= tl:
            u = min(1.0, (t - ta) / 1.0, (tl - t) / 1.0)      # descend / climb blend
            tree = nearest_tree(x, y)
            ap = apple_target(tree)
            # hover on the alley side (-y) of the tree, cutter pointing +y
            hx, hy, hz = ap[0], ap[1] - CANOPY_OUT - ARM_REACH, ap[2] + 0.02
            return dict(x=hx * u + x * (1 - u), y=hy * u + y * (1 - u), z=FLY_Z + (hz - FLY_Z) * u,
                        yaw=math.pi / 2, pitch=0.0, state="cut", tree=tree, u=u)
    x, y, fl = pos_at(paths, t)
    if not fl:
        px, py = pad_position(j)
        return dict(x=px, y=py, z=0.0, yaw=0.0, pitch=0.0, state="pad", tree=None, u=0.0)
    hd = heading(paths, t)
    return dict(x=x, y=y, z=FLY_Z, yaw=(hd if hd is not None else 0.0), pitch=math.radians(-7),
                state="fly", tree=None, u=0.0)


def pad_position(j):
    cols = 2
    r, c = divmod(j, cols)
    n_rows = (K + cols - 1) // cols
    px = DEPOT[0] + 0.5 - c * 1.6           # pads centred on the json depot point
    py = DEPOT[1] + (r - (n_rows - 1) / 2) * 1.6
    return px, py


# ------------------------------------------------------------------- blender setup
def clean_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.images,
                 bpy.data.lights, bpy.data.cameras, bpy.data.node_groups):
        for b in list(coll):
            if b.users == 0:
                coll.remove(b)


clean_scene()
SC = bpy.context.scene
W, H = [int(v) for v in OPT["res"].lower().split("x")]
SC.render.resolution_x, SC.render.resolution_y = W, H
SC.render.resolution_percentage = 100
SC.render.film_transparent = False
SC.view_settings.view_transform = "AgX"
SC.view_settings.look = "AgX - Medium High Contrast"
SC.view_settings.exposure = 0.0
SC.render.image_settings.file_format = "PNG"
SC.render.image_settings.color_depth = "8"
SC.render.image_settings.compression = 60

ENGINE = OPT["engine"].lower()
if ENGINE == "cycles":
    SC.render.engine = "CYCLES"
    cyc = SC.cycles
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        want = d.type == "OPTIX" and (OPT["gpu"] == "all" or "3090" in d.name)
        d.use = want
        log("device", d.name, d.type, "use" if want else "-")
    cyc.device = "GPU"
    cyc.samples = OPT["samples"]
    cyc.use_adaptive_sampling = True
    cyc.adaptive_threshold = 0.02
    cyc.use_denoising = True
    cyc.denoiser = "OPTIX"
    cyc.denoising_use_gpu = True
    cyc.max_bounces = 6
    cyc.diffuse_bounces = 3
    cyc.glossy_bounces = 3
    cyc.transmission_bounces = 2
    cyc.transparent_max_bounces = 16
    cyc.caustics_reflective = False
    cyc.caustics_refractive = False
    cyc.blur_glossy = 1.0
    cyc.use_light_tree = True
else:
    try:
        SC.render.engine = "BLENDER_EEVEE"
    except TypeError:
        SC.render.engine = "BLENDER_EEVEE_NEXT"
    ee = SC.eevee
    ee.taa_render_samples = max(16, min(64, OPT["samples"]))
    ee.use_shadows = True
    ee.shadow_ray_count = 2
    ee.shadow_step_count = 4
    ee.use_raytracing = True
    ee.use_volumetric_shadows = False
    ee.shadow_resolution_scale = 1.0

COLL_ORCHARD = bpy.data.collections.new("orchard")
COLL_INFRA = bpy.data.collections.new("infrastructure")
COLL_DRONES = bpy.data.collections.new("drones")
COLL_OVERLAY = bpy.data.collections.new("overlay")
for c in (COLL_ORCHARD, COLL_INFRA, COLL_DRONES, COLL_OVERLAY):
    SC.collection.children.link(c)


def new_obj(name, mesh, coll, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), parent=None):
    o = bpy.data.objects.new(name, mesh)
    o.location = loc
    o.rotation_euler = rot
    o.scale = scale
    if parent is not None:
        o.parent = parent
    coll.objects.link(o)
    return o


# ------------------------------------------------------------------- materials
def tex_node(nodes, path, colorspace="sRGB", loc=(-600, 0)):
    img = bpy.data.images.get(os.path.basename(path))
    if img is None:
        img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = colorspace
    n = nodes.new("ShaderNodeTexImage")
    n.image = img
    n.location = loc
    n.interpolation = "Smart"
    return n


def pbr_material(name, folder, prefix, res, scale=1.0, rough_mult=1.0, bump=0.3,
                 tint=None, hue_sat=None):
    """Principled material from a Poly Haven texture set with object-space mapping."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0 / scale, 1.0 / scale, 1.0 / scale)
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    base = os.path.join(folder, f"{prefix}")
    d = tex_node(nodes, base + f"_diffuse_{res}.jpg", "sRGB", (-600, 300))
    n = tex_node(nodes, base + f"_nor_gl_{res}.jpg", "Non-Color", (-600, -300))
    r = tex_node(nodes, base + f"_rough_{res}.jpg", "Non-Color", (-600, 0))
    for t in (d, n, r):
        links.new(mapping.outputs["Vector"], t.inputs["Vector"])
    col_out = d.outputs["Color"]
    if hue_sat is not None:
        hs = nodes.new("ShaderNodeHueSaturation")
        hs.inputs["Hue"].default_value = hue_sat[0]
        hs.inputs["Saturation"].default_value = hue_sat[1]
        hs.inputs["Value"].default_value = hue_sat[2]
        links.new(col_out, hs.inputs["Color"])
        col_out = hs.outputs["Color"]
    if tint is not None:
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs[7].default_value = (*tint, 1.0)
        links.new(col_out, mix.inputs[6])
        col_out = mix.outputs[2]
    links.new(col_out, bsdf.inputs["Base Color"])
    rm = nodes.new("ShaderNodeMath")
    rm.operation = "MULTIPLY"
    rm.inputs[1].default_value = rough_mult
    links.new(r.outputs["Color"], rm.inputs[0])
    links.new(rm.outputs[0], bsdf.inputs["Roughness"])
    nm = nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = bump
    links.new(n.outputs["Color"], nm.inputs["Color"])
    links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def simple_material(name, rgb, rough=0.5, metal=0.0, emit=0.0, alpha=1.0, spec=0.5, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Coat Weight"].default_value = coat
    if emit > 0:
        b.inputs["Emission Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        m.surface_render_method = "BLENDED"
        m.use_transparency_overlap = True
    return m


def emission_material(name, rgb, strength=4.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


# ------------------------------------------------------------------- ground
def build_ground():
    """Large plane: mown grass alleys, bare herbicide strip under each tree row,
    compacted dirt headland near the depot."""
    pad = 4.0 * FIELD_SIZE + 800.0
    x0, x1 = FIELD[0] - pad, FIELD[1] + pad
    y0, y1 = FIELD[2] - pad, FIELD[3] + pad
    me = bpy.data.meshes.new("ground")
    me.from_pydata([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], [], [(0, 1, 2, 3)])
    me.update()
    ob = new_obj("ground", me, COLL_INFRA)

    grass_dir = os.path.join(HERE, "assets", "ambientcg", "Grass004")
    mud_dir = os.path.join(ASSETS, "textures", "brown_mud_dry")
    m = bpy.data.materials.new("ground_mat")
    m.use_nodes = True
    nt = m.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]
    coord = nodes.new("ShaderNodeTexCoord")

    def texset(folder, prefix, res, scale, loc_y, acg=False):
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1.0 / scale,) * 3
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        base = os.path.join(folder, prefix)
        if acg:   # ambientCG naming
            d = tex_node(nodes, base + f"_{res}-JPG_Color.jpg", "sRGB", (-900, loc_y))
            n = tex_node(nodes, base + f"_{res}-JPG_NormalGL.jpg", "Non-Color", (-900, loc_y - 300))
            r = tex_node(nodes, base + f"_{res}-JPG_Roughness.jpg", "Non-Color", (-900, loc_y - 600))
        else:     # Poly Haven naming
            d = tex_node(nodes, base + f"_diffuse_{res}.jpg", "sRGB", (-900, loc_y))
            n = tex_node(nodes, base + f"_nor_gl_{res}.jpg", "Non-Color", (-900, loc_y - 300))
            r = tex_node(nodes, base + f"_rough_{res}.jpg", "Non-Color", (-900, loc_y - 600))
        for t in (d, n, r):
            links.new(mapping.outputs["Vector"], t.inputs["Vector"])
        return d, n, r

    gd, gn, gr = texset(grass_dir, "Grass004", "2K", 2.5, 900, acg=True)
    gd2, _, _ = texset(grass_dir, "Grass004", "2K", 6.1, 0, acg=True)   # second scale to break tiling
    md, mn, mr = texset(mud_dir, "brown_mud_dry", "2k", 2.2, -900)

    # grass colour: blend two scales, darken slightly, add large-scale variation
    gmix = nodes.new("ShaderNodeMix")
    gmix.data_type = "RGBA"
    gmix.inputs["Factor"].default_value = 0.45
    links.new(gd.outputs["Color"], gmix.inputs[6])
    links.new(gd2.outputs["Color"], gmix.inputs[7])
    var = nodes.new("ShaderNodeTexNoise")
    var.inputs["Scale"].default_value = 0.035
    var.inputs["Detail"].default_value = 3.0
    links.new(coord.outputs["Object"], var.inputs["Vector"])
    vramp = nodes.new("ShaderNodeValToRGB")
    vramp.color_ramp.elements[0].position = 0.35
    vramp.color_ramp.elements[0].color = (0.55, 0.55, 0.40, 1)
    vramp.color_ramp.elements[1].position = 0.65
    vramp.color_ramp.elements[1].color = (0.90, 0.90, 0.82, 1)
    links.new(var.outputs["Fac"], vramp.inputs["Fac"])
    gcol = nodes.new("ShaderNodeMix")
    gcol.data_type = "RGBA"
    gcol.blend_type = "MULTIPLY"
    gcol.inputs["Factor"].default_value = 1.0
    links.new(gmix.outputs[2], gcol.inputs[6])
    links.new(vramp.outputs["Color"], gcol.inputs[7])

    # strip mask: distance to nearest row centre line
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["Object"], sep.inputs[0])
    sub = nodes.new("ShaderNodeMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = ROW_OFFSET
    links.new(sep.outputs["Y"], sub.inputs[0])
    wrap = nodes.new("ShaderNodeMath")
    wrap.operation = "WRAP"
    wrap.inputs[1].default_value = -ROW_PITCH / 2
    wrap.inputs[2].default_value = ROW_PITCH / 2
    links.new(sub.outputs[0], wrap.inputs[0])
    ab = nodes.new("ShaderNodeMath")
    ab.operation = "ABSOLUTE"
    links.new(wrap.outputs[0], ab.inputs[0])
    edge = nodes.new("ShaderNodeTexNoise")
    edge.inputs["Scale"].default_value = 0.6
    edge.inputs["Detail"].default_value = 4.0
    links.new(coord.outputs["Object"], edge.inputs["Vector"])
    edge_s = nodes.new("ShaderNodeMath")
    edge_s.operation = "MULTIPLY_ADD"
    edge_s.inputs[1].default_value = 0.35
    edge_s.inputs[2].default_value = -0.175
    links.new(edge.outputs["Fac"], edge_s.inputs[0])
    dist = nodes.new("ShaderNodeMath")
    dist.operation = "ADD"
    links.new(ab.outputs[0], dist.inputs[0])
    links.new(edge_s.outputs[0], dist.inputs[1])
    strip = nodes.new("ShaderNodeMapRange")
    strip.inputs["From Min"].default_value = 0.50
    strip.inputs["From Max"].default_value = 0.68
    strip.inputs["To Min"].default_value = 1.0
    strip.inputs["To Max"].default_value = 0.0
    links.new(dist.outputs[0], strip.inputs["Value"])
    # inside field extent in x only (plus headland dirt near the depot)
    inx = nodes.new("ShaderNodeMapRange")
    inx.inputs["From Min"].default_value = FIELD[0] - 1.0
    inx.inputs["From Max"].default_value = FIELD[0] + 0.5
    links.new(sep.outputs["X"], inx.inputs["Value"])
    inx2 = nodes.new("ShaderNodeMapRange")
    inx2.inputs["From Min"].default_value = FIELD[1] + 1.0
    inx2.inputs["From Max"].default_value = FIELD[1] - 0.5
    links.new(sep.outputs["X"], inx2.inputs["Value"])
    iny = nodes.new("ShaderNodeMapRange")
    iny.inputs["From Min"].default_value = FIELD[2] - 2.5
    iny.inputs["From Max"].default_value = FIELD[2] - 1.0
    links.new(sep.outputs["Y"], iny.inputs["Value"])
    iny2 = nodes.new("ShaderNodeMapRange")
    iny2.inputs["From Min"].default_value = FIELD[3] + 2.5
    iny2.inputs["From Max"].default_value = FIELD[3] + 1.0
    links.new(sep.outputs["Y"], iny2.inputs["Value"])
    m1 = nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
    links.new(strip.outputs[0], m1.inputs[0]); links.new(inx.outputs[0], m1.inputs[1])
    m2 = nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY"
    links.new(m1.outputs[0], m2.inputs[0]); links.new(inx2.outputs[0], m2.inputs[1])
    m3 = nodes.new("ShaderNodeMath"); m3.operation = "MULTIPLY"
    links.new(m2.outputs[0], m3.inputs[0]); links.new(iny.outputs[0], m3.inputs[1])
    m4 = nodes.new("ShaderNodeMath"); m4.operation = "MULTIPLY"
    links.new(m3.outputs[0], m4.inputs[0]); links.new(iny2.outputs[0], m4.inputs[1])
    # headland: tractor track in front of the depot (x < field xmin - 1.5)
    head = nodes.new("ShaderNodeMapRange")
    head.inputs["From Min"].default_value = FIELD[0] - 1.5
    head.inputs["From Max"].default_value = FIELD[0] - 3.0
    links.new(sep.outputs["X"], head.inputs["Value"])
    head2 = nodes.new("ShaderNodeMapRange")
    head2.inputs["From Min"].default_value = FIELD[0] - 9.5
    head2.inputs["From Max"].default_value = FIELD[0] - 7.0
    links.new(sep.outputs["X"], head2.inputs["Value"])
    hn = nodes.new("ShaderNodeTexNoise")
    hn.inputs["Scale"].default_value = 0.25
    links.new(coord.outputs["Object"], hn.inputs["Vector"])
    hm0 = nodes.new("ShaderNodeMath"); hm0.operation = "MULTIPLY"
    links.new(head.outputs[0], hm0.inputs[0]); links.new(head2.outputs[0], hm0.inputs[1])
    hm = nodes.new("ShaderNodeMath"); hm.operation = "MULTIPLY_ADD"
    hm.inputs[2].default_value = 0.25
    links.new(hm0.outputs[0], hm.inputs[0]); links.new(hn.outputs["Fac"], hm.inputs[1])
    hr = nodes.new("ShaderNodeMapRange")
    hr.inputs["From Min"].default_value = 0.30
    hr.inputs["From Max"].default_value = 0.55
    links.new(hm.outputs[0], hr.inputs["Value"])
    mask = nodes.new("ShaderNodeMath"); mask.operation = "MAXIMUM"
    links.new(m4.outputs[0], mask.inputs[0]); links.new(hr.outputs[0], mask.inputs[1])

    def mixer(a, b, fac, kind="RGBA"):
        mx = nodes.new("ShaderNodeMix")
        mx.data_type = kind
        links.new(fac, mx.inputs["Factor"])
        if kind == "RGBA":
            links.new(a, mx.inputs[6]); links.new(b, mx.inputs[7])
            return mx.outputs[2]
        links.new(a, mx.inputs[2]); links.new(b, mx.inputs[3])
        return mx.outputs[0]

    mud_tint = nodes.new("ShaderNodeMix")
    mud_tint.data_type = "RGBA"; mud_tint.blend_type = "MULTIPLY"
    mud_tint.inputs["Factor"].default_value = 1.0
    mud_tint.inputs[7].default_value = (0.85, 0.78, 0.68, 1)
    links.new(md.outputs["Color"], mud_tint.inputs[6])
    links.new(mixer(gcol.outputs[2], mud_tint.outputs[2], mask.outputs[0]), bsdf.inputs["Base Color"])
    rough = mixer(gr.outputs["Color"], mr.outputs["Color"], mask.outputs[0], "FLOAT")
    rmul = nodes.new("ShaderNodeMath"); rmul.operation = "MULTIPLY"; rmul.inputs[1].default_value = 1.15
    links.new(rough, rmul.inputs[0])
    links.new(rmul.outputs[0], bsdf.inputs["Roughness"])
    nmix = mixer(gn.outputs["Color"], mn.outputs["Color"], mask.outputs[0])
    nm = nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = 0.8
    links.new(nmix, nm.inputs["Color"])
    links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    ob.data.materials.append(m)
    return ob


# ------------------------------------------------------------------- grass clumps (close-ups)
def grass_clump_mesh(rng):
    """A small clump of bent grass blades (thin quads), ~10 cm tall."""
    vs, fs = [], []
    nb = 9
    for b in range(nb):
        a = rng.uniform(0, 6.283)
        h = rng.uniform(0.07, 0.15)
        w = rng.uniform(0.004, 0.007)
        bend = rng.uniform(0.25, 0.6)
        r0 = rng.uniform(0.0, 0.03)
        dx, dy = math.cos(a), math.sin(a)
        px, py = -dy * w, dx * w
        base = len(vs)
        nseg = 3
        for k in range(nseg + 1):
            u = k / nseg
            z = h * u
            off = bend * h * u * u
            ww = 1.0 - 0.8 * u
            cx, cy = r0 * dx + off * dx, r0 * dy + off * dy
            vs.append((cx - px * ww, cy - py * ww, z))
            vs.append((cx + px * ww, cy + py * ww, z))
        for k in range(nseg):
            i = base + 2 * k
            fs.append((i, i + 1, i + 3, i + 2))
    me = bpy.data.meshes.new("grass_clump")
    me.from_pydata(vs, [], fs)
    me.update()
    return me


def build_grass_scatter(center, radius=18.0, density=42.0):
    """Scatter grass clumps on a disc around `center` (close-up shots only)."""
    rng = random.Random(5)
    mat = bpy.data.materials.new("grass_blade")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.16, 0.30, 0.06, 1.0)
    b.inputs["Roughness"].default_value = 0.65
    b.inputs["Specular IOR Level"].default_value = 0.3
    b.inputs["Subsurface Weight"].default_value = 0.25
    b.inputs["Subsurface Radius"].default_value = (0.02, 0.05, 0.01)
    b.inputs["Subsurface Scale"].default_value = 0.01
    # colour variation per clump via object info random
    nt = mat.node_tree
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.12, 0.26, 0.05, 1)
    ramp.color_ramp.elements[1].color = (0.30, 0.42, 0.10, 1)
    nt.links.new(oi.outputs["Random"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    mat.use_backface_culling = False
    clumps = []
    for q in range(3):
        me = grass_clump_mesh(rng)
        me.materials.append(mat)
        o = new_obj(f"grass_clump_{q}", me, COLL_INFRA, (0, 0, -5.0))   # source, hidden below ground
        o.hide_render = True
        clumps.append(o)
    coll = bpy.data.collections.new("grass_clumps")
    COLL_INFRA.children.link(coll)
    for o in clumps:
        coll.objects.link(o)
    # scatter surface: disc
    n = 48
    vs = [(0, 0, 0)] + [(radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n), 0) for k in range(n)]
    fs = [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)]
    me = bpy.data.meshes.new("grass_disc")
    me.from_pydata(vs, [], fs)
    me.update()
    disc = new_obj("grass_scatter", me, COLL_INFRA, (center[0], center[1], 0.003))
    ng = bpy.data.node_groups.new("grass_scatter", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    dist = N.new("GeometryNodeDistributePointsOnFaces")
    dist.distribute_method = "RANDOM"
    # density falls off with distance from the disc centre (world position)
    pos = N.new("GeometryNodeInputPosition")
    dcen = N.new("ShaderNodeVectorMath"); dcen.operation = "DISTANCE"
    dcen.inputs[1].default_value = (center[0], center[1], 0.0)
    ng.links.new(pos.outputs[0], dcen.inputs[0])
    fall = N.new("ShaderNodeMapRange")
    fall.inputs["From Min"].default_value = radius * 0.35
    fall.inputs["From Max"].default_value = radius
    fall.inputs["To Min"].default_value = density
    fall.inputs["To Max"].default_value = 0.0
    ng.links.new(dcen.outputs["Value"], fall.inputs["Value"])
    ng.links.new(fall.outputs[0], dist.inputs["Density"])
    # no grass on the herbicide strip under each row
    sep = N.new("ShaderNodeSeparateXYZ")
    ng.links.new(pos.outputs[0], sep.inputs[0])
    sub = N.new("ShaderNodeMath"); sub.operation = "SUBTRACT"; sub.inputs[1].default_value = ROW_OFFSET
    ng.links.new(sep.outputs["Y"], sub.inputs[0])
    wrap = N.new("ShaderNodeMath"); wrap.operation = "WRAP"
    wrap.inputs[1].default_value = -ROW_PITCH / 2; wrap.inputs[2].default_value = ROW_PITCH / 2
    ng.links.new(sub.outputs[0], wrap.inputs[0])
    ab = N.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"
    ng.links.new(wrap.outputs[0], ab.inputs[0])
    noise = N.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 0.6
    ng.links.new(pos.outputs[0], noise.inputs["Vector"])
    nm = N.new("ShaderNodeMath"); nm.operation = "MULTIPLY_ADD"
    nm.inputs[1].default_value = 0.3; nm.inputs[2].default_value = -0.15
    ng.links.new(noise.outputs["Fac"], nm.inputs[0])
    add = N.new("ShaderNodeMath"); add.operation = "ADD"
    ng.links.new(ab.outputs[0], add.inputs[0]); ng.links.new(nm.outputs[0], add.inputs[1])
    cmp = N.new("FunctionNodeCompare"); cmp.operation = "GREATER_THAN"; cmp.inputs[1].default_value = 0.62
    ng.links.new(add.outputs[0], cmp.inputs[0])
    ng.links.new(cmp.outputs[0], dist.inputs["Selection"])
    ng.links.new(gi.outputs[0], dist.inputs["Mesh"])
    iop = N.new("GeometryNodeInstanceOnPoints")
    ci = N.new("GeometryNodeCollectionInfo")
    ci.inputs[0].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    ng.links.new(ci.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    rid = N.new("FunctionNodeRandomValue"); rid.data_type = "INT"
    rid.inputs["Min"].default_value = 0; rid.inputs["Max"].default_value = 2
    ng.links.new(rid.outputs["Value"], iop.inputs["Instance Index"])
    rr = N.new("FunctionNodeRandomValue"); rr.data_type = "FLOAT_VECTOR"
    rr.inputs["Min"].default_value = (-0.15, -0.15, 0.0); rr.inputs["Max"].default_value = (0.15, 0.15, 6.283)
    ng.links.new(rr.outputs["Value"], iop.inputs["Rotation"])
    rs = N.new("FunctionNodeRandomValue"); rs.data_type = "FLOAT"
    rs.inputs["Min"].default_value = 0.7; rs.inputs["Max"].default_value = 1.5
    ng.links.new(rs.outputs["Value"], iop.inputs["Scale"])
    ng.links.new(dist.outputs["Points"], iop.inputs["Points"])
    ng.links.new(iop.outputs["Instances"], go.inputs[0])
    mod = disc.modifiers.new("scatter", "NODES")
    mod.node_group = ng
    log(f"grass scatter: disc r={radius} m, density {density}/m2")
    return disc


# ------------------------------------------------------------------- tree assets
def append_from(blend, datablock, names):
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        setattr(dst, datablock, [n for n in names if n in getattr(src, datablock)])
    return getattr(dst, datablock)


def fix_image_paths(folder):
    for im in bpy.data.images:
        p = bpy.path.abspath(im.filepath)
        if not os.path.exists(p):
            cand = os.path.join(folder, "textures", os.path.basename(im.filepath))
            if os.path.exists(cand):
                im.filepath = cand
            else:
                log("WARNING missing image", im.filepath)


def load_tree_mesh(lod):
    folder = os.path.join(ASSETS, "models", "tree_small_02")
    blend = os.path.join(folder, "tree_small_02_2k.blend")
    name = f"tree_small_02_LOD{lod}"
    if name in bpy.data.meshes:
        return bpy.data.meshes[name]
    objs = append_from(blend, "objects", [name])
    fix_image_paths(folder)
    me = objs[0].data
    me.name = name
    bpy.data.objects.remove(objs[0], do_unlink=True)
    # recentre: trunk base at origin
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    zmin = co[:, 2].min()
    base = co[co[:, 2] < zmin + 0.15]
    cx, cy = base[:, 0].mean(), base[:, 1].mean()
    co[:, 0] -= cx
    co[:, 1] -= cy
    co[:, 2] -= zmin
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    # leaf material tweaks: a touch of translucency comes with the asset; keep
    for m in me.materials:
        m.use_backface_culling = False
    log(f"tree mesh LOD{lod}: {len(me.polygons)} polys, height {co[:,2].max():.2f} m")
    return me


def load_apple_mesh():
    folder = os.path.join(ASSETS, "models", "food_apple_01")
    blend = os.path.join(folder, "food_apple_01_2k.blend")
    objs = append_from(blend, "objects", ["food_apple_01"])
    fix_image_paths(folder)
    ob = objs[0]
    SC.collection.objects.link(ob)
    # decimate (7k tris -> ~1.4k) keeping UVs, apply modifiers via depsgraph
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.ratio = 0.2
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    me.name = "apple_base"
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(ob.data.materials[0])
    bpy.data.objects.remove(ob, do_unlink=True)
    # centre + scale to ~7.6 cm diameter
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co -= (co.max(0) + co.min(0)) / 2
    diam = (co.max(0) - co.min(0))[:2].mean()
    co *= 0.076 / diam
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    log(f"apple mesh: {len(me.polygons)} polys, diameter {(co.max(0)-co.min(0))[0]*100:.1f} cm")
    return me


def apple_materials(base_mat):
    """red (ripe, scanned texture) / green (unripe, hue shifted) apple materials."""
    red = base_mat.copy()
    red.name = "apple_red"
    nt = red.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Coat Weight"].default_value = 0.35
    b.inputs["Coat Roughness"].default_value = 0.15
    b.inputs["Subsurface Weight"].default_value = 0.08
    b.inputs["Subsurface Radius"].default_value = (0.03, 0.01, 0.005)
    b.inputs["Subsurface Scale"].default_value = 0.02
    green = red.copy()
    green.name = "apple_green"
    nt = green.node_tree
    b = nt.nodes["Principled BSDF"]
    img = [n for n in nt.nodes if n.type == "TEX_IMAGE" and "diff" in n.image.name][0]
    hs = nt.nodes.new("ShaderNodeHueSaturation")
    hs.inputs["Hue"].default_value = 0.70       # red -> yellow-green
    hs.inputs["Saturation"].default_value = 0.70
    hs.inputs["Value"].default_value = 0.95
    nt.links.new(img.outputs["Color"], hs.inputs["Color"])
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"; tint.blend_type = "MULTIPLY"
    tint.inputs["Factor"].default_value = 1.0
    tint.inputs[7].default_value = (0.62, 0.85, 0.30, 1)
    nt.links.new(hs.outputs["Color"], tint.inputs[6])
    nt.links.new(tint.outputs[2], b.inputs["Base Color"])
    b.inputs["Subsurface Radius"].default_value = (0.01, 0.03, 0.005)
    return red, green


def mesh_from_instances(name, base, mats, material):
    """Join copies of `base` mesh transformed by each 4x4 matrix in `mats`."""
    nv = len(base.vertices)
    co = np.empty(nv * 3)
    base.vertices.foreach_get("co", co)
    co = np.c_[co.reshape(-1, 3), np.ones(nv)]
    nl = len(base.loops)
    lv = np.empty(nl, dtype=np.int32)
    base.loops.foreach_get("vertex_index", lv)
    npoly = len(base.polygons)
    ls = np.empty(npoly, dtype=np.int32)
    lt = np.empty(npoly, dtype=np.int32)
    base.polygons.foreach_get("loop_start", ls)
    base.polygons.foreach_get("loop_total", lt)
    uv = np.empty(nl * 2)
    base.uv_layers[0].data.foreach_get("uv", uv)
    n = len(mats)
    all_co = np.concatenate([(co @ np.array(M).T)[:, :3] for M in mats])
    all_lv = np.concatenate([lv + i * nv for i in range(n)])
    all_ls = np.concatenate([ls + i * nl for i in range(n)])
    all_lt = np.tile(lt, n)
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv * n)
    me.vertices.foreach_set("co", all_co.ravel())
    me.loops.add(nl * n)
    me.loops.foreach_set("vertex_index", all_lv)
    me.polygons.add(npoly * n)
    me.polygons.foreach_set("loop_start", all_ls)
    me.polygons.foreach_set("loop_total", all_lt)
    me.polygons.foreach_set("use_smooth", np.ones(npoly * n, dtype=bool))
    me.update(calc_edges=True)
    me.uv_layers.new(name="UVMap")
    me.uv_layers[0].data.foreach_set("uv", np.tile(uv, n))
    me.materials.append(material)
    me.validate()
    return me


def leaf_points(tree_me):
    """Centres of leaf polygons (material slot 'leaves') in tree-local coords."""
    li = [i for i, m in enumerate(tree_me.materials) if "leaves" in m.name][0]
    npoly = len(tree_me.polygons)
    mi = np.empty(npoly, dtype=np.int32)
    tree_me.polygons.foreach_get("material_index", mi)
    cen = np.empty(npoly * 3)
    tree_me.polygons.foreach_get("center", cen)
    cen = cen.reshape(-1, 3)
    return cen[mi == li]


def apple_matrices(points, n, rng, hang=True):
    """Pick n apple positions near the outer canopy, return 4x4 matrices."""
    # bias towards the outside of the canopy: radial distance from canopy axis
    c = points.mean(0)
    r = np.linalg.norm(points[:, :2] - c[:2], axis=1)
    w = (r / (r.max() + 1e-6)) ** 1.5 + 0.08
    w *= (points[:, 2] > 0.6)                     # no fruit on the lowest 60 cm
    w /= w.sum()
    idx = rng.choice(len(points), size=n, replace=False, p=w)
    mats = []
    for i in idx:
        p = points[i].copy()
        out = p - c
        out[2] *= 0.3
        nrm = np.linalg.norm(out) + 1e-6
        p += out / nrm * rng.uniform(-0.02, 0.04)  # at / just inside the canopy surface
        p[2] -= 0.04                                # hanging below the leaves
        s = rng.uniform(0.92, 1.08)
        rot = Euler((rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0, 6.283)), "XYZ")
        M = Matrix.Translation(Vector(p)) @ rot.to_matrix().to_4x4() @ Matrix.Scale(s, 4)
        mats.append(M)
    return mats


def fallen_matrices(n, rng, radius=0.9):
    mats = []
    for _ in range(n):
        a = rng.uniform(0, 6.283)
        rr = radius * math.sqrt(rng.uniform(0.05, 1.0))
        p = Vector((rr * math.cos(a) * 1.3, rr * math.sin(a) * 0.7, 0.036))
        rot = Euler((rng.uniform(0, 6.283), rng.uniform(0, 6.283), rng.uniform(0, 6.283)), "XYZ")
        mats.append(Matrix.Translation(p) @ rot.to_matrix().to_4x4() @ Matrix.Scale(rng.uniform(0.92, 1.08), 4))
    return mats


class TreeKit:
    """Tree variants (mesh + baked apple clouds) built once and instanced."""

    def __init__(self, lod_far=1, lod_near=0):
        self.tree_far = load_tree_mesh(lod_far)
        self.tree_near = load_tree_mesh(lod_near) if lod_near != lod_far else self.tree_far
        self.apple = load_apple_mesh()
        self.mat_red, self.mat_green = apple_materials(self.apple.materials[0])
        self.apple.materials[0] = self.mat_red
        pts_far = leaf_points(self.tree_far)
        # variant transforms (local tree frame -> world-aligned frame at origin)
        # scanned tree is widest along its local y; rotate so the wide axis
        # follows the row (x), then squash across-row and shorten to ~3.3 m
        self.variants = []
        rng = np.random.default_rng(11)
        for v in range(4):
            rz = math.pi / 2 + (math.pi if v % 2 else 0.0)
            mirror = -1.0 if v >= 2 else 1.0
            sx, sy, sz = 0.60 * mirror, 0.72, TREE_H / 4.52
            M = Matrix.Rotation(rz, 4, "Z") @ Matrix.Diagonal((sx, sy, sz, 1.0))
            pts_w = (np.c_[pts_far, np.ones(len(pts_far))] @ np.array(M).T)[:, :3]
            n_red, n_green = 90, 50
            red = mesh_from_instances(f"apples_red_{v}", self.apple,
                                      apple_matrices(pts_w, n_red, rng), self.mat_red)
            green = mesh_from_instances(f"apples_green_{v}", self.apple,
                                        apple_matrices(pts_w, n_green, rng), self.mat_green)
            fallen = mesh_from_instances(f"apples_fallen_{v}", self.apple,
                                         fallen_matrices(n_red - 6, rng), self.mat_red)
            self.variants.append(dict(M=M, red=red, green=green, fallen=fallen, pts=pts_w))
        log(f"tree kit ready: {len(self.variants)} variants")

    def place(self, i, state, near=False, jitter=None):
        """state: 'ripe' | 'unripe' | 'done' | 'harvesting'. Returns (tree, apples, fallen)."""
        v = self.variants[i % 4]
        rng = random.Random(i * 7919 + 3)
        yaw = rng.uniform(-0.12, 0.12) if jitter is None else jitter
        sc = rng.uniform(0.93, 1.06)
        loc = (XS[i], YS[i], 0.0)
        tree = new_obj(f"tree_{i}", self.tree_near if near else self.tree_far, COLL_ORCHARD, loc)
        tree.matrix_world = Matrix.Translation(Vector(loc)) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(sc, 4) @ v["M"]
        # second, rotated copy of the scanned tree at the same spot: the two
        # crowns interleave into the dense canopy of a high-density orchard
        v2 = self.variants[(i + 2) % 4]
        tree2 = new_obj(f"tree_{i}b", self.tree_near if near else self.tree_far, COLL_ORCHARD, loc)
        tree2.matrix_world = Matrix.Translation(Vector((loc[0] + 0.15, loc[1], 0.0))) @ Matrix.Rotation(yaw + math.pi, 4, "Z") @ Matrix.Scale(sc * 0.96, 4) @ v2["M"]
        if near:
            v3 = self.variants[(i + 1) % 4]
            tree3 = new_obj(f"tree_{i}c", self.tree_near, COLL_ORCHARD, loc)
            tree3.matrix_world = Matrix.Translation(Vector((loc[0] - 0.45, loc[1] + 0.05, 0.0))) @ Matrix.Rotation(yaw + math.pi + 0.2, 4, "Z") @ Matrix.Scale(sc * 0.90, 4) @ v3["M"]
        apples = fallen = None
        R = Matrix.Translation(Vector(loc)) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(sc, 4)
        if state in ("ripe", "harvesting"):
            apples = new_obj(f"apples_{i}", v["red"], COLL_ORCHARD)
            apples.matrix_world = R
        elif state == "unripe":
            apples = new_obj(f"apples_{i}", v["green"], COLL_ORCHARD)
            apples.matrix_world = R
        if state in ("done", "harvesting"):
            fallen = new_obj(f"fallen_{i}", v["fallen"], COLL_ORCHARD)
            fallen.matrix_world = Matrix.Translation(Vector(loc)) @ Matrix.Rotation(yaw, 4, "Z")
        return tree, apples, fallen


def canopy_apples_world(i):
    """Centres of the (red) canopy apples of tree i in world coordinates."""
    v = KIT.variants[i % 4]
    rng = random.Random(i * 7919 + 3)
    yaw = rng.uniform(-0.12, 0.12)
    sc = rng.uniform(0.93, 1.06)
    R = Matrix.Translation(Vector((XS[i], YS[i], 0.0))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(sc, 4)
    me = v["red"]
    nv = len(KIT.apple.vertices)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    cents = co.reshape(-1, nv, 3).mean(1)
    return (np.c_[cents, np.ones(len(cents))] @ np.array(R).T)[:, :3]


_TARGET_CACHE = {}


def apple_target(i):
    """World position of the apple the cutter touches on tree i (alley side,
    ~2.2-2.6 m high)."""
    if i in _TARGET_CACHE:
        return _TARGET_CACHE[i]
    v = KIT.variants[i % 4]
    rng = random.Random(i * 7919 + 3)
    yaw = rng.uniform(-0.12, 0.12)
    sc = rng.uniform(0.93, 1.06)
    R = Matrix.Translation(Vector((XS[i], YS[i], 0.0))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(sc, 4)
    me = v["red"]
    nv = len(KIT.apple.vertices)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    cents = co.reshape(-1, nv, 3).mean(1)
    cw = (np.c_[cents, np.ones(len(cents))] @ np.array(R).T)[:, :3]
    ok = (cw[:, 2] > 2.05) & (cw[:, 2] < 2.7)
    cand = np.where(ok)[0]
    if len(cand) == 0:
        cand = np.arange(len(cw))
    j = cand[np.argmin(cw[cand, 1])]      # most -y (alley side)
    _TARGET_CACHE[i] = cw[j]
    return cw[j]


# ------------------------------------------------------------------- trellis / depot
def cylinder_mesh(name, r, h, segs=12, cap=True):
    me = bpy.data.meshes.new(name)
    vs, fs = [], []
    for k in range(segs):
        a = 2 * math.pi * k / segs
        vs.append((r * math.cos(a), r * math.sin(a), 0.0))
        vs.append((r * math.cos(a), r * math.sin(a), h))
    for k in range(segs):
        k2 = (k + 1) % segs
        fs.append((2 * k, 2 * k2, 2 * k2 + 1, 2 * k + 1))
    if cap:
        fs.append(tuple(2 * k + 1 for k in range(segs)))
        fs.append(tuple(2 * k for k in reversed(range(segs))))
    me.from_pydata(vs, [], fs)
    me.update()
    for p in me.polygons:
        p.use_smooth = segs > 6
    return me


def box_mesh(name, sx, sy, sz, origin_bottom=True):
    me = bpy.data.meshes.new(name)
    z0 = 0.0 if origin_bottom else -sz / 2
    x, y = sx / 2, sy / 2
    vs = [(-x, -y, z0), (x, -y, z0), (x, y, z0), (-x, y, z0),
          (-x, -y, z0 + sz), (x, -y, z0 + sz), (x, y, z0 + sz), (-x, y, z0 + sz)]
    fs = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me.from_pydata(vs, [], fs)
    me.update()
    return me


def build_trellis():
    wood = pbr_material("post_wood", os.path.join(ASSETS, "textures", "bark_brown_02"),
                        "bark_brown_02", "1k", scale=0.6, rough_mult=1.2, bump=0.6,
                        tint=(0.75, 0.68, 0.6))
    wire = simple_material("wire", (0.55, 0.55, 0.58), rough=0.35, metal=1.0)
    post_me = cylinder_mesh("post", 0.055, 3.25, segs=10)
    post_me.materials.append(wood)
    wire_me = cylinder_mesh("wire", 0.0025, 1.0, segs=6, cap=False)
    wire_me.materials.append(wire)
    n = 0
    for r in ROWS:
        y = r * ROW_PITCH + ROW_OFFSET
        xs_row = XS[row_ids == r]
        x0, x1 = xs_row.min() - 1.0, xs_row.max() + 1.0
        npost = max(2, int(round((x1 - x0) / 10.0)) + 1)
        for k in range(npost):
            x = x0 + (x1 - x0) * k / (npost - 1)
            new_obj(f"post_{r}_{k}", post_me, COLL_INFRA, (x, y, -0.05),
                    (random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(0, 6.28)))
            n += 1
        for zw in (1.0, 1.9, 2.9):
            new_obj(f"wire_{r}_{zw}", wire_me, COLL_INFRA, (x0, y, zw), (0, math.pi / 2, 0),
                    (1, 1, x1 - x0))
    log(f"trellis: {n} posts, {len(ROWS)} rows")


def build_depot():
    """Trailer with harvest bins + landing pads at the headland."""
    x, y = DEPOT
    steel = simple_material("steel_painted", (0.18, 0.20, 0.22), rough=0.45, metal=0.3)
    red_paint = simple_material("trailer_paint", (0.55, 0.08, 0.06), rough=0.4, coat=0.3)
    rubber = simple_material("rubber", (0.02, 0.02, 0.02), rough=0.8)
    deck = simple_material("deck_wood", (0.35, 0.26, 0.17), rough=0.8)
    pad_mat = simple_material("pad_rubber", (0.06, 0.06, 0.065), rough=0.9)
    # trailer parked along the headland (its long axis along y)
    tx, ty = x - 4.6, y
    frame = new_obj("trailer_frame", box_mesh("trailer_frame", 2.3, 5.4, 0.18), COLL_INFRA, (tx, ty, 0.72))
    frame.data.materials.append(red_paint)
    d = new_obj("trailer_deck", box_mesh("trailer_deck", 2.4, 5.6, 0.05), COLL_INFRA, (tx, ty, 0.90))
    d.data.materials.append(deck)
    for sx in (-1.0, 1.0):
        s = new_obj("trailer_side", box_mesh("trailer_side", 0.04, 5.6, 0.5), COLL_INFRA, (tx + sx * 1.18, ty, 0.95))
        s.data.materials.append(red_paint)
    bar = new_obj("drawbar", box_mesh("drawbar", 0.12, 1.6, 0.12), COLL_INFRA, (tx, ty + 3.4, 0.55), (0.0, 0, 0))
    bar.data.materials.append(steel)
    wheel = cylinder_mesh("wheel", 0.42, 0.26, segs=20)
    wheel.materials.append(rubber)
    for sx in (-1.0, 1.0):
        for sy in (-0.8, 0.8):
            new_obj("wheel", wheel, COLL_INFRA, (tx + sx * 1.1 - 0.13 * sx, ty + sy, 0.42), (0, math.pi / 2, 0))
    axle = new_obj("axle", cylinder_mesh("axle", 0.04, 2.2, segs=8), COLL_INFRA, (tx - 1.1, ty, 0.42), (0, math.pi / 2, 0))
    axle.data.materials.append(steel)
    # harvest bins (plastic crates, CC0 Poly Haven)
    folder = os.path.join(ASSETS, "models", "plastic_crate_02")
    objs = append_from(os.path.join(folder, "plastic_crate_02_1k.blend"), "objects", ["plastic_crate_02"])
    fix_image_paths(folder)
    crate = objs[0].data
    bpy.data.objects.remove(objs[0], do_unlink=True)
    # crate dims from bbox
    co = np.empty(len(crate.vertices) * 3)
    crate.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    csz = co.max(0) - co.min(0)
    cz0 = co.min(0)[2]
    ci = 0
    for ix in range(2):
        for iy in range(4):
            for iz in range(3 if (ix + iy) % 3 else 2):
                new_obj(f"crate_{ci}", crate, COLL_INFRA,
                        (tx - 0.6 + ix * (csz[0] + 0.03), ty - 2.0 + iy * (csz[1] + 0.05), 0.95 - cz0 + iz * csz[2]),
                        (0, 0, random.uniform(-0.03, 0.03)))
                ci += 1
    # a few bins on the ground next to the trailer with harvested apples
    for ib in range(3):
        new_obj(f"crate_g{ib}", crate, COLL_INFRA, (tx + 1.9, ty - 1.5 + ib * (csz[1] + 0.08), -cz0),
                (0, 0, random.uniform(-0.1, 0.1)))
    # landing pads
    pad_me = box_mesh("pad", 1.1, 1.1, 0.02)
    pad_me.materials.append(pad_mat)
    for j in range(K):
        px, py = pad_position(j)
        new_obj(f"pad_{j}", pad_me, COLL_INFRA, (px, py, 0.0))
        ring = bpy.data.curves.new(f"padring_{j}", "CURVE")
        ring.dimensions = "3D"
        ring.bevel_depth = 0.02
        ring.resolution_u = 2
        sp = ring.splines.new("POLY")
        sp.points.add(3)
        for q, (dx, dy) in enumerate(((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))):
            sp.points[q].co = (px + dx, py + dy, 0.03, 1.0)
        sp.use_cyclic_u = True
        ring.materials.append(simple_material(f"padcol_{j}", DRONE_COLORS[j % 16], rough=0.5))
        o = bpy.data.objects.new(f"padring_{j}", ring)
        COLL_INFRA.objects.link(o)
    # charging / control cabinet
    cab = new_obj("cabinet", box_mesh("cabinet", 0.8, 1.2, 1.4), COLL_INFRA, (x - 7.5, y - 4.5, 0))
    cab.data.materials.append(simple_material("cabinet_paint", (0.75, 0.76, 0.78), rough=0.4, metal=0.2))
    log("depot built")


# ------------------------------------------------------------------- drone
class DroneKit:
    def __init__(self):
        self.carbon = simple_material("carbon", (0.02, 0.02, 0.022), rough=0.32, metal=0.0, coat=0.5)
        self.body_mat = simple_material("drone_body", (0.08, 0.085, 0.09), rough=0.38, coat=0.4)
        self.alu = simple_material("alu", (0.65, 0.66, 0.68), rough=0.3, metal=1.0)
        self.prop_disc = simple_material("prop_disc", (0.04, 0.04, 0.04), rough=0.6, alpha=0.14)
        self.prop_blade = simple_material("prop_blade", (0.06, 0.06, 0.06), rough=0.5, alpha=0.45)
        self.lens = simple_material("lens", (0.02, 0.02, 0.03), rough=0.05, coat=1.0)
        self.orange = simple_material("safety_orange", (0.9, 0.25, 0.03), rough=0.45, coat=0.3)
        self.parts = self._parts()

    def _parts(self):
        P = []
        b = box_mesh("d_body", 0.26, 0.19, 0.085, origin_bottom=False)
        P.append((b, self.body_mat, Matrix.Translation((0.0, 0, 0.0))))
        top = box_mesh("d_top", 0.18, 0.14, 0.045, origin_bottom=False)
        P.append((top, self.body_mat, Matrix.Translation((-0.01, 0, 0.065))))
        batt = box_mesh("d_batt", 0.15, 0.075, 0.05, origin_bottom=False)
        P.append((batt, self.carbon, Matrix.Translation((-0.09, 0, 0.0)) @ Matrix.Translation((0, 0, 0.0))))
        arm = cylinder_mesh("d_arm", 0.013, 0.36, segs=10)
        motor = cylinder_mesh("d_motor", 0.024, 0.032, segs=14)
        mcap = cylinder_mesh("d_mcap", 0.009, 0.015, segs=8)
        disc = cylinder_mesh("d_disc", 0.19, 0.003, segs=40)
        blade = box_mesh("d_blade", 0.37, 0.028, 0.004, origin_bottom=False)
        for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            ang = math.atan2(sy, sx)
            R = Matrix.Rotation(ang, 4, "Z")
            P.append((arm, self.carbon, Matrix.Translation((sx * 0.10, sy * 0.07, 0.01)) @ R @ Matrix.Rotation(math.pi / 2, 4, "Y")))
            mx, my = sx * 0.10 + 0.34 * math.cos(ang), sy * 0.07 + 0.34 * math.sin(ang)
            P.append((motor, self.alu, Matrix.Translation((mx, my, 0.01))))
            P.append((mcap, self.alu, Matrix.Translation((mx, my, 0.042))))
            P.append((disc, self.prop_disc, Matrix.Translation((mx, my, 0.054))))
            P.append((blade, self.prop_blade, Matrix.Translation((mx, my, 0.056)) @ Matrix.Rotation(random.uniform(0, 3.14), 4, "Z")))
        # landing legs (two skids)
        leg = cylinder_mesh("d_leg", 0.006, 0.16, segs=8)
        skid = cylinder_mesh("d_skid", 0.007, 0.34, segs=8)
        for sy in (-1, 1):
            for sx in (-1, 1):
                P.append((leg, self.carbon, Matrix.Translation((sx * 0.09, sy * 0.11, -0.19)) @ Matrix.Rotation(sy * -0.25, 4, "X")))
            P.append((skid, self.carbon, Matrix.Translation((-0.17, sy * 0.15, -0.20)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        # forward camera / sensor pod
        cam = cylinder_mesh("d_cam", 0.022, 0.04, segs=12)
        P.append((cam, self.body_mat, Matrix.Translation((0.11, 0, -0.03)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        lens = cylinder_mesh("d_lens", 0.014, 0.004, segs=12)
        P.append((lens, self.lens, Matrix.Translation((0.15, 0, -0.03)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        # cutter arm: carbon tube forward, slight downward elbow, end effector
        tube = cylinder_mesh("d_tube", 0.010, 0.30, segs=8)
        P.append((tube, self.carbon, Matrix.Translation((0.13, 0, -0.01)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        tube2 = cylinder_mesh("d_tube2", 0.006, 0.17, segs=8)
        P.append((tube2, self.carbon, Matrix.Translation((0.43, 0, -0.01)) @ Matrix.Rotation(math.pi / 2 + 0.12, 4, "Y")))
        joint = cylinder_mesh("d_joint", 0.014, 0.03, segs=10)
        P.append((joint, self.alu, Matrix.Translation((0.43, 0, -0.025))))
        head = box_mesh("d_head", 0.06, 0.04, 0.036, origin_bottom=False)
        P.append((head, self.alu, Matrix.Translation((0.565, 0, -0.03))))
        band = box_mesh("d_band", 0.02, 0.043, 0.039, origin_bottom=False)
        P.append((band, self.orange, Matrix.Translation((0.55, 0, -0.03))))
        puck = cylinder_mesh("d_puck", 0.03, 0.012, segs=16)
        P.append((puck, self.alu, Matrix.Translation((-0.04, 0, 0.088))))
        ant = cylinder_mesh("d_ant", 0.004, 0.09, segs=6)
        P.append((ant, self.body_mat, Matrix.Translation((-0.11, 0.06, 0.04))))
        P.append((ant, self.body_mat, Matrix.Translation((-0.11, -0.06, 0.04))))
        bl = box_mesh("d_cutblade", 0.05, 0.014, 0.003, origin_bottom=False)
        P.append((bl, self.alu, Matrix.Translation((0.615, 0.011, -0.03)) @ Matrix.Rotation(0.22, 4, "Z")))
        P.append((bl, self.alu, Matrix.Translation((0.615, -0.011, -0.03)) @ Matrix.Rotation(-0.22, 4, "Z")))
        return P

    def make(self, j):
        """Create drone j (an empty root with mesh children). Returns root object."""
        root = bpy.data.objects.new(f"drone_{j}", None)
        root.empty_display_size = 0.3
        COLL_DRONES.objects.link(root)
        for q, (me, mat, M) in enumerate(self.parts):
            if not me.materials:
                me.materials.append(mat)
            o = new_obj(f"drone_{j}_p{q}", me, COLL_DRONES, parent=root)
            o.matrix_parent_inverse = Matrix.Identity(4)
            o.matrix_basis = M
            if me.name in ("d_body", "d_top", "d_batt", "d_head"):
                bv = o.modifiers.new("bevel", "BEVEL")
                bv.width = 0.012 if me.name != "d_head" else 0.005
                bv.segments = 3
        # colour LED stripes along the two rear arms + status light
        led = simple_material(f"led_{j}", DRONE_COLORS[j % 16], rough=0.4, emit=3.0)
        stripe = box_mesh(f"d_stripe_{j}", 0.22, 0.012, 0.006, origin_bottom=False)
        stripe.materials.append(led)
        for sy in (-1, 1):
            ang = math.atan2(sy, -1)
            o = new_obj(f"drone_{j}_led{sy}", stripe, COLL_DRONES, parent=root)
            o.matrix_parent_inverse = Matrix.Identity(4)
            o.matrix_basis = Matrix.Translation((-0.27, sy * 0.19, 0.025)) @ Matrix.Rotation(ang, 4, "Z")
        tail = box_mesh(f"d_tail_{j}", 0.03, 0.10, 0.012, origin_bottom=False)
        tail.materials.append(led)
        o = new_obj(f"drone_{j}_tail", tail, COLL_DRONES, parent=root)
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = Matrix.Translation((-0.135, 0, 0.03))
        return root

    @staticmethod
    def pose(root, p):
        root.location = (p["x"], p["y"], p["z"] + (0.21 if p["state"] == "pad" else 0.0))
        root.rotation_euler = Euler((0.0, -p["pitch"], p["yaw"]), "XYZ")


# ------------------------------------------------------------------- overlay (routes)
def route_curve(name, pts, color, radius, z, strength=0.9):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    cu.resolution_u = 1
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for q, (x, y) in enumerate(pts):
        sp.points[q].co = (x, y, z, 1.0)
    cu.materials.append(emission_material(name + "_m", color, strength))
    o = bpy.data.objects.new(name, cu)
    COLL_OVERLAY.objects.link(o)
    o.visible_shadow = False
    o.visible_diffuse = False
    o.visible_glossy = False
    return o


def build_overlay(rnd, markers=True):
    """Route of each drone in round `rnd` as thin emissive tube + drone position marker."""
    scale = max(1.0, FIELD_SIZE / 60.0)
    r_tube = 0.11 * scale
    for j, paths in enumerate(S["drones"]):
        if rnd >= len(paths):
            continue
        path = paths[rnd]
        pts = []
        for (t, x, y) in path:
            if not pts or abs(pts[-1][0] - x) + abs(pts[-1][1] - y) > 1e-6:
                pts.append((x, y))
        col = tuple(c * 0.8 for c in DRONE_COLORS[j % 16])
        route_curve(f"route_{j}", pts, col, r_tube, FLY_Z + 0.05)
        if markers:
            p = drone_pose(j, T_NOW)
            if p["state"] == "pad":
                continue
            # vertical stalk + ring: diagram marker, obviously not an object
            stalk = route_curve(f"marker_{j}", [(p["x"], p["y"]), (p["x"] + 1e-4, p["y"])], col, r_tube * 0.6, 0.0, 1.1)
            stalk.data.splines[0].points[0].co = (p["x"], p["y"], 0.0, 1.0)
            stalk.data.splines[0].points[1].co = (p["x"], p["y"], FLY_Z + 0.6 * scale, 1.0)
            cu = bpy.data.curves.new(f"ring_{j}", "CURVE")
            cu.dimensions = "3D"
            cu.bevel_depth = r_tube * 0.9
            cu.resolution_u = 3
            sp = cu.splines.new("NURBS")
            n = 24
            sp.points.add(n - 1)
            rr = 1.4 * scale
            for q in range(n):
                a = 2 * math.pi * q / n
                sp.points[q].co = (p["x"] + rr * math.cos(a), p["y"] + rr * math.sin(a), FLY_Z + 0.6 * scale, 1.0)
            sp.use_cyclic_u = True
            sp.order_u = 4
            cu.materials.append(emission_material(f"ring_{j}_m", col, 1.1))
            o = bpy.data.objects.new(f"ring_{j}", cu)
            COLL_OVERLAY.objects.link(o)
            o.visible_shadow = False
    log("overlay built")


# ------------------------------------------------------------------- lighting
def build_lighting(sun_az=math.radians(215), sun_el=math.radians(48)):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    SC.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(os.path.join(ASSETS, "hdris", "kloofendal_48d_partly_cloudy_puresky_4k.hdr"))
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, sun_az - math.radians(90))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.75
    nt.links.new(bg.outputs[0], out.inputs[0])
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4.2
    sun.angle = math.radians(0.6)
    sun.color = (1.0, 0.96, 0.90)
    so = bpy.data.objects.new("sun", sun)
    SC.collection.objects.link(so)
    so.rotation_euler = Euler((math.pi / 2 - sun_el, 0, sun_az), "XYZ")


# ------------------------------------------------------------------- cameras
def make_camera(name, loc, target, lens=35.0, fstop=None, focus=None, sensor=36.0):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.sensor_width = sensor
    cam.clip_end = 5000
    if fstop is not None:
        cam.dof.use_dof = True
        cam.dof.aperture_fstop = fstop
        cam.dof.focus_distance = focus if focus is not None else (Vector(target) - Vector(loc)).length
        cam.dof.aperture_blades = 7
    o = bpy.data.objects.new(name, cam)
    SC.collection.objects.link(o)
    o.location = loc
    d = Vector(target) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    SC.camera = o
    return o


def fit_overview_camera(pitch_deg=38.0, lens=32.0):
    """Oblique aerial view from the depot side (-x) looking +x over the whole field."""
    from bpy_extras.object_utils import world_to_camera_view
    pts = [Vector((x, y, z)) for x in (FIELD[0] - 12, FIELD[1] + 2) for y in (FIELD[2] - 2, FIELD[3] + 2) for z in (0, 4)]
    pts.append(Vector((DEPOT[0] - 10, DEPOT[1] - 5, 0)))
    pts.append(Vector((DEPOT[0] - 10, DEPOT[1] + 5, 0)))
    cx = (FIELD[0] + FIELD[1]) / 2 - 0.06 * FIELD_W
    cy = (FIELD[2] + FIELD[3]) / 2
    pitch = math.radians(pitch_deg)
    cam = make_camera("cam_overview", (cx - 10, cy, 10), (cx, cy, 0), lens=lens)
    cam.rotation_euler = Euler((math.pi / 2 - pitch, 0, -math.pi / 2), "XYZ")
    dist = FIELD_SIZE
    for _ in range(40):
        cam.location = Vector((cx, cy, 0)) + Vector((-math.cos(pitch) * dist, 0, math.sin(pitch) * dist))
        bpy.context.view_layer.update()
        uv = [world_to_camera_view(SC, cam, p) for p in pts]
        xs = [u.x for u in uv]
        ys = [u.y for u in uv]
        ok = min(xs) > 0.03 and max(xs) < 0.97 and min(ys) > 0.04 and max(ys) < 0.96
        if ok:
            over = max(0.03 - min(xs), max(xs) - 0.97, 0.04 - min(ys), max(ys) - 0.96)
            if over > -0.015:
                break
            dist *= 0.97
        else:
            dist *= 1.05
    # centre the content vertically: tilt the camera so the projected bbox is centred
    for _ in range(3):
        bpy.context.view_layer.update()
        uv = [world_to_camera_view(SC, cam, p) for p in pts]
        ys = [u.y for u in uv]
        mid = 0.5 * (min(ys) + max(ys))
        vfov = 2 * math.atan(0.5 * cam.data.sensor_width * SC.render.resolution_y / SC.render.resolution_x / cam.data.lens)
        cam.rotation_euler.x -= (0.5 - mid) * vfov * 0.9
    log(f"overview camera: dist {dist:.1f} m, height {cam.location.z:.1f} m, pitch {90 - math.degrees(cam.rotation_euler.x):.1f} deg")
    return cam


# ------------------------------------------------------------------- shots
def pick_closeup_event():
    """Choose (drone j, event) : a cutting dwell whose tree has ripe neighbours
    on both sides and an unripe tree nearby, preferring the requested drone."""
    best = None
    for j, paths in enumerate(S["drones"]):
        if OPT["drone"] is not None and j != OPT["drone"]:
            continue
        for ev in cut_events(paths):
            if ev[0] < fly0["t0"] + 15 or ev[1] > fly0["t1"]:
                continue
            i = nearest_tree(ev[2], ev[3])
            d = np.hypot(XS - XS[i], YS - YS[i])
            near = np.where((d < 4.5) & (d > 0.5))[0]
            n_ripe = int(RIPE[near].sum())
            n_unripe = len(near) - n_ripe
            # trees already harvested at that time look bare -> avoid
            done = sum(1 for q in near if S["done_t"][q] is not None and S["done_t"][q] < ev[0])
            score = min(n_ripe, 3) * 2 + min(n_unripe, 3) - 3 * done + (2 if ev[1] - ev[0] > 5 else 0)
            if best is None or score > best[0]:
                best = (score, j, ev, i)
    return best[1], best[2], best[3]


KIT = TreeKit(lod_far=1, lod_near=(1 if OPT["preview"] or OPT["shot"] == "overview" else 0))
DK = DroneKit()


def pick_topcut_time(half_w, half_h):
    """Mission time (round 0 flight) and frame centre where the most drones
    are cutting simultaneously inside a half_w x half_h window."""
    best = None
    t = fly0["t0"] + 0.35 * (fly0["t1"] - fly0["t0"])     # some trees already harvested
    while t < fly0["t1"] - 2.0:
        cut = []
        for j in range(K):
            p = drone_pose(j, t)
            if p["state"] == "cut" and p["u"] > 0.99:
                cut.append((p["x"], p["y"]))
        for (cx, cy) in cut:
            for (ox, oy) in cut:
                mx, my = 0.5 * (cx + ox), 0.5 * (cy + oy)
                near = [(x, y) for (x, y) in cut if abs(x - mx) < half_w and abs(y - my) < half_h]
                inside = np.where((np.abs(XS - mx) < half_w) & (np.abs(YS - my) < half_h))[0]
                n_done = sum(1 for q in inside if tree_state(q, t) == "done")
                n_unripe = int((~RIPE[inside]).sum())
                score = 10 * len(near) + 2 * min(n_done, 8) + min(n_unripe, 4)
                if best is None or score > best[0]:
                    best = (score, t, mx, my, len(near))
        t += 0.5
    return best


def tree_state(i, t):
    dt = S["done_t"][i]
    if dt is not None and dt <= t:
        return "done"
    return "ripe" if RIPE[i] else "unripe"


def build_orchard(t, near_center=None, near_radius=0.0, hero=None):
    n_states = {"ripe": 0, "unripe": 0, "done": 0, "harvesting": 0}
    for i in range(NT):
        st = tree_state(i, t)
        if hero is not None and i == hero:
            st = "harvesting"
        near = near_center is not None and math.hypot(XS[i] - near_center[0], YS[i] - near_center[1]) < near_radius
        KIT.place(i, st, near=near)
        n_states[st] += 1
    log(f"orchard: {n_states}")


def place_drones(t, hero_j=None):
    roots = []
    for j in range(K):
        root = DK.make(j)
        p = drone_pose(j, t)
        DK.pose(root, p)
        roots.append((root, p))
    return roots


def shot_overview():
    build_ground()
    build_trellis()
    build_depot()
    build_orchard(T_NOW)
    place_drones(T_NOW)
    build_overlay(OPT["round"], markers=True)
    build_lighting()
    fit_overview_camera()


def shot_closeup(detail=False):
    j, ev, hero = pick_closeup_event()
    t = 0.5 * (ev[0] + ev[1])
    global T_NOW
    T_NOW = t
    log(f"close-up: drone {j} cutting tree {hero} at t={t:.1f}s (dwell {ev[0]:.1f}-{ev[1]:.1f})")
    build_ground()
    build_trellis()
    build_depot()
    build_orchard(t, near_center=(XS[hero], YS[hero]), near_radius=14.0, hero=hero)
    roots = place_drones(t)
    build_lighting(sun_az=math.radians(335), sun_el=math.radians(40))
    p = roots[j][1]
    tip = Vector((p["x"], p["y"] + ARM_REACH, p["z"] - 0.03))
    # apples mid-fall below the cutter (illustration)
    fall = KIT.apple
    cutting = new_obj("apple_cutting", fall, COLL_ORCHARD, (tip.x + 0.01, tip.y + 0.045, tip.z - 0.045), (0.25, 0.1, 1.2))
    for q, dz in enumerate((0.40, 1.05)):
        a = new_obj(f"falling_{q}", fall, COLL_ORCHARD, (tip.x + 0.06 * q, tip.y - 0.12 - 0.08 * q, tip.z - dz),
                    (random.uniform(0, 3), random.uniform(0, 3), 0))
    build_grass_scatter((p["x"] - 1.0, p["y"] - 0.6))
    if detail:
        # tree + apples detail from the alley; the drone works in the background
        # focus on a real canopy apple of the hero tree near the wanted spot,
        # and add two cluster mates right next to it (inside the leaf layer)
        want = np.array((p["x"] - 0.9, YS[hero] - 0.95, 1.95))
        cw = canopy_apples_world(hero)
        side = cw[(cw[:, 1] < YS[hero] - 0.45) & (cw[:, 2] > 1.3) & (cw[:, 2] < 2.5)]
        d = np.linalg.norm(side - want, axis=1)
        cand = side[d < 0.7] if (d < 0.7).any() else side[[np.argmin(d)]]
        focus = Vector(cand[np.argmin(cand[:, 1])])      # outermost (alley-side) apple near the spot
        rng = random.Random(21)
        for q, off in enumerate(((0.075, 0.02, -0.03), (-0.03, 0.05, -0.07))):
            new_obj(f"apple_fg_{q}", fall, COLL_ORCHARD, (focus.x + off[0], focus.y + off[1], focus.z + off[2]),
                    (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28)))
        log(f"detail focus apple at {tuple(round(v, 2) for v in focus)}")
        loc = Vector((focus.x - 1.6, YS[hero] - 2.05, focus.z + 0.1))
        make_camera("cam_detail", loc, focus, lens=60.0, fstop=3.2, focus=(focus - loc).length)
    else:
        mid = Vector((p["x"], p["y"] + 0.35, p["z"] - 0.05))
        loc = Vector((p["x"] - 2.3, YS[hero] - 2.0, p["z"] + 0.12))
        make_camera("cam_closeup", loc, mid, lens=55.0, fstop=4.5, focus=(Vector((tip.x, tip.y, tip.z)) - loc).length)


def shot_topcut():
    """Low-altitude view looking (nearly) straight down on a patch of the
    orchard where several drones are cutting at the same moment; apples
    fall from each cutter, harvested trees show their apples on the ground."""
    global T_NOW
    fw = OPT["height"] * 36.0 / OPT["lens"]               # ground footprint (pitch 90)
    _, t, mx, my, ncut = pick_topcut_time(0.5 * fw - 2.0, 0.5 * fw * H / W - 1.5)
    if OPT["t"] is not None and "--t" in sys.argv:
        t = OPT["t"]
    T_NOW = t
    log(f"top-cut: t={t:.1f}s, {ncut} drones cutting near ({mx:.1f}, {my:.1f})")
    build_ground()
    build_trellis()
    build_depot()
    build_orchard(t, near_center=(mx, my), near_radius=16.0)
    roots = place_drones(t)
    build_lighting(sun_az=math.radians(300), sun_el=math.radians(68))
    build_grass_scatter((mx, my), radius=16.0)
    # thin ring in the drone's colour around every drone in view (marker only, drones stay true scale)
    for j, (root, p) in enumerate(roots):
        if p["state"] == "pad":
            continue
        cu = bpy.data.curves.new(f"tring_{j}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.035
        sp = cu.splines.new("NURBS")
        n = 32
        sp.points.add(n - 1)
        for q in range(n):
            a = 2 * math.pi * q / n
            sp.points[q].co = (p["x"] + 0.95 * math.cos(a), p["y"] + 0.95 * math.sin(a), p["z"] + 0.35, 1.0)
        sp.use_cyclic_u = True
        sp.order_u = 4
        cu.materials.append(emission_material(f"tring_{j}_m", DRONE_COLORS[j % 16], 1.4))
        o = bpy.data.objects.new(f"tring_{j}", cu)
        COLL_OVERLAY.objects.link(o)
        o.visible_shadow = False
    fall = KIT.apple
    rng = random.Random(11)
    for j, (root, p) in enumerate(roots):
        if p["state"] != "cut":
            continue
        tip = Vector((p["x"], p["y"] + ARM_REACH, p["z"] - 0.03))
        new_obj(f"apple_cut_{j}", fall, COLL_ORCHARD, (tip.x + 0.01, tip.y + 0.045, tip.z - 0.045), (0.25, 0.1, 1.2))
        for q, dz in enumerate((0.45, 1.15)):
            new_obj(f"falling_{j}_{q}", fall, COLL_ORCHARD,
                    (tip.x + rng.uniform(-0.08, 0.08), tip.y - 0.10 - 0.06 * q, tip.z - dz),
                    (rng.uniform(0, 3), rng.uniform(0, 3), 0))
    pitch = math.radians(OPT["pitch"])
    h = OPT["height"]
    target = Vector((mx, my, 1.2))
    if OPT["pitch"] >= 89.5:
        cam = make_camera("cam_top", (mx, my, h), target, lens=OPT["lens"])
        cam.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")      # image x = row direction
    else:
        back = (h - target.z) / math.tan(pitch)
        loc = Vector((mx, my - back, h))
        cam = make_camera("cam_topcut", loc, target, lens=OPT["lens"])


def shot_anim():
    """Keyframed version: drones move, apples disappear when harvested."""
    fps = OPT["fps"]
    SC.render.fps = fps
    speed = OPT["speed"]
    t_end = S["t_end"]
    n_frames = int(t_end / speed * fps) + 1
    SC.frame_start, SC.frame_end = 1, n_frames
    build_ground()
    build_trellis()
    build_depot()
    # trees with apples; apple objects get keyframed hide_render at done_t
    for i in range(NT):
        st = "ripe" if RIPE[i] else "unripe"
        tree, apples, _ = KIT.place(i, st)
        dt = S["done_t"][i]
        if dt is not None and apples is not None:
            f = int(dt / speed * fps) + 1
            for attr in ("hide_render", "hide_viewport"):
                setattr(apples, attr, False)
                apples.keyframe_insert(attr, frame=f - 1)
                setattr(apples, attr, True)
                apples.keyframe_insert(attr, frame=f)
            v = KIT.variants[i % 4]
            fallen = new_obj(f"fallen_{i}", v["fallen"], COLL_ORCHARD)
            fallen.matrix_world = Matrix.Translation(Vector((XS[i], YS[i], 0))) @ Matrix.Rotation(random.Random(i * 7919 + 3).uniform(-0.12, 0.12), 4, "Z")
            for attr in ("hide_render", "hide_viewport"):
                setattr(fallen, attr, True)
                fallen.keyframe_insert(attr, frame=f - 1)
                setattr(fallen, attr, False)
                fallen.keyframe_insert(attr, frame=f)
    roots = place_drones(0.0)
    for j, (root, _) in enumerate(roots):
        for f in range(1, n_frames + 1, OPT["step"]):
            t = (f - 1) / fps * speed
            p = drone_pose(j, t)
            DK.pose(root, p)
            root.keyframe_insert("location", frame=f)
            root.keyframe_insert("rotation_euler", frame=f)
    build_overlay(0, markers=False)
    build_lighting()
    fit_overview_camera()
    log(f"animation: {n_frames} frames at {fps} fps, {speed} s mission / s")


shot = OPT["shot"]
if shot == "overview":
    shot_overview()
elif shot == "closeup":
    shot_closeup(detail=False)
elif shot == "detail":
    shot_closeup(detail=True)
elif shot == "topcut":
    shot_topcut()
elif shot == "anim":
    shot_anim()
else:
    raise SystemExit("unknown shot " + shot)

log(f"scene: {len(bpy.data.objects)} objects, {len(bpy.data.meshes)} meshes")

if OPT["save"]:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(OPT["save"]))
    log("saved", OPT["save"])

if OPT["out"]:
    out = os.path.abspath(OPT["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if shot == "anim":
        SC.render.filepath = out
        if OPT["frames"]:
            a, b = [int(v) for v in OPT["frames"].split("-")]
            SC.frame_start, SC.frame_end = a, b
        SC.frame_step = OPT["step"]
        t0 = time.time()
        bpy.ops.render.render(animation=True)
        log(f"rendered frames {SC.frame_start}-{SC.frame_end} step {SC.frame_step} in {time.time()-t0:.0f}s")
    else:
        SC.render.filepath = out
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        log(f"rendered {out} in {time.time()-t0:.0f}s")
