"""Photorealistic scene of the tethered-drone harvest-platform scenario (v3).

Run with Blender 5.x in background mode, e.g.

  blender -b --python build_scene_v3.py -- --data ../anim3_syn1ha.json --shot overview \
          --out out/fig_v3_overview.png --res 2400x1350 --samples 160

Scenario: fresh-market apple selective harvest in a high-density Tall Spindle
orchard (0.9 m x 3.3 m, 3-wire trellis).  Small electric ground platforms,
each carrying one apple bin and a tether hub, drive along the alleys; at each
stop the four tethered drones of a platform pick ripe fruit on the two canopy
faces bounding the alley and place it into the bin.  Full bins are swapped at
the bin station on the headland.

Everything that is *data* comes from the json written by
paper2/sim/export_platform.py (tree positions and cultivars, initial ripe /
unripe fruit per canopy face, stop cells, pick events, platform traces, bin
station, bin capacity).  Everything that is *illustration* (tree model, where
on the canopy each apple hangs, the platform / drone / bin / tractor models,
trellis, ground textures, lighting, cameras) is generated here from CC0 assets
(see assets/SOURCES.md) and procedural geometry.  See README.md, section v3.

Options (after "--"):
  --data PATH            anim3 json (required)
  --shot NAME            overview | platform | closeup | station
  --t SECONDS            mission time to depict (default: chosen per shot from the data)
  --platform P           platform index for the platform/closeup/station shots (default: chosen from the data)
  --out PATH             output png
  --res WxH              resolution (default 2400x1350)
  --samples N            Cycles samples (default 160)
  --preview              960x540, 32 samples, LOD1 trees everywhere
  --apple-div N          render one apple per N fruit of the data (default 1)
  --engine cycles|eevee
  --gpu 3090|all
  --save PATH            also save the .blend
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
        "data": None, "shot": "overview", "t": None, "platform": None, "out": None,
        "res": "2400x1350", "samples": 160, "preview": False, "engine": "cycles",
        "gpu": "3090", "save": None, "seed": 7, "apple-div": 1,
    }
    i = 0
    while i < len(argv):
        key = argv[i][2:]
        if key in ("preview",):
            opt[key] = True
            i += 1
            continue
        opt[key] = argv[i + 1]
        i += 2
    for key in ("samples", "seed", "apple-div"):
        opt[key] = int(opt[key])
    if opt["t"] is not None:
        opt["t"] = float(opt["t"])
    if opt["platform"] is not None:
        opt["platform"] = int(opt["platform"])
    if opt["preview"]:
        opt["res"] = "960x540"
        opt["samples"] = 32
    return opt


OPT = parse_args()
random.seed(OPT["seed"])
np.random.seed(OPT["seed"])
T0 = time.time()
APPLE_DIV = max(1, OPT["apple-div"])


def log(*a):
    print(f"[build {time.time() - T0:6.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------- data
with open(OPT["data"]) as f:
    DATA = json.load(f)
TR = DATA["trees"]
XS = np.array(TR["x"])
YS = np.array(TR["y"])
CULT = np.array(TR["cultivar"], dtype=int)
R0 = np.array(TR["ripe"], dtype=int)          # (NT, 2) initial ripe fruit per face
U0 = np.array(TR["unripe"], dtype=int)        # (NT, 2) initial unripe fruit per face
NT = len(XS)
ALLEYS = DATA["alleys"]
ALLEY_Y = np.array([a["y"] for a in ALLEYS])
NODES = DATA["nodes"]
NN = len(NODES["x"])
PICKS = DATA["picks"]                          # [t_end, node, ripe_picked, unripe_picked, platform]
PLATS = DATA["platforms"]                      # [platform][trip][[t, x, y, state]]
K = int(DATA["k"])
M_DRONES = int(DATA["drones_per_platform"])
BIN_CAP = float(DATA["bin_capacity"])
STATION = tuple(DATA["station"])
PAR = DATA.get("params", {})
CULT_NAMES = DATA.get("cultivars", ["red", "yellow", "green"])
MISSION_S = float(DATA.get("summary", {}).get("mission_s", max(p[0] for p in PICKS)))

ROW_PITCH = float(PAR.get("ROW", 330.0)) / 100.0
ALLEY_OFF = float(ALLEY_Y[0])                              # alley centre lines: ALLEY_OFF + k * ROW_PITCH
ROW_OFFSET = ALLEY_OFF + ROW_PITCH / 2                     # row centre lines
row_ids = np.rint((YS - ROW_OFFSET) / ROW_PITCH).astype(int)
ROW_Y = ROW_OFFSET + row_ids * ROW_PITCH
ROWS = sorted(set(row_ids.tolist()))
FIELD = (min(a["x0"] for a in ALLEYS), max(a["x1"] for a in ALLEYS),
         float(ALLEY_Y[0]) - ROW_PITCH / 2, float(ALLEY_Y[-1]) + ROW_PITCH / 2)
FIELD_W = FIELD[1] - FIELD[0]
FIELD_H = FIELD[3] - FIELD[2]
FIELD_SIZE = max(FIELD_W, FIELD_H)
NODE_XY = {(round(x, 3), round(y, 3)): q for q, (x, y) in enumerate(zip(NODES["x"], NODES["y"]))}
NODE_ALLEY_Y = np.array(NODES["y"])

# geometry constants (metres)
TREE_H = 3.1              # tall spindle height
CANOPY_HALF = 0.55        # across-row half thickness of the fruiting wall
REACH = 0.52              # drone centre -> suction cup tip
HOVER_Z = (1.55, 2.45)    # picking heights of the two drones on one face
DECK_Z = 0.45             # platform deck height
BIN_SZ = (1.2, 1.2, 0.75) # standard apple bin
HUB_LOCAL = (1.0, 0.0, 2.75)   # tether hub in the platform frame
ESCORT = [(0.55, 0.42, 1.95), (0.55, -0.42, 1.95), (-0.55, 0.42, 1.95), (-0.55, -0.42, 1.95)]
STATION_SLOT = 3.4        # spacing of platforms queued at the station (illustration)

PLATFORM_COLORS = [
    (0.90, 0.20, 0.15), (0.15, 0.45, 0.95), (0.95, 0.65, 0.10), (0.20, 0.75, 0.35),
    (0.80, 0.25, 0.85), (0.10, 0.80, 0.85), (0.95, 0.40, 0.55), (0.55, 0.55, 0.20),
    (0.95, 0.30, 0.00), (0.30, 0.30, 0.95), (0.60, 0.85, 0.10), (0.85, 0.10, 0.45),
    (0.15, 0.65, 0.60), (0.75, 0.50, 0.25), (0.45, 0.20, 0.80), (0.90, 0.80, 0.30),
]

log(f"data {os.path.basename(OPT['data'])}: {NT} trees, {len(ROWS)} rows, {NN} stop cells, "
    f"k={K} platforms x {M_DRONES} drones, {len(PICKS)} pick events, mission {MISSION_S:.0f} s")


# ------------------------------------------------------------------- platform traces
def _stops_of(p):
    """[(t_arrive, t_leave, node, x, y, trip)] picking stops of platform p."""
    out = []
    for ti, trip in enumerate(PLATS[p]):
        for a, b in zip(trip[:-1], trip[1:]):
            if a[3] == "pick" and b[3] == "pick" and abs(a[1] - b[1]) + abs(a[2] - b[2]) < 1e-6 and b[0] > a[0]:
                q = NODE_XY.get((round(a[1], 3), round(a[2], 3)))
                if q is None:
                    d = (np.array(NODES["x"]) - a[1]) ** 2 + (np.array(NODES["y"]) - a[2]) ** 2
                    q = int(np.argmin(d))
                out.append((a[0], b[0], q, a[1], a[2], ti))
    return out


STOPS = [_stops_of(p) for p in range(K)]
TRIP_SPAN = [[(trip[0][0], trip[-1][0]) for trip in PLATS[p]] for p in range(K)]


def pick_events():
    """PICKS with the arrival time of the stop attached: (t_arr, t_end, node, ripe, unripe, platform, trip)."""
    ev = []
    for (te, q, rp, up, p) in PICKS:
        best = None
        for (ta, tl, qq, x, y, ti) in STOPS[p]:
            if qq == q and abs(tl - te) < 0.6:
                if best is None or abs(tl - te) < abs(best[1] - te):
                    best = (ta, tl, ti)
        if best is None:
            ti = next((i for i, (a, b) in enumerate(TRIP_SPAN[p]) if a <= te <= b + 1e-6), 0)
            best = (te - 1.0, te, ti)
        ev.append((best[0], te, q, rp, up, p, best[2]))
    return ev


EVENTS = pick_events()


def node_faces(q):
    """[(tree, face)] canopy faces bounding stop cell q."""
    return [(i, 0) for i in NODES["trees_up"][q]] + [(i, 1) for i in NODES["trees_lo"][q]]


def _distribute(amount, weights):
    """Largest-remainder split of an integer `amount` over faces proportional to `weights`."""
    w = np.asarray(weights, float)
    if amount <= 0 or w.sum() <= 0:
        return np.zeros(len(w), int)
    raw = amount * w / w.sum()
    base = np.floor(raw).astype(int)
    rest = int(round(amount - base.sum()))
    order = np.argsort(-(raw - base))
    base[order[:rest]] += 1
    return np.minimum(base, w.astype(int))


def fruit_removed(t):
    """(rem_r, rem_u): fruit removed from each tree face by mission time t (ints; the
    stop in progress contributes its elapsed fraction)."""
    rem_r = np.zeros((NT, 2), int)
    rem_u = np.zeros((NT, 2), int)
    for (ta, te, q, rp, up, p, ti) in sorted(EVENTS, key=lambda e: e[1]):
        if ta >= t:
            continue
        frac = 1.0 if te <= t else (t - ta) / max(te - ta, 1e-6)
        faces = node_faces(q)
        if not faces:
            continue
        idx = np.array([i for i, _ in faces])
        fc = np.array([f for _, f in faces])
        avail_r = R0[idx, fc] - rem_r[idx, fc]
        avail_u = U0[idx, fc] - rem_u[idx, fc]
        rem_r[idx, fc] += _distribute(int(round(rp * frac)), avail_r)
        rem_u[idx, fc] += _distribute(int(round(up * frac)), avail_u)
    return rem_r, rem_u


def trips_done(t):
    return sum(1 for p in range(K) for (a, b) in TRIP_SPAN[p] if b <= t)


def bin_fill(p, t):
    """(fraction, picked count, cultivar/ripeness class counts) of the bin on platform p at t."""
    ti = next((i for i, (a, b) in enumerate(TRIP_SPAN[p]) if a <= t <= b), None)
    cls = np.zeros((3, 2))
    if ti is None:
        return 0.0, 0, cls
    n = 0.0
    for (ta, te, q, rp, up, pp, tj) in EVENTS:
        if pp != p or tj != ti or ta >= t:
            continue
        frac = 1.0 if te <= t else (t - ta) / max(te - ta, 1e-6)
        n += (rp + up) * frac
        faces = node_faces(q)
        if faces:
            c = int(np.bincount(CULT[[i for i, _ in faces]], minlength=3).argmax())
            cls[c, 0] += rp * frac
            cls[c, 1] += up * frac
    return min(1.0, n / BIN_CAP), int(n), cls


def platform_pose(p, t):
    """dict(x, y, yaw, state, trip, node, stop=(ta, tl)) of platform p at time t."""
    trips = PLATS[p]
    last_dir = (0.0, 1.0)

    def station_pose(ti):
        return dict(x=STATION[0], y=STATION[1], yaw=math.pi / 2, state="station", trip=ti, node=None, stop=None)

    for ti, trip in enumerate(trips):
        if t < trip[0][0]:
            return station_pose(ti)
        if t > trip[-1][0]:
            continue
        for k, (a, b) in enumerate(zip(trip[:-1], trip[1:])):
            dx, dy = b[1] - a[1], b[2] - a[2]
            if math.hypot(dx, dy) > 1e-6:
                last_dir = (dx, dy)
            if a[0] <= t <= b[0]:
                u = 0.0 if b[0] <= a[0] else (t - a[0]) / (b[0] - a[0])
                x, y = a[1] + u * dx, a[2] + u * dy
                yaw = math.atan2(last_dir[1], last_dir[0])
                if a[3] == "pick" and b[3] == "pick" and math.hypot(dx, dy) < 1e-6:
                    q = NODE_XY.get((round(a[1], 3), round(a[2], 3)))
                    return dict(x=x, y=y, yaw=yaw, state="pick", trip=ti, node=q, stop=(a[0], b[0]))
                if math.hypot(x - STATION[0], y - STATION[1]) < 0.05:
                    return dict(x=x, y=y, yaw=math.pi / 2, state="station", trip=ti, node=None, stop=None)
                return dict(x=x, y=y, yaw=yaw, state="move", trip=ti, node=None, stop=None)
    return station_pose(len(trips))


def all_platform_poses(t):
    poses = [platform_pose(p, t) for p in range(K)]
    # platforms queued at the station are spread along the headland lane (illustration)
    at_station = [p for p, ps in enumerate(poses) if ps["state"] == "station"]
    for k, p in enumerate(at_station):
        off = ((k + 1) // 2) * STATION_SLOT * (1 if k % 2 else -1)
        poses[p]["y"] = STATION[1] + off
    return poses


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
    SC.eevee.taa_render_samples = max(16, min(64, OPT["samples"]))

COLL_ORCHARD = bpy.data.collections.new("orchard")
COLL_INFRA = bpy.data.collections.new("infrastructure")
COLL_FLEET = bpy.data.collections.new("fleet")
COLL_OVERLAY = bpy.data.collections.new("overlay")
for c in (COLL_ORCHARD, COLL_INFRA, COLL_FLEET, COLL_OVERLAY):
    SC.collection.children.link(c)


def new_obj(name, mesh, coll, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), parent=None):
    o = bpy.data.objects.new(name, mesh)
    o.location = loc
    o.rotation_euler = rot
    o.scale = scale
    if parent is not None:
        o.parent = parent
        o.matrix_parent_inverse = Matrix.Identity(4)
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


def pbr_material(name, folder, prefix, res, scale=1.0, rough_mult=1.0, bump=0.3, tint=None, hue_sat=None):
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


class NodeMath:
    """Tiny helper to write shader/geometry math graphs."""

    def __init__(self, nodes, links):
        self.N, self.L = nodes, links

    def _plug(self, node, idx, v):
        if isinstance(v, (int, float)):
            node.inputs[idx].default_value = v
        else:
            self.L.new(v, node.inputs[idx])

    def op(self, operation, a, b=None, c=None):
        n = self.N.new("ShaderNodeMath")
        n.operation = operation
        self._plug(n, 0, a)
        if b is not None:
            self._plug(n, 1, b)
        if c is not None:
            self._plug(n, 2, c)
        return n.outputs[0]

    def ramp(self, v, a, b, lo=0.0, hi=1.0):
        n = self.N.new("ShaderNodeMapRange")
        n.inputs["From Min"].default_value = a
        n.inputs["From Max"].default_value = b
        n.inputs["To Min"].default_value = lo
        n.inputs["To Max"].default_value = hi
        self._plug(n, 0, v)
        return n.outputs[0]

    def noise(self, vec, scale, detail=3.0):
        n = self.N.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        self.L.new(vec, n.inputs["Vector"])
        return n.outputs["Fac"]

    def mix_rgb(self, a, b, fac, blend="MIX"):
        mx = self.N.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        mx.blend_type = blend
        self._plug(mx, 0, fac)
        for idx, v in ((6, a), (7, b)):
            if isinstance(v, tuple):
                mx.inputs[idx].default_value = (*v, 1.0)
            else:
                self.L.new(v, mx.inputs[idx])
        return mx.outputs[2]

    def mix_f(self, a, b, fac):
        mx = self.N.new("ShaderNodeMix")
        mx.data_type = "FLOAT"
        self._plug(mx, 0, fac)
        self._plug(mx, 2, a)
        self._plug(mx, 3, b)
        return mx.outputs[0]


# ------------------------------------------------------------------- ground
def build_ground():
    """One large plane: mown grass alleys with two wheel tracks, bare herbicide
    strip under each row, dirt headland lane around the block and a compacted
    pad at the bin station."""
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
    NM = NodeMath(nodes, links)

    def texset(folder, prefix, res, scale, loc_y, acg=False):
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1.0 / scale,) * 3
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        base = os.path.join(folder, prefix)
        if acg:
            d = tex_node(nodes, base + f"_{res}-JPG_Color.jpg", "sRGB", (-900, loc_y))
            n = tex_node(nodes, base + f"_{res}-JPG_NormalGL.jpg", "Non-Color", (-900, loc_y - 300))
            r = tex_node(nodes, base + f"_{res}-JPG_Roughness.jpg", "Non-Color", (-900, loc_y - 600))
        else:
            d = tex_node(nodes, base + f"_diffuse_{res}.jpg", "sRGB", (-900, loc_y))
            n = tex_node(nodes, base + f"_nor_gl_{res}.jpg", "Non-Color", (-900, loc_y - 300))
            r = tex_node(nodes, base + f"_rough_{res}.jpg", "Non-Color", (-900, loc_y - 600))
        for t in (d, n, r):
            links.new(mapping.outputs["Vector"], t.inputs["Vector"])
        return d, n, r

    gd, gn, gr = texset(grass_dir, "Grass004", "2K", 2.5, 900, acg=True)
    gd2, _, _ = texset(grass_dir, "Grass004", "2K", 6.1, 0, acg=True)
    md, mn, mr = texset(mud_dir, "brown_mud_dry", "2k", 2.2, -900)
    obj = coord.outputs["Object"]

    # grass colour: two scales + large-scale variation
    gmix = NM.mix_rgb(gd.outputs["Color"], gd2.outputs["Color"], 0.45)
    var = NM.noise(obj, 0.035, 3.0)
    vramp = nodes.new("ShaderNodeValToRGB")
    vramp.color_ramp.elements[0].position = 0.35
    vramp.color_ramp.elements[0].color = (0.55, 0.55, 0.40, 1)
    vramp.color_ramp.elements[1].position = 0.65
    vramp.color_ramp.elements[1].color = (0.90, 0.90, 0.82, 1)
    links.new(var, vramp.inputs["Fac"])
    gcol = NM.mix_rgb(gmix, vramp.outputs["Color"], 1.0, "MULTIPLY")

    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(obj, sep.inputs[0])
    X, Y = sep.outputs["X"], sep.outputs["Y"]
    edge = NM.op("MULTIPLY_ADD", NM.noise(obj, 0.6, 4.0), 0.35, -0.175)

    # distance to the nearest row line / alley centre line
    d_row = NM.op("ABSOLUTE", NM.op("WRAP", NM.op("SUBTRACT", Y, ROW_OFFSET), -ROW_PITCH / 2, ROW_PITCH / 2))
    d_alley = NM.op("ABSOLUTE", NM.op("WRAP", NM.op("SUBTRACT", Y, ALLEY_OFF), -ROW_PITCH / 2, ROW_PITCH / 2))
    # inside the block (soft box)
    inside = NM.op("MULTIPLY", NM.op("MULTIPLY", NM.ramp(X, FIELD[0] - 0.8, FIELD[0] + 0.3), NM.ramp(X, FIELD[1] + 0.8, FIELD[1] - 0.3)),
                   NM.op("MULTIPLY", NM.ramp(Y, FIELD[2] - 0.5, FIELD[2] + 0.3), NM.ramp(Y, FIELD[3] + 0.5, FIELD[3] - 0.3)))
    strip = NM.op("MULTIPLY", NM.ramp(NM.op("ADD", d_row, edge), 0.46, 0.60, 1.0, 0.0), inside)
    # wheel tracks of the platforms (track width 1.0 m, tyre width 0.18 m) in every alley
    track = NM.op("MULTIPLY", NM.ramp(NM.op("ADD", NM.op("ABSOLUTE", NM.op("SUBTRACT", d_alley, 0.50)), NM.op("MULTIPLY", edge, 0.15)),
                                      0.07, 0.13, 0.45, 0.0), inside)
    # headland lane around the block (distance outside the block box 0.5 .. 4.5 m)
    d_out = NM.op("MAXIMUM", NM.op("MAXIMUM", NM.op("SUBTRACT", FIELD[0], X), NM.op("SUBTRACT", X, FIELD[1])),
                  NM.op("MAXIMUM", NM.op("SUBTRACT", FIELD[2], Y), NM.op("SUBTRACT", Y, FIELD[3])))
    hn = NM.op("MULTIPLY_ADD", NM.noise(obj, 0.25, 3.0), 1.6, -0.8)
    d_outn = NM.op("ADD", d_out, hn)
    head = NM.op("MULTIPLY", NM.ramp(d_outn, 0.3, 1.2), NM.ramp(d_outn, 5.2, 4.0))
    # bin station pad (compacted, in front of the bins)
    d_st = NM.op("SQRT", NM.op("ADD", NM.op("POWER", NM.op("MULTIPLY", NM.op("SUBTRACT", X, STATION[0] - 4.5), 1.3), 2.0),
                                NM.op("POWER", NM.op("SUBTRACT", Y, STATION[1] - 1.0), 2.0)))
    stpad = NM.ramp(NM.op("ADD", d_st, hn), 9.0, 7.0)
    mask = NM.op("MAXIMUM", NM.op("MAXIMUM", strip, head), NM.op("MAXIMUM", track, stpad))

    mud_tint = NM.mix_rgb(md.outputs["Color"], (0.78, 0.74, 0.68), 1.0, "MULTIPLY")
    links.new(NM.mix_rgb(gcol, mud_tint, mask), bsdf.inputs["Base Color"])
    rough = NM.op("MULTIPLY", NM.mix_f(gr.outputs["Color"], mr.outputs["Color"], mask), 1.15)
    links.new(rough, bsdf.inputs["Roughness"])
    nmix = NM.mix_rgb(gn.outputs["Color"], mn.outputs["Color"], mask)
    nm = nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = 0.8
    links.new(nmix, nm.inputs["Color"])
    links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    ob.data.materials.append(m)
    return ob


# ------------------------------------------------------------------- grass clumps (close-ups)
def grass_clump_mesh(rng):
    vs, fs = [], []
    for b in range(9):
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
    """Scatter grass clumps on a disc around `center` (close-up shots only);
    none on the herbicide strips, thinner on the wheel tracks."""
    rng = random.Random(5)
    mat = bpy.data.materials.new("grass_blade")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.65
    b.inputs["Specular IOR Level"].default_value = 0.3
    b.inputs["Subsurface Weight"].default_value = 0.25
    b.inputs["Subsurface Radius"].default_value = (0.02, 0.05, 0.01)
    b.inputs["Subsurface Scale"].default_value = 0.01
    nt = mat.node_tree
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.12, 0.26, 0.05, 1)
    ramp.color_ramp.elements[1].color = (0.30, 0.42, 0.10, 1)
    nt.links.new(oi.outputs["Random"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    mat.use_backface_culling = False
    coll = bpy.data.collections.new("grass_clumps")
    COLL_INFRA.children.link(coll)
    for q in range(3):
        me = grass_clump_mesh(rng)
        me.materials.append(mat)
        o = new_obj(f"grass_clump_{q}", me, COLL_INFRA, (0, 0, -5.0))
        o.hide_render = True
        coll.objects.link(o)
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
    NM = NodeMath(N, ng.links)
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    dist = N.new("GeometryNodeDistributePointsOnFaces")
    dist.distribute_method = "RANDOM"
    pos = N.new("GeometryNodeInputPosition")
    dcen = N.new("ShaderNodeVectorMath")
    dcen.operation = "DISTANCE"
    dcen.inputs[1].default_value = (center[0], center[1], 0.0)
    ng.links.new(pos.outputs[0], dcen.inputs[0])
    sep = N.new("ShaderNodeSeparateXYZ")
    ng.links.new(pos.outputs[0], sep.inputs[0])
    Y = sep.outputs["Y"]
    d_alley = NM.op("ABSOLUTE", NM.op("WRAP", NM.op("SUBTRACT", Y, ALLEY_OFF), -ROW_PITCH / 2, ROW_PITCH / 2))
    track = NM.ramp(NM.op("ABSOLUTE", NM.op("SUBTRACT", d_alley, 0.50)), 0.10, 0.18, 0.35, 1.0)
    dens = NM.op("MULTIPLY", NM.ramp(dcen.outputs["Value"], radius * 0.35, radius, density, 0.0), track)
    ng.links.new(dens, dist.inputs["Density"])
    d_row = NM.op("ABSOLUTE", NM.op("WRAP", NM.op("SUBTRACT", Y, ROW_OFFSET), -ROW_PITCH / 2, ROW_PITCH / 2))
    nz = NM.op("MULTIPLY_ADD", NM.noise(pos.outputs[0], 0.6, 3.0), 0.3, -0.15)
    cmp = N.new("FunctionNodeCompare")
    cmp.operation = "GREATER_THAN"
    cmp.inputs[1].default_value = 0.62
    ng.links.new(NM.op("ADD", d_row, nz), cmp.inputs[0])
    ng.links.new(cmp.outputs[0], dist.inputs["Selection"])
    ng.links.new(gi.outputs[0], dist.inputs["Mesh"])
    iop = N.new("GeometryNodeInstanceOnPoints")
    ci = N.new("GeometryNodeCollectionInfo")
    ci.inputs[0].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    ng.links.new(ci.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    rid = N.new("FunctionNodeRandomValue")
    rid.data_type = "INT"
    rid.inputs["Min"].default_value = 0
    rid.inputs["Max"].default_value = 2
    ng.links.new(rid.outputs["Value"], iop.inputs["Instance Index"])
    rr = N.new("FunctionNodeRandomValue")
    rr.data_type = "FLOAT_VECTOR"
    rr.inputs["Min"].default_value = (-0.15, -0.15, 0.0)
    rr.inputs["Max"].default_value = (0.15, 0.15, 6.283)
    ng.links.new(rr.outputs["Value"], iop.inputs["Rotation"])
    rs = N.new("FunctionNodeRandomValue")
    rs.data_type = "FLOAT"
    rs.inputs["Min"].default_value = 0.7
    rs.inputs["Max"].default_value = 1.5
    ng.links.new(rs.outputs["Value"], iop.inputs["Scale"])
    ng.links.new(dist.outputs["Points"], iop.inputs["Points"])
    ng.links.new(iop.outputs["Instances"], go.inputs[0])
    mod = disc.modifiers.new("scatter", "NODES")
    mod.node_group = ng
    log(f"grass scatter: disc r={radius} m")
    return disc


# ------------------------------------------------------------------- assets
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
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    zmin = co[:, 2].min()
    base = co[co[:, 2] < zmin + 0.15]
    co[:, 0] -= base[:, 0].mean()
    co[:, 1] -= base[:, 1].mean()
    co[:, 2] -= zmin
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    for m in me.materials:
        m.use_backface_culling = False
    log(f"tree mesh LOD{lod}: {len(me.polygons)} polys, height {co[:, 2].max():.2f} m")
    return me


def tint_leaf_material(me):
    """Multiply the leaf diffuse texture by the per-object colour (per-tree variation)."""
    for m in me.materials:
        if "leaves" not in m.name or m.get("v3_tinted"):
            continue
        nt = m.node_tree
        diff = [n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image and "diff" in n.image.name]
        if not diff:
            continue
        diff = diff[0]
        targets = [(l.to_node, l.to_socket) for l in list(nt.links) if l.from_node == diff and l.from_socket.name == "Color"]
        if not targets:
            continue
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        mx = nt.nodes.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        mx.blend_type = "MULTIPLY"
        mx.inputs["Factor"].default_value = 1.0
        nt.links.new(diff.outputs["Color"], mx.inputs[6])
        nt.links.new(oi.outputs["Color"], mx.inputs[7])
        for node, sock in targets:
            nt.links.new(mx.outputs[2], sock)
        m["v3_tinted"] = True


def load_apple_mesh():
    folder = os.path.join(ASSETS, "models", "food_apple_01")
    blend = os.path.join(folder, "food_apple_01_2k.blend")
    objs = append_from(blend, "objects", ["food_apple_01"])
    fix_image_paths(folder)
    ob = objs[0]
    SC.collection.objects.link(ob)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.ratio = 0.2
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    me.name = "apple_base"
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(ob.data.materials[0])
    bpy.data.objects.remove(ob, do_unlink=True)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co -= (co.max(0) + co.min(0)) / 2
    diam = (co.max(0) - co.min(0))[:2].mean()
    co *= 0.076 / diam
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    log(f"apple mesh: {len(me.polygons)} polys, diameter {(co.max(0) - co.min(0))[0] * 100:.1f} cm")
    return me


def apple_material(base_mat, name, hue, sat, val, tint, blush=0.0):
    """Apple material from the scanned (red/yellow striped) apple texture: hue
    shift + tint; `blush` blends part of the original red texture back in."""
    m = base_mat.copy()
    m.name = name
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Coat Weight"].default_value = 0.35
    b.inputs["Coat Roughness"].default_value = 0.15
    b.inputs["Subsurface Weight"].default_value = 0.08
    b.inputs["Subsurface Radius"].default_value = (0.03, 0.02, 0.005)
    b.inputs["Subsurface Scale"].default_value = 0.02
    img = [n for n in nt.nodes if n.type == "TEX_IMAGE" and "diff" in n.image.name][0]
    hs = nt.nodes.new("ShaderNodeHueSaturation")
    hs.inputs["Hue"].default_value = hue
    hs.inputs["Saturation"].default_value = sat
    hs.inputs["Value"].default_value = val
    nt.links.new(img.outputs["Color"], hs.inputs["Color"])
    NM = NodeMath(nt.nodes, nt.links)
    col = NM.mix_rgb(hs.outputs["Color"], tint, 1.0, "MULTIPLY")
    if blush > 0:
        col = NM.mix_rgb(col, img.outputs["Color"], blush)
    # slight per-instance brightness variation
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    v = NM.ramp(oi.outputs["Random"], 0.0, 1.0, 0.88, 1.08)
    col = NM.mix_rgb(col, v, 1.0, "MULTIPLY")
    nt.links.new(col, b.inputs["Base Color"])
    return m


# apple colour classes (cultivar type, ripe): (name, hue (0.5 = unchanged), saturation, value, tint rgb, blush)
# The scanned apple texture is red with yellow stripes; the hue shift moves it towards yellow/green.
APPLE_CLASSES = {
    (0, 1): ("apple_red_ripe", 0.50, 1.25, 0.95, (0.95, 0.55, 0.45), 0.0),       # red-type ripe: red / striped
    (0, 0): ("apple_red_unripe", 0.70, 0.65, 0.95, (0.70, 0.86, 0.34), 0.18),     # yellow-green with slight blush
    (1, 1): ("apple_yellow_ripe", 0.62, 0.80, 1.10, (1.00, 0.80, 0.30), 0.0),    # golden yellow
    (1, 0): ("apple_yellow_unripe", 0.70, 0.55, 1.00, (0.74, 0.90, 0.42), 0.0),   # light green
    (2, 1): ("apple_green_ripe", 0.74, 0.60, 1.00, (0.55, 0.85, 0.28), 0.0),      # bright grass green
    (2, 0): ("apple_green_unripe", 0.76, 0.60, 0.80, (0.36, 0.60, 0.20), 0.0),    # darker green
}


class AppleKit:
    """Apple mesh + six colour classes, each as a hidden source object that
    geometry-node instancers point at."""

    def __init__(self):
        self.base = load_apple_mesh()
        base_mat = self.base.materials[0]
        self.src = {}
        for key, (name, hue, sat, val, tint, blush) in APPLE_CLASSES.items():
            mat = apple_material(base_mat, name, hue, sat, val, tint, blush)
            me = self.base.copy()
            me.name = "mesh_" + name
            me.materials[0] = mat
            o = new_obj("src_" + name, me, COLL_ORCHARD, (0, 0, -20.0))
            o.hide_render = True
            o.hide_viewport = True
            self.src[key] = o
        self.n_inst = 0

    def instance(self, name, pts, key, tilt=0.35, seed=1, scale=(0.9, 1.1)):
        """Instance the apple of colour class `key` on the points `pts` (world coords)."""
        if len(pts) == 0:
            return None
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(p) for p in pts], [], [])
        me.update()
        ob = new_obj(name, me, COLL_ORCHARD)
        ng = bpy.data.node_groups.new(name + "_ng", "GeometryNodeTree")
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        N = ng.nodes
        gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
        m2p = N.new("GeometryNodeMeshToPoints")
        ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
        iop = N.new("GeometryNodeInstanceOnPoints")
        oi = N.new("GeometryNodeObjectInfo")
        oi.inputs["Object"].default_value = self.src[key]
        oi.inputs["As Instance"].default_value = True
        ng.links.new(oi.outputs["Geometry"], iop.inputs["Instance"])
        rr = N.new("FunctionNodeRandomValue")
        rr.data_type = "FLOAT_VECTOR"
        rr.inputs["Min"].default_value = (-tilt, -tilt, 0.0)
        rr.inputs["Max"].default_value = (tilt, tilt, 6.283)
        rr.inputs["Seed"].default_value = seed
        ng.links.new(rr.outputs["Value"], iop.inputs["Rotation"])
        rs = N.new("FunctionNodeRandomValue")
        rs.data_type = "FLOAT"
        rs.inputs["Min"].default_value = scale[0]
        rs.inputs["Max"].default_value = scale[1]
        rs.inputs["Seed"].default_value = seed + 1
        ng.links.new(rs.outputs["Value"], iop.inputs["Scale"])
        ng.links.new(m2p.outputs["Points"], iop.inputs["Points"])
        ng.links.new(iop.outputs["Instances"], go.inputs[0])
        mod = ob.modifiers.new("inst", "NODES")
        mod.node_group = ng
        self.n_inst += len(pts)
        return ob


def leaf_points(tree_me):
    li = [i for i, m in enumerate(tree_me.materials) if "leaves" in m.name][0]
    npoly = len(tree_me.polygons)
    mi = np.empty(npoly, dtype=np.int32)
    tree_me.polygons.foreach_get("material_index", mi)
    cen = np.empty(npoly * 3)
    tree_me.polygons.foreach_get("center", cen)
    return cen.reshape(-1, 3)[mi == li]


def xform(pts, M):
    return (np.c_[pts, np.ones(len(pts))] @ np.array(M).T)[:, :3]


# ------------------------------------------------------------------- trees
class TreeKit:
    """Tall-spindle tree = two interleaved, rescaled crowns of the scanned tree
    (upper crown 1.0-3.1 m, lower crown 0.5-1.9 m) around a central stake.
    Per variant a pool of apple slots per canopy face is sampled on the leaf
    clusters, biased to the outer canopy."""

    N_POOL = 520

    def __init__(self, apples, lod_far=1, lod_near=0):
        self.tree_far = load_tree_mesh(lod_far)
        self.tree_near = load_tree_mesh(lod_near) if lod_near != lod_far else self.tree_far
        tint_leaf_material(self.tree_far)
        if self.tree_near is not self.tree_far:
            tint_leaf_material(self.tree_near)
        self.apples = apples
        pts = leaf_points(self.tree_far)
        self.variants = []
        rng = np.random.default_rng(11)
        for v in range(4):
            rz = math.pi / 2 + (math.pi if v % 2 else 0.0)
            mirror = -1.0 if v >= 2 else 1.0
            MA = Matrix.Rotation(rz, 4, "Z") @ Matrix.Diagonal((0.40 * mirror, 0.40, TREE_H / 4.53, 1.0))
            MB = (Matrix.Translation((0, 0, -0.15)) @ Matrix.Rotation(rz + math.pi, 4, "Z")
                  @ Matrix.Diagonal((0.33 * mirror, 0.33, 0.45, 1.0)))
            P = np.concatenate([xform(pts, MA), xform(pts, MB)])
            P = P[(P[:, 2] > 0.5) & (P[:, 2] < TREE_H - 0.05)]
            pools = []
            for face, sgn in ((0, 1.0), (1, -1.0)):
                side = P[sgn * P[:, 1] > 0.08]
                outer = np.abs(side[:, 1]) / (np.abs(side[:, 1]).max() + 1e-6)
                w = outer ** 1.3 + 0.12
                w /= w.sum()
                idx = rng.choice(len(side), size=min(self.N_POOL, len(side)), replace=False, p=w)
                S = side[idx].copy()
                # hang at / just outside the leaf cluster, 4 cm below it
                S[:, 1] += sgn * rng.uniform(0.02, 0.11, len(S))
                S[:, 2] -= 0.04
                outer = np.abs(S[:, 1]) / (np.abs(S[:, 1]).max() + 1e-6)
                upper = (S[:, 2] - 0.5) / (TREE_H - 0.5)
                pools.append((S, 0.5 * outer + 0.3 * upper))
            self.variants.append(dict(MA=MA, MB=MB, pools=pools))
        log(f"tree kit: {len(self.variants)} variants, {self.N_POOL} apple slots per face")

    def frame(self, i):
        rng = random.Random(i * 7919 + 3)
        yaw = rng.uniform(-0.10, 0.10)
        sc = rng.uniform(0.95, 1.05)
        return Matrix.Translation(Vector((XS[i], YS[i], 0.0))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(sc, 4), rng

    def place(self, i, near=False):
        v = self.variants[i % 4]
        R, rng = self.frame(i)
        me = self.tree_near if near else self.tree_far
        a = new_obj(f"tree_{i}a", me, COLL_ORCHARD)
        a.matrix_world = R @ v["MA"]
        b = new_obj(f"tree_{i}b", me, COLL_ORCHARD)
        b.matrix_world = R @ v["MB"]
        # per-tree leaf colour variation + a modest cultivar foliage tint
        # (red-type slightly darker, yellow-type neutral, green-type lighter yellow-green)
        ct = ((0.93, 0.96, 0.94), (1.0, 1.0, 0.93), (1.0, 1.06, 0.86))[int(CULT[i])]
        g = rng.uniform(0.86, 1.06)
        col = (ct[0] * g * rng.uniform(0.95, 1.05), ct[1] * g, ct[2] * g * rng.uniform(0.92, 1.04), 1.0)
        a.color = col
        b.color = col
        return a, b

    def face_slots(self, i, face):
        """(world points sorted by pick priority, ripe count, unripe count) of tree i, face."""
        v = self.variants[i % 4]
        R, rng = self.frame(i)
        S, score = v["pools"][face]
        n_r, n_u = int(R0[i, face]), int(U0[i, face])
        n = min(len(S), n_r + n_u)
        rs = np.random.default_rng(i * 131 + face)
        idx = rs.choice(len(S), size=n, replace=False)
        sc = score[idx] + rs.uniform(0.0, 0.3, n)        # ripe prefers outer/upper, but mixed
        order = idx[np.argsort(-sc)]
        return xform(S[order], R), n_r, n_u


def build_orchard(t, near_center=None, near_radius=0.0, div=APPLE_DIV):
    """Trees, stakes and the apples of every canopy face at mission time t."""
    rem_r, rem_u = fruit_removed(t)
    stake = cylinder_mesh("stake", 0.011, TREE_H - 0.1, segs=8)
    stake.materials.append(simple_material("conduit", (0.55, 0.56, 0.58), rough=0.5, metal=0.8))
    pts = {key: [] for key in APPLE_CLASSES}
    shown = np.zeros(2, int)
    for i in range(NT):
        near = near_center is not None and math.hypot(XS[i] - near_center[0], YS[i] - near_center[1]) < near_radius
        KIT.place(i, near=near)
        new_obj(f"stake_{i}", stake, COLL_ORCHARD, (XS[i], YS[i], 0.0))
        c = int(CULT[i])
        for face in (0, 1):
            P, n_r, n_u = KIT.face_slots(i, face)
            if len(P) == 0:
                continue
            n_r = min(n_r, len(P))
            n_u = min(n_u, len(P) - n_r)
            ripe = P[:n_r][rem_r[i, face]:]            # outer ripe fruit is picked first
            unripe = P[n_r:n_r + n_u][rem_u[i, face]:]
            if div > 1:
                ripe, unripe = ripe[::div], unripe[::div]
            pts[(c, 1)].append(ripe)
            pts[(c, 0)].append(unripe)
            shown += (len(ripe), len(unripe))
    for key, lst in pts.items():
        arr = np.concatenate(lst) if lst else np.zeros((0, 3))
        KIT.apples.instance("apples_" + APPLE_CLASSES[key][0], arr, key, seed=key[0] * 2 + key[1])
    log(f"orchard at t={t:.0f}s: {NT} trees, apples shown ripe={shown[0]} unripe={shown[1]} "
        f"(1 apple per {div} fruit), removed so far ripe={rem_r.sum()} unripe={rem_u.sum()}")
    return rem_r, rem_u


def ripe_targets(q, rem_r):
    """Next-to-pick ripe apples (world points) per face set of stop cell q: {0: pts, 1: pts}."""
    out = {0: [], 1: []}
    for (i, face) in node_faces(q):
        P, n_r, n_u = KIT.face_slots(i, face)
        nxt = P[:min(n_r, len(P))][rem_r[i, face]:rem_r[i, face] + 10]
        if len(nxt):
            out[face].append(nxt)
    return {f: (np.concatenate(v) if v else np.zeros((0, 3))) for f, v in out.items()}


# ------------------------------------------------------------------- primitives
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


def tube_mesh(name, r_out, r_in, h, segs=24):
    """Open ring (prop guard)."""
    me = bpy.data.meshes.new(name)
    vs, fs = [], []
    for k in range(segs):
        a = 2 * math.pi * k / segs
        for r in (r_out, r_in):
            vs.append((r * math.cos(a), r * math.sin(a), 0.0))
            vs.append((r * math.cos(a), r * math.sin(a), h))
    for k in range(segs):
        k2 = (k + 1) % segs
        o0, o1 = 4 * k, 4 * k2
        fs.append((o0, o1, o1 + 1, o0 + 1))
        fs.append((o0 + 2, o0 + 3, o1 + 3, o1 + 2))
        fs.append((o0 + 1, o1 + 1, o1 + 3, o0 + 3))
        fs.append((o0, o0 + 2, o1 + 2, o1))
    me.from_pydata(vs, [], fs)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    return me


def curve_obj(name, pts, radius, mat, coll, cyclic=False, smooth=False):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    cu.resolution_u = 4 if smooth else 1
    cu.use_fill_caps = True
    sp = cu.splines.new("NURBS" if smooth else "POLY")
    sp.points.add(len(pts) - 1)
    for q, p in enumerate(pts):
        sp.points[q].co = (p[0], p[1], p[2], 1.0)
    sp.use_cyclic_u = cyclic
    if smooth:
        sp.order_u = 4
        sp.use_endpoint_u = not cyclic
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    coll.objects.link(o)
    return o


# ------------------------------------------------------------------- trellis
def build_trellis():
    """3-wire trellis: wooden posts every ~9 m (10 trees), wires at 0.9 / 1.8 / 2.7 m."""
    wood = pbr_material("post_wood", os.path.join(ASSETS, "textures", "bark_brown_02"),
                        "bark_brown_02", "1k", scale=0.6, rough_mult=1.2, bump=0.6, tint=(0.75, 0.68, 0.6))
    wire = simple_material("wire", (0.55, 0.55, 0.58), rough=0.35, metal=1.0)
    post_me = cylinder_mesh("post", 0.055, 3.25, segs=10)
    post_me.materials.append(wood)
    wire_me = cylinder_mesh("wire", 0.0025, 1.0, segs=6, cap=False)
    wire_me.materials.append(wire)
    n = 0
    for r in ROWS:
        y = ROW_OFFSET + r * ROW_PITCH
        xs_row = XS[row_ids == r]
        x0, x1 = xs_row.min() - 0.45, xs_row.max() + 0.45
        npost = max(2, int(round((x1 - x0) / 9.0)) + 1)
        for k in range(npost):
            x = x0 + (x1 - x0) * k / (npost - 1)
            new_obj(f"post_{r}_{k}", post_me, COLL_INFRA, (x, y, -0.05),
                    (random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(0, 6.28)))
            n += 1
        for zw in (0.9, 1.8, 2.7):
            new_obj(f"wire_{r}_{zw}", wire_me, COLL_INFRA, (x0, y, zw), (0, math.pi / 2, 0), (1, 1, x1 - x0))
    log(f"trellis: {n} posts, {len(ROWS)} rows")


# ------------------------------------------------------------------- bins, platform, drones
class Mats:
    def __init__(self):
        self.plywood = simple_material("bin_plywood", (0.62, 0.48, 0.30), rough=0.75)
        self.plywood_dark = simple_material("bin_plywood_in", (0.30, 0.22, 0.13), rough=0.85)
        self.pile = simple_material("apple_pile", (0.30, 0.10, 0.05), rough=0.8)
        self.steel = simple_material("steel_painted", (0.18, 0.20, 0.22), rough=0.45, metal=0.3)
        self.alu = simple_material("alu", (0.65, 0.66, 0.68), rough=0.3, metal=1.0)
        self.body = simple_material("platform_body", (0.82, 0.83, 0.80), rough=0.4, coat=0.3)
        self.orange = simple_material("safety_orange", (0.9, 0.25, 0.03), rough=0.45, coat=0.3)
        self.rubber = simple_material("rubber", (0.02, 0.02, 0.02), rough=0.8)
        self.tyre = simple_material("tyre", (0.03, 0.03, 0.03), rough=0.85)
        self.cable = simple_material("tether", (0.03, 0.03, 0.035), rough=0.35, coat=0.3)
        self.carbon = simple_material("carbon", (0.02, 0.02, 0.022), rough=0.32, coat=0.5)
        self.drone_body = simple_material("drone_body", (0.90, 0.90, 0.88), rough=0.35, coat=0.4)
        self.prop_disc = simple_material("prop_disc", (0.04, 0.04, 0.04), rough=0.6, alpha=0.14)
        self.prop_blade = simple_material("prop_blade", (0.06, 0.06, 0.06), rough=0.5, alpha=0.45)
        self.lens = simple_material("lens", (0.02, 0.02, 0.03), rough=0.05, coat=1.0)
        self.cup = simple_material("suction_cup", (0.04, 0.04, 0.045), rough=0.55)
        self.tractor = simple_material("tractor_paint", (0.08, 0.30, 0.12), rough=0.35, coat=0.4)
        self.rim = simple_material("rim_paint", (0.85, 0.75, 0.15), rough=0.4, coat=0.3)
        self.trailer = simple_material("trailer_paint", (0.55, 0.08, 0.06), rough=0.4, coat=0.3)
        self.glass = simple_material("glass", (0.1, 0.12, 0.14), rough=0.05, alpha=0.4, coat=1.0)
        self.led = {}

    def led_mat(self, p):
        if p not in self.led:
            self.led[p] = simple_material(f"led_{p}", PLATFORM_COLORS[p % 16], rough=0.4, emit=3.0)
        return self.led[p]


MT = None   # Mats(), created after the scene setup


def bin_top_layer(cx, cy, z, cls, seed=1):
    """Points of one apple layer on the fill surface of a bin, with classes
    sampled proportional to `cls` (3x2 counts)."""
    rng = np.random.default_rng(seed)
    pitch = 0.080
    pts = []
    nx = int(1.08 / pitch)
    for ix in range(nx):
        for iy in range(nx):
            x = -0.54 + 0.04 + ix * pitch + (0.04 if iy % 2 else 0.0) + rng.uniform(-0.012, 0.012)
            y = -0.54 + 0.04 + iy * pitch * 0.92 + rng.uniform(-0.012, 0.012)
            if abs(x) < 0.55 and abs(y) < 0.55:
                pts.append((cx + x, cy + y, z + rng.uniform(-0.015, 0.012)))
    pts = np.array(pts)
    tot = cls.sum()
    if tot <= 0:
        probs = np.zeros(6)
        probs[0] = 1.0
    else:
        probs = cls.ravel() / tot
    keys = [(c, 1 - r) for c in range(3) for r in range(2)]   # cls[c,0] = ripe, cls[c,1] = unripe
    lab = rng.choice(6, size=len(pts), p=probs)
    return {keys[k]: pts[lab == k] for k in range(6)}


def build_bin(name, loc, yaw, fill=0.0, cls=None, coll=None, seed=1):
    """Standard wooden apple bin 1.2 x 1.2 x 0.75 m; `fill` = fraction of the
    interior height filled with apples (top layer instanced, pile underneath)."""
    coll = coll or COLL_INFRA
    root = bpy.data.objects.new(name, None)
    root.location = loc
    root.rotation_euler = (0, 0, yaw)
    coll.objects.link(root)
    sx, sy, sz = BIN_SZ
    th = 0.03
    parts = []
    # bottom board + three runners
    parts.append((box_mesh(name + "_floor", sx, sy, th), MT.plywood, (0, 0, 0.09)))
    for rx in (-0.5, 0.0, 0.5):
        parts.append((box_mesh(name + "_runner", 0.1, sy, 0.09), MT.plywood_dark, (rx, 0, 0.0)))
    # walls (plywood) and corner posts, battens
    for s in (-1, 1):
        parts.append((box_mesh(name + "_wx", th, sy, sz - 0.09), MT.plywood, (s * (sx / 2 - th / 2), 0, 0.09)))
        parts.append((box_mesh(name + "_wy", sx, th, sz - 0.09), MT.plywood, (0, s * (sy / 2 - th / 2), 0.09)))
        parts.append((box_mesh(name + "_bx", 0.05, sy + 0.02, 0.05), MT.plywood_dark, (s * (sx / 2 + 0.01), 0, sz - 0.05)))
        parts.append((box_mesh(name + "_by", sx + 0.02, 0.05, 0.05), MT.plywood_dark, (0, s * (sy / 2 + 0.01), sz - 0.05)))
        parts.append((box_mesh(name + "_bx2", 0.04, sy + 0.02, 0.05), MT.plywood_dark, (s * (sx / 2 + 0.01), 0, 0.38)))
        parts.append((box_mesh(name + "_by2", sx + 0.02, 0.04, 0.05), MT.plywood_dark, (0, s * (sy / 2 + 0.01), 0.38)))
        for s2 in (-1, 1):
            parts.append((box_mesh(name + "_post", 0.06, 0.06, sz), MT.plywood_dark, (s * (sx / 2 + 0.015), s2 * (sy / 2 + 0.015), 0.0)))
    for q, (me, mat, off) in enumerate(parts):
        me.materials.append(mat)
        new_obj(f"{name}_p{q}", me, coll, off, parent=root)
    if fill > 0.0:
        h = 0.12 + fill * (sz - 0.12 - 0.04)
        pile = box_mesh(name + "_pile", sx - 2 * th - 0.01, sy - 2 * th - 0.01, h - 0.09 - 0.05)
        pile.materials.append(MT.pile)
        new_obj(name + "_pile", pile, coll, (0, 0, 0.09 + th), parent=root)
        R = Matrix.Translation(Vector(loc)) @ Matrix.Rotation(yaw, 4, "Z")
        layer = bin_top_layer(0.0, 0.0, h - 0.025, cls if cls is not None else np.array([[1.0, 0], [0, 0], [0, 0]]), seed)
        for key, pts in layer.items():
            if len(pts):
                KIT.apples.instance(f"{name}_apples_{APPLE_CLASSES[key][0]}", xform(pts, R), key, tilt=1.2, seed=seed + 7)
    return root


class DroneKit:
    """Tethered picking quadcopter: 0.70 m motor-to-motor, prop guards, forward
    suction-cup gripper with vacuum hose, tether anchor underneath."""

    REACH = REACH

    def __init__(self):
        self.parts = self._parts()

    def _parts(self):
        P = []
        P.append((box_mesh("d_body", 0.24, 0.17, 0.08, origin_bottom=False), MT.drone_body, Matrix.Translation((0, 0, 0))))
        P.append((box_mesh("d_top", 0.16, 0.12, 0.04, origin_bottom=False), MT.carbon, Matrix.Translation((-0.01, 0, 0.06))))
        P.append((box_mesh("d_vac", 0.14, 0.09, 0.06, origin_bottom=False), MT.carbon, Matrix.Translation((-0.06, 0, -0.07))))
        arm = cylinder_mesh("d_arm", 0.012, 0.27, segs=10)
        motor = cylinder_mesh("d_motor", 0.022, 0.03, segs=14)
        disc = cylinder_mesh("d_disc", 0.145, 0.003, segs=40)
        blade = box_mesh("d_blade", 0.28, 0.024, 0.004, origin_bottom=False)
        guard = tube_mesh("d_guard", 0.165, 0.158, 0.014, segs=28)
        for (sx, sy) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            ang = math.atan2(sy, sx)
            R = Matrix.Rotation(ang, 4, "Z")
            P.append((arm, MT.carbon, Matrix.Translation((sx * 0.09, sy * 0.065, 0.01)) @ R @ Matrix.Rotation(math.pi / 2, 4, "Y")))
            mx, my = sx * 0.09 + 0.26 * math.cos(ang), sy * 0.065 + 0.26 * math.sin(ang)
            P.append((motor, MT.alu, Matrix.Translation((mx, my, 0.01))))
            P.append((disc, MT.prop_disc, Matrix.Translation((mx, my, 0.05))))
            P.append((blade, MT.prop_blade, Matrix.Translation((mx, my, 0.052)) @ Matrix.Rotation(random.uniform(0, 3.14), 4, "Z")))
            P.append((guard, MT.drone_body, Matrix.Translation((mx, my, 0.04))))
            for a2 in (ang + 0.5, ang - 0.5):
                P.append((cylinder_mesh("d_gstrut", 0.004, 0.16, segs=6), MT.drone_body,
                          Matrix.Translation((mx, my, 0.045)) @ Matrix.Rotation(a2, 4, "Z") @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        # landing feet
        leg = cylinder_mesh("d_leg", 0.006, 0.12, segs=8)
        for sy in (-1, 1):
            for sx in (-1, 1):
                P.append((leg, MT.carbon, Matrix.Translation((sx * 0.08, sy * 0.07, -0.16)) @ Matrix.Rotation(sy * -0.2, 4, "X")))
        # stereo camera pod on the front
        cam = box_mesh("d_cam", 0.03, 0.11, 0.03, origin_bottom=False)
        P.append((cam, MT.carbon, Matrix.Translation((0.125, 0, 0.0))))
        lens = cylinder_mesh("d_lens", 0.009, 0.004, segs=12)
        for sy in (-0.035, 0.035):
            P.append((lens, MT.lens, Matrix.Translation((0.142, sy, 0.0)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        # suction gripper: tube forward, bellows, cup; tip at x = REACH
        tube = cylinder_mesh("d_tube", 0.014, 0.30, segs=10)
        P.append((tube, MT.carbon, Matrix.Translation((0.12, 0, -0.03)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        for q in range(4):
            ring = cylinder_mesh("d_bellow", 0.024 - 0.002 * (q % 2), 0.012, segs=14)
            P.append((ring, MT.cup, Matrix.Translation((0.42 + 0.014 * q, 0, -0.03)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        cup = cylinder_mesh("d_cup", 0.034, 0.025, segs=18, cap=False)
        P.append((cup, MT.cup, Matrix.Translation((self.REACH - 0.025, 0, -0.03)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        cupb = cylinder_mesh("d_cupb", 0.022, 0.02, segs=14)
        P.append((cupb, MT.cup, Matrix.Translation((self.REACH - 0.045, 0, -0.03)) @ Matrix.Rotation(math.pi / 2, 4, "Y")))
        # tether anchor + swivel below the body
        P.append((cylinder_mesh("d_anchor", 0.02, 0.03, segs=10), MT.alu, Matrix.Translation((-0.02, 0, -0.10))))
        P.append((cylinder_mesh("d_ant", 0.004, 0.08, segs=6), MT.carbon, Matrix.Translation((-0.1, 0.05, 0.04))))
        return P

    def make(self, name, p):
        root = bpy.data.objects.new(name, None)
        root.empty_display_size = 0.3
        COLL_FLEET.objects.link(root)
        for q, (me, mat, M) in enumerate(self.parts):
            if not me.materials:
                me.materials.append(mat)
            o = new_obj(f"{name}_p{q}", me, COLL_FLEET, parent=root)
            o.matrix_basis = M
            if me.name in ("d_body", "d_top", "d_vac", "d_cam"):
                bv = o.modifiers.new("bevel", "BEVEL")
                bv.width = 0.008
                bv.segments = 3
        led = MT.led_mat(p)
        stripe = box_mesh("d_stripe", 0.10, 0.012, 0.006, origin_bottom=False)
        stripe.materials.append(led)
        for sy in (-1, 1):
            o = new_obj(f"{name}_led{sy}", stripe, COLL_FLEET, parent=root)
            o.matrix_basis = Matrix.Translation((-0.06, sy * 0.09, 0.03))
        return root

    @staticmethod
    def pose(root, d):
        root.location = (d["x"], d["y"], d["z"])
        root.rotation_euler = Euler((0.0, -d.get("pitch", 0.0), d["yaw"]), "XYZ")

    def hose(self, name, root_loc, yaw):
        """Vacuum hose from the gripper tube to the pump box (illustration)."""
        R = Matrix.Translation(Vector(root_loc)) @ Matrix.Rotation(yaw, 4, "Z")
        pts = [R @ Vector((0.30, 0.0, -0.045)), R @ Vector((0.20, 0.03, -0.10)), R @ Vector((0.02, 0.03, -0.11))]
        return curve_obj(name, pts, 0.008, MT.cable, COLL_FLEET, smooth=True)


def tether(name, hub, anchor, sag=None):
    hub, anchor = Vector(hub), Vector(anchor)
    L = (anchor - hub).length
    sag = 0.10 * L if sag is None else sag
    pts = []
    for k in range(9):
        u = k / 8
        p = hub.lerp(anchor, u)
        p.z -= sag * math.sin(math.pi * u)
        pts.append(p)
    return curve_obj(name, pts, 0.0045, MT.cable, COLL_FLEET, smooth=True)


class PlatformKit:
    """Small electric orchard platform 2.5 x 1.2 m, deck 0.45 m, four wheels,
    battery pack between the wheels, one apple bin on the rear deck, tether
    mast (2.75 m hub) and the fruit-detection camera pods on the mast."""

    def __init__(self):
        self.dk = DroneKit()

    def make(self, p, ps, t, drone_specs, cables=True):
        name = f"platform_{p}"
        root = bpy.data.objects.new(name, None)
        root.location = (ps["x"], ps["y"], 0.0)
        root.rotation_euler = (0, 0, ps["yaw"])
        COLL_FLEET.objects.link(root)

        def part(me, mat, loc, rot=(0, 0, 0), bevel=0.0):
            if not me.materials:
                me.materials.append(mat)
            o = new_obj(f"{name}_{me.name}", me, COLL_FLEET, loc, rot, parent=root)
            if bevel > 0:
                bv = o.modifiers.new("bevel", "BEVEL")
                bv.width = bevel
                bv.segments = 3
            return o

        part(box_mesh("pl_deck", 2.5, 1.2, 0.06), MT.body, (0, 0, DECK_Z - 0.06), bevel=0.01)
        part(box_mesh("pl_frame", 2.3, 1.0, 0.12), MT.steel, (0, 0, DECK_Z - 0.18))
        part(box_mesh("pl_batt", 1.1, 0.8, 0.22), MT.steel, (-0.15, 0, 0.10), bevel=0.01)
        for s in (-1, 1):
            part(box_mesh("pl_rail", 2.3, 0.04, 0.05), MT.orange, (0, s * 0.58, DECK_Z))
            part(box_mesh("pl_bumper", 0.06, 1.1, 0.08), MT.orange, (s * 1.25, 0, 0.30))
            part(box_mesh("pl_id", 0.6, 0.012, 0.10), MT.led_mat(p), (0.55, s * 0.606, DECK_Z - 0.17))
        tyre = cylinder_mesh("pl_tyre", 0.30, 0.18, segs=24)
        rim = cylinder_mesh("pl_rim", 0.17, 0.19, segs=16)
        for sx in (-0.95, 0.95):
            for sy in (-0.52, 0.52):
                part(tyre, MT.tyre, (sx, sy - 0.09, 0.30), (math.pi / 2, 0, 0))
                part(rim, MT.alu, (sx, sy - 0.095, 0.30), (math.pi / 2, 0, 0))
        # mast with tether hub, GNSS antenna, two camera pods per side (fruit detection)
        hx, hy, hz = HUB_LOCAL
        part(cylinder_mesh("pl_mast", 0.035, hz - DECK_Z - 0.15, segs=12), MT.alu, (hx, 0, DECK_Z))
        part(box_mesh("pl_hub", 0.34, 0.34, 0.22), MT.steel, (hx, 0, hz - 0.30), bevel=0.01)
        part(cylinder_mesh("pl_hubtop", 0.09, 0.06, segs=16), MT.body, (hx, 0, hz - 0.08))
        part(cylinder_mesh("pl_gnss", 0.06, 0.03, segs=16), MT.body, (hx, 0, hz - 0.02))
        for (sx, sy) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            a = math.atan2(sy, sx)
            part(cylinder_mesh("pl_hubarm", 0.012, 0.22, segs=8), MT.alu,
                 (hx + 0.17 * math.cos(a), 0.17 * math.sin(a), hz - 0.14), (0, math.pi / 2, a))
        for s in (-1, 1):
            part(box_mesh("pl_campod", 0.08, 0.05, 0.14), MT.carbon, (hx, s * 0.06, 1.95), bevel=0.005)
            part(box_mesh("pl_campod2", 0.08, 0.05, 0.14), MT.carbon, (hx, s * 0.06, 1.35), bevel=0.005)
        part(box_mesh("pl_ctrl", 0.35, 0.3, 0.25), MT.body, (hx - 0.35, 0.0, DECK_Z), bevel=0.01)
        part(cylinder_mesh("pl_beacon", 0.04, 0.07, segs=12), MT.orange, (hx - 0.35, 0.0, DECK_Z + 0.25))
        # bin on the rear deck
        fill, picked, cls = bin_fill(p, t) if ps["state"] != "station" else (0.0, 0, None)
        R = Matrix.Translation(Vector((ps["x"], ps["y"], 0))) @ Matrix.Rotation(ps["yaw"], 4, "Z")
        bloc = R @ Vector((-0.6, 0, DECK_Z))
        build_bin(f"{name}_bin", tuple(bloc), ps["yaw"], fill=fill, cls=cls, coll=COLL_FLEET, seed=p + 100)
        # drones + tethers
        hub_w = R @ Vector(HUB_LOCAL)
        for j, d in enumerate(drone_specs):
            droot = self.dk.make(f"{name}_drone_{j}", p)
            self.dk.pose(droot, d)
            self.dk.hose(f"{name}_hose_{j}", (d["x"], d["y"], d["z"]), d["yaw"])
            if cables:
                tether(f"{name}_tether_{j}", hub_w, (d["x"] - 0.02 * math.cos(d["yaw"]), d["y"] - 0.02 * math.sin(d["yaw"]), d["z"] - 0.11))
        return root, fill, picked


def drone_specs(ps, rem_r, t, seed=0):
    """Poses of the four drones of a platform with pose `ps` at time t.
    pick: two drones per canopy face, suction cup on the next ripe apple;
    move/station: escort formation hovering above the deck."""
    R = Matrix.Translation(Vector((ps["x"], ps["y"], 0))) @ Matrix.Rotation(ps["yaw"], 4, "Z")
    if ps["state"] != "pick" or ps["node"] is None:
        out = []
        for (ex, ey, ez) in ESCORT:
            w = R @ Vector((ex, ey, ez))
            out.append(dict(x=w.x, y=w.y, z=ez, yaw=ps["yaw"], pitch=math.radians(-4) if ps["state"] == "move" else 0.0, target=None))
        return out
    q = ps["node"]
    tg = ripe_targets(q, rem_r)
    ay = NODE_ALLEY_Y[q]
    rng = random.Random(seed * 7 + q)
    # face 0 = trees south of the alley (+y face), gripper points -y; face 1 = north row, gripper points +y
    faces = [f for f in (0, 1) if node_faces_has(q, f)]
    if not faces:
        faces = [0, 1]
    plan = []
    for k in range(M_DRONES):
        f = faces[k % len(faces)]
        kk = k // len(faces)
        xo = (-1.55, 1.55, -0.6, 0.6)[kk] if len(faces) == 1 else (-1.55, 1.55)[kk]
        zo = HOVER_Z[(kk + f) % 2]
        plan.append((f, xo, zo))
    out = []
    used = []
    for (f, xo, zo) in plan:
        gy = -1.0 if f == 0 else 1.0          # gripper direction
        yaw = -math.pi / 2 if f == 0 else math.pi / 2
        P = tg[f]
        tx = ps["x"] + xo
        if len(P):
            cost = np.abs(P[:, 0] - tx) + 0.8 * np.abs(P[:, 2] - zo)
            for u in used:
                cost += np.where(np.linalg.norm(P - u, axis=1) < 0.9, 5.0, 0.0)
            j = int(np.argmin(cost))
            a = P[j]
            used.append(a)
            cx, cy, cz = a[0], a[1] - gy * (REACH + 0.035), a[2] + 0.03
            out.append(dict(x=float(cx), y=float(cy), z=float(cz), yaw=yaw, target=a))
        else:   # nothing ripe left on this face: scanning hover
            fy = ay + gy * (ROW_PITCH / 2 - CANOPY_HALF - REACH - 0.25)
            out.append(dict(x=tx, y=fy, z=zo, yaw=yaw, target=None))
    return out


def node_faces_has(q, f):
    return len(NODES["trees_up"][q] if f == 0 else NODES["trees_lo"][q]) > 0


# ------------------------------------------------------------------- bin station
def build_station(t, near=False):
    """Bin station on the headland: full and empty bins on the ground, a tractor
    with front forks and a bin trailer (illustration; position from station[])."""
    sx, sy = STATION
    n_full = min(8, trips_done(t))
    rng = random.Random(3)
    bx = sx - 4.6
    # full bins (two rows), empty bins stacked
    cls_full = np.array([[0.75, 0.1], [0.0, 0.0], [0.0, 0.0]])
    for k in range(n_full):
        row, col = divmod(k, 4)
        build_bin(f"stbin_full_{k}", (bx - row * 1.5, sy - 3.2 + col * 1.45, 0.0), rng.uniform(-0.08, 0.08),
                  fill=rng.uniform(0.9, 1.0), cls=np.array([[rng.uniform(0.6, 0.9), 0.1], [rng.uniform(0, 0.3), 0.05], [rng.uniform(0, 0.2), 0.02]]), seed=40 + k)
    for k in range(4):
        row, col = divmod(k, 2)
        build_bin(f"stbin_empty_{k}", (bx - 3.2, sy + 3.6 + col * 1.45, row * 0.78), rng.uniform(-0.05, 0.05), fill=0.0)
    # tractor (compact orchard tractor, front forks) + flat bin trailer with two full bins,
    # parked behind the full bins, facing +y along the headland
    tx, ty = sx - 8.6, sy - 7.5
    tyaw = math.pi / 2
    troot = bpy.data.objects.new("tractor", None)
    troot.location = (tx, ty, 0)
    troot.rotation_euler = (0, 0, tyaw)
    COLL_INFRA.objects.link(troot)

    def tp(me, mat, loc, rot=(0, 0, 0), bevel=0.0):
        if not me.materials:
            me.materials.append(mat)
        o = new_obj("tr_" + me.name, me, COLL_INFRA, loc, rot, parent=troot)
        if bevel:
            bv = o.modifiers.new("bevel", "BEVEL")
            bv.width = bevel
            bv.segments = 3
        return o

    tp(box_mesh("tr_chassis", 2.6, 0.7, 0.4), MT.steel, (0.1, 0, 0.45))
    tp(box_mesh("tr_hood", 1.5, 0.85, 0.6), MT.tractor, (0.85, 0, 0.75), bevel=0.04)
    tp(box_mesh("tr_grille", 0.08, 0.7, 0.5), MT.steel, (1.62, 0, 0.8))
    tp(box_mesh("tr_floor", 1.1, 1.3, 0.1), MT.steel, (-0.45, 0, 0.95))
    tp(box_mesh("tr_seatb", 0.45, 0.5, 0.45), MT.rubber, (-0.85, 0, 1.05), bevel=0.03)
    tp(box_mesh("tr_seatk", 0.1, 0.5, 0.55), MT.rubber, (-1.05, 0, 1.45), bevel=0.03)
    tp(box_mesh("tr_dash", 0.3, 0.6, 0.5), MT.tractor, (-0.1, 0, 1.05), bevel=0.02)
    tp(cylinder_mesh("tr_wheelcol", 0.025, 0.5, segs=8), MT.steel, (-0.25, 0, 1.3), (0, 0.8, 0))
    tp(cylinder_mesh("tr_steer", 0.18, 0.02, segs=20, cap=False), MT.rubber, (-0.55, 0, 1.63), (0, 0.8, 0))
    for s in (-1, 1):   # ROPS frame + roof
        tp(cylinder_mesh("tr_rops", 0.035, 1.6, segs=10), MT.steel, (-0.35, s * 0.62, 1.0))
        tp(cylinder_mesh("tr_rops2", 0.035, 1.6, segs=10), MT.steel, (-1.35, s * 0.62, 1.0))
        tp(cylinder_mesh("tr_fender", 0.72, 0.3, segs=24, cap=False), MT.tractor, (-0.95, s * 0.75 + (0.0 if s < 0 else -0.3), 0.72), (math.pi / 2, 0, 0))
    tp(box_mesh("tr_roof", 1.4, 1.5, 0.08), MT.tractor, (-0.85, 0, 2.6), bevel=0.02)
    tp(cylinder_mesh("tr_exhaust", 0.04, 1.3, segs=10), MT.steel, (0.6, 0.3, 1.0))
    big = cylinder_mesh("tr_bigtyre", 0.72, 0.38, segs=28)
    bigrim = cylinder_mesh("tr_bigrim", 0.42, 0.40, segs=20)
    small = cylinder_mesh("tr_smalltyre", 0.42, 0.24, segs=24)
    smallrim = cylinder_mesh("tr_smallrim", 0.24, 0.26, segs=16)
    for s in (-1, 1):
        tp(big, MT.tyre, (-0.95, s * 0.72 - 0.19, 0.72), (math.pi / 2, 0, 0))
        tp(bigrim, MT.rim, (-0.95, s * 0.72 - 0.20, 0.72), (math.pi / 2, 0, 0))
        tp(small, MT.tyre, (1.05, s * 0.60 - 0.12, 0.42), (math.pi / 2, 0, 0))
        tp(smallrim, MT.rim, (1.05, s * 0.60 - 0.13, 0.42), (math.pi / 2, 0, 0))
    # front bin forks (raised, empty)
    tp(box_mesh("tr_forkmast", 0.08, 1.1, 1.3), MT.steel, (1.85, 0, 0.25))
    for s in (-1, 1):
        tp(box_mesh("tr_forkarm", 0.1, 0.1, 0.6), MT.steel, (1.75, s * 0.45, 0.6))
        tp(box_mesh("tr_fork", 1.15, 0.1, 0.04), MT.steel, (2.45, s * 0.45, 0.52))
    # trailer behind (drawbar along -x in tractor frame)
    tp(box_mesh("tr_drawbar", 1.4, 0.1, 0.1), MT.steel, (-2.5, 0, 0.5))
    tp(box_mesh("tr_tdeck", 3.6, 1.5, 0.08), MT.trailer, (-4.9, 0, 0.62))
    tp(box_mesh("tr_tframe", 3.4, 1.2, 0.14), MT.steel, (-4.9, 0, 0.48))
    tt = cylinder_mesh("tr_ttyre", 0.40, 0.22, segs=24)
    for s in (-1, 1):
        for xx in (-4.4, -5.4):
            tp(tt, MT.tyre, (xx, s * 0.72 - 0.11, 0.40), (math.pi / 2, 0, 0))
    Rt = Matrix.Translation(Vector((tx, ty, 0))) @ Matrix.Rotation(tyaw, 4, "Z")
    for k, xx in enumerate((-4.15, -5.6)):
        w = Rt @ Vector((xx, 0, 0.66))
        build_bin(f"trailer_bin_{k}", (w.x, w.y, w.z), tyaw, fill=0.95,
                  cls=np.array([[0.7, 0.1], [0.1, 0.0], [0.1, 0.0]]), seed=60 + k)
    # bin-station marker: a few pallets and a small shade/tool cabinet
    cab = box_mesh("st_cabinet", 0.6, 0.9, 1.1)
    cab.materials.append(simple_material("cabinet_paint", (0.55, 0.57, 0.60), rough=0.4, metal=0.2))
    new_obj("st_cabinet", cab, COLL_INFRA, (bx - 3.6, sy - 5.2, 0), (0, 0, 0.3))
    log(f"bin station: {n_full} full bins on the ground ({trips_done(t)} trips completed by t)")


# ------------------------------------------------------------------- overlay / lighting / cameras
def platform_ring(p, ps, z=3.7, r=2.3):
    n = 32
    pts = [(ps["x"] + r * math.cos(2 * math.pi * q / n), ps["y"] + r * math.sin(2 * math.pi * q / n), z) for q in range(n)]
    o = curve_obj(f"ring_{p}", pts, 0.07, emission_material(f"ring_{p}_m", PLATFORM_COLORS[p % 16], 1.3), COLL_OVERLAY, cyclic=True, smooth=True)
    o.visible_shadow = False
    o.visible_diffuse = False
    o.visible_glossy = False
    return o


def build_lighting(sun_az=math.radians(215), sun_el=math.radians(48), strength=0.75, energy=4.2):
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
    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs[0], out.inputs[0])
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = energy
    sun.angle = math.radians(0.6)
    sun.color = (1.0, 0.96, 0.90)
    so = bpy.data.objects.new("sun", sun)
    SC.collection.objects.link(so)
    so.rotation_euler = Euler((math.pi / 2 - sun_el, 0, sun_az), "XYZ")


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


def fit_overview_camera(pitch_deg=34.0, lens=32.0, az_deg=42.0):
    """Oblique aerial view looking along azimuth az_deg (0 = +x) over the whole
    block; from the south-west corner by default so that the south canopy face
    of every row (and its fruit colour) is visible."""
    from bpy_extras.object_utils import world_to_camera_view
    pts = [Vector((x, y, z)) for x in (FIELD[0] - 4, FIELD[1] + 1) for y in (FIELD[2] - 1, FIELD[3] + 1) for z in (0, 4)]
    pts.append(Vector((STATION[0] - 10, STATION[1] - 8, 0)))
    pts.append(Vector((STATION[0] - 10, STATION[1] + 4, 0)))
    cx = (FIELD[0] + FIELD[1]) / 2
    cy = (FIELD[2] + FIELD[3]) / 2
    pitch = math.radians(pitch_deg)
    az = math.radians(az_deg)
    cam = make_camera("cam_overview", (cx - 10, cy, 10), (cx, cy, 0), lens=lens)
    cam.rotation_euler = Euler((math.pi / 2 - pitch, 0, az - math.pi / 2), "XYZ")
    dist = FIELD_SIZE
    for _ in range(40):
        cam.location = Vector((cx, cy, 0)) + Vector((-math.cos(pitch) * dist * math.cos(az), -math.cos(pitch) * dist * math.sin(az), math.sin(pitch) * dist))
        bpy.context.view_layer.update()
        uv = [world_to_camera_view(SC, cam, p) for p in pts]
        xs = [u.x for u in uv]
        ys = [u.y for u in uv]
        ok = min(xs) > 0.015 and max(xs) < 0.985 and min(ys) > 0.03 and max(ys) < 0.97
        if ok:
            over = max(0.015 - min(xs), max(xs) - 0.985, 0.03 - min(ys), max(ys) - 0.97)
            if over > -0.015:
                break
            dist *= 0.97
        else:
            dist *= 1.05
    for _ in range(3):
        bpy.context.view_layer.update()
        uv = [world_to_camera_view(SC, cam, p) for p in pts]
        ys = [u.y for u in uv]
        mid = 0.5 * (min(ys) + max(ys))
        vfov = 2 * math.atan(0.5 * cam.data.sensor_width * SC.render.resolution_y / SC.render.resolution_x / cam.data.lens)
        cam.rotation_euler.x -= (0.5 - mid) * vfov * 0.9
    log(f"overview camera: dist {dist:.1f} m, height {cam.location.z:.1f} m, pitch {90 - math.degrees(cam.rotation_euler.x):.1f} deg")
    return cam


# ------------------------------------------------------------------- time selection rules
def choose_overview_time():
    """Sample t every 60 s in [25 %, 65 %] of the mission; maximise
    10 * min(12, alleys with a platform at a pick stop) + 5 * min(2, platforms moving)
    + 2 if at least one platform is at the station; earliest wins ties."""
    best = None
    t = 0.25 * MISSION_S
    while t <= 0.65 * MISSION_S:
        poses = [platform_pose(p, t) for p in range(K)]
        alleys = {int(NODES["alley"][ps["node"]]) for ps in poses if ps["state"] == "pick" and ps["node"] is not None}
        n_move = sum(1 for ps in poses if ps["state"] == "move")
        n_st = sum(1 for ps in poses if ps["state"] == "station")
        score = 10 * min(12, len(alleys)) + 5 * min(2, n_move) + (2 if n_st else 0)
        if best is None or score > best[0]:
            best = (score, t, len(alleys), n_move, n_st)
        t += 60.0
    log(f"overview time rule -> t={best[1]:.0f}s: {best[2]} alleys with a picking platform, {best[3]} moving, {best[4]} at station")
    return best[1]


def choose_platform_stop():
    """Among all picking stops with both canopy faces present and bin fill in
    [0.35, 0.75] at mid-stop, maximise min(ripe fruit remaining on the two faces)
    (ties: red cultivar first). Returns (platform, t_mid, node)."""
    best = None
    for p in range(K):
        if OPT["platform"] is not None and p != OPT["platform"]:
            continue
        for (ta, tl, q, x, y, ti) in STOPS[p]:
            if not (node_faces_has(q, 0) and node_faces_has(q, 1)) or tl - ta < 30:
                continue
            tm = 0.5 * (ta + tl)
            fill, _, _ = bin_fill(p, tm)
            if not (0.35 <= fill <= 0.75):
                continue
            faces = node_faces(q)
            c = int(np.bincount(CULT[[i for i, _ in faces]], minlength=3).argmax())
            r_up = sum(R0[i, 0] for i in NODES["trees_up"][q])
            r_lo = sum(R0[i, 1] for i in NODES["trees_lo"][q])
            score = min(r_up, r_lo) + (0.5 if c == 0 else 0.0)
            if best is None or score > best[0]:
                best = (score, p, tm, q, ta, tl, fill)
    log(f"platform stop rule -> platform {best[1]} at cell {best[3]} (stop {best[4]:.0f}-{best[5]:.0f}s), t={best[2]:.0f}s, fill {best[6]:.2f}")
    return best[1], best[2], best[3]


def choose_closeup_stop():
    """Stop + canopy face for the close-up: among all picking stops, the face set
    (trees_up or trees_lo of the cell) whose remaining ripe fraction at mid-stop
    is closest to 0.5 with at least 40 fruit remaining (red-type cultivar
    preferred, both faces present). Pre-filtered by the initial counts to keep
    it fast. Returns (platform, t_mid, node, face)."""
    cand = []
    for p in range(K):
        if OPT["platform"] is not None and p != OPT["platform"]:
            continue
        for (ta, tl, q, x, y, ti) in STOPS[p]:
            if not (node_faces_has(q, 0) and node_faces_has(q, 1)) or tl - ta < 30:
                continue
            for f in (0, 1):
                idx = NODES["trees_up"][q] if f == 0 else NODES["trees_lo"][q]
                nr, nu = int(R0[idx, f].sum()), int(U0[idx, f].sum())
                c = int(np.bincount(CULT[idx], minlength=3).argmax())
                if nr + nu < 60:
                    continue
                cand.append((abs(nr / (nr + nu) - 0.5) + (0.0 if c == 0 else 0.15), p, 0.5 * (ta + tl), q, f))
    cand.sort()
    best = None
    for (_, p, tm, q, f) in cand[:40]:
        rem_r, rem_u = fruit_removed(tm)
        idx = NODES["trees_up"][q] if f == 0 else NODES["trees_lo"][q]
        nr = int((R0[idx, f] - rem_r[idx, f]).sum())
        nu = int((U0[idx, f] - rem_u[idx, f]).sum())
        if nr + nu < 40 or nr < 8:
            continue
        c = int(np.bincount(CULT[idx], minlength=3).argmax())
        score = -abs(nr / (nr + nu) - 0.5) - (0.0 if c == 0 else 0.15)
        if best is None or score > best[0]:
            best = (score, p, tm, q, f, nr, nu)
    log(f"close-up rule -> platform {best[1]} at cell {best[3]}, face {best[4]}, t={best[2]:.0f}s, remaining ripe {best[5]} / unripe {best[6]}")
    return best[1], best[2], best[3], best[4]


def choose_station_time():
    """For every trip end (platform arriving at the station with a full bin) take
    t = t_arrival - 9 s (platform ~9 m from the station); maximise
    (other platforms at the station) * 2 + min(trips completed, 4); earliest wins ties."""
    best = None
    for p in range(K):
        if OPT["platform"] is not None and p != OPT["platform"]:
            continue
        for ti, (a, b) in enumerate(TRIP_SPAN[p]):
            if ti == len(TRIP_SPAN[p]) - 1 and b >= MISSION_S - 1:
                pass
            t = b - 9.0
            if t <= a:
                continue
            others = sum(1 for pp in range(K) if pp != p and platform_pose(pp, t)["state"] == "station")
            score = 2 * min(others, 2) + min(trips_done(t), 4)
            if best is None or score > best[0]:
                best = (score, p, t, others)
    log(f"station time rule -> platform {best[1]} returning, t={best[2]:.0f}s, {best[3]} other platform(s) at the station")
    return best[1], best[2]


# ------------------------------------------------------------------- shots
MT = Mats()
APPLES = AppleKit()
KIT = TreeKit(APPLES, lod_far=1, lod_near=(1 if OPT["preview"] or OPT["shot"] == "overview" else 0))
PK = PlatformKit()


def place_fleet(t, rem_r, rings=False, cables=True, focus_p=None):
    poses = all_platform_poses(t)
    out = []
    for p, ps in enumerate(poses):
        specs = drone_specs(ps, rem_r, t, seed=p)
        root, fill, picked = PK.make(p, ps, t, specs, cables=cables)
        if rings:
            platform_ring(p, ps)
        out.append((ps, specs, fill, picked))
    st = {}
    for ps, _, _, _ in out:
        st[ps["state"]] = st.get(ps["state"], 0) + 1
    log(f"fleet at t={t:.0f}s: {st}")
    return out


def shot_overview():
    t = OPT["t"] if OPT["t"] is not None else choose_overview_time()
    build_ground()
    build_trellis()
    rem_r, _ = build_orchard(t)
    build_station(t)
    place_fleet(t, rem_r, rings=True, cables=False)
    build_lighting(sun_az=math.radians(300), sun_el=math.radians(48))
    fit_overview_camera()
    return t


def shot_platform(closeup=False):
    want_face = None
    if closeup:
        p, t, q, want_face = choose_closeup_stop()
    else:
        p, t, q = choose_platform_stop()
    if OPT["t"] is not None:
        t = OPT["t"]
    ps = platform_pose(p, t)
    cx, cy = ps["x"], ps["y"]
    build_ground()
    build_trellis()
    rem_r, _ = build_orchard(t, near_center=(cx, cy), near_radius=14.0)
    build_station(t)
    fleet = place_fleet(t, rem_r)
    build_grass_scatter((cx, cy), radius=16.0)
    build_lighting(sun_az=math.radians(300), sun_el=math.radians(50))
    specs = fleet[p][1]
    dirx = math.cos(ps["yaw"])
    log(f"platform {p}: state {ps['state']}, bin fill {fleet[p][2]:.2f} ({fleet[p][3]} fruit), yaw {math.degrees(ps['yaw']):.0f} deg")
    if not closeup:
        loc = Vector((cx - 10.5 * dirx, cy - 0.7, 4.6))
        tgt = Vector((cx + 0.3 * dirx, cy, 1.5))
        make_camera("cam_platform", loc, tgt, lens=40.0, fstop=8.0, focus=(Vector((cx, cy, 1.5)) - loc).length)
    else:
        # the drone on the wanted face whose suction cup is on an apple (lower one preferred)
        rem_r2, rem_u2 = fruit_removed(t)
        best = None
        for j, d in enumerate(specs):
            if d["target"] is None:
                continue
            f = 0 if d["yaw"] < 0 else 1
            if want_face is not None and f != want_face:
                continue
            idx = NODES["trees_up"][q] if f == 0 else NODES["trees_lo"][q]
            nr = sum(R0[i, f] - rem_r2[i, f] for i in idx)
            nu = sum(U0[i, f] - rem_u2[i, f] for i in idx)
            score = -abs(nr / max(nr + nu, 1) - 0.5) - 0.1 * d["z"]
            if best is None or score > best[0]:
                best = (score, j, nr, nu)
        j = best[1]
        d = specs[j]
        log(f"close-up: drone {j} (face {'south' if d['yaw'] < 0 else 'north'} row), remaining ripe {best[2]} / unripe {best[3]} on that face")
        tip = Vector((d["x"] + REACH * math.cos(d["yaw"]), d["y"] + REACH * math.sin(d["yaw"]), d["z"] - 0.03))
        side = -1.0 if d["x"] > cx else 1.0            # towards the platform along the row
        back = -math.sin(d["yaw"])                      # out into the alley
        loc = Vector((tip.x + 1.3 * side, tip.y + 0.75 * back, tip.z + 0.95))
        tgt = Vector((tip.x, tip.y + 0.10 * back, tip.z - 0.02))
        make_camera("cam_closeup", loc, tgt, lens=55.0, fstop=4.0, focus=(tip - loc).length)
    return t


def shot_station():
    p, t = choose_station_time()
    if OPT["t"] is not None:
        t = OPT["t"]
    build_ground()
    build_trellis()
    rem_r, _ = build_orchard(t, near_center=STATION, near_radius=16.0)
    build_station(t)
    fleet = place_fleet(t, rem_r)
    build_grass_scatter((STATION[0] - 2, STATION[1] + 3), radius=18.0, density=30.0)
    build_lighting(sun_az=math.radians(200), sun_el=math.radians(45))
    ps = fleet[p][0]
    log(f"returning platform {p}: at ({ps['x']:.1f}, {ps['y']:.1f}) state {ps['state']}, bin fill {fleet[p][2]:.2f}")
    sx, sy = STATION
    loc = Vector((sx - 12.5, sy + 13.0, 5.2))
    tgt = Vector((sx - 3.5, sy - 1.5, 0.9))
    make_camera("cam_station", loc, tgt, lens=32.0, fstop=8.0, focus=(Vector((sx - 3, sy - 1, 1)) - loc).length)
    return t


def shot_swatch():
    """Six apple colour classes side by side (colour check / documentation)."""
    build_lighting(sun_az=math.radians(300), sun_el=math.radians(50))
    g = box_mesh("sw_ground", 4, 4, 0.02)
    g.materials.append(simple_material("sw_grey", (0.4, 0.4, 0.4), rough=0.8))
    new_obj("sw_ground", g, COLL_INFRA, (0, 0, -0.05))
    rng = np.random.default_rng(1)
    for c in range(3):
        for r in (1, 0):
            pts = np.array([(c * 0.24 + k * 0.085 - 0.3, (0.5 - r) * 0.3, 0.04) for k in range(2)])
            APPLES.instance(f"sw_{c}_{r}", pts, (c, r), tilt=0.5, seed=c * 2 + r)
    make_camera("cam_swatch", (0.0, -0.9, 0.9), (0.0, 0.0, 0.03), lens=50.0)
    return 0.0


shot = OPT["shot"]
if shot == "overview":
    T_NOW = shot_overview()
elif shot == "platform":
    T_NOW = shot_platform(closeup=False)
elif shot == "closeup":
    T_NOW = shot_platform(closeup=True)
elif shot == "station":
    T_NOW = shot_station()
elif shot == "swatch":
    T_NOW = shot_swatch()
else:
    raise SystemExit("unknown shot " + shot)

log(f"scene: {len(bpy.data.objects)} objects, {len(bpy.data.meshes)} meshes, {APPLES.n_inst} apple instances, t={T_NOW:.1f}s")

if OPT["save"]:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(OPT["save"]))
    log("saved", OPT["save"])

if OPT["out"]:
    out = os.path.abspath(OPT["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    SC.render.filepath = out
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    log(f"rendered {out} in {time.time() - t0:.0f}s")
