"""3D frames of the explainer video of the v4 harvest simulation (square-grid
orchard, tethered-drone platforms, k-d regions with end take-over).

Run inside Blender (background), one call per shot (or per frame range of a shot):

  blender -b -t 2 --python video_v4.py -- --shot mission --out out/video/frames/mission \
          [--res 1920x1080] [--samples 64] [--gpu 3090|4060|all] [--frames 1-720] [--step 2] [--draft]

Shots (frame counts / mission windows in video_shots_v4.py):
  intro     aerial fly-in over the block to the bin station, drones lift off (t < 0, illustration)
  map       one straight-down still at t = 0 (compose_v4.py draws grid / k-d cuts / rings on it)
  mission   oblique aerial time-lapse, mission 0 -> 2 h at 300x, slow orbit
  closeup   one platform at an alley crossing, real time: drones fly to the crown, pick, carry, drop in the bin
  takeover  time-lapse of the first end take-over (platform 5 at the far end of region 1)
  finish    last trees, return to the station, aerial pull-back

The script imports build_scene_v4.py in library mode (--shot none) and reuses
its data model (poses along the recorded L-shaped traces, pick events, bin
fill, k-d regions, tree / apple / platform / drone / station kits, ground
masks, lighting).  New here: everything time dependent is driven per frame
(platform and drone poses, tethers, bin fill, bins at the station, cameras) by
a frame_change handler, and apples carry their removal time as a point
attribute so that one instancer serves the whole shot (geometry nodes delete
every apple whose removal time is before the frame's mission time; the
threshold is a driver on the frame).  The close-up choreography (approach,
suction grasp, detach, carry, drop into the bin, failed grasp) is illustration
timed with the simulation's per-fruit pick time (T_PICK = 6 s per drone).
See README.md, section "video".
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ------------------------------------------------------------------ own options
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
VOPT = {"shot": "mission", "out": None, "res": "1920x1080", "samples": 64, "gpu": "3090", "frames": None,
        "step": 1, "draft": False, "data": os.path.normpath(os.path.join(HERE, "..", "anim4_rect1ha.json")),
        "prep": os.path.join(HERE, "out", "video", "video_prep.json"), "mblur": None, "save": None,
        "apple-div": 1}
_i = 0
while _i < len(_argv):
    k = _argv[_i][2:]
    if k in ("draft",):
        VOPT[k] = True
        _i += 1
        continue
    VOPT[k] = _argv[_i + 1]
    _i += 2
VOPT["samples"] = int(VOPT["samples"])
VOPT["step"] = int(VOPT["step"])
VOPT["apple-div"] = int(VOPT["apple-div"])
if VOPT["draft"]:
    VOPT["res"] = "960x540" if VOPT["res"] == "1920x1080" else VOPT["res"]
    VOPT["samples"] = min(VOPT["samples"], 16)
if VOPT["mblur"] is not None:
    VOPT["mblur"] = bool(int(VOPT["mblur"]))
if VOPT["out"] is None:
    VOPT["out"] = os.path.join(HERE, "out", "video", "frames", VOPT["shot"])

# build_scene_v4 in library mode
sys.argv = [sys.argv[0], "--", "--data", VOPT["data"], "--shot", "none", "--res", VOPT["res"],
            "--samples", str(VOPT["samples"]), "--gpu", "all", "--apple-div", str(VOPT["apple-div"])]
import bpy                                   # noqa: E402
import numpy as np                           # noqa: E402
from mathutils import Euler, Matrix, Vector  # noqa: E402
import build_scene_v4 as B                   # noqa: E402
from video_shots_v4 import FPS, SHOTS        # noqa: E402

log = B.log
SC = B.SC
SHOT = VOPT["shot"]
SPEC = SHOTS[SHOT]
NF = SPEC["frames"]
PREP = json.load(open(VOPT["prep"])) if os.path.exists(VOPT["prep"]) else None
PRM = (PREP or {}).get("params", dict(T_PICK=6.0, P_SUCC=0.8, REC=0.85, PREC=0.88, T_STOP=20.0, M_DRONES=4))
MBLUR = SPEC["mblur"] if VOPT["mblur"] is None else VOPT["mblur"]
G = 9.81

# GPU selection (build_scene_v4 enabled every OptiX device)
prefs = bpy.context.preferences.addons["cycles"].preferences
for d in prefs.devices:
    want = d.type == "OPTIX" and (VOPT["gpu"] == "all" or VOPT["gpu"] in d.name)
    d.use = want
    if want:
        log("render device", d.name)
SC.render.use_persistent_data = True
SC.cycles.adaptive_threshold = 0.03 if not VOPT["draft"] else 0.1
SC.render.use_motion_blur = MBLUR
SC.render.motion_blur_shutter = 0.5
SC.render.motion_blur_position = "CENTER"
SC.render.image_settings.compression = 30


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return a + (b - a) * u


def lerp_v(a, b, u):
    return Vector(a).lerp(Vector(b), u)


def lerp_ang(a, b, u):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * u


def attach(ob, parent):
    """Parent without moving (world transform kept)."""
    bpy.context.view_layer.update()
    ob.parent = parent
    ob.matrix_parent_inverse = parent.matrix_world.inverted()


def look_at(ob, loc, target):
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


# ------------------------------------------------------------------ apples with removal times
def removal_times(t_lo, t_hi, exclude=None):
    """Per tree: (t_rm ripe slots, t_rm unripe slots) in pick order; +1e9 = never
    (within the shot), -1 = already gone before t_lo / excluded."""
    ev_by_tree = {}
    for (ta, te, q, rp, up, p, ti) in B.EVENTS:
        ev_by_tree.setdefault(q, []).append((ta, te, rp, up))
    out = {}
    for i in range(B.NT):
        n_r, n_u = int(B.R0[i]), int(B.U0[i])
        tr = np.full(n_r, 1e9)
        tu = np.full(n_u, 1e9)
        cr = cu = 0
        for (ta, te, rp, up) in sorted(ev_by_tree.get(i, [])):
            rp, up = min(rp, n_r - cr), min(up, n_u - cu)
            if rp > 0:
                tr[cr:cr + rp] = ta + (te - ta) * (np.arange(rp) + 1) / rp
            if up > 0:
                tu[cu:cu + up] = ta + (te - ta) * (np.arange(up) + 1) / up
            cr += rp
            cu += up
        tr[tr < t_lo] = -1.0
        tu[tu < t_lo] = -1.0
        tr[tr > t_hi] = 1e9
        tu[tu > t_hi] = 1e9
        if exclude and i in exclude:
            tr[list(exclude[i])] = -1.0
        out[i] = (tr, tu)
    return out


APPLE_MODS = []


def instance_timed(name, pts, t_rm, key, seed):
    """Apple instancer like AppleKit.instance plus a per-point attribute t_rm and
    a group input T: apples with t_rm <= T are deleted."""
    if len(pts) == 0:
        return None
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in pts], [], [])
    me.update()
    at = me.attributes.new("t_rm", "FLOAT", "POINT")
    at.data.foreach_set("value", np.ascontiguousarray(t_rm, dtype=np.float32))
    ob = B.new_obj(name, me, B.COLL_ORCHARD)
    ng = bpy.data.node_groups.new(name + "_ng", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    sT = ng.interface.new_socket("T", in_out="INPUT", socket_type="NodeSocketFloat")
    N = ng.nodes
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    na = N.new("GeometryNodeInputNamedAttribute")
    na.data_type = "FLOAT"
    na.inputs["Name"].default_value = "t_rm"
    cmp = N.new("FunctionNodeCompare")
    cmp.data_type = "FLOAT"
    cmp.operation = "LESS_EQUAL"
    ng.links.new(na.outputs["Attribute"], cmp.inputs[0])
    ng.links.new(gi.outputs["T"], cmp.inputs[1])
    dele = N.new("GeometryNodeDeleteGeometry")
    dele.domain = "POINT"
    ng.links.new(gi.outputs[0], dele.inputs["Geometry"])
    ng.links.new(cmp.outputs["Result"], dele.inputs["Selection"])
    m2p = N.new("GeometryNodeMeshToPoints")
    ng.links.new(dele.outputs["Geometry"], m2p.inputs["Mesh"])
    iop = N.new("GeometryNodeInstanceOnPoints")
    oi = N.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = B.APPLES.src[key]
    oi.inputs["As Instance"].default_value = True
    ng.links.new(oi.outputs["Geometry"], iop.inputs["Instance"])
    rr = N.new("FunctionNodeRandomValue")
    rr.data_type = "FLOAT_VECTOR"
    rr.inputs["Min"].default_value = (-0.35, -0.35, 0.0)
    rr.inputs["Max"].default_value = (0.35, 0.35, 6.283)
    rr.inputs["Seed"].default_value = seed
    ng.links.new(rr.outputs["Value"], iop.inputs["Rotation"])
    rs = N.new("FunctionNodeRandomValue")
    rs.data_type = "FLOAT"
    rs.inputs["Min"].default_value = 0.9
    rs.inputs["Max"].default_value = 1.1
    rs.inputs["Seed"].default_value = seed + 1
    ng.links.new(rs.outputs["Value"], iop.inputs["Scale"])
    ng.links.new(m2p.outputs["Points"], iop.inputs["Points"])
    ng.links.new(iop.outputs["Instances"], go.inputs[0])
    mod = ob.modifiers.new("inst", "NODES")
    mod.node_group = ng
    sock = getattr(mod.properties.inputs, sT.identifier)     # Blender 5.x modifier input API
    sock.value = -1e9
    APPLE_MODS.append(sock)
    B.APPLES.n_inst += len(pts)
    return ob


def build_orchard_timed(t_lo, t_hi, near_center=None, near_radius=0.0, exclude=None, div=1):
    """Trees, trunks and all apples present at some time in [t_lo, t_hi] with their removal times."""
    rt = removal_times(t_lo, t_hi, exclude)
    trunk = B.cylinder_mesh("trunk", 0.055, 1.05, segs=10)
    trunk.materials.append(B.pbr_material("trunk_bark", os.path.join(B.ASSETS, "textures", "bark_brown_02"),
                                          "bark_brown_02", "1k", scale=0.5, rough_mult=1.2, bump=0.7, tint=(0.8, 0.74, 0.66)))
    pts = {key: ([], []) for key in B.APPLE_CLASSES}
    rnd = np.random.default_rng(3)
    for i in range(B.NT):
        near = near_center is not None and math.hypot(B.XS[i] - near_center[0], B.YS[i] - near_center[1]) < near_radius
        B.KIT.place(i, near=near)
        B.new_obj(f"trunk_{i}", trunk, B.COLL_ORCHARD, (B.XS[i], B.YS[i], -0.02), (0, 0, rnd.uniform(0, 6.28)),
                  (rnd.uniform(0.85, 1.15), rnd.uniform(0.85, 1.15), 1.0))
        c = int(B.CULT[i])
        P, D, n_r, n_u = B.KIT.tree_slots(i)
        if len(P) == 0:
            continue
        n_r = min(n_r, len(P))
        n_u = min(n_u, len(P) - n_r)
        tr, tu = rt[i][0][:n_r], rt[i][1][:n_u]
        for key, Pk, tk in (((c, 1), P[:n_r], tr), ((c, 0), P[n_r:n_r + n_u], tu)):
            m = tk >= 0
            if div > 1:
                m &= (np.arange(len(tk)) % div) == 0
            pts[key][0].append(Pk[m])
            pts[key][1].append(tk[m])
    shown = 0
    for key, (lp, lt) in pts.items():
        arr = np.concatenate(lp) if lp else np.zeros((0, 3))
        tarr = np.concatenate(lt) if lt else np.zeros(0)
        instance_timed("apples_" + B.APPLE_CLASSES[key][0], arr, tarr, key, seed=key[0] * 2 + key[1])
        shown += len(arr)
    log(f"orchard window {t_lo:.0f}-{t_hi:.0f}s: {shown} apple instances (timed removal)")


def drive_apple_threshold(expr):
    """Driver on every instancer's T input: expression in `frame`."""
    for sock in APPLE_MODS:
        fc = sock.driver_add("value")
        fc.driver.type = "SCRIPTED"
        fc.driver.expression = expr


# ------------------------------------------------------------------ animated fleet
class AnimPlatform:
    """Platform root with parts, bin (parented), 4 drones (roots), hoses (parented
    to the drones), tethers (curves updated per frame) and an optional ring."""

    def __init__(self, p, cables, ring, ring_z, ring_r, bin_cls):
        self.p = p
        MT, DECK_Z, HUB = B.MT, B.DECK_Z, B.HUB_LOCAL
        name = f"platform_{p}"
        root = bpy.data.objects.new(name, None)
        B.COLL_FLEET.objects.link(root)
        self.root = root

        def part(me, mat, loc, rot=(0, 0, 0), bevel=0.0):
            if not me.materials:
                me.materials.append(mat)
            o = B.new_obj(f"{name}_{me.name}", me, B.COLL_FLEET, loc, rot, parent=root)
            if bevel > 0:
                bv = o.modifiers.new("bevel", "BEVEL")
                bv.width = bevel
                bv.segments = 3
            return o

        part(B.box_mesh("pl_deck", 2.5, 1.2, 0.06), MT.body, (0, 0, DECK_Z - 0.06), bevel=0.01)
        part(B.box_mesh("pl_frame", 2.3, 1.0, 0.12), MT.steel, (0, 0, DECK_Z - 0.18))
        part(B.box_mesh("pl_batt", 1.1, 0.8, 0.22), MT.steel, (-0.15, 0, 0.10), bevel=0.01)
        for s in (-1, 1):
            part(B.box_mesh("pl_rail", 2.3, 0.04, 0.05), MT.orange, (0, s * 0.58, DECK_Z))
            part(B.box_mesh("pl_bumper", 0.06, 1.1, 0.08), MT.orange, (s * 1.25, 0, 0.30))
            part(B.box_mesh("pl_id", 0.6, 0.012, 0.10), MT.led_mat(p), (0.55, s * 0.606, DECK_Z - 0.17))
        tyre = B.cylinder_mesh("pl_tyre", 0.30, 0.18, segs=24)
        rim = B.cylinder_mesh("pl_rim", 0.17, 0.19, segs=16)
        for sx in (-0.95, 0.95):
            for sy in (-0.52, 0.52):
                part(tyre, MT.tyre, (sx, sy - 0.09, 0.30), (math.pi / 2, 0, 0))
                part(rim, MT.alu, (sx, sy - 0.095, 0.30), (math.pi / 2, 0, 0))
        hx, hy, hz = HUB
        part(B.cylinder_mesh("pl_mast", 0.035, hz - DECK_Z - 0.15, segs=12), MT.alu, (hx, 0, DECK_Z))
        part(B.box_mesh("pl_hub", 0.34, 0.34, 0.22), MT.steel, (hx, 0, hz - 0.30), bevel=0.01)
        part(B.cylinder_mesh("pl_hubtop", 0.09, 0.06, segs=16), MT.body, (hx, 0, hz - 0.08))
        part(B.cylinder_mesh("pl_gnss", 0.06, 0.03, segs=16), MT.body, (hx, 0, hz - 0.02))
        for (sx, sy) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            a = math.atan2(sy, sx)
            part(B.cylinder_mesh("pl_hubarm", 0.012, 0.22, segs=8), MT.alu,
                 (hx + 0.17 * math.cos(a), 0.17 * math.sin(a), hz - 0.14), (0, math.pi / 2, a))
        for s in (-1, 1):
            part(B.box_mesh("pl_campod", 0.08, 0.05, 0.14), MT.carbon, (hx, s * 0.06, 1.95), bevel=0.005)
            part(B.box_mesh("pl_campod2", 0.08, 0.05, 0.14), MT.carbon, (hx, s * 0.06, 1.35), bevel=0.005)
        part(B.box_mesh("pl_ctrl", 0.35, 0.3, 0.25), MT.body, (hx - 0.35, 0.0, DECK_Z), bevel=0.01)
        part(B.cylinder_mesh("pl_beacon", 0.04, 0.07, segs=12), MT.orange, (hx - 0.35, 0.0, DECK_Z + 0.25))
        # bin on the rear deck, built at a reference fill of 0.5 in platform coordinates, then parented
        self.fill_ref = 0.5
        self.bin_root = B.build_bin(f"{name}_bin", (-0.6, 0, DECK_Z), 0.0, fill=self.fill_ref, cls=bin_cls,
                                    coll=B.COLL_FLEET, seed=p + 100)
        self.bin_root.parent = root
        self.bin_root.matrix_parent_inverse = Matrix.Identity(4)
        self.pile = bpy.data.objects.get(f"{name}_bin_pile")
        self.layers = [o for o in bpy.data.objects if o.name.startswith(f"{name}_bin_apples_")]
        for o in self.layers:
            o.parent = root
            o.matrix_parent_inverse = Matrix.Identity(4)
        self.h_ref = 0.12 + self.fill_ref * (B.BIN_SZ[2] - 0.12 - 0.04)
        # drones, hoses, tethers
        self.drones, self.tethers = [], []
        for j in range(B.M_DRONES):
            droot = B.PK.dk.make(f"{name}_drone_{j}", p)
            hose = B.PK.dk.hose(f"{name}_hose_{j}", (0, 0, 0), 0.0)
            hose.parent = droot
            hose.matrix_parent_inverse = Matrix.Identity(4)
            self.drones.append(droot)
            if cables:
                self.tethers.append(B.tether(f"{name}_tether_{j}", (0, 0, 3), (1, 1, 2)))
        self.ring = None
        if ring:
            n = 32
            pts = [(ring_r * math.cos(2 * math.pi * q / n), ring_r * math.sin(2 * math.pi * q / n), ring_z) for q in range(n)]
            o = B.curve_obj(f"ring_{p}", pts, 0.10, B.emission_material(f"ring_{p}_m", B.PLATFORM_COLORS[p % 16], 1.3),
                            B.COLL_OVERLAY, cyclic=True, smooth=True)
            o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
            o.parent = root
            o.matrix_parent_inverse = Matrix.Identity(4)
            self.ring = o
        self.cache = {}

    # -- per frame
    def set_pose(self, ps):
        self.root.location = (ps["x"], ps["y"], 0.0)
        self.root.rotation_euler = (0, 0, ps["yaw"])
        self.R = Matrix.Translation(Vector((ps["x"], ps["y"], 0))) @ Matrix.Rotation(ps["yaw"], 4, "Z")
        self.hub_w = self.R @ Vector(B.HUB_LOCAL)

    def set_fill(self, fill):
        h = 0.12 + fill * (B.BIN_SZ[2] - 0.12 - 0.04)
        if self.pile is not None:
            self.pile.scale = (1, 1, max(0.001, (h - 0.14) / (self.h_ref - 0.14)) if fill > 0.03 else 0.001)
        for o in self.layers:
            o.location = (0, 0, (h - self.h_ref) if fill > 0.03 else -3.0)

    def set_drone(self, j, d):
        B.PK.dk.pose(self.drones[j], d)
        if self.tethers:
            hub, anc = self.hub_w, Vector((d["x"] - 0.02 * math.cos(d["yaw"]), d["y"] - 0.02 * math.sin(d["yaw"]), d["z"] - 0.11))
            L = (anc - hub).length
            sag = 0.10 * L
            sp = self.tethers[j].data.splines[0]
            for k in range(9):
                u = k / 8
                pt = hub.lerp(anc, u)
                pt.z -= sag * math.sin(math.pi * u)
                sp.points[k].co = (pt.x, pt.y, pt.z, 1.0)

    def escort(self, ps, pitch=0.0):
        out = []
        for (ex, ey, ez) in B.ESCORT:
            w = self.R @ Vector((ex, ey, ez))
            out.append(dict(x=w.x, y=w.y, z=ez, yaw=ps["yaw"], pitch=pitch, target=None))
        return out

    def pick_specs(self, ps):
        """Drone poses of the current stop, computed once per stop (fruit state at the stop's start)."""
        key = (ps["trip"], ps["node"], round(ps["stop"][0], 1))
        if key not in self.cache:
            rem_r, _ = B.fruit_removed(ps["stop"][0] + 1e-3)
            self.cache[key] = B.drone_specs(ps, rem_r, ps["stop"][0], seed=self.p)
        return self.cache[key]

    def timelapse_drones(self, ps, t):
        """pick: planned poses, blended from / to the escort formation during the
        first 20 s (T_STOP, setup) and the last 5 s of the stop; else escort."""
        esc = self.escort(ps, pitch=math.radians(-4) if ps["state"] == "move" else 0.0)
        if ps["state"] != "pick" or ps["node"] is None:
            return esc
        ta, tl = ps["stop"]
        b = smooth((t - ta) / float(PRM.get("T_STOP", 20.0))) * smooth((tl - t) / 5.0)
        pk = self.pick_specs(ps)
        out = []
        for e, d in zip(esc, pk):
            out.append(dict(x=lerp(e["x"], d["x"], b), y=lerp(e["y"], d["y"], b), z=lerp(e["z"], d["z"], b),
                            yaw=lerp_ang(e["yaw"], d["yaw"], b), pitch=0.0, target=None))
        return out


def platform_bin_classes(p):
    cls = np.zeros((3, 2))
    for (ta, te, q, rp, up, pp, ti) in B.EVENTS:
        if pp == p:
            cls[int(B.CULT[q]), 0] += rp
            cls[int(B.CULT[q]), 1] += up
    return cls if cls.sum() > 0 else np.array([[1.0, 0], [0, 0], [0, 0]])


class Fleet:
    def __init__(self, cables=False, rings=False, ring_only=None, ring_z=B.Z_OVERLAY + 0.05, ring_r=3.0):
        self.pl = [AnimPlatform(p, cables, rings and (ring_only is None or p in ring_only), ring_z, ring_r,
                                platform_bin_classes(p)) for p in range(B.K)]
        log(f"fleet: {B.K} platforms x {B.M_DRONES} drones, cables={cables}, rings={rings}")

    def update(self, t, drone_fn=None):
        poses = B.all_platform_poses(t)
        for p, ps in enumerate(poses):
            ap = self.pl[p]
            ap.set_pose(ps)
            fill = B.bin_fill(p, t)[0] if ps["state"] != "station" else 0.0
            ap.set_fill(fill)
            specs = drone_fn(ap, ps, t) if drone_fn else ap.timelapse_drones(ps, t)
            for j, d in enumerate(specs):
                ap.set_drone(j, d)
        return poses


class StationBins:
    """Full bins at the station appear as trips are completed (bins built for the
    shot's last time, moved below the ground until their trip is done)."""

    def __init__(self, t_hi):
        B.build_station(t_hi)
        self.roots = []
        for k in range(8):
            r = bpy.data.objects.get(f"stbin_full_{k}")
            if r is None:
                break
            for o in bpy.data.objects:
                if o.name.startswith(f"stbin_full_{k}_apples_"):
                    attach(o, r)
            self.roots.append(r)

    def update(self, t):
        n = B.trips_done(t)
        for k, r in enumerate(self.roots):
            r.location.z = 0.0 if k < n else -3.0


# ------------------------------------------------------------------ cameras
class OrbitCam:
    """Camera on a sphere around a centre: (azimuth, pitch, distance) per frame,
    initial distance from build_scene_v4.fit_camera."""

    def __init__(self, name, pts, pitch, lens, az, margin=(0.015, 0.985, 0.03, 0.97)):
        self.cam = B.fit_camera(name, pts, pitch_deg=pitch, lens=lens, az_deg=az, margin=margin)
        c = sum((Vector(p) for p in pts), Vector()) / len(pts)
        self.c = Vector((c.x, c.y, 0.0))
        self.dist = (self.cam.location - self.c).length
        self.dx = self.cam.rotation_euler.x - (math.pi / 2 - math.radians(pitch))

    def set(self, az_deg, pitch_deg, dist_mult=1.0, lens=None):
        az, pitch = math.radians(az_deg), math.radians(pitch_deg)
        d = self.dist * dist_mult
        self.cam.location = self.c + Vector((-math.cos(pitch) * d * math.cos(az), -math.cos(pitch) * d * math.sin(az), math.sin(pitch) * d))
        self.cam.rotation_euler = Euler((math.pi / 2 - pitch + self.dx, 0, az - math.pi / 2), "XYZ")
        if lens is not None:
            self.cam.data.lens = lens


# ------------------------------------------------------------------ shots
META = dict(shot=SHOT, fps=FPS, res=VOPT["res"], samples=VOPT["samples"], frames=NF, t=[], step=VOPT["step"],
            draft=VOPT["draft"], mblur=MBLUR)
HANDLER = None
TIMES = []


def frame_time(f):
    """Mission time of 0-based frame f (float frames during motion blur)."""
    if SHOT == "finish":
        u = min(1.0, f / float(SPEC["t_end_frame"]))
    else:
        u = f / float(max(1, NF - 1))
    return SPEC["t0"] + (SPEC["t1"] - SPEC["t0"]) * u


def shot_intro():
    sx, sy = B.STATION
    cx, cy = (B.FIELD[0] + B.FIELD[1]) / 2, (B.FIELD[2] + B.FIELD[3]) / 2
    B.OPT["region-tint"] = 0.0
    B.build_ground(0.0)
    build_orchard_timed(-1.0, 0.0)
    B.build_station(0.0)
    fleet = Fleet(cables=False, rings=False)
    B.build_lighting(sun_az=math.radians(300), sun_el=math.radians(50))
    cam = B.make_camera("cam_intro", (cx + 55, cy + 150, 110), (cx, cy + 10, 0), lens=32.0)
    loc0, tgt0 = Vector((cx + 60, cy + 165, 118)), Vector((cx, cy + 15, 0))
    loc1, tgt1 = Vector((sx + 26, sy - 21, 9.5)), Vector((sx - 5, sy + 7, 1.0))
    LAND = [(0.30, 0.42), (0.30, -0.42), (0.95, 0.42), (0.95, -0.42)]

    def drones(ap, ps, t):
        s = (t - SPEC["t0"])                       # shot seconds
        out = []
        for j in range(B.M_DRONES):
            t_lift = 3.0 + 0.3 * ap.p + 0.4 * j
            u = smooth((s - t_lift) / 2.2)
            lx, ly = LAND[j]
            ex, ey, ez = B.ESCORT[j]
            z = lerp(B.DECK_Z + 0.17, ez, u) + 0.35 * math.sin(math.pi * u) * (1 - u)
            x, y = lerp(lx, ex, smooth((u - 0.3) / 0.7)), lerp(ly, ey, smooth((u - 0.3) / 0.7))
            w = ap.R @ Vector((x, y, z))
            out.append(dict(x=w.x, y=w.y, z=z, yaw=ps["yaw"], pitch=0.0, target=None))
        return out

    def per_frame(f):
        t = frame_time(f)
        s = f / float(NF - 1)
        u = smooth(s)
        loc = loc0.lerp(loc1, u)
        loc.z += 18.0 * math.sin(math.pi * u) * (1 - u)
        tgt = tgt0.lerp(tgt1, smooth((s - 0.08) / 0.84))
        look_at(cam, loc, tgt)
        fleet.update(t, drones)
        return t
    return per_frame


def shot_map():
    sx, sy = B.STATION
    B.OPT["region-tint"] = 0.0
    B.build_ground(0.0)
    build_orchard_timed(-1.0, 0.0)
    B.build_station(0.0)
    fleet = Fleet(cables=False, rings=False)
    B.build_lighting(sun_az=math.radians(300), sun_el=math.radians(56))
    W_, H_ = B.W, B.H
    y_lo, y_hi = sy - 17.0, B.FIELD[3] + 3.0
    scale = 0.92 * H_ / (y_hi - y_lo)            # px per m
    cam_y = 0.5 * (y_lo + y_hi)
    cam_x = (B.FIELD[0] - 2.0) + (0.5 * W_ - 0.03 * W_) / scale
    f_px = W_ * 35.0 / 36.0
    Hc = f_px / scale
    cam = B.make_camera("cam_map", (cam_x, cam_y, Hc), (cam_x, cam_y, 0), lens=35.0)
    cam.rotation_euler = (0, 0, 0)
    META["map"] = dict(cam_x=cam_x, cam_y=cam_y, height=Hc, px_per_m=scale, cx_px=0.5 * W_, cy_px=0.5 * H_,
                       note="pixel = (cx_px + (x - cam_x) * px_per_m, cy_px - (y - cam_y) * px_per_m) for ground points")
    log(f"map camera at ({cam_x:.1f}, {cam_y:.1f}) height {Hc:.1f} m, {scale:.2f} px/m")

    def per_frame(f):
        fleet.update(0.0)
        return 0.0
    return per_frame


def shot_mission():
    t0, t1 = SPEC["t0"], SPEC["t1"]
    B.OPT["region-tint"] = 0.08
    B.build_ground(t1)
    build_orchard_timed(t0, t1)
    st = StationBins(t1)
    fleet = Fleet(cables=False, rings=True)
    for j in range(B.K):
        B.region_outline(j)
    B.build_lighting(sun_az=math.radians(300), sun_el=math.radians(52))
    oc = OrbitCam("cam_mission", B.field_points(), 36.0, 32.0, 48.0)

    def per_frame(f):
        t = frame_time(f)
        s = f / float(NF - 1)
        oc.set(lerp(40.0, 60.0, s), lerp(38.0, 33.0, smooth(s)), lerp(0.82, 0.72, smooth(s)))
        fleet.update(t)
        st.update(t)
        return t
    return per_frame


def shot_takeover():
    t0, t1 = SPEC["t0"], SPEC["t1"]
    REG = B.REG
    take = sorted(B.TAKEOVER)
    helper, owner = take[0][4], take[0][5]
    B.OPT["region-tint"] = 0.08
    B.build_ground(t1)
    bb = [REG["bbox"][owner], REG["bbox"][helper]]
    cx = 0.5 * (min(b[0] for b in bb) + max(b[2] for b in bb))
    cy = 0.5 * (min(b[1] for b in bb) + max(b[3] for b in bb))
    build_orchard_timed(t0, t1)
    st = StationBins(t1)
    fleet = Fleet(cables=False, rings=True, ring_only={helper, owner}, ring_z=4.3, ring_r=2.2)
    B.region_outline(owner, radius=0.12, strength=2.0)
    B.region_outline(helper, radius=0.12, strength=2.0)
    for j in range(B.K):
        if j not in (helper, owner):
            B.region_outline(j, radius=0.06, strength=0.8)
    B.build_lighting(sun_az=math.radians(300), sun_el=math.radians(50))
    # frame the northern halves of both regions (the take-over strip is the far / north end of the owner's region)
    y_cut = 0.5 * (REG["bbox"][owner][1] + REG["bbox"][owner][3])
    pts = [(x, y, z) for b in bb for x in (b[0] - 3, b[2] + 3) for y in (max(b[1], y_cut) - 3, b[3] + 3) for z in (0, 4.5)]
    oc = OrbitCam("cam_takeover", pts, 42.0, 35.0, 262.0, margin=(0.04, 0.96, 0.05, 0.95))
    META["takeover"] = dict(helper=helper, owner=owner, trees=[tk[1] for tk in take if tk[4] == helper])
    log(f"take-over shot: helper {helper} in region {owner}, centre ({cx:.1f}, {cy:.1f})")

    def per_frame(f):
        t = frame_time(f)
        s = f / float(NF - 1)
        oc.set(lerp(254.0, 270.0, s), lerp(44.0, 40.0, smooth(s)), lerp(0.92, 0.84, smooth(s)))
        fleet.update(t)
        st.update(t)
        return t
    return per_frame


def shot_finish():
    t0, t1 = SPEC["t0"], SPEC["t1"]
    B.OPT["region-tint"] = 0.08
    B.build_ground(t1)
    build_orchard_timed(t0, t1)
    st = StationBins(t1)
    fleet = Fleet(cables=False, rings=True)
    for j in range(B.K):
        B.region_outline(j)
    B.build_lighting(sun_az=math.radians(300), sun_el=math.radians(50))
    oc = OrbitCam("cam_finish", B.field_points(), 36.0, 32.0, 60.0)

    def per_frame(f):
        t = frame_time(f)
        s = f / float(NF - 1)
        oc.set(lerp(60.0, 48.0, s), lerp(34.0, 48.0, smooth(s)), lerp(0.74, 1.35, smooth(s)))
        fleet.update(t)
        st.update(t)
        return t
    return per_frame


# ------------------------------------------------------------------ close-up choreography
class Choreo:
    """Real-time picking of one platform: per drone a timeline of segments
    (name, duration, start pose, end pose, apple).  Durations add up to the
    simulation's T_PICK = 6 s per fruit and drone (approach 1.5, suction grasp
    0.6, detach 0.5, carry 1.5, release 0.5, return 1.4); one grasp fails
    (P_SUCC = 0.8) and is retried.  Apple positions are a pure function of the
    shot time (hanging -> on the cup -> free fall -> resting in the bin), so any
    frame can be rendered on its own.  Illustration."""

    PH = dict(approach=1.5, grasp=0.6, detach=0.5, carry=1.5, release=0.5, back=1.4, fail=0.5, re_approach=1.0)

    def __init__(self, ap, ps, q, rem_r, apple_objs, bin_fill_h, n_pick=12, n_cycles=3, fail_drone=1, fail_cycle=1):
        self.ap, self.q = ap, q
        P, D = B.ripe_targets(q, rem_r, n=len(apple_objs))
        cx, cy = B.XS[q], B.YS[q]
        a0 = math.atan2(ps["y"] - cy, ps["x"] - cx)
        used = []
        self.tl = []
        self.apples = {}                  # obj -> dict(s_detach, s_release, tip_rel, land, drone)
        bin_c = ap.R @ Vector((-0.6, 0, B.DECK_Z))
        self.bin_top = bin_c.z + B.BIN_SZ[2]
        self.land_z = bin_c.z + bin_fill_h + 0.045
        rnd = np.random.default_rng(q)
        n_used = 0
        plan = []
        for k in range(B.M_DRONES):
            az = a0 + math.radians(B.PICK_AZ[k])
            zo = B.PICK_Z[k]
            hov = Vector((cx + 1.9 * math.cos(az), cy + 1.9 * math.sin(az), zo))
            esc = ap.R @ Vector(B.ESCORT[k])
            tl = []
            tl.append(("wait", 0.6 + 0.45 * k, dict(pos=esc, yaw=ps["yaw"]), dict(pos=esc, yaw=ps["yaw"]), None))
            tl.append(("flyout", 2.4, dict(pos=esc, yaw=ps["yaw"]), dict(pos=hov, yaw=az + math.pi), None))
            tl.append(("wait", 0.3 + 1.1 * k, dict(pos=hov, yaw=az + math.pi), dict(pos=hov, yaw=az + math.pi), None))
            plan.append(dict(az=az, zo=zo, hov=hov, tl=tl, cur=dict(pos=hov, yaw=az + math.pi)))
        for cyc in range(n_cycles):
            for k in range(B.M_DRONES):
                pk = plan[k]
                az, zo, hov, tl = pk["az"], pk["zo"], pk["hov"], pk["tl"]
                if n_used >= n_pick or len(P) == 0:
                    break
                ang = np.arctan2(P[:, 1] - cy, P[:, 0] - cx)
                dang = np.abs((ang - az + math.pi) % (2 * math.pi) - math.pi)
                cost = 2.5 * dang + 0.8 * np.abs(P[:, 2] - zo) + 0.4 * cyc * rnd.uniform(0, 1, len(P))
                for u in used:
                    cost += np.where(np.linalg.norm(P - u, axis=1) < 0.45, 50.0, 0.0)
                for u in [v["slot"] for v in self.apples.values()]:
                    cost[u] = 1e6
                idx = int(np.argmin(cost))
                if cost[idx] >= 1e6:
                    break
                a, d = Vector(P[idx]), Vector((D[idx][0], D[idx][1], 0.0))
                used.append(P[idx])
                ob = apple_objs[idx]
                n_used += 1
                yaw_g = math.atan2(-d.y, -d.x)
                grasp = dict(pos=a + d * (B.REACH + 0.038) + Vector((0, 0, 0.03)), yaw=yaw_g)
                near = dict(pos=grasp["pos"] + d * 0.45 + Vector((0, 0, 0.10)), yaw=yaw_g)
                fail = (k == fail_drone and cyc == fail_cycle)
                tl.append(("approach", self.PH["approach"], pk["cur"], grasp, None))
                tl.append(("grasp", self.PH["grasp"], grasp, grasp, None))
                if fail:
                    tl.append(("fail", self.PH["fail"], grasp, near, None))
                    tl.append(("re_approach", self.PH["re_approach"], near, grasp, None))
                    tl.append(("grasp", self.PH["grasp"], grasp, grasp, None))
                s_det = sum(seg[1] for seg in tl)
                tl.append(("detach", self.PH["detach"], grasp, near, ob))
                off = Vector(((-1 if k % 2 else 1) * 0.22, (-1 if k < 2 else 1) * 0.22, 0))
                tip_b = bin_c + off + Vector((0, 0, B.BIN_SZ[2] + 0.30))
                yaw_b = math.atan2(tip_b.y - near["pos"].y, tip_b.x - near["pos"].x)
                rel = dict(pos=tip_b - Vector((B.REACH * math.cos(yaw_b), B.REACH * math.sin(yaw_b), -0.03)), yaw=yaw_b)
                tl.append(("carry", self.PH["carry"], near, rel, ob))
                s_rel = sum(seg[1] for seg in tl)
                tl.append(("release", self.PH["release"], rel, rel, ob))
                tl.append(("back", self.PH["back"], rel, dict(pos=hov, yaw=az + math.pi), None))
                pk["cur"] = dict(pos=hov, yaw=az + math.pi)
                self.apples[ob] = dict(slot=idx, s_detach=s_det, s_release=s_rel, drone=k, hang=a,
                                       tip_rel=self.tip(rel["pos"], rel["yaw"]),
                                       land=Vector((bin_c.x + off.x + rnd.uniform(-0.08, 0.08), bin_c.y + off.y + rnd.uniform(-0.08, 0.08), self.land_z)))
        for pk in plan:
            pk["tl"].append(("wait", 60.0, pk["cur"], pk["cur"], None))
            self.tl.append(pk["tl"])
        self.n_used = n_used
        self.events = sorted((v["s_detach"], v["s_release"], v["drone"]) for v in self.apples.values())
        log(f"choreography: {n_used} apples over {B.M_DRONES} drones (first detach at {self.events[0][0]:.1f} s), "
            f"bin top {self.bin_top:.2f} m, landing {self.land_z:.2f} m")

    @staticmethod
    def tip(pos, yaw):
        """Centre of an apple held on the cup of a drone at pos / yaw."""
        return Vector(pos) + Vector((math.cos(yaw) * (B.REACH + 0.036), math.sin(yaw) * (B.REACH + 0.036), -0.03))

    def drone_pose(self, k, s):
        tl = self.tl[k]
        acc = 0.0
        seg, u = tl[-1], 1.0
        for seg_ in tl:
            if s < acc + seg_[1]:
                seg = seg_
                u = (s - acc) / seg_[1]
                break
            acc += seg_[1]
        name, dur, a, b, ob = seg
        us = smooth(u)
        pos = a["pos"].lerp(b["pos"], us)
        if name in ("carry", "flyout"):
            pos.z += 0.45 * math.sin(math.pi * us)
        if name == "back":
            pos.z += 0.25 * math.sin(math.pi * us)
        if name == "grasp":
            pos = pos + Vector((math.cos(a["yaw"]), math.sin(a["yaw"]), 0)) * 0.012 * smooth(u / 0.4)
        yaw = lerp_ang(a["yaw"], b["yaw"], us)
        bob = 0.012 * math.sin(2 * math.pi * 0.9 * s + 1.7 * k)
        speed = (b["pos"] - a["pos"]).length / max(dur, 0.1)
        pitch = math.radians(-6.0) * min(1.0, speed / 1.5) * math.sin(math.pi * us) if name in ("carry", "back", "flyout", "approach") else 0.0
        return dict(x=pos.x, y=pos.y, z=pos.z + bob, yaw=yaw, pitch=pitch, target=None), name

    def poses(self, s):
        """s = seconds since the shot start -> drone specs; also places the picked apples."""
        out = [self.drone_pose(k, s) for k in range(B.M_DRONES)]
        for ob, st in self.apples.items():
            if s < st["s_detach"]:
                ob.location = st["hang"]
            elif s < st["s_release"]:
                d = out[st["drone"]][0]
                ob.location = self.tip((d["x"], d["y"], d["z"]), d["yaw"])
            else:
                tau = s - st["s_release"]
                z = st["tip_rel"].z - 0.5 * G * tau * tau
                land = st["land"]
                if z <= land.z:
                    ob.location = land
                else:
                    w = min(1.0, tau / 0.35)
                    ob.location = (lerp(st["tip_rel"].x, land.x, w), lerp(st["tip_rel"].y, land.y, w), z)
        return [d for d, _ in out]


def shot_closeup():
    p, tm, q = B.choose_platform_stop()
    stop = next((ta, tl) for (ta, tl, qq, x, y, ti) in B.STOPS[p] if qq == q and ta <= tm <= tl)
    t0 = stop[0] + float(PRM.get("T_STOP", 20.0)) - 3.0    # drones reach the crown when the setup time (T_STOP) is over
    t1 = t0 + (NF - 1) / float(FPS)
    SPEC["t0"], SPEC["t1"] = t0, t1
    ps = B.platform_pose(p, t0)
    cx, cy = ps["x"], ps["y"]
    rem_r, _ = B.fruit_removed(stop[0] + 1e-3)
    # the apples the choreography picks are separate objects: exclude their slots from the instancer
    P, D = B.ripe_targets(q, rem_r, n=90)
    n_pick = 12
    exclude = {q: set(range(rem_r[q], rem_r[q] + len(P)))}       # all candidate slots (the chosen ones are a subset)
    B.OPT["region-tint"] = 0.0
    B.build_ground(t1)
    build_orchard_timed(t0, t0, near_center=(cx, cy), near_radius=14.0, exclude=exclude)
    # candidate apples of tree q that the choreography does not use stay on the tree as objects
    key = (int(B.CULT[q]), 1)
    src = B.APPLES.src[key]
    apple_objs = []
    rnd = np.random.default_rng(q + 5)
    for i in range(len(P)):
        o = B.new_obj(f"pick_apple_{i}", src.data, B.COLL_ORCHARD, tuple(P[i]),
                      (rnd.uniform(-0.35, 0.35), rnd.uniform(-0.35, 0.35), rnd.uniform(0, 6.28)))
        o.scale = (rnd.uniform(0.92, 1.08),) * 3
        apple_objs.append(o)
    B.build_station(t1)
    fleet = Fleet(cables=True, rings=False)
    B.build_lighting(sun_az=math.radians(300), sun_el=math.radians(50))
    B.build_grass_scatter((cx, cy), radius=16.0)
    fill = B.bin_fill(p, t0)[0]
    h_fill = 0.12 + fill * (B.BIN_SZ[2] - 0.12 - 0.04)
    ap = fleet.pl[p]
    ap.set_pose(ps)
    ch = Choreo(ap, ps, q, rem_r, apple_objs, h_fill, n_pick=n_pick)
    loc0, tgt0 = Vector((cx + 8.6, cy + 1.0, 4.3)), Vector((cx - 1.2, cy - 1.0, 1.55))
    loc1, tgt1 = Vector((cx + 7.0, cy + 0.4, 3.5)), Vector((cx - 1.2, cy - 1.1, 1.5))
    cam = B.make_camera("cam_closeup", tuple(loc0), tuple(tgt0), lens=42.0, fstop=4.0, focus=(Vector((cx - 1.2, cy - 1.2, 1.9)) - loc0).length)
    META["closeup"] = dict(platform=p, tree=q, stop=[stop[0], stop[1]], fill=fill, cultivar=B.CULT_NAMES[int(B.CULT[q])],
                           ripe_initial=int(B.R0[q]), unripe_initial=int(B.U0[q]), picked_in_shot=ch.n_used,
                           events=ch.events, pick_s=PRM.get("T_PICK"), p_succ=PRM.get("P_SUCC"), rec=PRM.get("REC"), prec=PRM.get("PREC"))
    log(f"close-up: platform {p} at tree {q} ({B.CULT_NAMES[int(B.CULT[q])]}), stop {stop[0]:.0f}-{stop[1]:.0f}s, "
        f"shot {t0:.0f}-{t1:.0f}s, bin fill {fill:.2f}")

    def drones(apx, psx, t):
        if apx.p != p:
            return apx.timelapse_drones(psx, t)
        return ch.poses(t - t0)

    def per_frame(f):
        t = t0 + f / float(FPS)
        s = f / float(NF - 1)
        u = smooth(s)
        loc = loc0.lerp(loc1, u)
        tgt = tgt0.lerp(tgt1, u)
        look_at(cam, loc, tgt)
        cam.data.dof.focus_distance = (Vector((cx - 1.2, cy - 1.2, 1.9)) - loc).length
        fleet.update(t, drones)
        return t
    return per_frame


# ------------------------------------------------------------------ run
BUILDERS = dict(intro=shot_intro, map=shot_map, mission=shot_mission, closeup=shot_closeup,
                takeover=shot_takeover, finish=shot_finish)
per_frame = BUILDERS[SHOT]()
if SHOT != "closeup":
    t0, t1 = SPEC["t0"], SPEC["t1"]
    if SHOT == "finish":
        expr = f"min({t1!r}, {t0!r} + (frame - 1) * {(t1 - t0) / float(SPEC['t_end_frame'])!r})"
    else:
        expr = f"{t0!r} + (frame - 1) * {(t1 - t0) / float(max(1, NF - 1))!r}"
    drive_apple_threshold(expr)
else:
    drive_apple_threshold("-1e9")      # nothing vanishes from the instancers during the real-time close-up


def _handler(scene):
    f = getattr(scene, "frame_current_final", scene.frame_current + scene.frame_subframe) - 1.0
    per_frame(f)


bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(_handler)
SC.frame_start, SC.frame_end = 1, NF
SC.frame_step = VOPT["step"]
if VOPT["frames"]:
    a, b = VOPT["frames"].split("-")
    SC.frame_start, SC.frame_end = int(a), int(b)
for f in range(NF):
    META["t"].append(round(frame_time(f) if SHOT != "closeup" else SPEC["t0"] + f / float(FPS), 3))
META["t0"], META["t1"] = SPEC["t0"], SPEC["t1"]
os.makedirs(VOPT["out"], exist_ok=True)
if SC.frame_start == 1 or not os.path.exists(os.path.join(VOPT["out"], "meta.json")):
    json.dump(META, open(os.path.join(VOPT["out"], "meta.json"), "w"), indent=1)
log(f"scene: {len(bpy.data.objects)} objects, {B.APPLES.n_inst} apple instances; frames {SC.frame_start}-{SC.frame_end} step {SC.frame_step}, "
    f"{VOPT['res']} {VOPT['samples']} spp, motion blur {MBLUR}")
if VOPT["save"]:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(VOPT["save"]))
SC.render.filepath = os.path.join(VOPT["out"], "frame_")
t_start = time.time()
bpy.ops.render.render(animation=True)
n_done = len(range(SC.frame_start, SC.frame_end + 1, SC.frame_step))
log(f"rendered {n_done} frames in {time.time() - t_start:.0f}s ({(time.time() - t_start) / max(1, n_done):.1f} s/frame)")
