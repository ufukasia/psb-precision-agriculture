"""Shot table of the v4 explainer video, shared by video_v4.py (Blender, 3D frames)
and compose_v4.py (captions, clock, cards, encoding).  Pure python, no bpy.

Each 3D shot renders `frames` frames at FPS; the mission time depicted in frame f
(0-based) is written by video_v4.py into <frames dir>/meta.json ("t" list), so
compose_v4.py never recomputes it.  `t0`/`t1` here are the planned windows
(mission seconds of ../anim4_rect1ha.json); the close-up window is chosen from
the data by video_v4.py (same rule as the fig_v4_platform figure).
"""
FPS = 30
W, H = 1920, 1080

# order of the final cut; 2D-only sections (title / stats card / holds) are handled by compose_v4.py
SHOTS = {
    "intro":    dict(frames=270, t0=-9.0, t1=0.0, speed=1.0, mblur=False,
                     desc="aerial fly-in over the block towards the bin station; drones lift off (illustration, before t=0)"),
    "map":      dict(frames=1, t0=0.0, t1=0.0, speed=0.0, mblur=False,
                     desc="one straight-down still at t=0; grid / k-d cuts / rings are drawn by compose_v4.py"),
    "mission":  dict(frames=720, t0=0.0, t1=7200.0, speed=300.0, mblur=True,
                     desc="oblique aerial time-lapse 0 -> 2 h at 300x (fan-out, picking, first bin swaps)"),
    "closeup":  dict(frames=540, t0=None, t1=None, speed=1.0, mblur=False,
                     desc="one platform at an alley crossing in real time: drones fly out, pick, carry to the bin"),
    "takeover": dict(frames=330, t0=14300.0, t1=17400.0, speed=3100.0 / 11.0, mblur=True,
                     desc="time-lapse of the first end take-over (platform 5 enters the far end of region 1)"),
    "finish":   dict(frames=240, t0=17400.0, t1=18358.04, speed=958.04 / 5.0, mblur=True, t_end_frame=150,
                     desc="last trees, return to the station, aerial pull-back (time runs 17400 -> end in 5 s, then holds)"),
}
ORDER = ["intro", "map", "mission", "closeup", "takeover", "finish"]

# 2D sections (seconds) added by compose_v4.py
TITLE_S = 3.0          # title card before the intro (over the blurred first frame)
MAP_S = 16.0           # length of the map section (one still + overlays)
CARD_S = 6.0           # stats card at the end (over the last finish frame, which is held)
FADE_S = 0.4


def mission_time(shot, f):
    """Planned mission time of frame f of a 3D shot (video_v4.py writes the actual one to meta.json)."""
    s = SHOTS[shot]
    n = max(1, s["frames"] - 1)
    if shot == "finish":
        u = min(1.0, f / float(s["t_end_frame"]))
    else:
        u = f / float(n)
    if s["t0"] is None:
        return None
    return s["t0"] + (s["t1"] - s["t0"]) * u
