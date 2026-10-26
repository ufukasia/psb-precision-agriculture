#!/usr/bin/env python3
"""Compositing of the v4 explainer video: 3D frames (video_v4.py) + captions,
title / stats cards, mission clock, progress bar and the animated map overlay
(estimated axes, k-d cuts, region colours) -> H.264 MP4 via ffmpeg.

  python3 compose_v4.py --lang tr --out out/video/harvest_explainer_tr.mp4 [--frames out/video/frames] [--draft]

The same frames serve both languages; all numbers in the captions come from
../anim4_rect1ha.json (summary, k, drones_per_platform, bin_capacity, trees),
../anim4_rect1ha_regions.json (region sizes) and out/video/video_prep.json
(k-d cut sequence, axis angles, planted area, simulation parameters).
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from video_shots_v4 import FPS, SHOTS, ORDER, TITLE_S, MAP_S, CARD_S, FADE_S  # noqa: E402

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_B = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
PLATFORM_COLORS = [
    (0.90, 0.20, 0.15), (0.15, 0.45, 0.95), (0.95, 0.65, 0.10), (0.20, 0.75, 0.35),
    (0.80, 0.25, 0.85), (0.10, 0.80, 0.85), (0.95, 0.40, 0.55), (0.55, 0.55, 0.20),
]


def rgb(c, a=255):
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), a)


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def fmt_int(n, lang):
    s = f"{int(n):,}"
    return s.replace(",", " ") if lang == "tr" else s.replace(",", " ")


def fmt_dec(x, nd, lang):
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if lang == "tr" else s


def hhmm(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t - 3600 * h) // 60)
    return f"{h:02d}:{m:02d}"


# ------------------------------------------------------------------ texts
def texts(lang, D, REG, PREP, meta):
    k, m = int(D["k"]), int(D["drones_per_platform"])
    nt = len(D["trees"]["x"])
    area = PREP["area_ha"]
    q = int(D["bin_capacity"])
    S = D["summary"]
    P = PREP["params"]
    th = PREP["axis"]["theta_err_deg"]
    cu = meta.get("closeup", {}).get("closeup", {})
    tk = meta.get("takeover", {}).get("takeover", {})
    helper, owner = tk.get("helper", 5), tk.get("owner", 1)
    pct = 100.0 * S["picked"] / S["ripe_total"]
    T = {}
    if lang == "tr":
        T["title"] = "Bağlı drone platformlarıyla seçici elma hasadı"
        T["subtitle"] = "Hasat benzetiminin açıklayıcı videosu — çizgisel kare ızgara bahçe, k-d bölgeler, uç devralma"
        T["sec"] = {"intro": "1 · Filo", "map": "2 · Harita paylaşımı", "mission": "3 · Görev (hızlandırılmış)",
                    "closeup": "4 · Yakın plan (gerçek zaman)", "takeover": "5 · Uç devralma", "finish": "6 · Bitiş"}
        T["intro1"] = (f"{k} platform × {m} bağlı drone = {k * m} drone   ·   {nt} ağaç   ·   ≈{fmt_dec(area, 1, lang)} ha   ·   "
                       f"kasa {fmt_int(q, lang)} meyve")
        T["intro2"] = "Platformlar kasa istasyonunda bekliyor, droneler kalkıyor"
        T["map0"] = "Platformlar haritayı paylaşıyor: önce bahçe ekseni, sonra bölgeler"
        T["map_a"] = f"Eksen kestirimi: bahçe eksenleri ağaç konumlarından kestiriliyor — eksen hatası {fmt_dec(th, 2, lang)}°"
        T["map_b"] = "k-d bölütleme: kesme çizgileri koridorlar boyunca, her platform bir bölgenin sahibi"
        T["axis_lbl"] = f"eksen hatası {fmt_dec(th, 2, lang)}°"
        T["legend_title"] = "Platform → bölge (ağaç sayısı)"
        T["legend_row"] = lambda j, n: f"Platform {j + 1}   {n} ağaç"
        T["mission1"] = "Platformlar L-biçimli koridor yollarında ilerliyor ve kavşaklarda durup komşu ağacı topluyor"
        T["mission2"] = f"Kasa ({fmt_int(q, lang)} meyve) dolunca platform istasyona döner, kasa değişir ve yoluna devam eder"
        T["closeup1"] = "Algılama → toplama → kasa: dört drone olgun elmaları vantuzla alıp kasaya taşıyor"
        T["closeup2"] = (f"Meyve başına {fmt_dec(P['T_PICK'], 0, lang)} s, tutma başarısı %{fmt_dec(100 * P['P_SUCC'], 0, lang)}; "
                         f"algılama duyarlılığı {fmt_dec(P['REC'], 2, lang)}, kesinliği {fmt_dec(P['PREC'], 2, lang)} — ham meyve dalda kalır")
        T["closeup_fail"] = "başarısız tutma → yeniden deneme"
        T["takeover1"] = (f"Uç devralma: bölgesini bitiren platform {helper + 1}, en geride kalan komşusu olan "
                          f"platform {owner + 1}'in bölgesine uzak uçtan giriyor")
        T["takeover2"] = "İki platform birbirine doğru çalışır ve ortada buluşur; yeniden planlama gerekmez"
        T["finish1"] = "Son ağaçlar toplanıyor, tüm platformlar istasyona dönüyor"
        T["clock"] = "Görev saati"
        T["speed"] = lambda v: f"{int(round(v))}× hızlı" if v > 1.5 else "gerçek zaman"
        T["progress"] = "Olgun meyve toplandı"
        T["card_title"] = "Görev özeti (benzetim)"
        mm = int(round(S["mission_s"] / 60.0))
        T["card"] = [("Görev süresi", f"{mm // 60} sa {mm % 60:02d} dk"),
                     ("Toplanan olgun meyve", f"{fmt_int(S['picked'], lang)} / {fmt_int(S['ripe_total'], lang)}  (%{fmt_dec(pct, 1, lang)})"),
                     ("Yanlışlıkla toplanan ham meyve", fmt_int(S["unripe_picked"], lang)),
                     ("Kasa seferi", fmt_int(S["trips"], lang)),
                     ("Uç devralma", fmt_int(S["steals"], lang)),
                     ("Filo", f"{k} platform × {m} drone, {nt} ağaç, ≈{fmt_dec(area, 1, lang)} ha")]
        T["pick_lbl"] = f"Platform {cu.get('platform', 0) + 1}, ağaç {cu.get('tree', 0)} ({ {'red': 'kırmızı', 'yellow': 'sarı', 'green': 'yeşil'}.get(cu.get('cultivar', 'red'), '') } çeşit)"
    else:
        T["title"] = "Selective apple harvest with tethered-drone platforms"
        T["subtitle"] = "Explainer of the harvest simulation — square-grid orchard, k-d regions, end take-over"
        T["sec"] = {"intro": "1 · Fleet", "map": "2 · Map sharing", "mission": "3 · Mission (time-lapse)",
                    "closeup": "4 · Close-up (real time)", "takeover": "5 · End take-over", "finish": "6 · Finish"}
        T["intro1"] = (f"{k} platforms × {m} tethered drones = {k * m} drones   ·   {nt} trees   ·   ≈{fmt_dec(area, 1, lang)} ha   ·   "
                       f"bin {fmt_int(q, lang)} fruit")
        T["intro2"] = "The platforms wait at the bin station, the drones lift off"
        T["map0"] = "The platforms share one map: first the orchard axes, then the regions"
        T["map_a"] = f"Frame estimation: the orchard axes are estimated from the tree positions — axis error {fmt_dec(th, 2, lang)}°"
        T["map_b"] = "k-d partition: cuts along the alleys, each platform owns one region"
        T["axis_lbl"] = f"axis error {fmt_dec(th, 2, lang)}°"
        T["legend_title"] = "Platform → region (trees)"
        T["legend_row"] = lambda j, n: f"Platform {j + 1}   {n} trees"
        T["mission1"] = "Platforms drive their L-shaped alley paths and stop at crossings to pick the adjacent tree"
        T["mission2"] = f"When the bin ({fmt_int(q, lang)} fruit) is full the platform returns to the station, swaps the bin and continues"
        T["closeup1"] = "Detection → pick → bin: four drones take the ripe apples with a suction cup and carry them to the bin"
        T["closeup2"] = (f"{fmt_dec(P['T_PICK'], 0, lang)} s per fruit, {fmt_dec(100 * P['P_SUCC'], 0, lang)} % grasp success; "
                         f"detection recall {fmt_dec(P['REC'], 2, lang)}, precision {fmt_dec(P['PREC'], 2, lang)} — unripe fruit stays on the tree")
        T["closeup_fail"] = "failed grasp → retry"
        T["takeover1"] = (f"End take-over: platform {helper + 1}, finished with its region, enters the far end of the region of "
                          f"its most-behind neighbour, platform {owner + 1}")
        T["takeover2"] = "The two platforms work towards each other and meet in the middle; no re-planning needed"
        T["finish1"] = "Last trees harvested, all platforms return to the station"
        T["clock"] = "Mission clock"
        T["speed"] = lambda v: f"{int(round(v))}× speed" if v > 1.5 else "real time"
        T["progress"] = "Ripe fruit harvested"
        T["card_title"] = "Mission summary (simulation)"
        mm = int(round(S["mission_s"] / 60.0))
        T["card"] = [("Mission time", f"{mm // 60} h {mm % 60:02d} min"),
                     ("Ripe fruit harvested", f"{fmt_int(S['picked'], lang)} / {fmt_int(S['ripe_total'], lang)}  ({fmt_dec(pct, 1, lang)} %)"),
                     ("Unripe fruit picked by mistake", fmt_int(S["unripe_picked"], lang)),
                     ("Bin trips", fmt_int(S["trips"], lang)),
                     ("End take-overs", fmt_int(S["steals"], lang)),
                     ("Fleet", f"{k} platforms × {m} drones, {nt} trees, ≈{fmt_dec(area, 1, lang)} ha")]
        T["pick_lbl"] = f"Platform {cu.get('platform', 0) + 1}, tree {cu.get('tree', 0)} ({cu.get('cultivar', 'red')}-type)"
    return T


# ------------------------------------------------------------------ compositor
class Compositor:
    def __init__(self, lang, frames_root, draft, data, regions, prep):
        self.lang = lang
        self.root = frames_root
        self.draft = draft
        self.D = json.load(open(data))
        self.REG = json.load(open(regions))
        self.PREP = json.load(open(prep))
        self.meta = {}
        self.files = {}
        for name in ORDER:
            d = os.path.join(frames_root, name)
            mp = os.path.join(d, "meta.json")
            if not os.path.exists(mp):
                raise SystemExit(f"missing {mp} (render the shot first)")
            self.meta[name] = json.load(open(mp))
            fr = sorted(int(f[6:10]) for f in os.listdir(d) if f.startswith("frame_") and f.endswith(".png"))
            if not fr:
                raise SystemExit(f"no frames in {d}")
            self.files[name] = fr
        im = Image.open(self.path("intro", 1))
        self.W, self.H = im.size
        self.sc = self.W / 1920.0
        self.T = texts(lang, self.D, self.REG, self.PREP, self.meta)
        self.fonts = {}
        picks = sorted(self.D["picks"])
        self.pick_t = np.array([p[0] for p in picks])
        self.pick_r = np.cumsum([p[2] for p in picks])
        self.ripe_total = float(self.D["summary"]["ripe_total"])
        self.cache = {}

    # -- helpers
    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(FONT_B if bold else FONT, max(8, int(round(size * self.sc))))
        return self.fonts[key]

    def path(self, shot, f):
        fr = self.files[shot]
        i = np.searchsorted(fr, f, side="right") - 1
        return os.path.join(self.root, shot, f"frame_{fr[max(0, i)]:04d}.png")

    def frame(self, shot, f):
        p = self.path(shot, f)
        if self.cache.get("p") != p:
            self.cache = dict(p=p, im=Image.open(p).convert("RGB"))
        return self.cache["im"].copy()

    def progress(self, t):
        i = np.searchsorted(self.pick_t, t, side="right")
        return (self.pick_r[i - 1] if i > 0 else 0) / self.ripe_total

    def band_text(self, im, lines, alpha=1.0, y_frac=0.86, size=30, bold_first=False):
        """Caption band at the bottom: dark translucent rounded box, white text, safe margins."""
        if alpha <= 0.01 or not lines:
            return
        W, H = im.size
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        fonts = [self.font(size, bold=(bold_first and i == 0)) for i in range(len(lines))]
        widths = [dr.textlength(l, font=f) for l, f in zip(lines, fonts)]
        lh = [int(size * self.sc * 1.35) for _ in lines]
        tw = max(widths)
        th = sum(lh)
        pad = int(22 * self.sc)
        bw = min(W - 2 * int(0.05 * W), int(tw + 2 * pad))
        x0 = (W - bw) // 2
        y0 = int(y_frac * H) - th // 2 - pad
        a = int(alpha * 150)
        dr.rounded_rectangle((x0, y0, x0 + bw, y0 + th + 2 * pad), radius=int(14 * self.sc), fill=(0, 0, 0, a))
        y = y0 + pad
        for l, f, w_, h_ in zip(lines, fonts, widths, lh):
            dr.text(((W - w_) / 2, y), l, font=f, fill=(255, 255, 255, int(alpha * 255)))
            y += h_
        im.alpha_composite(ov)

    def label(self, im, text, xy, size=22, alpha=1.0, bold=False, color=(255, 255, 255), anchor="la", box=True):
        if alpha <= 0.01:
            return
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        f = self.font(size, bold)
        w = dr.textlength(text, font=f)
        h = int(size * self.sc * 1.25)
        x, y = xy
        if anchor == "ra":
            x -= w
        pad = int(8 * self.sc)
        if box:
            dr.rounded_rectangle((x - pad, y - pad // 2, x + w + pad, y + h + pad // 2), radius=int(8 * self.sc), fill=(0, 0, 0, int(alpha * 140)))
        dr.text((x, y), text, font=f, fill=(*color, int(alpha * 255)))
        im.alpha_composite(ov)

    def section_label(self, im, shot, alpha=1.0):
        self.label(im, self.T["sec"][shot], (int(0.05 * self.W), int(0.05 * self.H)), size=24, alpha=alpha)

    def clock(self, im, t, speed, alpha=1.0):
        """Mission clock, speed and the ripe-fruit progress bar, top right."""
        if alpha <= 0.01:
            return
        W, H = im.size
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        x1 = int(0.95 * W)
        y0 = int(0.05 * H)
        bw, bh = int(360 * self.sc), int(118 * self.sc)
        x0 = x1 - bw
        dr.rounded_rectangle((x0, y0, x1, y0 + bh), radius=int(12 * self.sc), fill=(0, 0, 0, int(alpha * 140)))
        pad = int(14 * self.sc)
        f1, f2, f3 = self.font(20), self.font(34, True), self.font(18)
        dr.text((x0 + pad, y0 + pad * 0.6), self.T["clock"], font=f1, fill=(220, 220, 220, int(alpha * 255)))
        dr.text((x0 + pad, y0 + pad * 0.6 + int(22 * self.sc)), hhmm(t), font=f2, fill=(255, 255, 255, int(alpha * 255)))
        sp = self.T["speed"](speed)
        dr.text((x1 - pad - dr.textlength(sp, font=f3), y0 + pad * 0.6 + int(30 * self.sc)), sp, font=f3, fill=(255, 220, 120, int(alpha * 255)))
        # progress bar
        pr = self.progress(t)
        by = y0 + bh - pad - int(14 * self.sc)
        txt = f"{self.T['progress']}  " + (f"%{pr * 100:.0f}" if self.lang == "tr" else f"{pr * 100:.0f} %")
        dr.text((x0 + pad, by - int(22 * self.sc)), txt, font=f3, fill=(220, 220, 220, int(alpha * 255)))
        dr.rounded_rectangle((x0 + pad, by, x1 - pad, by + int(10 * self.sc)), radius=int(5 * self.sc), fill=(255, 255, 255, int(alpha * 70)))
        if pr > 0:
            dr.rounded_rectangle((x0 + pad, by, x0 + pad + int((x1 - x0 - 2 * pad) * pr), by + int(10 * self.sc)), radius=int(5 * self.sc),
                                 fill=(235, 90, 60, int(alpha * 255)))
        im.alpha_composite(ov)

    def fade(self, im, a):
        """Multiply towards black (a = 1 full picture, 0 black)."""
        if a >= 0.999:
            return im
        return Image.blend(Image.new("RGBA", im.size, (0, 0, 0, 255)), im, max(0.0, a))

    # -- sections
    def sec_title(self, i, n):
        base = self.frame("intro", 1).filter(ImageFilter.GaussianBlur(int(14 * self.sc))).convert("RGBA")
        base = self.fade(base, 0.45)
        s = i / float(n - 1)
        a_in = smooth(s / 0.15) * (1 - smooth((s - 0.85) / 0.15))
        W, H = base.size
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        f1, f2 = self.font(54, True), self.font(26)
        lines = self.wrap(self.T["title"], f1, int(0.84 * W), dr)
        y = int(0.40 * H) - len(lines) * int(66 * self.sc) // 2
        for l in lines:
            w = dr.textlength(l, font=f1)
            dr.text(((W - w) / 2, y), l, font=f1, fill=(255, 255, 255, int(a_in * 255)))
            y += int(66 * self.sc)
        sub = self.wrap(self.T["subtitle"], f2, int(0.84 * W), dr)
        y += int(14 * self.sc)
        for l in sub:
            w = dr.textlength(l, font=f2)
            dr.text(((W - w) / 2, y), l, font=f2, fill=(230, 230, 230, int(a_in * 255)))
            y += int(36 * self.sc)
        base.alpha_composite(ov)
        return base

    def wrap(self, text, font, maxw, dr):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if dr.textlength(cand, font=font) <= maxw or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def caption_alpha(self, s, a, b, f=0.5):
        """1 inside [a, b] seconds with f-second fades."""
        return smooth((s - a) / f) * (1 - smooth((s - (b - f)) / f))

    def sec_intro(self, i, n):
        im = self.frame("intro", i + 1).convert("RGBA")
        s = i / float(FPS)
        dur = n / float(FPS)
        self.section_label(im, "intro", self.caption_alpha(s, 0.3, dur + 1))
        self.band_text(im, self.wrap2(self.T["intro1"]), self.caption_alpha(s, 0.8, 5.6), size=28)
        self.band_text(im, [self.T["intro2"]], self.caption_alpha(s, 5.6, dur + 1), size=28)
        return self.fade(im, smooth(s / FADE_S) * (1 - smooth((s - (dur - FADE_S)) / FADE_S)))

    def wrap2(self, text):
        dr = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        return self.wrap(text, self.font(28), int(0.86 * self.W), dr)

    # ---- map section (one still + 2D overlay)
    def map_px(self, x, y):
        m = self.meta["map"]["map"]
        return (m["cx_px"] + (x - m["cam_x"]) * m["px_per_m"], m["cy_px"] - (y - m["cam_y"]) * m["px_per_m"])

    def map_setup(self):
        D, REG, PREP = self.D, self.REG, self.PREP
        xs, ys = np.array(D["trees"]["x"]), np.array(D["trees"]["y"])
        au, aw = D["spacing"]
        # lattice (as build_scene_v4) for the region rectangles
        def off(v, a):
            ph = np.exp(2j * np.pi * v / a)
            return (np.angle(ph.mean()) / (2 * np.pi)) * a
        ox, oy = off(xs, au), off(ys, aw)
        ix, iy = np.rint((xs - ox) / au).astype(int), np.rint((ys - oy) / aw).astype(int)
        kd = np.array(REG["kd_region"])
        rects = {}
        for j in range(int(D["k"])):
            m = kd == j
            rects[j] = (ox + (ix[m].min() - 0.5) * au + 0.3, oy + (iy[m].min() - 0.5) * aw + 0.3,
                        ox + (ix[m].max() + 0.5) * au - 0.3, oy + (iy[m].max() + 0.5) * aw - 0.3)
        sx, sy = D["station"]
        plat = []
        for p in range(int(D["k"])):
            o = ((p + 1) // 2) * 3.4 * (1 if p % 2 else -1)
            plat.append((sx + o, sy))
        cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
        # cut -> regions finished by it
        cuts = []
        for c in PREP["cuts"]:
            done = []
            if c["j_left"] == 1:
                done.append(int(kd[c["left"][0]]))
            if c["j_right"] == 1:
                done.append(int(kd[c["right"][0]]))
            cuts.append((c["line"], done))
        self.map = dict(rects=rects, plat=plat, centre=(cx, cy), cuts=cuts, grid=PREP["grid"], kd=kd, xs=xs, ys=ys)

    def sec_map(self, i, n):
        if not hasattr(self, "map"):
            self.map_setup()
        M = self.map
        base = self.frame("map", 1).convert("RGBA")
        s = i / float(FPS)
        dur = n / float(FPS)
        W, H = base.size
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        cxp, cyp = self.map_px(*M["centre"])
        # (a) estimated axes: grid lines rotate from +8 deg onto the rows, then fade to a faint grid
        ga = self.caption_alpha(s, 1.0, 14.5, 0.6)
        rot = math.radians(8.0) * (1 - smooth((s - 1.2) / 2.6))
        ca, sa = math.cos(rot), math.sin(rot)
        gcol = (255, 255, 255, int(ga * (150 if s < 5.5 else 70)))
        lw = max(1, int(1.2 * self.sc))
        for (p0, p1) in M["grid"]:
            pts = []
            for (x, y) in (p0, p1):
                px, py = self.map_px(x, y)
                dx, dy = px - cxp, py - cyp
                pts.append((cxp + ca * dx - sa * dy, cyp + sa * dx + ca * dy))
            dr.line(pts, fill=gcol, width=lw)
        # (b) k-d cuts one after another
        t_cut0, dcut = 6.0, 1.05
        done_regions = {}
        for ci, (line, done) in enumerate(M["cuts"]):
            u = (s - (t_cut0 + ci * dcut)) / 0.8
            if u <= 0:
                continue
            u = min(1.0, u)
            (x0, y0), (x1, y1) = line
            p0, p1 = self.map_px(x0, y0), self.map_px(x1, y1)
            pe = (p0[0] + (p1[0] - p0[0]) * smooth(u), p0[1] + (p1[1] - p0[1]) * smooth(u))
            dr.line([p0, pe], fill=(255, 255, 255, 230), width=max(2, int(3 * self.sc)))
            if u >= 1.0:
                for j in done:
                    done_regions[j] = t_cut0 + ci * dcut + 0.8
        for j, t_done in done_regions.items():
            a = smooth((s - t_done) / 0.6)
            r = M["rects"][j]
            p0, p1 = self.map_px(r[0], r[3]), self.map_px(r[2], r[1])
            col = PLATFORM_COLORS[j % 8]
            dr.rectangle((p0, p1), fill=rgb(col, int(a * 70)), outline=rgb(col, int(a * 255)), width=max(2, int(3 * self.sc)))
            # platform ring at the station takes the region colour
            px, py = self.map_px(*M["plat"][j])
            rr = 3.0 * self.meta["map"]["map"]["px_per_m"] * 1.6
            dr.ellipse((px - rr, py - rr, px + rr, py + rr), outline=rgb(col, int(a * 255)), width=max(2, int(3 * self.sc)))
        # legend panel (right)
        if done_regions:
            x0 = int(0.62 * W)
            y0 = int(0.12 * H)
            f1, f2 = self.font(26, True), self.font(24)
            a_leg = smooth((s - min(done_regions.values())) / 0.6)
            dr.rounded_rectangle((x0 - int(20 * self.sc), y0 - int(16 * self.sc), int(0.95 * W), y0 + int(60 * self.sc) + 7 * int(40 * self.sc)),
                                 radius=int(12 * self.sc), fill=(0, 0, 0, int(a_leg * 130)))
            dr.text((x0, y0), self.T["legend_title"], font=f1, fill=(255, 255, 255, int(a_leg * 255)))
            for j in range(int(self.D["k"])):
                a = smooth((s - done_regions[j]) / 0.6) if j in done_regions else 0.0
                yy = y0 + int(52 * self.sc) + j * int(40 * self.sc)
                col = PLATFORM_COLORS[j % 8]
                dr.rounded_rectangle((x0, yy + int(4 * self.sc), x0 + int(26 * self.sc), yy + int(26 * self.sc)), radius=int(5 * self.sc), fill=rgb(col, int(a * 255)))
                dr.text((x0 + int(40 * self.sc), yy), self.T["legend_row"](j, self.REG["region_sizes"][j]), font=f2, fill=(255, 255, 255, int(a * 255)))
        base.alpha_composite(ov)
        # slow zoom towards the block
        z = 1.0 + 0.06 * smooth(s / dur)
        if z > 1.001:
            cw, ch = W / z, H / z
            zx, zy = cxp * 0.35 + W / 2 * 0.65, cyp * 0.35 + H / 2 * 0.65
            x0 = min(max(0, zx - cw / 2), W - cw)
            y0 = min(max(0, zy - ch / 2), H - ch)
            base = base.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.LANCZOS)
        self.section_label(base, "map")
        self.band_text(base, [self.T["map0"]], self.caption_alpha(s, 0.2, 1.6), size=28)
        self.band_text(base, self.wrap2(self.T["map_a"]), self.caption_alpha(s, 1.6, 6.2), size=28)
        if 3.9 < s < 6.4:
            px, py = self.map_px(M["xs"].max() + 4, M["ys"].max() - 6)
            self.label(base, self.T["axis_lbl"], (px, py), size=24, alpha=self.caption_alpha(s, 3.9, 6.4, 0.4), bold=True, color=(255, 230, 140))
        self.band_text(base, self.wrap2(self.T["map_b"]), self.caption_alpha(s, 6.2, dur + 1), size=28)
        return self.fade(base, smooth(s / FADE_S) * (1 - smooth((s - (dur - FADE_S)) / FADE_S)))

    # ---- time-lapse sections
    def sec_timelapse(self, shot, i, n, captions):
        im = self.frame(shot, i + 1).convert("RGBA")
        s = i / float(FPS)
        dur = n / float(FPS)
        t = self.meta[shot]["t"][min(i, len(self.meta[shot]["t"]) - 1)]
        self.section_label(im, shot)
        self.clock(im, t, SHOTS[shot]["speed"])
        for (a, b, key) in captions:
            self.band_text(im, self.wrap2(self.T[key]), self.caption_alpha(s, a, b), size=28)
        return self.fade(im, smooth(s / FADE_S) * (1 - smooth((s - (dur - FADE_S)) / FADE_S)))

    def sec_closeup(self, i, n):
        im = self.frame("closeup", i + 1).convert("RGBA")
        s = i / float(FPS)
        dur = n / float(FPS)
        t = self.meta["closeup"]["t"][min(i, len(self.meta["closeup"]["t"]) - 1)]
        self.section_label(im, "closeup")
        self.clock(im, t, 1.0)
        cu = self.meta["closeup"]["closeup"]
        self.label(im, self.T["pick_lbl"], (int(0.05 * self.W), int(0.05 * self.H) + int(44 * self.sc)), size=20, alpha=self.caption_alpha(s, 0.5, dur + 1))
        self.band_text(im, self.wrap2(self.T["closeup1"]), self.caption_alpha(s, 0.6, 9.0), size=28)
        self.band_text(im, self.wrap2(self.T["closeup2"]), self.caption_alpha(s, 9.0, dur + 1), size=26)
        # failed grasp label (drone 1, second cycle) from the choreography events
        ev = [e for e in cu.get("events", []) if e[2] == 1]
        if len(ev) >= 2:
            s_fail = ev[1][0] - 2.1       # the failed grasp happens 2.1 s (fail + re-approach + grasp) before the detach
            self.label(im, self.T["closeup_fail"], (int(0.05 * self.W), int(0.05 * self.H) + int(84 * self.sc)), size=22,
                       alpha=self.caption_alpha(s, s_fail - 0.3, s_fail + 2.4, 0.3), bold=True, color=(255, 200, 120))
        return self.fade(im, smooth(s / FADE_S) * (1 - smooth((s - (dur - FADE_S)) / FADE_S)))

    def sec_finish_card(self, i, n):
        last = self.files["finish"][-1]
        im = self.frame("finish", last).convert("RGBA")
        s = i / float(FPS)
        dur = n / float(FPS)
        t = self.meta["finish"]["t"][-1]
        a = smooth(s / 0.8)
        im = self.fade(im, 1 - 0.55 * a)
        W, H = im.size
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        f1, f2, f3 = self.font(40, True), self.font(28), self.font(28, True)
        rows = self.T["card"]
        bw, bh = int(0.70 * W), int((110 + 52 * len(rows)) * self.sc)
        x0, y0 = (W - bw) // 2, (H - bh) // 2
        dr.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=int(18 * self.sc), fill=(10, 14, 10, int(a * 215)))
        dr.text((x0 + int(40 * self.sc), y0 + int(30 * self.sc)), self.T["card_title"], font=f1, fill=(255, 255, 255, int(a * 255)))
        for r, (k, v) in enumerate(rows):
            ar = smooth((s - 0.6 - 0.25 * r) / 0.5)
            yy = y0 + int((100 + 52 * r) * self.sc)
            dr.text((x0 + int(40 * self.sc), yy), k, font=f2, fill=(210, 210, 210, int(ar * 255)))
            w = dr.textlength(v, font=f3)
            dr.text((x0 + bw - int(40 * self.sc) - w, yy), v, font=f3, fill=(255, 255, 255, int(ar * 255)))
        im.alpha_composite(ov)
        self.clock(im, t, SHOTS["finish"]["speed"], alpha=1 - a)
        return self.fade(im, 1 - smooth((s - (dur - 0.8)) / 0.8))

    def sections(self):
        n_title = int(TITLE_S * FPS)
        n_map = int(MAP_S * FPS)
        n_card = int(CARD_S * FPS)
        return [
            ("title", n_title, self.sec_title),
            ("intro", SHOTS["intro"]["frames"], self.sec_intro),
            ("map", n_map, self.sec_map),
            ("mission", SHOTS["mission"]["frames"], lambda i, n: self.sec_timelapse("mission", i, n, [(0.6, 12.5, "mission1"), (12.5, 25, "mission2")])),
            ("closeup", SHOTS["closeup"]["frames"], self.sec_closeup),
            ("takeover", SHOTS["takeover"]["frames"], lambda i, n: self.sec_timelapse("takeover", i, n, [(0.6, 6.5, "takeover1"), (6.5, 12, "takeover2")])),
            ("finish", SHOTS["finish"]["frames"], lambda i, n: self.sec_timelapse("finish", i, n, [(0.6, 9, "finish1")])),
            ("card", n_card, self.sec_finish_card),
        ]

    def run(self, out, crf=18, title=""):
        secs = self.sections()
        total = sum(n for _, n, _ in secs)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}",
               "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-threads", "4", "-movflags", "+faststart", "-metadata", f"title={title}", "-metadata", "comment=", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        k = 0
        for name, n, fn in secs:
            for i in range(n):
                im = fn(i, n)
                proc.stdin.write(im.convert("RGB").tobytes())
                k += 1
                if k % 300 == 0:
                    print(f"  {k}/{total} frames ({name})", flush=True)
        proc.stdin.close()
        proc.wait()
        print(f"written {out}: {total} frames, {total / FPS:.1f} s, {self.W}x{self.H}, exit {proc.returncode}")
        return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="tr", choices=("tr", "en"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", default=os.path.join(HERE, "out", "video", "frames"))
    ap.add_argument("--data", default=os.path.normpath(os.path.join(HERE, "..", "anim4_rect1ha.json")))
    ap.add_argument("--regions", default=None)
    ap.add_argument("--prep", default=os.path.join(HERE, "out", "video", "video_prep.json"))
    ap.add_argument("--draft", action="store_true", help="crf 23")
    a = ap.parse_args()
    regions = a.regions or os.path.splitext(a.data)[0] + "_regions.json"
    out = a.out or os.path.join(HERE, "out", "video", f"harvest_explainer_{a.lang}.mp4")
    c = Compositor(a.lang, a.frames, a.draft, a.data, regions, a.prep)
    title = c.T["title"]
    sys.exit(c.run(out, crf=23 if a.draft else 18, title=title))


if __name__ == "__main__":
    main()
