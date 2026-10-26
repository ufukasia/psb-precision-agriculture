# -*- coding: utf-8 -*-
"""Faz 1 raporu: anytime egrileri, butce tablolari, yetisme sureleri, makrolar.

Referans: her (ornek, k) icin tum yontemlerin tum tohumlardaki ve tum
zamanlardaki en iyi makespan'i (BEST). Sapma = mk(t) / BEST - 1; mk(t) = izde
t'ye kadarki en iyi deger (yoksa cozum yok = sonsuz).
Tekrarlar: her (ornek, yontem) icin mevcut tum tohumlar; ornek basina deger =
tohumlar uzerinden medyan (sonsuz dahil); "cozum yok" orani kosu basinadir.
Istatistik: ornek basina tohum ortalamasi nihai sapmalar uzerinde Wilcoxon
isaretli sira testi (iki yonlu), PSB+LNS'e karsi her LKH-3 varyanti.

Kosum:  python paper2/experiments/report.py
Cikti:  paper2/experiments/out/report/{summary.json, fig_anytime.pdf, ...}
        paper2/makale/sayilar.tex
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out", "report")
BENCH = os.path.join(HERE, "out", "bench")
METHODS = ("psb", "lkh_tuned", "lkh_warm", "lkh_def")
SOTA = ("hsnr", "ma", "mils", "hga")
LABEL = {"psb": "PSB+LNS (ours)", "lkh_tuned": "LKH-3 tuned", "lkh_warm": "LKH-3 warm (PSB init)",
         "lkh_def": "LKH-3 default", "hsnr": "HSNR", "ma": "MA", "mils": "MILS", "hga": "HGA"}
BUDGETS = (0.1, 1.0, 10.0, 60.0, 300.0, 600.0)


SLACK_REL, SLACK_ABS = 0.02, 2.0   # sure siniri asimi toleransi (tum yontemlere esit; mutlak pay yalniz t >= 60 s)


def mk_at(trace, t):
    best = math.inf
    for tt, c in trace:
        if tt <= t * (1 + SLACK_REL) + (SLACK_ABS if t >= 60 else 0.0):
            best = min(best, c)
    return best


def load():
    """recs[(grup, ad, k)][yontem] = {tohum: kayit}."""
    recs = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(BENCH, "*.json")):
        r = json.load(open(f))
        if r["group"] == "lit":
            continue
        recs[(r["group"], r["name"], r["k"])][r["method"]][r.get("seed", 1)] = r
    return recs


def tol(b):
    return b * (1 + SLACK_REL) + (SLACK_ABS if b >= 60 else 0.0)


def final_of(r):
    """Butce (cap) toleransi icindeki en iyi makespan: izde cap'e kadarki en iyi
    deger; cozum dosyasinin makespan'i yalniz kosu toleransta bittiyse sayilir
    (kaba iz basan kodlarda son cozum izde gorunmeyebilir)."""
    if r.get("skipped"):
        return math.inf
    cap = r["cap"]
    best = mk_at(r["trace"], cap)
    wall = r.get("wall") or r.get("lkh_wall") or r.get("elapsed") or math.inf
    if r.get("mk") and wall <= tol(cap):
        best = min(best, r["mk"])
    return best


def first_reach(trace, target):
    for t, c in sorted(trace):
        if c <= target * (1 + 1e-9):
            return t
    return math.inf


def macro_name(s):
    import re
    s = s.replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three") \
         .replace("4", "Four").replace("5", "Five").replace("6", "Six").replace("7", "Seven") \
         .replace("8", "Eight").replace("9", "Nine")
    return re.sub(r"[^A-Za-z]", "", s)


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = load()
    rows = []
    for key, ms in sorted(recs.items()):
        if not all(m in ms and 1 in ms[m] for m in METHODS):
            continue
        g, nm, k = key
        have = [m for m in METHODS + SOTA if m in ms and any(not r.get("skipped") for r in ms[m].values())]
        best = min(final_of(r) for m in have for r in ms[m].values() if not r.get("skipped"))
        p1 = ms["psb"][1]
        row = dict(group=g, name=nm, k=k, n=p1["n"], best=best, lb=p1["lb"], cap=p1["cap"],
                   psb_t0=p1["psb"]["t_total"], psb_mk0=p1["psb"]["mk"],
                   seeds={m: sorted(ms[m]) for m in have}, have=have)
        for m in have:
            runs = [ms[m][sd] for sd in sorted(ms[m]) if not ms[m][sd].get("skipped")]
            row[m] = {}
            for b in BUDGETS:
                v = [mk_at(r["trace"], b) / best - 1 for r in runs]
                row[m][str(b)] = float(np.median(v))
                row[m]["nosol_" + str(b)] = float(np.mean([not np.isfinite(x) for x in v]))
            fins = [final_of(r) / best - 1 for r in runs]
            row[m]["final"] = float(np.median(fins))
            row[m]["final_mean"] = float(np.mean(fins))
            row[m]["final_all"] = fins
            row[m]["first_t"] = float(np.median([min((t for t, _ in r["trace"]), default=math.inf) for r in runs]))
        # yetisme: LKH'nin PSB tek-atis ve PSB@10s kalitesine ulasma suresi (tohum medyani)
        for m in ("lkh_tuned", "lkh_def"):
            row[m]["reach_psb0"] = float(np.median([first_reach(r["trace"], row["psb_mk0"]) for r in ms[m].values()]))
            p10 = mk_at(p1["trace"], 10.0)
            row[m]["reach_psb10"] = float(np.median([first_reach(r["trace"], p10) for r in ms[m].values()]))
        rows.append(row)
    json.dump(rows, open(os.path.join(OUT, "rows.json"), "w"), default=lambda x: None)

    def wilcox(R, a, b):
        from scipy.stats import wilcoxon
        x = np.array([r[a]["final_mean"] for r in R])
        y = np.array([r[b]["final_mean"] for r in R])
        ok = np.isfinite(x) & np.isfinite(y)
        d = x[ok] - y[ok]
        if (np.abs(d) > 1e-12).sum() < 5:
            return None
        return float(wilcoxon(x[ok], y[ok], zero_method="wilcox").pvalue)

    def stats_sota(s, R):
        for m in SOTA:
            Rm = [r for r in R if m in r["have"]]
            s[f"{m}_count"] = len(Rm)
            if not Rm:
                continue
            for b in BUDGETS:
                vals = [r[m][str(b)] for r in Rm if r["cap"] >= b]
                s[f"{m}_gap_{b:g}"] = med_all([v * 100 for v in vals])
                ns = [r[m]["nosol_" + str(b)] for r in Rm if r["cap"] >= b]
                s[f"{m}_nosol_{b:g}"] = float(np.mean(ns)) if ns else None
            s[f"{m}_gap_final"] = med_all([r[m]["final"] * 100 for r in Rm])
            s[f"{m}_first_t_med"] = med([r[m]["first_t"] for r in Rm])
            s[f"psb_win_vs_{m}"] = float(np.mean([r["psb"]["final_mean"] < r[m]["final_mean"] - 1e-12 for r in Rm]))
            s[f"psb_win_vs_{m}_10"] = float(np.mean([r["psb"]["10.0"] < r[m]["10.0"] - 1e-12 for r in Rm]))
            s[f"psb_win_vs_{m}_1"] = float(np.mean([r["psb"]["1.0"] < r[m]["1.0"] - 1e-12 for r in Rm]))
            s[f"p_vs_{m}"] = wilcox(Rm, "psb", m)

    def stats_common(s, R):
        s["win_vs_tuned_final"] = float(np.mean([r["psb"]["final_mean"] < r["lkh_tuned"]["final_mean"] - 1e-12 for r in R]))
        s["win_vs_warm_final"] = float(np.mean([r["psb"]["final_mean"] < r["lkh_warm"]["final_mean"] - 1e-12 for r in R]))
        s["p_vs_tuned"] = wilcox(R, "psb", "lkh_tuned")
        s["p_vs_warm"] = wilcox(R, "psb", "lkh_warm")
        s["psb_seed_sd_med"] = float(np.median([np.std(np.array(r["psb"]["final_all"]) * 100) for r in R]))
        s["seeds_max"] = int(max(len(r["seeds"]["psb"]) for r in R))
        s["seeds_min"] = int(min(len(r["seeds"]["psb"]) for r in R))
        stats_sota(s, R)

    def med(vals):
        """Sonlu degerlerin medyani (sure olcutleri icin)."""
        v = [x for x in vals if np.isfinite(x)]
        return float(np.median(v)) if v else math.inf

    def med_all(vals):
        """Cozum yok = +sonsuz sayilarak medyan (sapma olcutleri icin)."""
        v = list(vals)
        return float(np.median(v)) if v else math.inf

    summ = {}
    for g in ("mtsplib", "orchard", "osm", "vlsi", "all"):
        R = [r for r in rows if g == "all" or r["group"] == g]
        if not R:
            continue
        s = dict(count=len(R))
        for m in METHODS:
            for b in BUDGETS:
                vals = [r[m][str(b)] for r in R if r["cap"] >= b]
                s[f"{m}_gap_{b:g}"] = med_all([v * 100 for v in vals])
                ns = [r[m]["nosol_" + str(b)] for r in R if r["cap"] >= b]
                s[f"{m}_nosol_{b:g}"] = float(np.mean(ns)) if ns else None
            s[f"{m}_gap_final"] = med_all([r[m]["final"] * 100 for r in R])
        stats_common(s, R)
        for b in BUDGETS:
            pairs = [(r["psb"][str(b)], r["lkh_tuned"][str(b)]) for r in R if r["cap"] >= b]
            s[f"win_vs_tuned_{b:g}"] = float(np.mean([p < q - 1e-12 for p, q in pairs])) if pairs else None
        s["psb_t0_med"] = med([r["psb_t0"] for r in R])
        s["psb_gap0_med"] = med([(r["psb_mk0"] / r["best"] - 1) * 100 for r in R])
        s["lb_gap_psb_final"] = med([((1 + r["psb"]["final"]) * r["best"] / r["lb"] - 1) * 100 for r in R])
        reach = [r["lkh_tuned"]["reach_psb0"] for r in R]
        s["tuned_reach_psb0_med"] = med(reach)
        s["tuned_reach_psb0_never"] = float(np.mean([not np.isfinite(x) for x in reach]))
        reach = [r["lkh_tuned"]["reach_psb10"] for r in R]
        s["tuned_reach_psb10_med"] = med(reach)
        s["tuned_reach_psb10_never"] = float(np.mean([not np.isfinite(x) for x in reach]))
        summ[g] = s
    # n sinifina gore
    for lo, hi, tag in ((0, 1000, "small"), (1000, 5000, "mid"), (5000, 10 ** 9, "large")):
        R = [r for r in rows if lo <= r["n"] < hi]
        if not R:
            continue
        s = dict(count=len(R))
        for m in METHODS:
            for b in BUDGETS:
                vals = [r[m][str(b)] for r in R if r["cap"] >= b]
                s[f"{m}_gap_{b:g}"] = med_all([v * 100 for v in vals])
                ns = [r[m]["nosol_" + str(b)] for r in R if r["cap"] >= b]
                s[f"{m}_nosol_{b:g}"] = float(np.mean(ns)) if ns else None
            s[f"{m}_gap_final"] = med_all([r[m]["final"] * 100 for r in R])
        stats_common(s, R)
        summ["n_" + tag] = s
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=lambda x: None)

    # anytime egrisi (medyan sapma) -- grup basina; cozum yok / > CLIP ust bantta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CLIP = 15.0
    ts = np.logspace(-2, math.log10(600), 80)
    PANELS = [("mTSPLib, n = 50–98 (120 s)", lambda r: r["group"] == "mtsplib"),
              ("n < 1000 (120 s)", lambda r: r["group"] != "mtsplib" and r["n"] < 1000),
              ("1000 ≤ n < 5000 (300 s)", lambda r: 1000 <= r["n"] < 5000),
              ("n ≥ 5000 (600 s)", lambda r: r["n"] >= 5000)]
    groups = [p for p in PANELS if any(p[1](r) for r in rows)]
    best_of = {(r["group"], r["name"], r["k"]): r["best"] for r in rows}
    fig, axs = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.4), squeeze=False)
    for ax, (title, sel) in zip(axs[0], groups):
        keys = [(r["group"], r["name"], r["k"]) for r in rows if sel(r)]
        tmax = min(recs[key]["psb"][1]["cap"] for key in keys)
        styles = dict(psb=("-", "C0", 2.2), lkh_tuned=("--", "C1", 1.6), lkh_warm=("-.", "C2", 1.6),
                      lkh_def=(":", "C3", 1.6), hsnr=((0, (1, 1)), "C4", 1.3), ma=((0, (5, 1)), "C5", 1.3),
                      mils=((0, (3, 1, 1, 1)), "C6", 1.6), hga=((0, (5, 2, 1, 2)), "C7", 1.3))
        for m in METHODS + SOTA:
            if not all(m in recs[key] and any(not r.get("skipped") for r in recs[key][m].values()) for key in keys):
                continue
            st, col, lw = styles[m]
            curve = []
            for t in ts:
                v = [np.median([mk_at(r["trace"], t) / best_of[key] - 1 for r in recs[key][m].values()
                                if not r.get("skipped")])
                     for key in keys] if t <= tmax * 1.001 else []
                curve.append(100 * np.median(v) if v else np.nan)
            curve = np.array(curve, float)
            curve[np.isinf(curve)] = CLIP
            curve = np.minimum(curve, CLIP)
            ax.plot(ts, curve, linestyle=st, color=col, label=LABEL[m], lw=lw)
        ax.axhspan(CLIP * 0.97, CLIP * 1.05, color="0.85", zorder=0)
        ax.text(0.011, CLIP * 0.93, f"> {CLIP:g}% or no solution yet", fontsize=7, va="top", color="0.35")
        ax.set_xscale("log")
        ax.set_ylim(-0.3, CLIP * 1.05)
        ax.set_xlabel("wall-clock time (s, single core)")
        ax.set_ylabel("median gap to best found (%)")
        ax.set_title(f"{title}, {len(keys)} runs", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axs[0][0].legend(fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_anytime.pdf"))
    fig.savefig(os.path.join(OUT, "fig_anytime.png"), dpi=150)
    print(json.dumps(summ, indent=1, default=lambda x: None)[:6000])


if __name__ == "__main__":
    main()
