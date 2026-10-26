# -*- coding: utf-8 -*-
"""Makale 2: tum sayilar ve tablolar ham sonuclardan (makrolar + LaTeX tablolari).

Girdi : out/report/{summary.json, rows.json} (report.py), out/bench/lit__*.json,
        out/sim/*.json
Cikti : generated/numbers.tex, tab_budget.tex, tab_lit.tex, tab_sim.tex
Kosum : python experiments/report.py && python experiments/make_numbers.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
MK = os.path.join(P2, "generated")
os.makedirs(os.path.join(MK, "figs"), exist_ok=True)
sys.path.insert(0, HERE)
import report as R      # noqa: E402
import lit_bks          # noqa: E402

M = {}


def put(name, val, fmt="{:.1f}"):
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        M[name] = r"\ensuremath{\infty}"
    elif isinstance(val, str):
        M[name] = val
    else:
        M[name] = fmt.format(val)


def pct(x):
    return 100.0 * x


def main_bench():
    S = json.load(open(os.path.join(HERE, "out", "report", "summary.json")))
    rows = json.load(open(os.path.join(HERE, "out", "report", "rows.json")))
    put("OrchCount", sum(1 for r in rows if r["group"] == "orchard"), "{:d}")
    put("VlsiCount", sum(1 for r in rows if r["group"] == "vlsi"), "{:d}")
    put("MainRuns", len(rows), "{:d}")
    on = [r["n"] for r in rows if r["group"] == "orchard"]
    vn = [r["n"] for r in rows if r["group"] == "vlsi"]
    put("OrchNmin", min(on), "{:d}")
    put("OrchNmax", max(on), "{:d}")
    put("VlsiNmin", min(vn), "{:d}")
    put("VlsiNmax", max(vn), "{:d}")
    a = S["all"]
    put("PsbOneShotMs", 1000 * a["psb_t0_med"], "{:.0f}")
    put("PsbOneShotGap", a["psb_gap0_med"])
    put("ReachNeverAll", pct(a["tuned_reach_psb0_never"]), "{:.0f}")
    put("ReachNeverTenAll", pct(a["tuned_reach_psb10_never"]), "{:.0f}")
    for g, tag in (("orchard", "Orch"), ("vlsi", "Vlsi"), ("mtsplib", "Mtsp")):
        s = S[g]
        put(f"PsbOneShotMs{tag}", 1000 * s["psb_t0_med"], "{:.0f}")
        put(f"WinFinal{tag}", pct(s["win_vs_tuned_final"]), "{:.0f}")
        put(f"WinOne{tag}", pct(s["win_vs_tuned_1"]), "{:.0f}")
        put(f"WinSixty{tag}", pct(s["win_vs_tuned_60"]), "{:.0f}")
        put(f"ReachNever{tag}", pct(s["tuned_reach_psb0_never"]), "{:.0f}")
        put(f"ReachMed{tag}", s["tuned_reach_psb0_med"])
        put(f"LbGap{tag}", s["lb_gap_psb_final"])
        put(f"TunedFinal{tag}", s["lkh_tuned_gap_final"])
        put(f"PsbFinal{tag}", s["psb_gap_final"], "{:.2f}")
    for cls, tag in (("n_small", "Small"), ("n_mid", "Mid"), ("n_large", "Large")):
        s = S[cls]
        put(f"Count{tag}", s["count"], "{:d}")
        put(f"WinFinal{tag}", pct(s["win_vs_tuned_final"]), "{:.0f}")
        put(f"WinWarm{tag}", pct(s["win_vs_warm_final"]), "{:.0f}")
        put(f"TunedFinal{tag}", s["lkh_tuned_gap_final"])
        put(f"WarmFinal{tag}", s["lkh_warm_gap_final"], "{:.2f}")
        put(f"PsbFinal{tag}", s["psb_gap_final"], "{:.2f}")
        put(f"DefFinal{tag}", s["lkh_def_gap_final"])
        for b in ("1", "10", "60"):
            put(f"PsbGap{tag}{ {'1': 'One', '10': 'Ten', '60': 'Sixty'}[b] }", s[f"psb_gap_{b}"], "{:.1f}")
            put(f"TunedNosol{tag}{ {'1': 'One', '10': 'Ten', '60': 'Sixty'}[b] }",
                pct(s[f"lkh_tuned_nosol_{b}"] or 0.0), "{:.0f}")
    # butce tablosu
    def cell(s, m, b):
        v = s.get(f"{m}_gap_{b}")
        ns = s.get(f"{m}_nosol_{b}")
        if v is None:
            return "--"
        txt = r"\ensuremath{\infty}" if not math.isfinite(v) else f"{v:.2f}"
        if ns:
            txt += rf"\,{{\scriptsize({100 * ns:.0f}\%)}}"
        return txt
    SHORT = {"psb": "PSB+LNS", "lkh_tuned": "LKH-3 tuned", "lkh_warm": "LKH-3 warm", "lkh_def": "LKH-3 default"}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"method & 0.1\,s & 1\,s & 10\,s & 60\,s & cap & final \\", r"\midrule"]
    caps = {"mtsplib": "120", "n_small": "120", "n_mid": "300", "n_large": "600"}
    names = {"mtsplib": "mTSPLib", "n_small": r"$n<1000$", "n_mid": r"$1000\le n<5000$", "n_large": r"$n\ge 5000$"}
    for cls in ("mtsplib", "n_small", "n_mid", "n_large"):
        s = S[cls]
        lines.append(r"\multicolumn{7}{l}{\emph{" + names[cls] + rf", {s['count']} instances, cap {caps[cls]}\,s" + r"}} \\")
        best = min(s[f"{m}_gap_final"] for m in R.METHODS)
        for m in R.METHODS:
            fin = s[f"{m}_gap_final"]
            txt = r"\ensuremath{\infty}" if not math.isfinite(fin) else f"{fin:.2f}"
            if fin == best:
                txt = r"\textbf{" + txt + "}"
            lines.append(f"{SHORT[m]} & " + " & ".join(cell(s, m, b) for b in ("0.1", "1", "10", "60", caps[cls]))
                         + f" & {txt} \\\\")
        lines.append(r"\midrule" if cls != "n_large" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(os.path.join(MK, "tab_budget.tex"), "w").write("\n".join(lines) + "\n")


def lit():
    recs = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(HERE, "out", "bench", "lit__*.json")):
        r = json.load(open(f))
        recs[(r["name"], r["k"])][r["method"]][r.get("seed", 1)] = r
    if not recs:
        return
    put("LitCount", len(lit_bks.LIT), "{:d}")
    MS = ("psb", "lkh_tuned", "lkh_warm") + R.SOTA
    rows = []
    for (nm, k), ms in recs.items():
        if "psb" not in ms or 1 not in ms["psb"]:
            continue
        ref = lit_bks.ref(nm, k)
        p1 = ms["psb"][1]
        row = dict(name=nm, k=k, n=p1["n"], ref=ref)
        for m in MS:
            runs = [x for x in ms.get(m, {}).values() if not x.get("skipped")]
            if not runs:
                continue
            fins, at = [], {b: [] for b in (1.0, 10.0, 60.0)}
            for x in runs:
                fins.append(R.final_of(x) / ref - 1)
                for b in at:
                    at[b].append(R.mk_at(x["trace"], b) / ref - 1)
            row[m] = {b: float(np.median(v)) for b, v in at.items()}
            row[m]["final"] = float(np.median(fins))
            row[m]["seeds"] = len(runs)
        row["psb_t0"] = p1["psb"]["t_total"]
        row["psb_g0"] = p1["psb"]["mk"] / ref - 1
        rows.append(row)
    json.dump(rows, open(os.path.join(HERE, "out", "report", "lit_rows.json"), "w"), default=lambda x: None)
    big = [r for r in rows if r["n"] >= 1300]
    small = [r for r in rows if r["n"] < 1300]

    def medp(R_, f):
        v = [f(r) for r in R_]
        return 100 * float(np.median(v)) if v else math.inf
    for tag, R_ in (("LitL", big), ("LitS", small)):
        if not R_:
            continue
        put(f"{tag}Count", len(R_), "{:d}")
        put(f"{tag}PsbOne", medp(R_, lambda r: r["psb"][1.0]))
        put(f"{tag}PsbTen", medp(R_, lambda r: r["psb"][10.0]))
        put(f"{tag}PsbSixty", medp(R_, lambda r: r["psb"][60.0]))
        put(f"{tag}PsbFinal", medp(R_, lambda r: r["psb"]["final"]))
        put(f"{tag}PsbZero", medp(R_, lambda r: r["psb_g0"]))
        for m, mt in (("lkh_tuned", "Tuned"), ("lkh_warm", "Warm"), ("hsnr", "Hsnr"), ("ma", "Ma"),
                      ("mils", "Mils"), ("hga", "Hga")):
            Rm = [r for r in R_ if m in r]
            if Rm:
                put(f"{tag}{mt}Final", medp(Rm, lambda r: r[m]["final"]))
                put(f"{tag}{mt}Ten", medp(Rm, lambda r: r[m][10.0]))
                put(f"{tag}{mt}One", medp(Rm, lambda r: r[m][1.0]))
        put(f"{tag}PsbWithinTwo", 100 * float(np.mean([r["psb"]["final"] <= 0.02 for r in R_])), "{:.0f}")
        put(f"{tag}PsbBetter", sum(1 for r in R_ if r["psb"]["final"] < -1e-6), "{:d}")
    # few (k<=5) vs many (k>=10) salesmen: PSB final vs the others
    for tag, R_ in (("LitFewK", [r for r in rows if r["k"] <= 5]), ("LitManyK", [r for r in rows if r["k"] >= 10])):
        put(f"{tag}Count", len(R_), "{:d}")
        for m, mt in (("lkh_tuned", "Tuned"), ("lkh_warm", "Warm"), ("hsnr", "Hsnr"), ("ma", "Ma"),
                      ("mils", "Mils"), ("hga", "Hga")):
            Rm = [r for r in R_ if m in r]
            if not Rm:
                continue
            put(f"{tag}PsbBeats{mt}", sum(1 for r in Rm if r["psb"]["final"] < r[m]["final"] - 1e-6), "{:d}")
            put(f"{tag}PsbTies{mt}", sum(1 for r in Rm if abs(r["psb"]["final"] - r[m]["final"]) <= 1e-6), "{:d}")
            put(f"{tag}{mt}Final", medp(Rm, lambda r: r[m]["final"]))
            put(f"{tag}{mt}Count", len(Rm), "{:d}")
        put(f"{tag}PsbFinal", medp(R_, lambda r: r["psb"]["final"]))
    # longtable body: the environment, caption and label live in the .tex file
    cols = [m for m in MS if any(m in r for r in rows)]
    labs = {"psb": "PSB", "lkh_tuned": "tuned", "lkh_warm": "warm", "hsnr": "HSNR", "ma": "MA",
            "mils": "MILS", "hga": "HGA"}
    nl = sum(1 for c in cols if c.startswith("lkh"))
    ns = sum(1 for c in cols if c in R.SOTA)
    grp = r" & & & & & " + (r"\multicolumn{%d}{c}{LKH-3}" % nl if nl else "") + \
          (" & " if nl and ns else "") + (r"\multicolumn{%d}{c}{other codes}" % ns if ns else "") + r" \\"
    cm = r"\cmidrule(lr){7-%d}" % (6 + nl) if nl else ""
    if ns:
        cm += r"\cmidrule(lr){%d-%d}" % (7 + nl, 6 + nl + ns)
    head = [r"\toprule", grp, cm,
            r"instance & $n$ & $k$ & ref. & one-shot & " + " & ".join(labs[c] for c in cols) + r" \\",
            r"\midrule"]
    lines = head + [r"\endfirsthead"] + head + [r"\endhead", r"\bottomrule", r"\endlastfoot"]

    def g(v):
        if not math.isfinite(v):
            return r"\ensuremath{\infty}"
        t = f"{100 * v:+.1f}"
        return "0.0" if t in ("+0.0", "-0.0") else t
    for r in sorted(rows, key=lambda r: (r["n"], r["k"])):
        fins = [r[c]["final"] if c in r else math.inf for c in cols]
        lo = min(fins)
        cells = [("--" if c not in r else (r"\textbf{" + g(v) + "}" if v <= lo + 1e-6 else g(v))) for c, v in zip(cols, fins)]
        lines.append(f"{r['name']} & {r['n']} & {r['k']} & {r['ref']:.1f} & {g(r['psb_g0'])} & " + " & ".join(cells) + r" \\")
    open(os.path.join(MK, "tab_lit.tex"), "w").write("\n".join(lines) + "\n")
    put("LitColumns", len(cols) + 5, "{:d}")

I_N = {}


def sim():
    fs = glob.glob(os.path.join(HERE, "out", "sim", "*.json"))
    if not fs:
        return
    D = defaultdict(dict)
    for f in fs:
        r = json.load(open(f))
        D[(r["tag"], r["rseed"], r["k"])][r["planner"]] = r
    planners = ("psb", "psb_lns1", "psb_lns10", "lkh_warm10", "lkh_tuned10", "lkh_tuned60")
    lab = {"psb": "PSB", "psb_lns1": "PSB+LNS 1\\,s", "psb_lns10": "PSB+LNS 10\\,s",
           "lkh_tuned10": "LKH-3 10\\,s", "lkh_tuned60": "LKH-3 60\\,s", "lkh_warm10": "LKH-3 warm 10\\,s"}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"planner & mission [s] & planning [s] & flight [s] & rounds & $\Delta$ [\%] & no plan \\",
             r"\midrule"]
    out = {}
    min_rel = {}
    for tag in ("small", "medium", "large"):
        keys = [k for k in D if k[0] == tag and "psb_lns1" in D[k]]
        min_rel[tag] = min(100 * np.mean([D[k][p]["summary"]["mission_s"] / D[k]["psb_lns1"]["summary"]["mission_s"] - 1
                                          for k in keys if p in D[k]]) for p in planners) if keys else 0.0
    for tag, tname in (("small", "Small"), ("medium", "Medium"), ("large", "Large")):
        # tum (tohum, k) senaryolari; LKH-3'un butcede plan yazamadigi kosular
        # ortalamadan cikarilir ve "no plan" sutununda sayilir (Delta: eslesik)
        keys = [k for k in D if k[0] == tag and "psb_lns1" in D[k]]
        if not keys:
            continue
        n = D[keys[0]]["psb_lns1"]["n"]
        T = {"small": "Small", "medium": "Medium", "large": "Large"}[tag]
        b = [D[k]["psb_lns1"] for k in keys]
        put(f"Sim{T}N", n, "{:d}")
        put(f"Sim{T}Targets", float(np.mean([r["rounds"][0]["n_targets"] for r in b])), "{:.0f}")
        put(f"Sim{T}Ripe", float(np.mean([r["summary"]["ripe_total"] for r in b])), "{:.0f}")
        put(f"Sim{T}Wasted", float(np.mean([r["summary"]["wasted"] for r in b])), "{:.0f}")
        put(f"Sim{T}Recall", 100 * float(min(D[k][p]["summary"]["recall"] for k in keys for p in planners if p in D[k])), "{:.0f}")
        lines.append(r"\multicolumn{7}{l}{\emph{" + f"{tname} orchard, $n={n}$" + r"}} \\")
        for i, p in enumerate(planners):
            ok = [k for k in keys if p in D[k]]
            ms = [D[k][p]["summary"]["mission_s"] for k in ok]
            pl = [D[k][p]["summary"]["plan_s"] for k in ok]
            fl = [D[k][p]["summary"]["flight_s"] for k in ok]
            rd = [D[k][p]["summary"]["rounds"] for k in ok]
            rel = [D[k][p]["summary"]["mission_s"] / D[k]["psb_lns1"]["summary"]["mission_s"] - 1 for k in ok]
            out[(tag, p)] = dict(mission=float(np.mean(ms)), plan=float(np.mean(pl)), flight=float(np.mean(fl)),
                                 rel=float(np.mean(rel)), runs=len(keys), fail=len(keys) - len(ok))
            nf = f"{len(keys) - len(ok)}/{len(keys)}" if len(ok) < len(keys) else "--"
            d = 100 * np.mean(rel)
            dtxt = r"\textbf{" + f"{d:+.1f}" + "}" if d <= min_rel[tag] + 1e-9 else f"{d:+.1f}"
            lines.append(f"{lab[p]} & {np.mean(ms):.1f} & {np.mean(pl):.1f} & {np.mean(fl):.1f} & "
                         f"{np.mean(rd):.1f} & {dtxt} & {nf} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    if all(v["fail"] == 0 for v in out.values()):          # hic basarisiz kosu yoksa sutunu kaldir
        lines = [ln.replace(r"{lrrrrrr}", r"{lrrrrr}").replace(r"\multicolumn{7}", r"\multicolumn{6}")
                 .replace(r" & no plan \\", r" \\").replace(r" & -- \\", r" \\") for ln in lines]
    put("SimFailTotal", sum(v["fail"] for v in out.values()), "{:d}")
    open(os.path.join(MK, "tab_sim.tex"), "w").write("\n".join(lines) + "\n")
    for (tag, p), v in out.items():
        T = {"small": "Small", "medium": "Medium", "large": "Large"}[tag]
        P = {"psb": "Psb", "psb_lns1": "LnsOne", "psb_lns10": "LnsTen", "lkh_tuned10": "LkhTen",
             "lkh_tuned60": "LkhSixty", "lkh_warm10": "WarmTen"}[p]
        put(f"Sim{T}{P}Mission", v["mission"])
        put(f"Sim{T}{P}Plan", v["plan"])
        put(f"Sim{T}{P}Rel", 100 * v["rel"], "{:+.1f}")
        put(f"Sim{T}{P}RelAbs", abs(100 * v["rel"]), "{:.1f}")
        put(f"Sim{T}Runs", v["runs"], "{:d}")
        put(f"Sim{T}{P}Fail", v["fail"], "{:d}")


def ablation():
    f = os.path.join(HERE, "out", "extra", "ablation.json")
    if not os.path.exists(f):
        return
    R = json.load(open(f))
    V = [("full", "PSB (full)"), ("no_balance", "no Balance"), ("no_polish", "no Polish, no Balance"),
         ("equal_split", "equal-length split"), ("ge_nols", "GE tour, no 2-opt/Or-opt"),
         ("nn_tour", "nearest-neighbour tour"), ("sweep_tsp", "sweep + TSP"), ("kmeans_tsp", "$k$-means + TSP"),
         ("kmeans_bal", "$k$-means + TSP + Balance")]
    G = [("mtsplib", "mTSPLib"), ("orchard", "synth."), ("osm", "real"), ("vlsi", "VLSI"), ("lit", "lit.")]
    G = [g for g in G if any(r["group"] == g[0] for r in R)]
    lines = [r"\begin{tabular}{l" + "r" * len(G) + "r}", r"\toprule",
             "variant & " + " & ".join(f"{lab} ({sum(1 for r in R if r['group'] == g)})" for g, lab in G)
             + r" & time [ms] \\", r"\midrule"]
    for v, lab in V:
        cells = []
        for g, _ in G:
            gap = [100 * (r[v]["mk"] / r["full"]["mk"] - 1) for r in R if r["group"] == g]
            cells.append("--" if v == "full" else f"{np.median(gap):+.2f}")
        t = np.median([1000 * r[v]["t"] for r in R])
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {t:.0f} \\\\")
        if v != "full":
            allg = [100 * (r[v]["mk"] / r["full"]["mk"] - 1) for r in R]
            put("Abl" + macro(v) + "All", float(np.median(allg)), "{:+.2f}")
            for g, _ in G:
                gap = [100 * (r[v]["mk"] / r["full"]["mk"] - 1) for r in R if r["group"] == g]
                put("Abl" + macro(v) + macro(g), float(np.median(gap)), "{:+.2f}")
                put("Abl" + macro(v) + macro(g) + "Beats",
                    100 * float(np.mean([r[v]["mk"] < r["full"]["mk"] - 1e-9 for r in R if r["group"] == g])), "{:.0f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(MK, "tab_ablation.tex"), "w").write("\n".join(lines) + "\n")
    f2 = os.path.join(HERE, "out", "extra", "lns_start.json")
    if os.path.exists(f2):
        L = json.load(open(f2))
        D = {}
        for r in L:
            D.setdefault((r["g"], r["nm"], r["k"]), {})[r["start"]] = r
        for g in ("orchard", "vlsi"):
            ks = [k for k in D if k[0] == g]
            put("LnsStart" + macro(g), 100 * float(np.median([D[k]["sweep"]["mk"] / D[k]["psb"]["mk"] - 1 for k in ks])),
                "{:+.2f}")


def macro(s):
    return "".join(w.capitalize() for w in s.replace("-", "_").split("_"))


def scaling():
    f = os.path.join(HERE, "out", "extra", "scaling.json")
    if not os.path.exists(f):
        return
    R = json.load(open(f))
    one = {(r["name"], r["k"]): r["t_med"] for r in R if r["workers"] == 1}
    sp = {}
    for r in R:
        sp.setdefault(r["workers"], []).append(one[(r["name"], r["k"])] / r["t_med"])
    for w, v in sp.items():
        put(f"ScaleSpeedup{ {1: 'One', 2: 'Two', 4: 'Four', 8: 'Eight', 16: 'Sixteen'}[w] }", float(np.median(v)), "{:.2f}")
    big = [r for r in R if r["workers"] == 1 and r["n"] > 9000]
    put("ScaleMsLargest", 1000 * max(r["t_med"] for r in big), "{:.0f}")
    put("ScaleNLargest", max(r["n"] for r in big), "{:d}")
    ns = np.array([r["n"] for r in R if r["workers"] == 1], float)
    ts = np.array([r["t_med"] for r in R if r["workers"] == 1], float)
    put("ScaleExponent", float(np.polyfit(np.log(ns), np.log(ts), 1)[0]), "{:.2f}")


def sens():
    fs = glob.glob(os.path.join(HERE, "out", "sim_sens", "*.json"))
    if not fs:
        return
    D = defaultdict(dict)
    for f in fs:
        r = json.load(open(f))
        D[(r["setting"], r["rseed"])][r["planner"]] = r
    P = ("psb", "psb_lns1", "lkh_warm10", "lkh_tuned10", "lkh_tuned60")
    lab = {"psb": "PSB", "psb_lns1": "PSB+LNS 1\\,s", "lkh_warm10": "LKH-3 warm 10\\,s",
           "lkh_tuned10": "LKH-3 10\\,s", "lkh_tuned60": "LKH-3 60\\,s"}
    order = ["base", "s_h=3", "s_h=12", "speed=300", "speed=800", "tau=0.6", "tau=0.8", "rho=300", "rho=1000"]
    nice = {"base": "baseline", "s_h=3": "$s_h=3$\\,s", "s_h=12": "$s_h=12$\\,s", "speed=300": "$v=3$\\,m/s",
            "speed=800": "$v=8$\\,m/s", "tau=0.6": "$\\tau=0.6$", "tau=0.8": "$\\tau=0.8$",
            "rho=300": "$\\rho=3$\\,m", "rho=1000": "$\\rho=10$\\,m"}
    lines = [r"\begin{tabular}{lr" + "r" * (len(P) - 1) + "}", r"\toprule",
             r" & PSB+LNS 1\,s & \multicolumn{" + str(len(P) - 1) + r"}{c}{$\Delta$ mission time [\%]} \\",
             r"\cmidrule(lr){3-" + str(len(P) + 1) + "}",
             r"setting & mission [s] & " + " & ".join(lab[p] for p in P if p != "psb_lns1") + r" \\", r"\midrule"]
    worst = []
    for st in order:
        keys = [k for k in D if k[0] == st and "psb_lns1" in D[k]]
        if not keys:
            continue
        base = np.mean([D[k]["psb_lns1"]["summary"]["mission_s"] for k in keys])
        cells = []
        for p in P:
            if p == "psb_lns1":
                continue
            rel = [D[k][p]["summary"]["mission_s"] / D[k]["psb_lns1"]["summary"]["mission_s"] - 1 for k in keys if p in D[k]]
            cells.append(f"{100 * np.mean(rel):+.1f}")
            if p.startswith("lkh"):
                worst.append(100 * np.mean(rel))
        lines.append(f"{nice[st]} & {base:.1f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(MK, "tab_sens.tex"), "w").write("\n".join(lines) + "\n")
    put("SensLkhMinRel", float(min(worst)), "{:+.1f}")
    put("SensLkhMaxRel", float(max(worst)), "{:+.1f}")
    put("SensSettings", len(order), "{:d}")


def harvest():
    fs = glob.glob(os.path.join(HERE, "out", "harvest", "*.json"))
    if not fs:
        return
    D = defaultdict(dict)
    for f in fs:
        r = json.load(open(f))
        D[(r["tag"], r["passname"], r["seed"])][r["planner"]] = r
    P_ = ("psb", "psb_lns1", "ortools1", "ortools10", "greedy", "static", "rowblock")
    lab = {"psb": "PSB-MT/R", "psb_lns1": "PSB-MT/R + LNS 1\\,s", "ortools1": "OR-Tools 1\\,s",
           "ortools10": "OR-Tools 10\\,s", "greedy": "greedy", "static": "static (no replanning)",
           "rowblock": "row blocks (practice)"}
    tags = [("syn1ha", "synthetic 1 ha"), ("osm1ha", "real 1.0 ha"), ("osm2ha", "real 2.0 ha"), ("osm3ha", "real 3.2 ha")]
    passes = [("pass1", "first pick (40\\,\\% ripe)"), ("sparse", "sparse pick (15\\,\\% ripe)")]
    head = [r"\toprule",
            r"planner & mission [h] & $\Delta$ [\%] & picked [\%] & travel [\%] & planning [s] \\",
            r"\midrule"]
    lines = head + [r"\endfirsthead"] + head + [r"\endhead", r"\bottomrule", r"\endlastfoot"]
    agg = defaultdict(list)
    for ps, plab in passes:
        for tg, tlab in tags:
            keys = [k for k in D if k[0] == tg and k[1] == ps and "psb" in D[k]]
            if not keys:
                continue
            r0 = D[keys[0]]["psb"]
            lines.append(r"\multicolumn{6}{l}{\emph{" + f"{tlab}, {plab}, $k={r0['k']}$ platforms, "
                         f"{r0['n_trees']} trees" + r"}} \\")
            for pl in P_:
                ok = [k for k in keys if pl in D[k]]
                if not ok:
                    continue
                ms = [D[k][pl]["mission_s"] / 3600 for k in ok]
                rel = [100 * (D[k][pl]["mission_s"] / D[k]["psb"]["mission_s"] - 1) for k in ok]
                rec = [100 * D[k][pl]["recall"] for k in ok]
                trav = [100 * D[k][pl]["travel_s"] / (D[k][pl]["travel_s"] + D[k][pl]["service_s"]) for k in ok]
                pt = [D[k][pl]["plan_time"] for k in ok]
                agg[pl] += rel
                agg[pl + "_rec"] += [D[k][pl]["recall"] - D[k]["psb"]["recall"] for k in ok]
                lines.append(f"{lab[pl]} & {np.mean(ms):.2f} & {np.mean(rel):+.1f} & {np.mean(rec):.1f} & "
                             f"{np.mean(trav):.1f} & {np.mean(pt):.1f} \\\\")
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    open(os.path.join(MK, "tab_harvest.tex"), "w").write("\n".join(lines) + "\n")
    for pl in P_:
        if agg[pl]:
            put("Hv" + macro(pl) + "RelMean", float(np.mean(agg[pl])), "{:+.1f}")
            put("Hv" + macro(pl) + "RelMin", float(np.min(agg[pl])), "{:+.1f}")
            put("Hv" + macro(pl) + "RelMax", float(np.max(agg[pl])), "{:+.1f}")
            put("Hv" + macro(pl) + "RecallDiff", 100 * float(np.mean(agg[pl + "_rec"])), "{:+.1f}")
            put("Hv" + macro(pl) + "Runs", len(agg[pl]), "{:d}")
    # Wilcoxon: PSB vs others (paired over orchard x pass x seed)
    from scipy.stats import wilcoxon
    for pl in ("ortools1", "greedy", "static", "rowblock"):
        pairs = [(D[k]["psb"]["mission_s"], D[k][pl]["mission_s"]) for k in D if "psb" in D[k] and pl in D[k]]
        if len(pairs) >= 6:
            a, b = zip(*pairs)
            put("HvP" + macro(pl), float(wilcoxon(a, b).pvalue), "{:.1e}")
            put("HvWin" + macro(pl), 100 * float(np.mean([x < y for x, y in pairs])), "{:.0f}")
    # planlama gecikmesi
    lat = defaultdict(list)
    for k in D:
        if "psb" not in D[k]:
            continue
        for pl in D[k]:
            if pl.startswith("delay"):
                lat[(k[0], float(pl[5:]))].append(100 * (D[k][pl]["mission_s"] / D[k]["psb"]["mission_s"] - 1))
    if lat:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for tg, tlab in tags:
            xs = sorted(L for (t, L) in lat if t == tg)
            if not xs:
                continue
            ys = [np.mean(lat[(tg, L)]) for L in xs]
            ax.plot([0] + xs, [0] + ys, "o-", label=tlab)
        ax.set_xlabel("planning latency per replan [s]")
        ax.set_ylabel("mission time increase [%]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(MK, "figs", "fig_latency.pdf"))
        for (tg, L), v in lat.items():
            put(f"HvLat{macro(tg)}{int(L)}", float(np.mean(v)), "{:+.1f}")
    rr = [D[k]["psb"] for k in D if "psb" in D[k]]
    if not rr:
        return
    put("HvPsbPlanMs", 1000 * float(np.median([r["plan_time"] / max(r["plans"], 1) for r in rr])), "{:.0f}")
    put("HvTripsMed", float(np.median([r["trips"] for r in rr])), "{:.0f}")
    put("HvReplanGapMin", float(np.median([r["mission_s"] / max(r["plans"], 1) / 60 for r in rr])), "{:.1f}")
    put("HvPrecision", 100 * float(np.median([r["precision"] for r in rr])), "{:.1f}")
    put("HvRecall", 100 * float(np.median([r["recall"] for r in rr])), "{:.1f}")


def harvest_sens():
    fs = glob.glob(os.path.join(HERE, "out", "harvest_sens", "*.json"))
    if not fs:
        return
    D = defaultdict(dict)
    for f in fs:
        r = json.load(open(f))
        D[(r["setting"], r["seed"])][r["planner"]] = r
    P_ = ("ortools1", "greedy", "static", "rowblock")
    lab = {"ortools1": "OR-Tools 1\\,s", "greedy": "greedy", "static": "static", "rowblock": "row blocks"}
    sets = sorted({k[0] for k in D}, key=lambda x: (x != "base", x))
    lines = [r"\begin{tabular}{lr" + "r" * len(P_) + "}", r"\toprule",
             r" & PSB-MT/R & \multicolumn{" + str(len(P_)) + r"}{c}{$\Delta$ mission time [\%]} \\",
             r"\cmidrule(lr){3-" + str(len(P_) + 2) + "}",
             r"setting & mission [h] & " + " & ".join(lab[p] for p in P_) + r" \\", r"\midrule"]
    allrel = defaultdict(list)
    for stn in sets:
        keys = [k for k in D if k[0] == stn and "psb" in D[k]]
        base = np.mean([D[k]["psb"]["mission_s"] / 3600 for k in keys])
        cells = []
        for p in P_:
            rel = [100 * (D[k][p]["mission_s"] / D[k]["psb"]["mission_s"] - 1) for k in keys if p in D[k]]
            cells.append(f"{np.mean(rel):+.1f}")
            allrel[p].append(np.mean(rel))
        nm = stn.replace("_", "\\_")
        lines.append(f"{nm} & {base:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(MK, "tab_harvest_sens.tex"), "w").write("\n".join(lines) + "\n")
    for p in P_:
        put("HsMin" + macro(p), float(np.min(allrel[p])), "{:+.1f}")
        put("HsMax" + macro(p), float(np.max(allrel[p])), "{:+.1f}")
    put("HsSettings", len(sets), "{:d}")


def hgrid():
    fs = glob.glob(os.path.join(HERE, "out", "hgrid", "*.json"))
    if not fs:
        return
    from scipy.stats import wilcoxon
    D = defaultdict(dict)
    for f in fs:
        r = json.load(open(f))
        D[(r["tag"], r["passname"], r["seed"])][r["planner"]] = r
    P_ = ("kdsteal", "kd", "psb", "ortools1", "greedy", "rowblock")
    lab = {"kdsteal": "k-d regions + end take-over (ours)", "kd": "k-d regions, no take-over",
           "psb": "global PSB replanning", "ortools1": "OR-Tools 1\\,s replanning", "greedy": "greedy dispatch",
           "rowblock": "row bands (current practice)"}
    tags = [("rect1ha", "notched 1.3 ha block"), ("osm1ha", "real 1.0 ha"), ("osm2ha", "real 2.0 ha"),
            ("osm3ha", "real 3.2 ha"), ("osm5ha", "real 5.4 ha")]
    passes = [("pass1", "first pick (40\\,\\% ripe)"), ("sparse", "sparse pick (15\\,\\% ripe)")]
    lines = [r"\begin{tabular}{l" + "r" * len(P_) + "}", r"\toprule",
             r"orchard & " + " & ".join({"kdsteal": "k-d+TO", "kd": "k-d", "psb": "PSB", "ortools1": "OR-Tools",
                                         "greedy": "greedy", "rowblock": "row bands"}[q] for q in P_) + r" \\",
             r"\midrule"]
    agg = defaultdict(list)
    for ps, plab in passes:
        lines.append(r"\multicolumn{" + str(len(P_) + 1) + r"}{l}{\emph{" + plab + r": mission time [h], mean of 5 seeds}} \\")
        for tg, tlab in tags:
            keys = [k for k in D if k[0] == tg and k[1] == ps and "psb" in D[k]]
            if not keys:
                continue
            vals = {q: np.mean([D[k][q]["mission_s"] / 3600 for k in keys if q in D[k]]) for q in P_
                    if any(q in D[k] for k in keys)}
            best = min(vals.values())
            cells = [(r"\textbf{%.2f}" % vals[q] if abs(vals[q] - best) < 1e-9 else "%.2f" % vals[q]) if q in vals else "--"
                     for q in P_]
            lines.append(f"{tlab} ($k={D[keys[0]]['kdsteal']['k']}$) & " + " & ".join(cells) + r" \\")
            for k in keys:
                for q in P_:
                    if q in D[k]:
                        agg[q].append(100 * (D[k][q]["mission_s"] / D[k]["psb"]["mission_s"] - 1))
                        agg[q + "_rec"].append(100 * (D[k][q]["recall"] - D[k]["psb"]["recall"]))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(MK, "tab_hgrid.tex"), "w").write("\n".join(lines) + "\n")
    for q in P_:
        if agg[q]:
            put("Hg" + macro(q) + "Rel", float(np.mean(agg[q])), "{:+.1f}")
            put("Hg" + macro(q) + "RelMin", float(np.min(agg[q])), "{:+.1f}")
            put("Hg" + macro(q) + "RelMax", float(np.max(agg[q])), "{:+.1f}")
            put("Hg" + macro(q) + "Rec", float(np.mean(agg[q + "_rec"])), "{:+.1f}")
        if q != "psb":
            pairs = [(D[k]["psb"]["mission_s"], D[k][q]["mission_s"]) for k in D if "psb" in D[k] and q in D[k]]
            if len(pairs) >= 6:
                a, b = zip(*pairs)
                put("HgP" + macro(q), float(wilcoxon(a, b).pvalue), "{:.1e}")
                put("HgWin" + macro(q), 100 * float(np.mean([x < y - 1e-9 for x, y in pairs])), "{:.0f}")
                put("HgN", len(pairs), "{:d}")
    for ps, pt in (("pass1", "First"), ("sparse", "Sparse")):
        for q in ("kdsteal", "kd", "ortools1", "greedy", "rowblock"):
            pr = [(D[k]["psb"], D[k][q]) for k in D if k[1] == ps and "psb" in D[k] and q in D[k]]
            if pr:
                put(f"Hg{pt}{macro(q)}Rel", 100 * float(np.mean([b["mission_s"] / a["mission_s"] - 1 for a, b in pr])), "{:+.1f}")
                put(f"Hg{pt}{macro(q)}Win", 100 * float(np.mean([a["mission_s"] < b["mission_s"] for a, b in pr])), "{:.0f}")
                put(f"Hg{pt}{macro(q)}Fruit", 100 * float(np.mean([a["picked"] / b["picked"] - 1 for a, b in pr])), "{:+.1f}")
    pr = [(D[k]["kdsteal"]["mission_s"], D[k]["kd"]["mission_s"]) for k in D if "kdsteal" in D[k] and "kd" in D[k]]
    if len(pr) >= 6:
        a, b = zip(*pr)
        put("HgTakeoverGain", 100 * float(np.mean([1 - x / y for x, y in pr])), "{:.1f}")
        put("HgTakeoverWin", 100 * float(np.mean([x < y for x, y in pr])), "{:.0f}")
        put("HgTakeoverP", float(wilcoxon(a, b).pvalue), "{:.0e}")
    rrk = [D[k]["kdsteal"] for k in D if "kdsteal" in D[k]]
    put("HgStealsMed", float(np.median([r["steals"] for r in rrk])), "{:.0f}")
    put("HgKdPlanMs", 1000 * float(np.median([r["plan_time"] / max(r["plans"], 1) for r in rrk])), "{:.2f}")
    # cerceve etkisi
    for var, tag in (("kdsteal@oracle", "Oracle"), ("kdsteal@axis0", "AxisZero"), ("psb@axis0", "PsbAxisZero")):
        rel = []
        for k in D:
            if var in D[k]:
                base = D[k]["psb" if var.startswith("psb") else "kdsteal"]["mission_s"]
                rel.append(100 * (D[k][var]["mission_s"] / base - 1))
        if rel:
            put("HgFrame" + tag, float(np.mean(rel)), "{:+.1f}")
            put("HgFrame" + tag + "Abs", abs(float(np.mean(rel))), "{:.1f}")
            put("HgFrame" + tag + "Max", float(np.max(rel)), "{:+.1f}")
    rr = [D[k]["psb"] for k in D if "psb" in D[k]]
    if not rr:
        return
    put("HgThetaErrMax", float(max(r["theta_err"] for r in rr)), "{:.2f}")
    put("HgThetaErrMed", float(np.median([r["theta_err"] for r in rr])), "{:.2f}")
    put("HgPlanMs", 1000 * float(np.median([r["plan_time"] / max(r["plans"], 1) for r in rr])), "{:.2f}")
    pp = [D[k]["psb"] for k in D if "psb" in D[k]]
    put("HgPsbPlanMs", 1000 * float(np.median([r["plan_time"] / max(r["plans"], 1) for r in pp])), "{:.0f}")
    put("HgRecall", 100 * float(np.median([r["recall"] for r in rr])), "{:.1f}")
    put("HgPrecision", 100 * float(np.median([r["precision"] for r in rr])), "{:.1f}")
    put("HgRuns", len(rr), "{:d}")
    put("HgTreesMin", min(r["n_trees"] for r in rr), "{:d}")
    put("HgTreesMax", max(r["n_trees"] for r in rr), "{:d}")
    put("HgKMin", min(r["k"] for r in rr), "{:d}")
    put("HgKMax", max(r["k"] for r in rr), "{:d}")
    # duyarlilik
    fs = glob.glob(os.path.join(HERE, "out", "hgrid_sens", "*.json"))
    if fs:
        E = defaultdict(dict)
        for f in fs:
            r = json.load(open(f))
            E[(r["setting"], r["seed"])][r["planner"]] = r
        Q_ = ("kdsteal", "kd", "greedy", "rowblock")
        sets = sorted({k[0] for k in E}, key=lambda x: (x != "base", x))
        lines = [r"\begin{tabular}{lr" + "r" * len(Q_) + "}", r"\toprule",
                 r" & PSB & \multicolumn{" + str(len(Q_)) + r"}{c}{$\Delta$ mission time vs.\ PSB [\%]} \\",
                 r"\cmidrule(lr){3-" + str(len(Q_) + 2) + "}",
                 r"setting & mission [h] & " + " & ".join({"kd": "k-d", "kdsteal": "k-d+TO", "greedy": "greedy",
                                                          "rowblock": "row bands"}[q] for q in Q_) + r" \\",
                 r"\midrule"]
        rng_ = defaultdict(list)
        for stn in sets:
            keys = [k for k in E if k[0] == stn and "psb" in E[k]]
            base = np.mean([E[k]["psb"]["mission_s"] / 3600 for k in keys])
            cells = []
            for q in Q_:
                rel = [100 * (E[k][q]["mission_s"] / E[k]["psb"]["mission_s"] - 1) for k in keys if q in E[k]]
                cells.append(f"{np.mean(rel):+.1f}")
                rng_[q].append(np.mean(rel))
            lines.append(stn.replace("_", "\\_") + f" & {base:.2f} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(MK, "tab_hgrid_sens.tex"), "w").write("\n".join(lines) + "\n")
        for q in Q_:
            put("HgsMin" + macro(q), float(np.min(rng_[q])), "{:+.1f}")
            put("HgsMax" + macro(q), float(np.max(rng_[q])), "{:+.1f}")


def sota():
    S = json.load(open(os.path.join(HERE, "out", "report", "summary.json")))
    rows = json.load(open(os.path.join(HERE, "out", "report", "rows.json")))
    M_ = [("psb", "PSB+LNS (ours)"), ("mils", "MILS"), ("hga", "HGA"), ("ma", "MA"), ("hsnr", "HSNR"),
          ("lkh_tuned", "LKH-3 tuned")]
    cls = [("n_small", r"$n<1000$"), ("n_mid", r"$1000\le n<5000$"), ("n_large", r"$n\ge 5000$")]

    def f(v):
        return r"\ensuremath{\infty}" if v is None or not math.isfinite(v) else f"{v:.2f}"
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"method & first sol.\ [s] & 1\,s & 10\,s & 60\,s & final & PSB better [\%] & $p$ \\", r"\midrule"]
    for c, cl in cls:
        x = S[c]
        lines.append(r"\multicolumn{8}{l}{\emph{" + cl + f", {x['count']} instances" + r"}} \\")
        R_ = [r for r in rows if (c == "n_small" and r["n"] < 1000) or (c == "n_mid" and 1000 <= r["n"] < 5000)
              or (c == "n_large" and r["n"] >= 5000)]
        for m, lab in M_:
            if m != "psb" and m not in ("lkh_tuned",) and not x.get(m + "_count"):
                continue
            ft = float(np.median([r[m]["first_t"] for r in R_ if m in r]))
            g = [x.get(f"{m}_gap_{b}") for b in ("1", "10", "60")]
            fin = x.get(f"{m}_gap_final")
            if m == "psb":
                win, pv = "--", "--"
            elif m == "lkh_tuned":
                win, pv = f"{100 * x['win_vs_tuned_final']:.0f}", f"{x['p_vs_tuned']:.0e}"
            else:
                win, pv = f"{100 * x['psb_win_vs_' + m]:.0f}", (f"{x['p_vs_' + m]:.0e}" if x.get("p_vs_" + m) else "--")
            lines.append(f"{lab} & {ft:.2f} & " + " & ".join(f(v) for v in g) + f" & {f(fin)} & {win} & {pv} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(MK, "tab_sota.tex"), "w").write("\n".join(lines) + "\n")
    for c, tag in (("n_small", "Small"), ("n_mid", "Mid"), ("n_large", "Large"), ("all", "All")):
        x = S[c]
        for m, mt_ in (("mils", "Mils"), ("hga", "Hga"), ("ma", "Ma"), ("hsnr", "Hsnr")):
            if x.get(m + "_count"):
                put(f"Sota{mt_}Win{tag}", 100 * x["psb_win_vs_" + m], "{:.0f}")
                put(f"Sota{mt_}WinOne{tag}", 100 * x["psb_win_vs_" + m + "_1"], "{:.0f}")
                put(f"Sota{mt_}WinTen{tag}", 100 * x["psb_win_vs_" + m + "_10"], "{:.0f}")
                put(f"Sota{mt_}Final{tag}", x[m + "_gap_final"], "{:.2f}")
                put(f"Sota{mt_}Ten{tag}", x[m + "_gap_10"], "{:.1f}")
                put(f"Sota{mt_}First{tag}", x[m + "_first_t_med"], "{:.1f}")
                if x.get("p_vs_" + m):
                    put(f"Sota{mt_}P{tag}", x["p_vs_" + m], "{:.1e}")
        put(f"PsbSeedSd{tag}", x["psb_seed_sd_med"], "{:.2f}")
        put(f"PsbTen{tag}", x["psb_gap_10"], "{:.2f}")
        put(f"PsbOne{tag}", x["psb_gap_1"], "{:.2f}")
        put(f"PsbFin{tag}", x["psb_gap_final"], "{:.2f}")
        put(f"PvsTuned{tag}", x["p_vs_tuned"], "{:.0e}")
        put(f"PvsWarm{tag}", x["p_vs_warm"], "{:.0e}")
    put("SeedsMain", S["all"]["seeds_max"], "{:d}")
    put("WinFinalAll", 100 * S["all"]["win_vs_tuned_final"], "{:.0f}")
    put("InstAll", S["all"]["count"], "{:d}")
    put("OsmCount", S["osm"]["count"], "{:d}")
    put("WinFinalOsm", 100 * S["osm"]["win_vs_tuned_final"], "{:.0f}")


def prag_tr_quality():
    """Mevcut hasat kayitlarindan bahce-gecis kumelemeli kalite ozeti; yeni kosu yok."""
    from scipy.stats import t as student_t

    out = os.path.join(P2, "generated")
    methods = (("rowblock", "sıra bantları"), ("greedy", "açgözlü"), ("ortools1", "OR-Tools 1\\,s"))
    macros = {}
    rows = json.load(open(os.path.join(HERE, "out", "report", "rows.json")))
    for class_name, pred in (("Small", lambda n: n < 1000),
                             ("Mid", lambda n: 1000 <= n < 5000),
                             ("Large", lambda n: n >= 5000)):
        selected = [r for r in rows if pred(r["n"])]
        for method, suffix in (("hga", "Hga"), ("ma", "Ma"), ("hsnr", "Hsnr"), ("mils", "Mils")):
            pairs = [r for r in selected if method in r and math.isfinite(r[method]["final_mean"])
                     and math.isfinite(r["psb"]["final_mean"])]
            macros[f"Sota{suffix}WilcoxN{class_name}"] = str(len(pairs))
    table = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"planlayıcı & $\Delta T$ [\%] & $\Delta$ olgun [yp] & $\Delta$ kesinlik [yp] & $\Delta$ olgun/saat [\%] \\",
             r"\midrule"]

    def fmt(x, digits=1):
        return f"{x:+.{digits}f}"

    def summary(values, digits=1):
        values = np.asarray(values, float)
        g = len(values)
        mean = float(np.mean(values))
        half = float(student_t.ppf(0.975, g - 1) * np.std(values, ddof=1) / np.sqrt(g))
        return mean, half, f"{fmt(mean, digits)} [{fmt(mean - half, digits)},{fmt(mean + half, digits)}]"

    for source, prefix, heading in (("hgrid", "HqGrid", "Kare dikimli bahçeler (10 bahçe--geçiş kümesi)"),
                                    ("harvest", "HqRows", "Tall Spindle bahçeler (8 bahçe--geçiş kümesi)")):
        D = defaultdict(dict)
        for f in glob.glob(os.path.join(HERE, "out", source, "*.json")):
            r = json.load(open(f))
            D[(r["tag"], r["passname"], r["seed"])][r["planner"]] = r
        table.append(r"\multicolumn{5}{l}{\emph{" + heading + r"}} \\")
        for method, label in methods:
            clusters = defaultdict(list)
            for (tag, passname, seed), runs in D.items():
                if "psb" not in runs or method not in runs:
                    continue
                a, b = runs["psb"], runs[method]
                assert a["ripe_total"] == b["ripe_total"]
                time_rel = 100 * (a["mission_s"] / b["mission_s"] - 1)
                ripe_pp = 100 * (a["recall"] - b["recall"])
                precision_pp = 100 * (a["precision"] - b["precision"])
                ha = a["picked"] / a["mission_s"]
                hb = b["picked"] / b["mission_s"]
                throughput_rel = 100 * (ha / hb - 1) if hb else math.nan
                clusters[(tag, passname)].append((time_rel, ripe_pp, precision_pp, throughput_rel))
            values = np.array([np.mean(v, axis=0) for _, v in sorted(clusters.items())])
            results = [summary(values[:, j], digits) for j, digits in enumerate((1, 1, 2, 1))]
            for suffix, result in zip(("Time", "Ripe", "Precision", "Throughput"), results):
                mean, half, cell = result
                digits = 2 if suffix == "Precision" else 1
                macros[f"{prefix}{macro(method)}{suffix}"] = cell
                macros[f"{prefix}{macro(method)}{suffix}Mean"] = fmt(mean, digits)
                macros[f"{prefix}{macro(method)}{suffix}Half"] = fmt(half, digits)
            table.append(label + " & " + " & ".join("$" + r[2] + "$" for r in results) + r" \\")
        table.append(r"\midrule")
    table[-1] = r"\bottomrule"
    table.append(r"\end{tabular}")
    with open(os.path.join(out, "tab_harvest_quality.tex"), "w") as f:
        f.write("\n".join(table) + "\n")
    with open(os.path.join(out, "numbers_quality.tex"), "w") as f:
        f.write("% generated by experiments/make_numbers.py --prag-tr-quality; do not edit by hand\n")
        for key in sorted(macros):
            f.write(f"\\newcommand{{\\{_tex_name(key)}}}{{{macros[key]}}}\n")
    print(len(macros), "hasat kalite makrosu")


def _tex_name(k):
    """LaTeX makro adi rakam iceremez: rakamlari kelimeye cevir."""
    w = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
    return "".join(w[int(ch)] if ch.isdigit() else ch for ch in k)


def write():
    global M
    M = {_tex_name(k): v for k, v in M.items()}
    with open(os.path.join(MK, "numbers.tex"), "w") as f:
        f.write("% generated by experiments/make_numbers.py; do not edit by hand\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    print(len(M), "makro")


if __name__ == "__main__":
    if "--prag-tr-quality" in sys.argv:
        prag_tr_quality()
    else:
        main_bench()
        lit()
        ablation()
        scaling()
        harvest()
        harvest_sens()
        hgrid()
        sota()
        write()
