# Literature-based parameters for a selective apple-harvest drone-fleet simulation

Compiled 2026-10-16. BibTeX keys refer to `harvest_refs.bib` in this folder.

## How the sources were checked

- Every DOI was resolved through the Crossref API. Authors, title, journal, volume, issue and pages in the `.bib` file come from that record, checked by hand.
- Each number was checked against the abstract (Crossref, OpenAlex, Semantic Scholar, ISHS or ASABE abstract pages) or against open full text (arXiv, SciELO, extension PDFs). Quotes are verbatim from those texts.
- **Status column:** `A` = the number appears in the peer-reviewed abstract. `F` = it appears in open full text or a preprint of the same paper. `G` = grey literature (extension bulletin, government magazine, manufacturer, trade press), not peer reviewed. `S` = secondary source quoting the paper; not checked against the original. `D` = derived by arithmetic from cited numbers (the formula is shown).
- **"NOT VERIFIED"** means the value was searched for and could not be confirmed. Do not cite it as a literature value.

---

## 1. Orchard geometry

| Parameter | Value | Unit | Context | Source | Status / quote |
|---|---|---|---|---|---|
| Density range, modern orchards | 1,000-6,000 | trees/ha | worldwide | `robinson_2011_advances` | F: "tree densities of modern apple orchards range from 1,000 to 6,000 trees/ha" |
| Typical high-density row spacing, tree height | 3.0-3.5 m rows; 2.7-3.2 m trees | m | Tree height / row spacing ratio 0.8-0.9 | `robinson_2011_advances` | F: "the ratio of tree height to row spacing should be 0.8-0.9"; "typical row spacings used in high density orchards of 3.0-3.5m should have mature tree heights of 2.7-3.2m" |
| Pedestrian (ground-picked) orchard | trees <2.5 m; alleys ≤2.8 m | m | — | `robinson_2011_advances` | F: "short stature trees (<2.5m tall) which can be picked from the ground they should reduce tractor alleys to 2.8m or less" |
| Tall Spindle density | 2,500-3,300 | trees/ha | NY optimum economic density | `robinson_2011_tallspindle`; `robinson_2014_yield` | A: "optimum economic planting density (2,500-3,300 trees/ha)" |
| Tall Spindle height | 3.3 m (limited annually) | m | mature tree | `robinson_2014_yield` | A: "tree height is limited each year to 3.3 m" |
| Height constraint | tree height ≤90 % of row spacing | — | lower canopy quality | `robinson_2011_tallspindle` | A: "If tree height exceeded 90% of the between row spacing, fruit quality in the lower part of the canopy was reduced" |
| System spacing table (Cornell) | Super Spindle 0.6×3.0 m (5,382/ha); Tall Spindle 0.9×3.3 m (3,262/ha); Slender Vertical Axis 1.2×3.6 m (2,244/ha); Vertical Axis 1.5×4.25 m (1,538/ha); Slender Pyramid 2.4×4.9 m (840/ha) | m (in-row × between-row) | NY trials, M.9/M.26 | `hoying_nd_tallspindle` | G (extension slides). Same slides: Tall Spindle "0.90-1.20 m in-row spacing", "3.0-3.5 m between rows", "3 m tall" |
| Peer-reviewed densities of the same systems | SS 5,382; TS 3,262; Triple Axis 2,243; VA 1,656 | trees/ha | 'Delicious', NY, 2007-2017 | `reig_2020_longterm` | A (the in-row × row spacings are not in the abstract; the Cornell table gives them) |
| Density trial range | 598-5,382 | trees/ha | 8 densities, 4 cultivars | `robinson_2007_effects` | A |
| US adoption | 400 → 1,500-3,000; economic optimum ~2,500 (or 5,000-6,000 with cheap trees) | trees/ha | USA | `robinson_2008_evolution` | A: "most growers have increased their planting density from 400 trees/ha to between 1,500 and 3,000 trees/ha" |
| Super spindle / Europe | Bodensee region: 2,600 → 3,600 (1990-1998); optimum ~4,000; rows <3 m "too narrow for high picking outputs" | trees/ha; m | Germany | `weber_2001_optimizing` | A |
| 2D planar cordon (fruiting wall) | 3 m between trees in-row; 1.5 or 2.0 m between rows (2,222 or 1,667 trees/ha); 3-3.5 m tall; <0.5 m wide; 5 uprights 30 cm apart per cordon arm | m; trees/ha | New Zealand prototype | `tustin_2018_planar`; `vanhooijdonk_2023_pomological` | A: "narrow planar trees, 3 m long, less than 0.5 m wide, and with an anticipated height of 3-3.5 m"; NZ commercial orchards have "wide inter-rows (3+ m)" |
| Semi-intensive (vertical axis, cider) | 1.8 m in-row × 3.7 m rows = 1,495 trees/ha; 3.0 m tall vertical axis | m; trees/ha | 'G.935', WA, USA | `brawner_2025_overtherow`; `brawner_2026_wsu` | A (1.8 m, 1,495 trees/ha); G (3.7 m rows, 3.0 m height in WSU summary) |
| Traditional low density (cider, Europe) | ~280 trees/acre (~690 trees/ha) | trees/acre | free-standing cider orchards | `brawner_2026_wsu` | G. Conversion to trees/ha is D (×2.471) |

Implication for drones: 2D planar canopies (<0.5 m thick, 1.5-2 m alleys) leave too little room for ground platforms but expose most of the fruit on both faces. This is a strong argument for aerial pickers. That argument is an inference from the sources, not a sourced claim.

## 2. Crop load

| Parameter | Value | Unit | Context | Source | Status / quote |
|---|---|---|---|---|---|
| Target fruit number | 130 fruit/tree → 75 t/ha | fruit/tree; t/ha | 'Gala', NY, target before experiment | `francescatto_2020_precision` | A: "initial desired fruit number for these trees was determined at 130 fruit tree-1 to achieve 100 count fruit size and 75 t ha-1" |
| Crop-value optimum | ~338 fruit/tree at ~150 g | fruit/tree; g | 'Gala', 4 years | `francescatto_2020_precision` | A: "maximum crop value was achieved when fruit size was about 150 g and fruit number tree-1 was 338 fruits tree-1" |
| Mean fruit mass at optimum value | 161-169 g (113-100 count) | g | 'Gala'/M.9 | `robinson_2008_pruning` | A |
| Early yield, Tall Spindle | 15-20 t/ha in year 2; >150 t/ha cumulative over first 5 years | t/ha | NY | `robinson_2011_tallspindle`; `robinson_2014_yield` | A |
| Yield, highest-density system | 50 t/ha in year 4 (lowest density never >25 t/ha) | t/ha | NY | `robinson_2007_effects` | A |
| Yield ceiling, conventional high density | 100-120 t/ha (≈60 % light interception) | t/ha | NZ analysis | `tustin_2018_planar` | A |
| Commercial reference vs planar | NZ upper-quartile 'Royal Gala' ~75 t/ha; planar cordon 173 (1.5 m rows) and 140 t/ha (2 m rows) at year 7 | t/ha | NZ | `vanhooijdonk_2023_pomological` | A |
| Fruit per ha | 130 fruit × 3,262 trees ≈ 4.2×10⁵; 75 t/ha ÷ 0.165 kg ≈ 4.5×10⁵ | fruit/ha | — | — | D |

## 3. Ripening heterogeneity and selective harvest

| Parameter | Value | Unit | Context | Source | Status / quote |
|---|---|---|---|---|---|
| Passes by cultivar | Red Delicious: 1 pass; Honeycrisp: up to 5 passes | passes | Washington | `simnitt_2023_advancements` | G: "Red Delicious apples ripen uniformly and are usually harvested with a single pass"; "Honeycrisp apples do not ripen all at one time, so they require up to five passes" |
| Spot picking | 3-4 picks for cultivars that mature over a couple of weeks | picks | extension guidance | `roper_2001_when` | G: "Fruit on these trees needs to be 'spot picked' three or four times" |
| 'Royal Gala' picking regime | 2 vs 4 selective picks compared; 4 picks gave less advanced, less variable maturity | picks | NZ export | `brookfield_1993_fruit` | A |
| 'Royal Gala' harvest window and criterion | Multiple harvests over ~3 weeks; fruit selected by background colour, not size; larger fruit picked first (odds ×2 per +20 g) | weeks | NZ | `hall_2001_effect` | A: "picked in multiple harvests spread over about three weeks"; "fruit are selected for picking according to background colour, not size" |
| 'Honeycrisp' | Harvested 2-3 times; exterior-canopy fruit more mature than interior | — | ME, MN (USA), ON (Canada) | `moran_2022_honeycrisp` | F (abstract in PDF): "Harvest maturity was more advanced in exterior compared with interior fruit"; "Apples are typically spot picked according to color" |
| Spatial ripening gradient | Outside → inside and top → bottom; exposed fruit up to ~2 weeks earlier | days | extension | `roper_2001_when` | G: "Exposed fruit on the outside of the tree may ripen two weeks earlier than fruit on the canopy interior" |
| Sampling guidance | Periphery fruit "tend to ripen earlier" | — | extension | `farcuh_2021_how` | G |
| Counter-evidence | Canopy position did NOT affect firmness, starch index, Streif index or respiration; it did affect size, blush, sugars | — | Lithuania | `kviklys_2022_apple` | A |
| Counter-evidence | Canopy region had no significant effect; flowering-time variance share is cultivar-dependent: Cox 18.4-18.7 %, Gala 13.42-14.71 %, Golden Delicious 5.86-6.15 %, Braeburn 2.88-3.37 %, Fuji 0.52-0.61 % | % of variance | UK | `tang_2025_flowering` | A |
| Maturity indices | Background colour green → yellow (esp. Gala, Fuji); firmness ≥15 lb (long-term storage), 13-15 lb (short-term); starch 3-5 (storage) vs 6-7 (fresh market) on 1-8 scale; SSC ~12-14 %; IAD (DA meter) 0.60-0.70 for long storage | lb; index; % | extension | `farcuh_2021_how` | G |
| Colour ≠ maturity | "some highly colored apples change to red weeks before they are mature, so color is not always a reliable indicator"; ground colour "useless on Red Delicious and other solid red apples" | — | extension | `roper_2001_when` | G |
| Cultivar colour classes | Colour requirements differ for "solid red, striped red, green or yellow" cultivars; Honeycrisp retail minimum "50-60% red" | — | extension | `farcuh_2021_how` | G |
| Share of fruit per pass | — | % | — | — | **NOT VERIFIED**: no peer-reviewed per-pass harvest fractions were found |
| Granny Smith-specific maturity statement | — | — | — | — | **NOT VERIFIED** with a citable source. That green-ripe cultivars cannot be judged by colour is consistent with `roper_2001_when` and `farcuh_2021_how`, but neither names Granny Smith |

## 4. Robotic apple harvesting performance

| System | Pick success | Cycle time | Other | Source | Status / quote |
|---|---|---|---|---|---|
| WSU, 7-DOF, planar canopy (WA) | 84 % (127/150) | 6.0 s/fruit | localisation 1.5 s/fruit | `silwal_2017_design` | A: "successfully picked 127 of the 150 fruit attempted for an overall success rate of 84% with an average picking time of 6.0 s per fruit" |
| WSU, soft 3D-printed end-effector | 67 % detachment | 7.3 s/fruit (separation → bin) | commercial orchard, 2017 | `hohimer_2019_design` | A |
| MSU/USDA vacuum robot | 82.4 % (young, well-pruned); 65.2 % (older, dense) | ~6 s | 4-DOF, vacuum soft end-effector | `zhang_2024_automated` | A |
| MSU/USDA prototype | — | 3.6 s (successful picks); approach/return 0.75-1.4 s, detach ~1 s, release ~0.5 s | MSU field 2021 | `zhang_2022_algorithm` | F (arXiv 2203.00582): "the average cycle time is approximately 3.6 s, including software algorithm processing and hardware execution" |
| MSU dual-arm | 80.7 % and 79.7 % | 5.97 s (dual-arm); 8.29 s single arm | >100 trees, ~1,500 apples; fruiting wall + UFO | `zhu_2025_advancement` | A; F for the single-arm value |
| Bu et al., China | 80.17 % (anthropomorphic), 82.93 % (horizontal pull + bend) | 17.17±0.36 s and 12.53±0.53 s | — | `bu_2022_design` | Success rates: S (search snippet of the ScienceDirect abstract). Cycle times: S (vendor page `nokov_nd_bu`). Check the original before citing |
| Review | Robots "not been commercialized due to their high costs and low efficiency" | — | — | `zhang_2016_development` | A |
| Abundant Robotics (vacuum) | — | "capable of picking one apple per second" | NZ commercial run: "10 to 20 bins in 24 hours"; target "30 to 40 bins per day"; claimed reach 70-90 % of fruit | `wheat_2019_robotic` | G (company statements in trade press; no peer-reviewed data found) |
| FFRobotics | — | "six bins per hour" (claim by a third party); 8 arms per side, 3-prong hands | — | `wheat_2019_robotic` | G (unverified claim) |
| Tevel | see §6 | not published | — | — | **NOT VERIFIED**: no measured picking rate found |

## 5. Fruit detection and occlusion

| Metric | Value | Context | Source | Status |
|---|---|---|---|---|
| Recall / accuracy | 0.821 / 0.853; 28 ms inference | LedNet, apple orchards | `kang_2020_fast` | A |
| Recall, precision, mAP, F1 | 91.48 %, 83.83 %, 86.75 %, 87.49 %; 0.015 s/image | improved YOLOv5s; separates "graspable" (unoccluded/leaf-occluded) from "ungraspable" (branch/fruit-occluded) apples | `yan_2021_realtime` | A |
| Precision / recall | 0.93 / 0.97 (YOLOv8) vs 0.85 / 0.88 (Mask R-CNN) | immature green apples, segmentation | `sapkota_2024_comparing` | A |
| Detection F1 | 0.861 | CNN + watershed, Australia | `bargoti_2017_image` | A |
| F1, 2D → 3D | 0.816 (2D) → 0.881 (3D SfM) | 11 Fuji trees, 1,455 apples | `genemola_2020_fruit` | A |
| F1; visibility | F1 0.86; sizing improves when fruit visibility >60 % | Fuji/Elstar, amodal segmentation | `genemola_2023_looking` | A |
| Yield-estimation accuracy | 95.56-97.83 % | counting pipeline | `hani_2020_comparative` | A |
| One side vs two sides | Crop-load estimate 58 % accurate with single-side imaging vs 82 % with dual-side imaging; identification accuracy 79.8 %; duplicate-matching error 21.1 % | tall spindle, over-the-row tunnel | `gongal_2016_apple` | A (from the abstract on the CMU publication page) |
| Occlusion makes fruit unharvestable | "some apples were heavily occluded by branches or foliage, rendering them effectively 'unharvestable'" | MI field tests | `zhu_2025_advancement` | F (arXiv) |
| Fraction of fruit hidden from one side | — | — | — | **NOT VERIFIED** as a direct measurement. The closest proxy is the 58 % vs 82 % single/dual-side figure from `gongal_2016_apple` |

## 6. Drone-based fruit picking

| Item | Finding | Source | Status |
|---|---|---|---|
| Tevel FAR architecture | Drones tethered to an autonomous ground vehicle that supplies power: "the drones are tethered to the battery in an autonomous mobile base that enables the drones to fly continuously without stopping to recharge" | `oitzman_2021_tevel` | G |
| Drones per vehicle | 4 quadcopters ("four quadcopter drones that are electrically tethered to that vehicle"); picked fruit placed "onto the deck of the base vehicle" | `coxworth_2022_fruitpicking` (Jan 2022) | G |
| Drones per vehicle (earlier) | "Each ground unit vehicle contains six flying disc-shaped robots" | `tothstub_2020_fresh` (Nov 2020) | G. Conflicts with the 4 above; the design probably changed |
| Ripeness decision | Selection by "color grading or weight and size"; per-fruit logging of size, weight, ripeness | `ats_2022_fruitpicking`; `tevel_nd_website` | G (company claims) |
| Field use | 5-week non-stop pilot in an Italian apple orchard; "about 20 robots" (2022); >10 % grading difference in favour of the robots (company claim) | `ellis_2022_meet` | G |
| Picking rate | not published | — | **NOT VERIFIED**; no peer-reviewed or technical paper found. No patent could be confirmed |
| UAV-mounted harvesting mechanism | Single-motor mechanism on a hexacopter harvested a ~300 g Fuji apple "with no adverse effects observed on the hexacopter's operation" | `park_2024_fruit` | A |
| Aerial grasping/harvesting | Full GPS-denied aerial grasping system; "apple-like fruit harvesting, indoors in a vertical farming setting" | `kumar_2024_design` | A (no field orchard data) |
| Background review | Aerial manipulation state of the art | `ruggiero_2018_aerial` | A |

## 7. Multirotor endurance, battery logistics, speed

| Parameter | Value | Context | Source | Status |
|---|---|---|---|---|
| Small UAV endurance | 8-10 min per battery charge | 3 UAVs, 3-h persistent mission, >100 automated battery swaps | `ure_2015_automated` | A |
| Automated swap station | 8-battery dual-drum "hot" swap (vehicle stays powered) with on-board recharging | indoor tests | `michini_2011_automated` | A. The swap duration in seconds was **NOT VERIFIED** (full text not accessible) |
| Heavy agricultural multirotor | Hover 18 min without payload (50 kg take-off); 7 min with full spray payload (90 kg); 6 min with full spread payload (101 kg) | DJI Agras T40 | `dji_t40_specs` | G (manufacturer) |
| Fast recharge | 9-12 min per 30 Ah flight battery with the dedicated charger/generator | DJI Agras T40 | `dji_t40_specs` | G |
| Mid-size multirotor | 55 min max flight, no payload, ~8 m/s, windless; max take-off 9.2 kg | DJI M350 RTK | `dji_m350_specs` | G |
| In-orchard UAV operating speed | 2, 3, 4, 5 m/s tested | peach orchard spraying (gas-powered helicopter, not a multirotor) | `meng_2020_experimental` | A |
| General agricultural-UAV references | UAV types and payloads; bibliometric review | `radoglou_2020_compilation`; `rejeb_2022_drones` | A (descriptive) |

## 8. Processing/cider harvest and bruising

| Item | Value / finding | Source | Status / quote |
|---|---|---|---|
| Cider apples traditionally harvested from the ground | "harvested at full ripeness when the fruit naturally drops or is easily shaken off the tree"; FSMA rule: fruit that has touched the ground may not be used for human consumption without sufficient processing steps | `ewing_2018_food` | A |
| Europe mechanises cider harvest | "In Europe, much of the cider apple harvesting is done with machinery"; mechanical harvest cut cumulative harvest costs by 23-53 % (5-60 acre models, 30 years) | `karl_2022_mechanically` | A |
| Shake-and-sweep | "European mechanical harvest shakes trees to knock fruit to the ground and sweeps fruit up from the orchard floor"; groundfalls usable for hard cider but must be segregated | `brawner_2026_wsu` | G (author web summary) |
| Over-the-row cider harvester | 82 % captured, 9 % dropped to the ground, 9 % left on tree; 5.3 s/tree machine vs 229 s/tree by hand; all captured fruit bruised | `brawner_2025_overtherow`; `brawner_2026_wsu` | A (capture, timing); G ("all ... were bruised") |
| Fresh-market apples hand-picked | Semi-automatic harvesters not commercialised "due to unacceptably high incidence of apple bruising" | `zhang_2016_development` | A |
| Bin-filler bruising | Platform-assisted harvest unsuccessful in the US "due to greater bruising with the mechanized bin fillers" | `robinson_2011_advances` | F |
| Bruise threshold (very low) | Bruising initiated at an estimated 2.0 mm drop onto steel (≈20 peak g) for large McIntosh, day after harvest | `schulte_1992_apple` | A |
| Definition | Bruise threshold = drop height at which bruising just begins for a given fruit and impact surface | `hyde_2003_increasing` | A |
| Drop height vs bruise | 20-80 cm drops onto steel: pressure area rises linearly with drop height and fruit weight | `li_2023_effect` | A |
| Fruit-on-fruit | Contact areas >1 cm² closely related to bruise area; below that bruising rare | `pang_1992_analysis` | A |
| Catch systems for fresh market | ≥95 % removal, 86-95 % recovered, 59-84 % Extra Fancy pack-out; stem pulls 20-57 % | `peterson_2003_freshmarket` | A |
| Shake-and-catch (fruiting wall) | 76.6-89.5 % removal; 84.4-90.6 % marketable | `he_2019_effect` | A |
| Handling guidance | "Never throw fruit into a box, bucket or bag ... This will certainly cause bruising." | `roper_2001_when` | G |

## 9. Human harvest labour

| Item | Value | Source | Status |
|---|---|---|---|
| Piece rate and earnings | $20-28 per bin; pickers earned $16-18 per hour (WA, 2021) | `simnitt_2023_advancements` | G |
| Implied picking rate | ≈0.57-0.90 bins/h per picker | — | D: ($16-18 h⁻¹)/($20-28 bin⁻¹), range 16/28 to 18/20 |
| Implied fruit rate | ≈1,200-1,900 fruit/h (0.33-0.53 fruit/s) | — | D. Assumes a 20-bushel bin (`wells_2017_apple` uses "twenty-bushel bins"), ~19 kg/bushel (assumed, NOT VERIFIED) and 0.18 kg/fruit. Run sensitivity analysis on this value |
| Non-picking time | Pickers spend "up to a third of their time moving and positioning ladders rather than picking" | `simnitt_2023_advancements` | G |
| Time breakdown | Picking = 64 % (ladder), 78 % (DBR vacuum), 83 % (Huron platform) of total time; detaching = 30-32 % of picking time | `zhang_2020_time` | A |
| Harvest-assist gain | +95 % apples picked per second (upper canopy) vs ladders | `zhang_2016_design` | A |
| Platform gains | Harvest gains of 13-16.8 % in 3 of 5 trials (one-pick); spot-pick gains estimated at 25-50 % | `wells_2017_apple` | G |
| Platform vs ladder (multi-task) | Efficiency gains: 34 % (peach pruning) to 59 % (peach harvest) in PA; 19-67 % for apple tasks in WA | `baugher_2009_mobile` | A |
| Selective vs strip (pear) | Ladder crews 3.5 (selective) vs 5.5 (strip) 500-kg bins/worker/day; platform 1.8 vs 3.4 | `elkins_2011_comparison` | A (pear, not apple) |
| Hand harvest per tree | 229 s/tree (3.8 min) | `brawner_2025_overtherow` | A (cider orchard, 1,495 trees/ha) |

---

## Recommended simulation parameter set

Values marked **[assumption]** have no direct literature value. They must appear as assumptions in the paper and be covered by the sensitivity analysis.

| Parameter | Baseline | Range for sensitivity | Justification |
|---|---|---|---|
| Orchard layout (baseline) | Tall Spindle, 0.9 m in-row × 3.3 m between rows (3,262 trees/ha) | 0.9-1.2 × 3.0-3.5 m | `robinson_2011_tallspindle`, `reig_2020_longterm`, `hoying_nd_tallspindle` |
| Alternative layouts | Super Spindle 0.6 × 3.0 m; 2D planar 3.0 × 1.5-2.0 m; semi-intensive 1.5-1.8 × 3.7-4.25 m | — | `hoying_nd_tallspindle`, `tustin_2018_planar`, `vanhooijdonk_2023_pomological`, `brawner_2025_overtherow` |
| Tree height | 3.0 m | 2.7-3.3 m (≤0.9 × row spacing) | `robinson_2011_advances`, `robinson_2014_yield`, `robinson_2011_tallspindle` |
| Canopy thickness | 2D planar <0.5 m; tall spindle **[assumption]** ~1 m | 0.5-1.2 m | `tustin_2018_planar` (planar only) |
| Fruit per tree | 130 | 100-340 | `francescatto_2020_precision` |
| Mean fruit mass | 0.165 kg | 0.15-0.19 kg | `robinson_2008_pruning`, `francescatto_2020_precision` |
| Implied yield check | ≈70 t/ha | 50-120 t/ha | `francescatto_2020_precision`, `tustin_2018_planar`, `vanhooijdonk_2023_pomological` |
| Number of picking passes | 3 | 1 (Red Delicious) to 5 (Honeycrisp) | `simnitt_2023_advancements`, `roper_2001_when`, `brookfield_1993_fruit`, `moran_2022_honeycrisp` |
| Ripe share per pass | **[assumption]** 0.40 / 0.35 / 0.25 | uniform to front-loaded | No verified data. `hall_2001_effect` supports picking over ~3 weeks |
| Pass interval | **[assumption]** 7 d | 5-10 d | Implied by a ~3-week window with 3-4 picks (`hall_2001_effect`, `roper_2001_when`) |
| Spatial ripening bias | Exterior/top fruit ripen earlier, offset up to 14 d | 0 d (null, per `kviklys_2022_apple`, `tang_2025_flowering`) to 14 d | `roper_2001_when`, `farcuh_2021_how`, `moran_2022_honeycrisp` |
| Ripeness observability by cultivar class | Solid red: colour weak; bicolour (Gala/Fuji): background colour informative; green/yellow: colour uninformative, so classification uses non-colour indices | per-class misclassification rate **[assumption]** | `roper_2001_when`, `farcuh_2021_how`, `hall_2001_effect` |
| Detection precision / recall | 0.88 / 0.85 | P 0.84-0.93; R 0.82-0.97 | `kang_2020_fast`, `yan_2021_realtime`, `sapkota_2024_comparing`, `genemola_2020_fruit` |
| Visible fraction from one side | 0.60 | 0.55-0.80 (two sides: 0.82-0.97) | Proxy: `gongal_2016_apple` (58 % single vs 82 % dual side). Direct value NOT VERIFIED |
| Pick success per attempt | 0.80 | 0.65-0.84 | `silwal_2017_design`, `zhang_2024_automated`, `zhu_2025_advancement`, `hohimer_2019_design` |
| Per-fruit service time at the tree (ground robot reference) | 6.0 s | 3.6-7.3 s | `silwal_2017_design`, `zhang_2024_automated`, `zhang_2022_algorithm`, `hohimer_2019_design` |
| Detach sub-step | 1.0 s | — | `zhang_2022_algorithm` |
| Drone endurance (free-flying, light payload) | **[assumption]** 15 min | 8-25 min | Bracketed by `ure_2015_automated` (8-10 min, small UAV) and `dji_t40_specs` (7 min full / 18 min empty, heavy-lift) |
| Battery swap time | **[assumption]** 60 s | 30-120 s | Automated hot-swap stations exist (`michini_2011_automated`, `ure_2015_automated`); duration not verified |
| Battery recharge time | 10 min (fast) | 9-12 min; up to ~60 min **[assumption]** for standard chargers | `dji_t40_specs` |
| Tethered variant | unlimited endurance; tether-length constraint **[assumption]** | — | `oitzman_2021_tevel`, `coxworth_2022_fruitpicking` (descriptive only) |
| Cruise speed in alley | 2 m/s | 1-5 m/s | `meng_2020_experimental` (2-5 m/s orchard UAV operation) |
| Fruit drop policy | Fresh market: no free drop (place in bin or on a cushioned catcher). Processing/cider: drop to ground allowed, with later ground collection | — | `schulte_1992_apple`, `pang_1992_analysis`, `li_2023_effect`, `peterson_2003_freshmarket`, `ewing_2018_food`, `karl_2022_mechanically`, `brawner_2025_overtherow` |
| Human benchmark | ≈1,500 fruit/h per picker (≈0.4 fruit/s) | 1,200-1,900 fruit/h | D from `simnitt_2023_advancements` (rate assumptions above); `zhang_2020_time` for the time breakdown |

## Open gaps (search done; no verified value found)

1. Per-pass fraction of fruit harvested in multi-pick apple harvest.
2. A direct measurement of the fraction of fruit not visible from one side of a canopy.
3. Tevel picking rate, tether length, payload, and any peer-reviewed or patent description.
4. Measured swap duration of automated UAV battery stations (Michini et al. 2011; Lee et al. 2015, ICUAS, doi:10.1109/ICUAS.2015.7152282, gives no number in its abstract).
5. Peer-reviewed absolute human apple-picking rate in fruit/h. The available values are derived, or come from pears or ratios.
6. Original-text confirmation of the Bu et al. (2022) cycle times and the Abundant Robotics and FFRobotics throughput claims.
7. A citable source naming Granny Smith as a cultivar where skin colour does not indicate maturity.
