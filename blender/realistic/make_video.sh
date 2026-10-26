#!/usr/bin/env bash
# Explainer video of the v4 harvest simulation: data prep -> 3D frames on two GPUs -> TR / EN MP4s.
#
#   ./make_video.sh draft      # 960x540, 16 spp, every other frame -> out/video/draft_tr.mp4 (and draft_en.mp4)
#   ./make_video.sh final      # 1920x1080, 64 spp, all frames    -> out/video/harvest_explainer_{tr,en}.mp4
#   ./make_video.sh compose    # only re-run the compositing (frames already rendered)
#
# Blender runs with 2 CPU threads (the CPU is shared with long benchmark jobs) and renders on the GPUs:
# the RTX 3090 takes the heavy shots, the RTX 4060 Ti the lighter ones; frame ranges of a shot can be
# split between the two queues (see JOBS_*).  Logs: out/video/log/<shot>[_<range>].log
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MODE="${1:-draft}"
OUT="$HERE/out/video"
LOG="$OUT/log"
mkdir -p "$OUT" "$LOG"
BLENDER="${BLENDER:-blender}"
THREADS="${THREADS:-2}"

if [ "$MODE" = "draft" ]; then
    FRAMES="$OUT/frames_draft"; EXTRA="--draft --step 2"; SUFFIX="draft"
elif [ "$MODE" = "final" ] || [ "$MODE" = "compose" ]; then
    FRAMES="$OUT/frames"; EXTRA="--res 1920x1080 --samples ${SAMPLES:-64}"; SUFFIX="final"
else
    echo "usage: $0 draft|final|compose"; exit 1
fi

# ---------------------------------------------------------------- 1. data preparation (needs the sim modules)
if [ ! -f "$OUT/video_prep.json" ]; then
    python3 "$HERE/video_prep_v4.py" --data "$HERE/../anim4_rect1ha.json" || exit 1
fi

# ---------------------------------------------------------------- 2. frames
# job = "<shot> <first>-<last>" ; frame counts: intro 270, map 1, mission 720, closeup 540, takeover 330, finish 240
JOBS_3090=("mission 1-720" "closeup 1-540")
JOBS_4060=("map 1-1" "intro 1-270" "takeover 1-330" "finish 1-240")
if [ "$MODE" = "final" ]; then
    # measured at 1920x1080 / 64 spp (see README): 3090 9.6 s/frame (motion-blur shots) and 8.1 s (close-up),
    # 4060 Ti about 1.45x slower -> the 4060 Ti gets the lighter shots plus the tail of the close-up
    JOBS_3090=("mission 1-720" "closeup 1-470")
    JOBS_4060=("map 1-1" "intro 1-270" "takeover 1-330" "finish 1-240" "closeup 471-540")
fi

render_queue () {
    local gpu="$1"; shift
    for job in "$@"; do
        set -- $job
        local shot="$1" range="$2"
        local log="$LOG/${SUFFIX}_${shot}_${range}.log"
        echo "[$(date +%H:%M:%S)] $gpu: $shot $range" | tee -a "$LOG/${SUFFIX}_queue_$gpu.log"
        "$BLENDER" -b -t "$THREADS" --python "$HERE/video_v4.py" -- --shot "$shot" --out "$FRAMES/$shot" \
            --gpu "$gpu" --frames "$range" $EXTRA > "$log" 2>&1
        grep -E "rendered|Traceback|Error" "$log" | tail -3 | tee -a "$LOG/${SUFFIX}_queue_$gpu.log"
    done
}

if [ "$MODE" != "compose" ]; then
    T0=$(date +%s)
    render_queue 3090 "${JOBS_3090[@]}" &
    P1=$!
    render_queue 4060 "${JOBS_4060[@]}" &
    P2=$!
    wait $P1 $P2
    echo "frames done in $(( ($(date +%s) - T0) / 60 )) min" | tee -a "$LOG/${SUFFIX}_queue_3090.log"
fi

# ---------------------------------------------------------------- 3. compositing (captions, cards, clock, map overlay)
if [ "$MODE" = "draft" ]; then
    python3 "$HERE/compose_v4.py" --lang tr --frames "$FRAMES" --out "$OUT/draft_tr.mp4" --draft
    python3 "$HERE/compose_v4.py" --lang en --frames "$FRAMES" --out "$OUT/draft_en.mp4" --draft
else
    python3 "$HERE/compose_v4.py" --lang tr --frames "$FRAMES" --out "$OUT/harvest_explainer_tr.mp4"
    python3 "$HERE/compose_v4.py" --lang en --frames "$FRAMES" --out "$OUT/harvest_explainer_en.mp4"
fi
