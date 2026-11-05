#!/usr/bin/env bash
# =============================================================================
# get_data.sh - Shell port of construct/get_data.py (no Python required)
#
# Target: Thunder (Ubuntu Server, GPU box) which has no native Python.
#
#   - Real data is written under  $DATA_ROOT           (default /data/CameraArray)
#   - $LINK_ROOT/<session> (default ./collected_data/<session>) is a SYMLINK
#     into $DATA_ROOT, so the analysis code keeps its usual relative path
#     without the bulk data sitting on the workspace filesystem.
#   - Unreachable / failing e{NN} hosts are LOGGED and SKIPPED. A single dead
#     Pi never aborts the run.
#   - Safe to re-run: complete hosts are skipped, partial ones resume (rsync).
#   - Designed to survive SSH disconnects via tmux (--tmux).
#   - A PRE-FLIGHT SURVEY runs before anything is transferred: every Pi is
#     asked what it actually holds (session, file count, bytes, span, disk,
#     clock) and the array is checked for agreement. Hosts whose data does
#     not belong to the run being collected - the Pi that never woke up, the
#     one that fell back to the SD card, the one that died mid-capture - are
#     reported and excluded instead of being silently merged in.
#   - Two sessions are "the same session" when they hold the SAME NUMBER OF
#     FILES. Directory names are per-Pi wall-clock stamps and never match, so
#     the timestamp is used only to break a tie between two sessions on one
#     Pi that hold the same count, and as a sanity warning afterwards.
#
# Usage:
#   ./get_data.sh [options]
#
#   --tmux                 (re)launch itself inside a detached tmux session
#   --session NAME         collect this session dir instead of auto-detecting
#   --hosts e00,e03,e07    only these hosts (default e00..e15)
#   --jobs N               parallel transfers (default 8)
#   --retries N            rsync retries per host (default 2)
#   --data-root DIR        default /data/CameraArray
#   --link-root DIR        default $PWD/collected_data
#   --dry-run              rsync --dry-run, no symlink/dir side effects
#   --progress-interval S  seconds between progress reports (default 30, 0 off)
#
#  pre-flight survey (see "PRE-FLIGHT SURVEY" below):
#   --check                survey only: print the report and exit, copy nothing
#   --no-check             skip the survey and transfer straight away
#   -y | --yes             transfer even though the survey reported warnings
#   --check-sessions N     newest sessions inspected per Pi (default 5)
#   --expect-files N       the per-Pi file count of the run to collect
#                          (default: the count most Pis agree on)
#   --tolerance N          files a Pi may differ from that count (default 0)
#   --window SEC           start times further apart than this only WARN
#                          (default 900) - they never decide identity
#   --include-mismatched   also copy hosts whose file count does not match
#   --detail               also list every session found on every Pi
#   -h | --help
#
# Environment overrides: SSH_PASSWORD, SSH_USER, HOST_SUFFIX, SSH_TIMEOUT,
#                        REMOTE_DATA_DIR_SSD, REMOTE_DATA_DIR_LOCAL, TMUX_SESSION
#
# Requires: bash 4.3+, rsync, ssh (+ sshpass when SSH_PASSWORD is set), tmux.
#   sudo apt install rsync sshpass tmux
# =============================================================================

# Bash only (arrays, associative arrays, wait -n). Re-exec when someone runs
# this as `sh get_data.sh`, where dash chokes on `set -o pipefail`.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -uo pipefail   # deliberately NOT -e: a failing Pi must not kill the run

# ----------------------------- configuration ---------------------------------
SSH_USER="${SSH_USER:-pi}"
SSH_PASSWORD="${SSH_PASSWORD-pi}"      # SSH_PASSWORD="" => key-based auth only
HOST_SUFFIX="${HOST_SUFFIX-.local}"    # tried first; bare hostname is fallback
SSH_TIMEOUT="${SSH_TIMEOUT:-10}"       # seconds, per connection attempt

REMOTE_DATA_DIR_SSD="${REMOTE_DATA_DIR_SSD:-/media/pi/HIKSEMI/data}"
REMOTE_DATA_DIR_LOCAL="${REMOTE_DATA_DIR_LOCAL:-/home/pi/PiCameraArray/data}"

DATA_ROOT="${DATA_ROOT:-/data/CameraArray}"
LINK_ROOT="${LINK_ROOT:-$PWD/collected_data}"
TMUX_SESSION="${TMUX_SESSION:-picam_get_data}"

MAX_JOBS=8
RETRIES=2
DRY_RUN=0
FORCE_SESSION=""
CHECK_MODE=auto          # auto = survey then transfer | only (--check) | off
ASSUME_YES=0             # 1 = transfer despite survey warnings
CHECK_SESSIONS=5         # newest session dirs inspected per Pi
MATCH_WINDOW=900         # s; only a warning: same count but far apart in time
COUNT_TOLERANCE=0        # files a host may differ from the expected count
EXPECT_FILES=""          # --expect-files N overrides the count the array votes on
INCLUDE_MISMATCHED=0     # 1 = copy out-of-window sessions instead of skipping
DETAIL=0                 # 1 = list every inspected session of every host
PROGRESS_INTERVAL=30     # s between progress reports; 0 disables them
HOSTS=()
for i in $(seq -w 0 15); do HOSTS+=("e${i}"); done

# ------------------------------ arg parsing ----------------------------------
USE_TMUX=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tmux)        USE_TMUX=1; shift ;;
        --session)     FORCE_SESSION="$2"; shift 2 ;;
        --hosts)       IFS=',' read -r -a HOSTS <<< "$2"; shift 2 ;;
        --jobs)        MAX_JOBS="$2"; shift 2 ;;
        --retries)     RETRIES="$2"; shift 2 ;;
        --data-root)   DATA_ROOT="$2"; shift 2 ;;
        --link-root)   LINK_ROOT="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --check)       CHECK_MODE=only; shift ;;
        --no-check)    CHECK_MODE=off; shift ;;
        -y|--yes)      ASSUME_YES=1; shift ;;
        --check-sessions) CHECK_SESSIONS="$2"; shift 2 ;;
        --window)      MATCH_WINDOW="$2"; shift 2 ;;
        --tolerance)   COUNT_TOLERANCE="$2"; shift 2 ;;
        --expect-files) EXPECT_FILES="$2"; shift 2 ;;
        --include-mismatched) INCLUDE_MISMATCHED=1; shift ;;
        --detail)      DETAIL=1; shift ;;
        --progress-interval) PROGRESS_INTERVAL="$2"; shift 2 ;;
        # print the comment header, whatever length it has grown to
        -h|--help)     awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 0 ;;
        *)             echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

DATA_ROOT="${DATA_ROOT%/}"
LINK_ROOT="${LINK_ROOT%/}"

# --------------------------- tmux self-launch --------------------------------
# Re-exec inside tmux so the collection survives an SSH disconnect.
if [[ $USE_TMUX -eq 1 && -z "${TMUX:-}" ]]; then
    command -v tmux >/dev/null 2>&1 || {
        echo "tmux not installed: sudo apt install tmux" >&2; exit 1; }
    SELF="$(readlink -f "$0")"
    ARGS=()
    [[ -n "$FORCE_SESSION" ]] && ARGS+=(--session "$FORCE_SESSION")
    ARGS+=(--jobs "$MAX_JOBS" --retries "$RETRIES"
           --data-root "$DATA_ROOT" --link-root "$LINK_ROOT"
           --check-sessions "$CHECK_SESSIONS" --window "$MATCH_WINDOW"
           --tolerance "$COUNT_TOLERANCE")
    [[ $DRY_RUN -eq 1 ]] && ARGS+=(--dry-run)
    [[ $CHECK_MODE == only ]] && ARGS+=(--check)
    [[ $CHECK_MODE == off  ]] && ARGS+=(--no-check)
    [[ $INCLUDE_MISMATCHED -eq 1 ]] && ARGS+=(--include-mismatched)
    [[ $DETAIL -eq 1 ]] && ARGS+=(--detail)
    [[ -n "$EXPECT_FILES" ]] && ARGS+=(--expect-files "$EXPECT_FILES")
    ARGS+=(--progress-interval "$PROGRESS_INTERVAL")
    # A detached pane has nobody to answer a prompt: inside tmux the survey
    # either passes or stops the run, so -y must be given up front.
    [[ $ASSUME_YES -eq 1 ]] && ARGS+=(-y)
    ARGS+=(--hosts "$(IFS=,; echo "${HOSTS[*]}")")
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "tmux session '$TMUX_SESSION' already exists (a run is in progress?)." >&2
        echo "  attach: tmux attach -t $TMUX_SESSION" >&2
        exit 1
    fi
    # keep the pane alive after exit so the summary stays readable
    tmux new-session -d -s "$TMUX_SESSION" -c "$PWD" \
        "'$SELF' $(printf "'%s' " "${ARGS[@]}"); echo; echo '[finished - press q]'; read -r -n1"
    echo "Started in tmux session '$TMUX_SESSION'."
    echo "  attach : tmux attach -t $TMUX_SESSION   (detach: Ctrl-b d)"
    echo "  logs   : tail -f $DATA_ROOT/logs/get_data_*.log"
    echo "  status : watch -n5 cat $DATA_ROOT/logs/get_data_*.progress"
    exit 0
fi

# ------------------------------- logging -------------------------------------
LOG_DIR="$DATA_ROOT/logs"
mkdir -p "$LOG_DIR" 2>/dev/null || {
    echo "Cannot create $LOG_DIR (check permissions on $DATA_ROOT)" >&2; exit 1; }
RUN_ID="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/get_data_${RUN_ID}.log"
FAIL_LOG="$LOG_DIR/get_data_${RUN_ID}.failed"
: > "$MAIN_LOG"
: > "$FAIL_LOG"
STATUS_DIR="$(mktemp -d)"
SURVEY_DIR="$(mktemp -d)"
# Ctrl-C / kill must take the whole crew down: the progress monitor, the
# per-host workers, and the rsync/ssh each worker is waiting on. Anything
# left behind would keep writing into a collection nobody is watching.
stop_children() {
    local p
    kill "${PROGRESS_PID:-}" 2>/dev/null
    for p in $(pgrep -P $$ 2>/dev/null); do
        pkill -P "$p" 2>/dev/null
        kill "$p" 2>/dev/null
    done
}
trap 'stop_children; rm -rf "$STATUS_DIR" "$SURVEY_DIR"' EXIT
trap 'echo; log WARN "interrupted -- stopping; re-run the same command to resume"; exit 130' INT TERM

# log LEVEL MESSAGE...   flock keeps parallel workers from interleaving lines
log() {
    local level="$1"; shift
    local line
    line="$(date '+%Y-%m-%d %H:%M:%S') [$level] $*"
    {
        flock 9
        printf '%s\n' "$line" | tee -a "$MAIN_LOG"
    } 9>>"$MAIN_LOG"
}

fail_note() {   # record a problem host without ever stopping the run
    { flock 8; printf '%s\n' "$*" >> "$FAIL_LOG"; } 8>>"$FAIL_LOG"
}

# ------------------------------ ssh helpers ----------------------------------
SSH_OPTS=(-o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null
          -o LogLevel=ERROR
          -o ConnectTimeout="$SSH_TIMEOUT"
          -o ServerAliveInterval=30
          -o ServerAliveCountMax=6)

if [[ -n "$SSH_PASSWORD" ]]; then
    command -v sshpass >/dev/null 2>&1 || {
        echo "sshpass not found: sudo apt install sshpass rsync" >&2; exit 1; }
    SSH_WRAP=(sshpass -p "$SSH_PASSWORD")
    SSH_OPTS+=(-o PubkeyAuthentication=no -o PreferredAuthentications=password)
else
    SSH_WRAP=()
    SSH_OPTS+=(-o BatchMode=yes)   # never hang on a password prompt
fi
command -v rsync >/dev/null 2>&1 || {
    echo "rsync not found: sudo apt install rsync" >&2; exit 1; }

# resolve_target HOST -> prints a reachable "host[.local]" name, rc=1 if none.
# Tries $HOST_SUFFIX first (mDNS), then the bare name (/etc/hosts, DNS).
resolve_target() {
    local host="$1" cand seen=""
    for cand in "${host}${HOST_SUFFIX}" "$host"; do
        [[ " $seen " == *" $cand "* ]] && continue
        seen="$seen $cand"
        if "${SSH_WRAP[@]}" ssh "${SSH_OPTS[@]}" "${SSH_USER}@${cand}" true \
               >/dev/null 2>&1; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

# ssh_run TARGET 'remote command'  -> stdout of the remote command
ssh_run() {
    local target="$1"; shift
    "${SSH_WRAP[@]}" ssh "${SSH_OPTS[@]}" "${SSH_USER}@${target}" "$*" 2>/dev/null
}

remote_data_dir() {   # external SSD if mounted, otherwise the SD card
    local target="$1" out
    out="$(ssh_run "$target" "test -d '$REMOTE_DATA_DIR_SSD' && echo ssd || echo local")"
    [[ "$out" == "ssd" ]] && printf '%s\n' "$REMOTE_DATA_DIR_SSD" \
                          || printf '%s\n' "$REMOTE_DATA_DIR_LOCAL"
}

latest_session() {    # newest session directory name under the data dir
    local target="$1" dir="$2"
    ssh_run "$target" "ls -1t '$dir' 2>/dev/null | head -1" | tr -d '\r'
}

remote_usage() {  # "files bytes" of the .png/.npz below the remote session dir
    local target="$1" dir="$2" name="$3" out
    out="$(ssh_run "$target" \
        "find '$dir/$name' -type f \\( -name '*.png' -o -name '*.npz' \\) -printf '%s\\n' 2>/dev/null |
         awk '{c++; b += \$1} END {print c+0, b+0}'")"
    out="$(printf '%s' "$out" | tr -dc '0-9 ')"
    [[ -n "$out" ]] && printf '%s\n' "$out" || echo "0 0"
}

local_file_count() {
    local d="$1"
    [[ -d "$d" ]] || { echo 0; return; }
    find "$d" \( -name '*.png' -o -name '*.npz' \) 2>/dev/null | wc -l | tr -dc '0-9'
}

# local_usage DIR -> "files bytes".  Counting files is not enough to call a
# host finished: an interrupted rsync leaves a file that exists at its final
# name but is short, and a count-only check calls that host complete forever.
local_usage() {
    local d="$1"
    [[ -d "$d" ]] || { echo "0 0"; return; }
    find "$d" -type f \( -name '*.png' -o -name '*.npz' \) -printf '%s\n' 2>/dev/null |
        awk '{c++; b += $1} END {print c+0, b+0}'
}

# --------------------------- pre-flight survey -------------------------------
# Nothing below transfers data. Before a single byte moves it answers:
#   * which Pis answer at all
#   * where each one wrote  (external SSD vs the SD-card fallback)
#   * which session each one would contribute: file count, bytes, span
#   * whether that session really belongs to the SAME run as everybody else
#
# Session directory names are per-Pi wall-clock stamps taken when that Pi's
# agent entered capture mode, so they are never byte-identical across the
# array (e00 may say 20260828_101502 while e07 says ..._101504), and a stamp
# is only as trustworthy as that Pi's clock. What a shared run really leaves
# behind is a shared FILE COUNT: every camera answers the same triggers, so
# every camera writes the same number of bursts.
#
# Hence: identity is the file count. The array votes on the expected count,
# each Pi contributes the session that matches it, and the timestamp is used
# only to break a tie between two sessions on one Pi with the same count --
# plus as a warning afterwards, when a count matches but the start time is
# far from everybody else's.

SURVEY_SESSION=""          # session dir name adopted for the collection folder
SURVEY_WARN=()             # human-readable problems found by the survey

# One SSH round trip per Pi. Prints tab-separated records; keys are stable so
# the local side can grow without touching the remote script.
REMOTE_SURVEY_BODY=$(cat <<'REMOTE_EOF'
if [ -d "$ssd" ]; then d="$ssd"; store=ssd; else d="$loc"; store=local; fi
printf 'NOW\t%s\n' "$(date +%s)"
printf 'DIR\t%s\n' "$d"
printf 'STORE\t%s\n' "$store"
if pgrep -f camera_agent >/dev/null 2>&1; then
    printf 'AGENT\trunning\n'
else
    printf 'AGENT\tstopped\n'
fi
df -Pk "$d" 2>/dev/null | awk 'NR==2 {printf "DF\t%s\t%s\n", $2, $4}'
[ -d "$d" ] || { printf 'NODIR\t%s\n' "$d"; exit 0; }
ls -1t "$d" 2>/dev/null | head -n "$nkeep" | while IFS= read -r n; do
    [ -d "$d/$n" ] || continue
    find "$d/$n" -type f \( -name '*.npz' -o -name '*.png' \) \
         -printf '%s %T@\n' 2>/dev/null |
    awk -v n="$n" '
        {c++; b += $1; if (mn == "" || $2 < mn) mn = $2; if ($2 > mx) mx = $2}
        END {printf "SESSION\t%s\t%d\t%.0f\t%.0f\t%.0f\n", n, c+0, b+0, mn+0, mx+0}'
done
REMOTE_EOF
)

remote_survey_cmd() {
    printf "ssd='%s'; loc='%s'; nkeep=%s\n%s\n" \
        "$REMOTE_DATA_DIR_SSD" "$REMOTE_DATA_DIR_LOCAL" "$CHECK_SESSIONS" \
        "$REMOTE_SURVEY_BODY"
}

# report LINE...  -- like log() but without the timestamp/level prefix, so the
# survey table stays readable both on screen and in the log file.
report() {
    { flock 9; printf '%s\n' "$*" | tee -a "$MAIN_LOG"; } 9>>"$MAIN_LOG"
}

survey_pi() {   # never fails; leaves its findings in $SURVEY_DIR/<host>.*
    local host="$1" target out
    date +%s > "$SURVEY_DIR/$host.asked"
    if ! target="$(resolve_target "$host")"; then
        echo unreachable > "$SURVEY_DIR/$host.state"
        return 0
    fi
    printf '%s\n' "$target" > "$SURVEY_DIR/$host.target"
    out="$("${SSH_WRAP[@]}" ssh "${SSH_OPTS[@]}" "${SSH_USER}@${target}" \
           "$(remote_survey_cmd)" 2>/dev/null)"
    if [[ -z "$out" ]]; then
        echo error > "$SURVEY_DIR/$host.state"
        return 0
    fi
    printf '%s\n' "$out" > "$SURVEY_DIR/$host.raw"
    grep '^SESSION' "$SURVEY_DIR/$host.raw" | cut -f2- > "$SURVEY_DIR/$host.sessions"
    echo ok > "$SURVEY_DIR/$host.state"
    return 0
}

session_epoch() {   # 20260828_101502 -> epoch seconds; 0 when unparsable
    local n="$1"
    if [[ "$n" =~ ^([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})([0-9]{2}) ]]; then
        date -d "${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-${BASH_REMATCH[3]} ${BASH_REMATCH[4]}:${BASH_REMATCH[5]}:${BASH_REMATCH[6]}" \
             +%s 2>/dev/null || echo 0
    else
        echo 0
    fi
}

human_bytes() {
    awk -v b="${1:-0}" 'BEGIN {
        split("B KiB MiB GiB TiB", u, " "); i = 1
        while (b >= 1024 && i < 5) {b /= 1024; i++}
        printf (i == 1 ? "%.0f%s" : "%.1f%s"), b, u[i] }'
}

human_dur() {
    local s="${1:-0}"
    (( s < 0 )) && s=$(( -s ))
    if   (( s >= 86400 )); then printf '%dd%02dh' $(( s / 86400 )) $(( (s % 86400) / 3600 ))
    elif (( s >= 3600 ));  then printf '%dh%02dm' $(( s / 3600 ))  $(( (s % 3600) / 60 ))
    elif (( s >= 60 ));    then printf '%dm%02ds' $(( s / 60 ))    $(( s % 60 ))
    else                        printf '%ds' "$s"
    fi
}

signed_int() {  # "0" / "+2" / "-1", for the file-count difference column
    local v="${1:-0}"
    if (( v > 0 )); then printf '+%d' "$v"; else printf '%d' "$v"; fi
}

signed_dur() {  # "+2s" / "-2d01h", for gap columns
    local s="${1:-0}"
    (( s == 0 )) && { printf '0s'; return; }
    (( s > 0 )) && printf '+%s' "$(human_dur "$s")" || printf -- '-%s' "$(human_dur "$s")"
}

median_of() {   # median of the integers given as arguments (0 when none)
    printf '%s\n' "$@" | sort -n | awk '
        {v[NR] = $1}
        END {if (NR == 0) {print 0; exit}
             print (NR % 2 ? v[(NR + 1) / 2] : int((v[NR / 2] + v[NR / 2 + 1]) / 2))}'
}

# host_latest_epoch HOST -> start time of that host's newest session (0 = none)
host_latest_epoch() {
    local h="$1" name cnt bytes first last ep best=0
    [[ -s "$SURVEY_DIR/$h.sessions" ]] || { echo 0; return; }
    while IFS=$'\t' read -r name cnt bytes first last; do
        [[ -n "$name" ]] || continue
        ep="$(session_epoch "$name")"
        (( ep == 0 )) && ep="${first:-0}"
        (( ep > best )) && best="$ep"
    done < "$SURVEY_DIR/$h.sessions"
    printf '%s\n' "$best"
}

# mode_of VALUES...  -- the most frequent value; a tie goes to the larger one,
# i.e. to the more complete capture.
mode_of() {
    printf '%s\n' "$@" | sort -n | uniq -c | sort -k1,1n -k2,2n | tail -1 | awk '{print $2}'
}

# pick_session HOST REF_COUNT REF_EPOCH
#   -> "name<TAB>count<TAB>bytes<TAB>epoch<TAB>span<TAB>gap<TAB>count_diff"
# The session whose FILE COUNT is closest to the count the array agreed on.
# Only when two sessions on this host are equally close does the start time
# decide, and then it merely picks the one nearer the rest of the array.
pick_session() {
    local h="$1" refc="$2" ref="$3"
    local name cnt bytes first last ep span cdiff adiff agap
    local best="" bestdiff=-1 bestgap=-1
    [[ -s "$SURVEY_DIR/$h.sessions" ]] || return 1
    while IFS=$'\t' read -r name cnt bytes first last; do
        [[ -n "$name" ]] || continue
        cnt="${cnt:-0}"; bytes="${bytes:-0}"; first="${first:-0}"; last="${last:-0}"
        ep="$(session_epoch "$name")"
        (( ep == 0 )) && ep="$first"
        cdiff=$(( cnt - refc ))
        adiff=$(( cdiff < 0 ? -cdiff : cdiff ))
        agap=$(( ep > ref ? ep - ref : ref - ep ))
        span=$(( last > first ? last - first : 0 ))
        if (( bestdiff < 0 || adiff < bestdiff || (adiff == bestdiff && agap < bestgap) )); then
            bestdiff="$adiff"; bestgap="$agap"
            best="$name"$'\t'"$cnt"$'\t'"$bytes"$'\t'"$ep"$'\t'"$span"$'\t'"$(( ep - ref ))"$'\t'"$cdiff"
        fi
    done < "$SURVEY_DIR/$h.sessions"
    [[ -n "$best" ]] || return 1
    printf '%s\n' "$best"
}

# report_runs REF_COUNT -- every candidate session the array still holds,
# grouped by FILE COUNT, which is what "the same session" means here. A group
# whose members also started within seconds of each other is a real
# simultaneous run; a group whose SPREAD is days wide shares a file count by
# coincidence. That column is how you tell the two apart before committing.
report_runs() {
    local refc="$1"
    [[ -s "$SURVEY_DIR/all.tsv" ]] || return 0

    awk -F'\t' -v OFS='\t' '
        {
            ep = $1; h = $2; c = $4; b = $5
            if (!(c in nh)) {nh[c] = 0; first[c] = ep; last[c] = ep}
            if (ep < first[c]) first[c] = ep
            if (ep > last[c]) last[c] = ep
            if (!((c SUBSEP h) in seen)) {
                seen[c SUBSEP h] = 1
                hosts[c] = hosts[c] (hosts[c] == "" ? "" : " ") h
                nh[c]++
            }
            bytes[c] += b; nses[c]++
        }
        END {for (c in nh) print nh[c], c, nses[c], bytes[c], first[c], last[c], hosts[c]}
    ' "$SURVEY_DIR/all.tsv" |
    sort -t"$(printf '\t')" -k1,1nr -k6,6nr > "$SURVEY_DIR/runs.tsv"

    report ""
    report "candidate sessions grouped by file count ($(wc -l < "$SURVEY_DIR/runs.tsv") distinct count(s) in the $CHECK_SESSIONS newest session(s) per Pi):"
    report "$(printf '  %7s %6s %9s  %-19s %8s  %s' \
             FILES HOSTS SIZE 'EARLIEST START' SPREAD 'HOSTS WITHOUT IT')"

    local nh c nses rbytes fstart lstart rhosts mark miss h
    while IFS=$'\t' read -r nh c nses rbytes fstart lstart rhosts; do
        mark="  "; (( c == refc )) && mark=" *"
        miss=""
        for h in "${HOSTS[@]}"; do
            [[ " $rhosts " == *" $h "* ]] || miss+="$h "
        done
        [[ -z "$miss" ]] && miss="-"
        report "$(printf '%s%7s %6s %9s  %-19s %8s  %s' \
                 "$mark" "$c" "$nh/${#HOSTS[@]}" "$(human_bytes "$rbytes")" \
                 "$(date -d "@$fstart" '+%Y-%m-%d %H:%M:%S')" \
                 "$(human_dur $(( lstart - fstart )))" "${miss% }")"
    done < "$SURVEY_DIR/runs.tsv"
    report "  (* = the group this collection would take; choose another with --expect-files N)"
    report "  (SPREAD = how far apart the members of a group started; seconds = one real run)"
    report ""
}

stamp() {   # frame mtime -> "MM-DD HH:MM:SS"; "-" when the session is empty
    local e="${1:-0}"
    if (( e > 0 )); then date -d "@$e" '+%m-%d %H:%M:%S'; else printf -- '-'; fi
}

# report_detail -- every session of every host, newest first (--detail)
report_detail() {
    local h name cnt bytes first last ep pickmark
    report ""
    report "all sessions inspected per host (* = the one that would be collected):"
    for h in "${HOSTS[@]}"; do
        if [[ ! -s "$SURVEY_DIR/$h.sessions" ]]; then
            report "$(printf '  %-4s %s' "$h" "(nothing to list)")"
            continue
        fi
        while IFS=$'\t' read -r name cnt bytes first last; do
            [[ -n "$name" ]] || continue
            pickmark=" "
            [[ "$name" == "$(cut -f2 "$SURVEY_DIR/$h.pick" 2>/dev/null)" ]] && pickmark="*"
            report "$(printf '  %-4s %s %-16s %6s files %9s  %14s -> %-14s' \
                     "$h" "$pickmark" "$name" "${cnt:-0}" "$(human_bytes "${bytes:-0}")" \
                     "$(stamp "${first:-0}")" "$(stamp "${last:-0}")")"
        done < "$SURVEY_DIR/$h.sessions"
    done
    report ""
}

# run_survey -- fills $SURVEY_DIR/<host>.pick for every host that should be
# transferred, prints the report, and returns 1 when anything looks wrong.
run_survey() {
    local h
    log INFO "pre-flight survey: inspecting ${#HOSTS[@]} host(s), $CHECK_SESSIONS newest session(s) each"

    local running=0
    for h in "${HOSTS[@]}"; do
        survey_pi "$h" &
        running=$((running + 1))
        if (( running >= MAX_JOBS )); then
            wait -n 2>/dev/null || true
            running=$((running - 1))
        fi
    done
    wait

    declare -A STATE DIR STORE AGENT SKEW DFTOT DFAVAIL LATEST
    local now_remote asked key a b
    for h in "${HOSTS[@]}"; do
        STATE[$h]="$(cat "$SURVEY_DIR/$h.state" 2>/dev/null || echo error)"
        DIR[$h]="-"; STORE[$h]="-"; AGENT[$h]="-"; SKEW[$h]=0
        DFTOT[$h]=0; DFAVAIL[$h]=0; LATEST[$h]=0
        [[ "${STATE[$h]}" == ok ]] || continue
        now_remote=0
        asked="$(cat "$SURVEY_DIR/$h.asked" 2>/dev/null || echo 0)"
        while IFS=$'\t' read -r key a b; do
            case "$key" in
                NOW)   now_remote="$a" ;;
                DIR)   DIR[$h]="$a" ;;
                STORE) STORE[$h]="$a" ;;
                AGENT) AGENT[$h]="$a" ;;
                DF)    DFTOT[$h]="${a:-0}"; DFAVAIL[$h]="${b:-0}" ;;
                NODIR) STATE[$h]=nodir; DIR[$h]="$a" ;;
            esac
        done < "$SURVEY_DIR/$h.raw"
        (( now_remote > 0 && asked > 0 )) && SKEW[$h]=$(( now_remote - asked ))
        LATEST[$h]="$(host_latest_epoch "$h")"
        [[ "${STATE[$h]}" == ok && "${LATEST[$h]}" -eq 0 ]] && STATE[$h]=nosession
    done

    # ---- every session every host still holds, one row each -------------
    # This is the evidence everything below is decided on: not just the one
    # session each Pi happens to have written last, but every session the
    # survey inspected, so a wrong newest-directory cannot mislead the run.
    local name cnt bytes first last ep
    : > "$SURVEY_DIR/all.tsv"
    for h in "${HOSTS[@]}"; do
        [[ "${STATE[$h]}" == ok ]] || continue
        while IFS=$'\t' read -r name cnt bytes first last; do
            [[ -n "$name" ]] || continue
            ep="$(session_epoch "$name")"
            (( ep == 0 )) && ep="${first:-0}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$ep" "$h" "$name" "${cnt:-0}" "${bytes:-0}" "${first:-0}" "${last:-0}" \
                >> "$SURVEY_DIR/all.tsv"
        done < "$SURVEY_DIR/$h.sessions"
    done
    if [[ ! -s "$SURVEY_DIR/all.tsv" ]]; then
        report ""
        report "PRE-FLIGHT SURVEY: no host offered a usable session directory."
        SURVEY_WARN+=("no host has any session data")
        for h in "${HOSTS[@]}"; do
            report "  $h : ${STATE[$h]}"
        done
        return 1
    fi

    # ---- the file count that identifies the run being collected ----------
    # The array votes with the newest session of every host that answered;
    # 0-file sessions never vote, and a tie goes to the larger count.
    local ref_count=0 ref_src="" newest=() c
    for h in "${HOSTS[@]}"; do
        [[ "${STATE[$h]}" == ok ]] || continue
        c="$(head -1 "$SURVEY_DIR/$h.sessions" 2>/dev/null | cut -f2)"
        [[ -n "${c:-}" ]] && (( c > 0 )) && newest+=("$c")
    done
    if [[ -n "$EXPECT_FILES" ]]; then
        ref_count="$EXPECT_FILES"
        ref_src="from --expect-files"
    elif [[ -n "$FORCE_SESSION" ]]; then
        ref_count="$(awk -F'\t' -v s="$FORCE_SESSION" '$3 == s {print $4; exit}' "$SURVEY_DIR/all.tsv")"
        if [[ -n "${ref_count:-}" ]]; then
            ref_src="file count of --session $FORCE_SESSION"
        else
            # Each Pi names its session after its own start, so the collection
            # name is usually on no host at all. Then the session that STARTED
            # nearest to it identifies the run - never the array vote, which
            # with one host is just that host's newest session (that is how
            # e10's 2026-07-29 run was filed under 20260728_073439).
            local fe; fe="$(session_epoch "$FORCE_SESSION")"
            ref_count="$(awk -F'\t' -v fe="$fe" -v w="$MATCH_WINDOW" '
                $4 > 0 {d = $1 - fe; if (d < 0) d = -d
                        if (d <= w && (best == "" || d < bd)) {bd = d; best = $4}}
                END {print best + 0}' "$SURVEY_DIR/all.tsv")"
            if (( fe > 0 && ref_count > 0 )); then
                ref_src="file count of the session starting nearest to --session $FORCE_SESSION"
            else
                ref_count=0
                log WARN "--session '$FORCE_SESSION': no host has a session starting within $(human_dur "$MATCH_WINDOW") of it; falling back to the array vote (hosts far from it will be excluded)"
            fi
        fi
    fi
    if (( ref_count == 0 )); then
        if (( ${#newest[@]} > 0 )); then
            ref_count="$(mode_of "${newest[@]}")"
            ref_src="agreed by the array"
        else
            # every newest session is empty: fall back to the largest count seen
            ref_count="$(awk -F'\t' 'BEGIN{m=0} $4 > m {m = $4} END {print m}' "$SURVEY_DIR/all.tsv")"
            ref_src="largest count found"
        fi
    fi

    # Hosts holding a session with exactly that count, and the median start
    # time of those sessions -- the only thing timestamps are used for.
    local agree ref=0 eps=()
    agree="$(awk -F'\t' -v c="$ref_count" '$4 == c {h[$2] = 1} END {print length(h)}' "$SURVEY_DIR/all.tsv")"
    while read -r ep; do [[ -n "$ep" ]] && eps+=("$ep"); done \
        < <(awk -F'\t' -v c="$ref_count" '$4 == c {print $1}' "$SURVEY_DIR/all.tsv")
    (( ${#eps[@]} == 0 )) && \
        while read -r ep; do [[ -n "$ep" ]] && eps+=("$ep"); done \
            < <(cut -f1 "$SURVEY_DIR/all.tsv")
    ref="$(median_of "${eps[@]:-0}")"
    # a named collection fixes the reference start: GAP is then measured from
    # it, and a host whose pick is far from it is a different run
    if [[ -n "$FORCE_SESSION" ]] && (( $(session_epoch "$FORCE_SESSION") > 0 )); then
        ref="$(session_epoch "$FORCE_SESSION")"
    fi

    # ---- per-host pick ---------------------------------------------------
    declare -A PICK PCOUNT PBYTES PSPAN PGAP PCDIFF
    local line
    for h in "${HOSTS[@]}"; do
        PICK[$h]=""; PCOUNT[$h]=0; PBYTES[$h]=0; PSPAN[$h]=0; PGAP[$h]=0; PCDIFF[$h]=0
        [[ "${STATE[$h]}" == ok ]] || continue
        line="$(pick_session "$h" "$ref_count" "$ref")" || continue
        IFS=$'\t' read -r PICK[$h] PCOUNT[$h] PBYTES[$h] _ PSPAN[$h] PGAP[$h] PCDIFF[$h] <<< "$line"
    done

    # ---- report table ----
    report ""
    report "======================= PRE-FLIGHT SURVEY ======================================"
    report "expected file count : $ref_count per Pi  ($ref_src; $agree of ${#HOSTS[@]} host(s) hold such a session)"
    report "same-session rule   : the file count decides, tolerance +/- $COUNT_TOLERANCE file(s);"
    report "                      timestamps only break ties between equal counts"
    report "reference start     : $(date -d "@$ref" '+%Y-%m-%d %H:%M:%S')  (median of the matching sessions; GAP is measured from it)"
    report ""
    report "$(printf '%-4s %-5s %-5s %-16s %7s %6s %9s %8s %8s %6s %9s' \
             HOST STORE AGENT SESSION FILES 'vsEXP' SIZE SPAN GAP SKEW FREE)"
    report "--------------------------------------------------------------------------------"

    local total_bytes=0 participants=0 excluded=0 state store agentcol
    for h in "${HOSTS[@]}"; do
        state="${STATE[$h]}"
        if [[ "$state" != ok ]]; then
            case "$state" in
                unreachable) report "$(printf '%-4s %s' "$h" 'UNREACHABLE  (ssh timeout / refused / auth failed)')"
                             SURVEY_WARN+=("$h : unreachable -- it will NOT contribute to this session") ;;
                nodir)       report "$(printf '%-4s %s' "$h" "NO DATA DIRECTORY at ${DIR[$h]}")"
                             SURVEY_WARN+=("$h : no data directory (${DIR[$h]}) -- did the agent ever start?") ;;
                nosession)   report "$(printf '%-4s %s' "$h" "NO SESSION under ${DIR[$h]}")"
                             SURVEY_WARN+=("$h : data directory is empty -- this Pi captured nothing") ;;
                *)           report "$(printf '%-4s %s' "$h" 'SURVEY FAILED (ssh ok, remote query returned nothing)')"
                             SURVEY_WARN+=("$h : survey command failed") ;;
            esac
            fail_note "$h ${state} (pre-check)"
            excluded=$((excluded + 1))
            continue
        fi

        store="${STORE[$h]}"
        [[ "$store" == local ]] && store="SD!"
        case "${AGENT[$h]}" in running) agentcol=run ;; stopped) agentcol=stop ;; *) agentcol="-" ;; esac
        local cdiff=${PCDIFF[$h]}
        report "$(printf '%-4s %-5s %-5s %-16s %7s %6s %9s %8s %8s %6s %9s' \
                 "$h" "$store" "$agentcol" "${PICK[$h]:--}" "${PCOUNT[$h]}" \
                 "$(signed_int "$cdiff")" \
                 "$(human_bytes "${PBYTES[$h]}")" "$(human_dur "${PSPAN[$h]}")" \
                 "$(signed_dur "${PGAP[$h]}")" "$(signed_dur "${SKEW[$h]}")" \
                 "$(human_bytes $(( ${DFAVAIL[$h]} * 1024 )) )")"

        # -- anomalies ---------------------------------------------------------
        # File count decides membership; the start time only raises a doubt.
        local gap_abs=$(( ${PGAP[$h]} < 0 ? -${PGAP[$h]} : ${PGAP[$h]} ))
        local cdiff_abs=$(( cdiff < 0 ? -cdiff : cdiff ))
        local mismatched=0
        if (( cdiff_abs > COUNT_TOLERANCE )); then
            mismatched=1
            SURVEY_WARN+=("$h : its best session '${PICK[$h]}' holds ${PCOUNT[$h]} files, not $ref_count -- a DIFFERENT session")
            local other ocnt
            while IFS=$'\t' read -r other ocnt _ _ _; do
                [[ -n "$other" ]] && SURVEY_WARN+=("       sessions on $h: $other ($ocnt files)")
            done < "$SURVEY_DIR/$h.sessions"
        fi
        if (( mismatched == 1 && INCLUDE_MISMATCHED == 0 )); then
            SURVEY_WARN+=("       -> $h EXCLUDED from this collection (--include-mismatched forces it in,")
            SURVEY_WARN+=("          --tolerance N allows a difference of N files, --expect-files N sets the count)")
            fail_note "$h count-mismatch (pre-check)"
            excluded=$((excluded + 1))
            continue
        fi
        if (( ${PCOUNT[$h]} == 0 )); then
            SURVEY_WARN+=("$h : session '${PICK[$h]}' contains 0 png/npz files -- nothing to fetch")
            fail_note "$h empty-session (pre-check)"
            excluded=$((excluded + 1))
            continue
        fi
        if [[ -n "$FORCE_SESSION" ]] && (( gap_abs > MATCH_WINDOW )); then
            SURVEY_WARN+=("$h : session '${PICK[$h]}' starts $(human_dur "$gap_abs") from --session $FORCE_SESSION -- a DIFFERENT run")
            if (( INCLUDE_MISMATCHED == 0 )); then
                SURVEY_WARN+=("       -> $h EXCLUDED (filing it under $FORCE_SESSION would mix two runs)")
                fail_note "$h wrong-run (pre-check)"
                excluded=$((excluded + 1))
                continue
            fi
        fi
        (( mismatched == 1 )) && \
            SURVEY_WARN+=("       -> $h INCLUDED anyway (--include-mismatched)")
        (( mismatched == 0 && cdiff_abs > 0 )) && \
            SURVEY_WARN+=("$h : ${PCOUNT[$h]} files, $cdiff_abs off the expected $ref_count -- accepted by --tolerance $COUNT_TOLERANCE")
        # Same file count, wildly different start time: the count may be a
        # coincidence, so say so instead of quietly merging it.
        (( gap_abs > MATCH_WINDOW )) && \
            SURVEY_WARN+=("$h : matches on file count but starts $(human_dur "$gap_abs") from the rest -- confirm it is the same run")
        [[ "${STORE[$h]}" == local ]] && \
            SURVEY_WARN+=("$h : recorded on the SD CARD, not the SSD (${DIR[$h]}) -- the SSD was not mounted")
        (( ${SKEW[$h]} > 5 || ${SKEW[$h]} < -5 )) && \
            SURVEY_WARN+=("$h : clock is $(signed_dur "${SKEW[$h]}") off this machine -- frame timestamps will not line up")
        (( ${DFTOT[$h]} > 0 && ${DFAVAIL[$h]} * 100 / ${DFTOT[$h]} < 5 )) && \
            SURVEY_WARN+=("$h : only $(human_bytes $(( ${DFAVAIL[$h]} * 1024 ))) free on its capture disk")

        printf '%s\t%s\t%s\t%s\n' "${DIR[$h]}" "${PICK[$h]}" "${PCOUNT[$h]}" "${PBYTES[$h]}" \
            > "$SURVEY_DIR/$h.pick"
        SURVEY_TOTAL_BYTES=$(( SURVEY_TOTAL_BYTES + ${PBYTES[$h]} ))
        SURVEY_TOTAL_FILES=$(( SURVEY_TOTAL_FILES + ${PCOUNT[$h]} ))
        total_bytes=$(( total_bytes + ${PBYTES[$h]} ))
        participants=$((participants + 1))
        # the collection folder is named after the first participating host
        [[ -z "$SURVEY_SESSION" ]] && SURVEY_SESSION="${PICK[$h]}"
    done
    report "--------------------------------------------------------------------------------"

    report_runs "$ref_count"
    (( DETAIL == 1 )) && report_detail

    # ---- totals and local disk ----
    local avail_kb; avail_kb="$(df -Pk "$DATA_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
    avail_kb="${avail_kb:-0}"
    report "participating hosts : $participants / ${#HOSTS[@]}   (excluded: $excluded)"
    report "expected file count : $ref_count per host"
    report "to transfer         : $(human_bytes "$total_bytes")  (already-present files are skipped by rsync)"
    report "local free space    : $(human_bytes $(( avail_kb * 1024 ))) at $DATA_ROOT"
    if [[ -n "${FORCE_SESSION:-}${SURVEY_SESSION:-}" ]]; then
        report "collection folder   : $DATA_ROOT/${FORCE_SESSION:-$SURVEY_SESSION}"
    else
        report "collection folder   : (none -- no host would contribute)"
    fi
    if (( avail_kb * 1024 < total_bytes )); then
        SURVEY_WARN+=("local disk holds $(human_bytes $(( avail_kb * 1024 ))) but the transfer needs up to $(human_bytes "$total_bytes")")
    fi
    (( participants == 0 )) && SURVEY_WARN+=("no host would contribute anything -- nothing to transfer")

    if (( ${#SURVEY_WARN[@]} > 0 )); then
        local issues
        issues="$(printf '%s\n' "${SURVEY_WARN[@]}" | grep -c '^[^ ]')"
        report ""
        report "WARNINGS ($issues):"
        local w
        for w in "${SURVEY_WARN[@]}"; do report "  ! $w"; done
    else
        report ""
        report "all ${#HOSTS[@]} hosts hold the same session: $ref_count files each."
    fi
    report "================================================================================"
    report ""

    (( ${#SURVEY_WARN[@]} == 0 )) && return 0
    return 1
}

# ------------------------------- progress ------------------------------------
# A 600 GB pull takes hours, so the run must be able to answer "how far along
# is it?" at any moment - including from another terminal, hence the status
# file. The denominator comes from the survey (expected files/bytes per Pi),
# so the percentage is real rather than a guess at rsync's output.
#
# Note on counting: rsync runs with --partial --inplace, so a file being
# written already exists at its final name with a partial size. Bytes are
# therefore accurate; the file count leads by at most one file per host.

SURVEY_TOTAL_BYTES=0       # what run_survey expects to fetch, in total
SURVEY_TOTAL_FILES=0
PROGRESS_FILE=""           # live status table, rewritten every tick
PROGRESS_PID=""

dest_usage() {   # DEST -> "files bytes" of the png/npz already on this disk
    local d="$1"
    [[ -d "$d" ]] || { echo "0 0"; return; }
    find "$d" -type f \( -name '*.png' -o -name '*.npz' \) -printf '%s\n' 2>/dev/null |
        awk '{c++; b += $1} END {print c+0, b+0}'
}

progress_bar() {
    local pct="${1:-0}" w=20 filled
    (( pct < 0 )) && pct=0
    (( pct > 100 )) && pct=100
    filled=$(( pct * w / 100 ))
    printf '[%s%s]' "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
                    "$(printf '%*s' $(( w - filled )) '')"
}

# progress_monitor SESSION -- background loop; exits once every host has a
# final status, and is killed by the main run as a backstop.
progress_monitor() {
    local session="$1"
    local start_t; start_t="$(date +%s)"
    local prev_b=0 prev_t="$start_t" ema=0 first=1 stalled=0
    local h dest expf expb f b now dt rate eta pct
    local cur_f cur_b exp_f exp_b done_hosts running waiting state line

    # bytes already on disk when the transfer started: they are part of the
    # goal (a resumed run is legitimately 80% done) but not of the rate.
    while :; do
        sleep "$PROGRESS_INTERVAL"
        cur_f=0; cur_b=0; exp_f=0; exp_b=0
        done_hosts=0; running=0; waiting=0
        : > "$PROGRESS_FILE.tmp"
        for h in "${HOSTS[@]}"; do
            dest=""; expf=0; expb=0
            [[ -s "$SURVEY_DIR/$h.expect" ]] &&
                IFS=$'\t' read -r dest expf expb < "$SURVEY_DIR/$h.expect"
            expf="${expf:-0}"; expb="${expb:-0}"
            read -r f b <<< "$(dest_usage "$dest")"
            cur_f=$(( cur_f + f )); cur_b=$(( cur_b + b ))
            exp_f=$(( exp_f + expf )); exp_b=$(( exp_b + expb ))

            if [[ -s "$STATUS_DIR/$h" ]]; then
                state="$(cat "$STATUS_DIR/$h")"
                done_hosts=$((done_hosts + 1))
            elif [[ -e "$SURVEY_DIR/$h.started" ]]; then
                state=running; running=$((running + 1))
            else
                state=queued; waiting=$((waiting + 1))
            fi

            pct=0
            if (( expb > 0 )); then pct=$(( b * 100 / expb ))
            elif (( expf > 0 )); then pct=$(( f * 100 / expf ))
            fi
            printf '  %-4s %-10s %6s/%-6s %9s/%-9s %3s%% %s\n' \
                "$h" "$state" "$f" "$expf" \
                "$(human_bytes "$b")" "$( (( expb > 0 )) && human_bytes "$expb" || printf '?' )" \
                "$pct" "$(progress_bar "$pct")" >> "$PROGRESS_FILE.tmp"
        done

        # the survey's totals win; without a survey, sum what the workers found
        (( SURVEY_TOTAL_BYTES > 0 )) && exp_b="$SURVEY_TOTAL_BYTES"
        (( SURVEY_TOTAL_FILES > 0 )) && exp_f="$SURVEY_TOTAL_FILES"

        now="$(date +%s)"
        dt=$(( now - prev_t )); (( dt <= 0 )) && dt=1
        rate=$(( (cur_b - prev_b) / dt )); (( rate < 0 )) && rate=0
        if (( first == 1 )); then ema="$rate"; first=0
        else ema=$(( (ema * 7 + rate * 3) / 10 )); fi
        if (( rate == 0 )); then stalled=$(( stalled + dt )); else stalled=0; fi

        pct=0
        if (( exp_b > 0 )); then pct=$(( cur_b * 100 / exp_b ))
        elif (( exp_f > 0 )); then pct=$(( cur_f * 100 / exp_f ))
        fi
        eta=""
        (( ema > 0 && exp_b > cur_b )) && eta="$(human_dur $(( (exp_b - cur_b) / ema )))"

        line="progress $(printf '%3s' "$pct")%  $(human_bytes "$cur_b")/$( (( exp_b > 0 )) && human_bytes "$exp_b" || printf '?' )"
        line="$line  $cur_f/$exp_f files  $(human_bytes "$ema")/s"
        [[ -n "$eta" ]] && line="$line  ETA $eta"
        line="$line  ($done_hosts done, $running running, $waiting queued)"
        (( stalled >= 300 )) && line="$line  STALLED $(human_dur "$stalled")"

        {
            printf 'PiCameraArray collection   run=%s   session=%s\n' "$RUN_ID" "$session"
            printf 'updated %s   elapsed %s\n' \
                "$(date '+%Y-%m-%d %H:%M:%S')" "$(human_dur $(( now - start_t )))"
            printf '%s\n\n' "$line"
            printf '  %-4s %-10s %13s %19s %4s %s\n' \
                HOST STATE FILES BYTES '' 'PROGRESS'
            cat "$PROGRESS_FILE.tmp"
        } > "$PROGRESS_FILE.new"
        mv -f "$PROGRESS_FILE.new" "$PROGRESS_FILE"
        rm -f "$PROGRESS_FILE.tmp"
        log INFO "$line"

        prev_b="$cur_b"; prev_t="$now"

        # every host reported a final status -> the transfer is over
        local pending=0
        for h in "${HOSTS[@]}"; do
            [[ -s "$STATUS_DIR/$h" ]] || { pending=1; break; }
        done
        (( pending == 0 )) && break
    done
}

# --------------------------- per-host worker ---------------------------------
# Writes one word (ok|partial|skipped|unreachable|failed) to $STATUS_DIR/$host.
# Always returns 0 -- the caller must never die because of one Pi.
process_pi() {
    local host="$1" session="$2"
    local hostlog="$LOG_DIR/${host}_${RUN_ID}.log"
    local target="" rdir="" rname="" rcount=""

    # The survey already resolved the host and chose which session to take;
    # reuse that verdict so the transfer copies exactly what was reviewed.
    if [[ -s "$SURVEY_DIR/$host.pick" ]]; then
        IFS=$'\t' read -r rdir rname rcount _ < "$SURVEY_DIR/$host.pick"
        target="$(cat "$SURVEY_DIR/$host.target" 2>/dev/null)"
    fi

    if [[ -z "$target" ]] && ! target="$(resolve_target "$host")"; then
        log WARN "$host : UNREACHABLE (ssh timeout/refused/auth) -- skipped, continuing"
        fail_note "$host unreachable"
        echo unreachable > "$STATUS_DIR/$host"
        return 0
    fi
    [[ "$target" != "${host}${HOST_SUFFIX}" ]] && \
        log INFO "$host : resolved as '$target'"

    if [[ -z "$rname" ]]; then
        rdir="$(remote_data_dir "$target")"
        rname="$(latest_session "$target" "$rdir")"
        if [[ -z "$rname" ]]; then
            log WARN "$host : no session directory under $rdir -- skipped"
            fail_note "$host no-session-dir"
            echo failed > "$STATUS_DIR/$host"
            return 0
        fi
        # Prefer the reference session name when this host actually has it.
        if [[ -n "$session" && "$rname" != "$session" ]]; then
            if [[ "$(ssh_run "$target" "test -d '$rdir/$session' && echo yes")" == "yes" ]]; then
                rname="$session"
            else
                # without a survey nothing else checks this: the latest session
                # may be a later run, and it must not land in this collection
                local se re gap
                se="$(session_epoch "$session")"; re="$(session_epoch "$rname")"
                gap=$(( re > se ? re - se : se - re ))
                if (( se > 0 && re > 0 && gap > MATCH_WINDOW )); then
                    log ERROR "$host : its latest session '$rname' starts $(human_dur "$gap") from '$session' -- a different run, skipped"
                    fail_note "$host wrong-run"
                    echo failed > "$STATUS_DIR/$host"
                    return 0
                fi
                log WARN "$host : session '$session' absent, using its latest '$rname'"
            fi
        fi
    fi

    local dest="$DATA_ROOT/${session:-$rname}/${host}_${rname}"
    local lcount rbytes=0
    if [[ -s "$SURVEY_DIR/$host.pick" ]]; then
        rbytes="$(cut -f4 "$SURVEY_DIR/$host.pick")"
    fi
    # Without a survey (--no-check) nobody has asked the Pi yet. Ask now, for
    # files AND bytes: the byte total is what tells a finished host apart from
    # one whose files are all present but one of them cut short.
    if [[ -z "$rcount" || "${rbytes:-0}" -le 0 ]]; then
        read -r rcount rbytes <<< "$(remote_usage "$target" "$rdir" "$rname")"
    fi
    rcount="${rcount:-0}"; rbytes="${rbytes:-0}"
    # tell the progress monitor where to look and what "finished" means here
    printf '%s\t%s\t%s\n' "$dest" "$rcount" "${rbytes:-0}" > "$SURVEY_DIR/$host.expect"
    local lbytes
    read -r lcount lbytes <<< "$(local_usage "$dest")"

    if [[ -d "$dest" && "$rcount" -gt 0 && "$rcount" -eq "$lcount" ]]; then
        if [[ "${rbytes:-0}" -le 0 || "$lbytes" -eq "$rbytes" ]]; then
            log INFO "$host : already complete ($lcount files) -- skipped"
            echo skipped > "$STATUS_DIR/$host"
            return 0
        fi
        # Same number of files, fewer bytes: one of them was cut short. rsync
        # sees the size difference and fetches those files again.
        log WARN "$host : $lcount/$rcount files but $(( rbytes - lbytes )) byte(s) short -- repairing"
    fi

    log INFO "$host : syncing $target:$rdir/$rname -> $dest (remote=$rcount local=$lcount)"
    : > "$SURVEY_DIR/$host.started"   # the progress monitor reads this as "running"
    [[ $DRY_RUN -eq 0 ]] && mkdir -p "$dest"

    local rsync_opts=(-a --partial --inplace --timeout=600 --info=stats1
                      -e "ssh $(printf '%s ' "${SSH_OPTS[@]}")")
    [[ $DRY_RUN -eq 1 ]] && rsync_opts+=(--dry-run)

    local attempt=1 rc=1
    while (( attempt <= RETRIES + 1 )); do
        printf '\n=== %s attempt %d/%d %s ===\n' \
            "$host" "$attempt" "$((RETRIES + 1))" "$(date '+%F %T')" >> "$hostlog"
        "${SSH_WRAP[@]}" rsync "${rsync_opts[@]}" \
            "${SSH_USER}@${target}:$rdir/$rname/" "$dest/" >>"$hostlog" 2>&1
        rc=$?
        (( rc == 0 )) && break
        log WARN "$host : rsync attempt $attempt failed (rc=$rc)"
        attempt=$((attempt + 1))
        (( attempt <= RETRIES + 1 )) && sleep 5
    done

    if (( rc == 0 )); then
        local after afterbytes; read -r after afterbytes <<< "$(local_usage "$dest")"
        if [[ $DRY_RUN -eq 1 ]]; then
            log INFO "$host : dry-run OK (would sync $rcount file(s))"
            echo ok > "$STATUS_DIR/$host"
        elif [[ "$rcount" -gt 0 && "$after" -lt "$rcount" ]]; then
            log WARN "$host : PARTIAL ($after/$rcount files) -- see $hostlog"
            fail_note "$host partial:$after/$rcount"
            echo partial > "$STATUS_DIR/$host"
        elif [[ "${rbytes:-0}" -gt 0 && "$afterbytes" -lt "$rbytes" ]]; then
            log WARN "$host : PARTIAL ($after/$rcount files but $(( rbytes - afterbytes )) byte(s) short) -- see $hostlog"
            fail_note "$host short-bytes:$afterbytes/$rbytes"
            echo partial > "$STATUS_DIR/$host"
        else
            log INFO "$host : OK ($after files, $(human_bytes "$afterbytes"))"
            echo ok > "$STATUS_DIR/$host"
        fi
    else
        log ERROR "$host : rsync FAILED after $((RETRIES + 1)) attempt(s) (rc=$rc) -- skipped, see $hostlog"
        fail_note "$host rsync-failed rc=$rc"
        echo failed > "$STATUS_DIR/$host"
    fi
    return 0
}

# --------------------------------- main --------------------------------------
log INFO "======================================================================"
log INFO "PiCameraArray data collection (shell)  run=$RUN_ID  pid=$$"
log INFO "hosts=${HOSTS[*]}"
log INFO "data root=$DATA_ROOT  link root=$LINK_ROOT  jobs=$MAX_JOBS  retries=$RETRIES  dry-run=$DRY_RUN"
log INFO "======================================================================"

while read -r line; do log INFO "free space: $line"; done \
    < <(df -h "$DATA_ROOT" 2>/dev/null | tail -1)

# 1) pre-flight survey: look before leaping
SURVEY_RC=0
if [[ "$CHECK_MODE" != off ]]; then
    run_survey || SURVEY_RC=$?
    if [[ "$CHECK_MODE" == only ]]; then
        log INFO "--check: survey only, nothing was transferred"
        log INFO "log file: $MAIN_LOG"
        exit "$SURVEY_RC"
    fi
    if (( SURVEY_RC != 0 && ASSUME_YES == 0 )); then
        if [[ -t 0 && -z "${TMUX:-}" ]]; then
            read -r -p "The survey reported warnings. Transfer anyway? [y/N] " ans
            if [[ ! "$ans" =~ ^[yY] ]]; then
                log INFO "aborted by the user after the pre-flight survey"
                exit 1
            fi
        else
            log ERROR "the pre-flight survey reported warnings and nobody can answer a prompt here."
            log ERROR "  review the report above, then re-run with -y to accept it as it is,"
            log ERROR "  or narrow the run with --hosts / --session / --include-mismatched."
            exit 1
        fi
    fi
fi

# 2) decide the session name -- the survey's choice, else first reachable host
SESSION="$FORCE_SESSION"
[[ -z "$SESSION" ]] && SESSION="$SURVEY_SESSION"
if [[ -z "$SESSION" ]]; then
    for h in "${HOSTS[@]}"; do
        if t="$(resolve_target "$h")"; then
            d="$(remote_data_dir "$t")"
            SESSION="$(latest_session "$t" "$d")"
            if [[ -n "$SESSION" ]]; then
                log INFO "reference host $h ($t) -> session '$SESSION' (under $d)"
                break
            fi
            log WARN "$h : reachable but no session directory under $d"
        else
            log WARN "$h : UNREACHABLE while looking for a reference host"
        fi
    done
fi
if [[ -z "$SESSION" ]]; then
    log ERROR "no reachable host could provide a session name -- aborting"
    exit 1
fi

# 3) real storage under $DATA_ROOT + symlink the analysis code reads through
COLLECTION="$DATA_ROOT/$SESSION"
if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$COLLECTION" || { log ERROR "cannot create $COLLECTION"; exit 1; }
    if [[ -L "$LINK_ROOT" ]]; then
        log INFO "link root is itself a symlink -> $(readlink -f "$LINK_ROOT"); no per-session link needed"
    else
        mkdir -p "$LINK_ROOT"
        if [[ -d "$LINK_ROOT/$SESSION" && ! -L "$LINK_ROOT/$SESSION" ]]; then
            if rmdir "$LINK_ROOT/$SESSION" 2>/dev/null; then
                ln -sfn "$COLLECTION" "$LINK_ROOT/$SESSION"
                log INFO "symlink: $LINK_ROOT/$SESSION -> $COLLECTION"
            else
                log WARN "$LINK_ROOT/$SESSION is a real non-empty directory -- symlink NOT created"
                log WARN "  move or remove it, then re-run; data still goes to $COLLECTION"
            fi
        else
            ln -sfn "$COLLECTION" "$LINK_ROOT/$SESSION"
            log INFO "symlink: $LINK_ROOT/$SESSION -> $COLLECTION"
        fi
    fi
fi

# 4) parallel collection; per-host failures are logged, never fatal
PROGRESS_FILE="$LOG_DIR/get_data_${RUN_ID}.progress"
# Seed the monitor with the survey's expectations so hosts that have not
# started yet already show their real denominator instead of 0/0.
if [[ "$CHECK_MODE" != off ]]; then
    for h in "${HOSTS[@]}"; do
        [[ -s "$SURVEY_DIR/$h.pick" ]] || continue
        IFS=$'\t' read -r _ pname pcount pbytes < "$SURVEY_DIR/$h.pick"
        printf '%s\t%s\t%s\n' "$DATA_ROOT/$SESSION/${h}_${pname}" "$pcount" "$pbytes" \
            > "$SURVEY_DIR/$h.expect"
    done
fi
read -r _ TRANSFER_START_BYTES <<< "$(dest_usage "$COLLECTION")"
TRANSFER_START_TIME="$(date +%s)"
if (( PROGRESS_INTERVAL > 0 && DRY_RUN == 0 )); then
    progress_monitor "$SESSION" &
    PROGRESS_PID=$!
    log INFO "progress every ${PROGRESS_INTERVAL}s in this log, live table: $PROGRESS_FILE"
    log INFO "  watch it from anywhere with:  watch -n5 cat $PROGRESS_FILE"
fi

running=0
for h in "${HOSTS[@]}"; do
    # hosts the survey ruled out are never contacted again
    if [[ "$CHECK_MODE" != off && ! -s "$SURVEY_DIR/$h.pick" ]]; then
        echo excluded > "$STATUS_DIR/$h"
        continue
    fi
    process_pi "$h" "$SESSION" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
done
wait

if [[ -n "$PROGRESS_PID" ]]; then
    kill "$PROGRESS_PID" 2>/dev/null
    wait "$PROGRESS_PID" 2>/dev/null
fi

# 5) summary
declare -A COUNT=([ok]=0 [partial]=0 [skipped]=0 [unreachable]=0 [failed]=0 [excluded]=0)
for h in "${HOSTS[@]}"; do
    s="$(cat "$STATUS_DIR/$h" 2>/dev/null)"
    [[ -z "$s" ]] && s=failed
    COUNT[$s]=$(( ${COUNT[$s]:-0} + 1 ))
done

log INFO "----------------------------------------------------------------------"
log INFO "SUMMARY  session=$SESSION"
log INFO "  copied            : ${COUNT[ok]}"
log INFO "  partial           : ${COUNT[partial]}"
log INFO "  already complete  : ${COUNT[skipped]}"
log INFO "  unreachable       : ${COUNT[unreachable]}"
log INFO "  failed            : ${COUNT[failed]}"
log INFO "  excluded by check : ${COUNT[excluded]}"
read -r TRANSFER_END_FILES TRANSFER_END_BYTES <<< "$(dest_usage "$COLLECTION")"
TRANSFER_SECS=$(( $(date +%s) - TRANSFER_START_TIME )); (( TRANSFER_SECS < 1 )) && TRANSFER_SECS=1
TRANSFER_BYTES=$(( TRANSFER_END_BYTES - TRANSFER_START_BYTES )); (( TRANSFER_BYTES < 0 )) && TRANSFER_BYTES=0
log INFO "  fetched this run  : $(human_bytes "$TRANSFER_BYTES") in $(human_dur "$TRANSFER_SECS") (avg $(human_bytes $(( TRANSFER_BYTES / TRANSFER_SECS )))/s)"
log INFO "  on disk now       : $(human_bytes "$TRANSFER_END_BYTES") in $TRANSFER_END_FILES file(s)"
log INFO "  data              : $COLLECTION"
log INFO "  read path         : $LINK_ROOT/$SESSION"
if [[ -s "$FAIL_LOG" ]]; then
    log WARN "problem hosts (also in $FAIL_LOG):"
    while read -r l; do log WARN "  $l"; done < "$FAIL_LOG"
    log WARN "re-run the same command later to pick these up; finished hosts are skipped"
else
    log INFO "all hosts processed without error"
fi
log INFO "log file: $MAIN_LOG"
log INFO "----------------------------------------------------------------------"

exit 0