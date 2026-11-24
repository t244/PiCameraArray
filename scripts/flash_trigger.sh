#!/usr/bin/env bash
#
# flash_trigger.sh - build and upload exposure_trigger.ino to the Uno attached
# to this Pi (normally e00), then write the capture settings to its EEPROM.
#
#   sudo bash flash_trigger.sh <sketch_dir>
#
# Environment overrides:
#   EXPOSURE_US (30000)  PERIOD_MS (120000)  FPS (30)  BURST_MS (2000)
#   PORT  (auto: first /dev/ttyACM* or /dev/ttyUSB*)
#   FQBN  (auto from "arduino-cli board list"; the trigger box is an
#          Arduino UNO R4 Minima = arduino:renesas_uno:minima)
#   CLI   (arduino-cli path; auto-installed to /usr/local/bin if missing and
#          the Pi has internet access)
#
# The capture agent holds the serial port, so it is stopped for the upload
# and started again at the end (also on failure).
set -euo pipefail

SKETCH="${1:?usage: flash_trigger.sh <sketch_dir>}"
EXPOSURE_US="${EXPOSURE_US:-30000}"
PERIOD_MS="${PERIOD_MS:-120000}"
FPS="${FPS:-30}"
BURST_MS="${BURST_MS:-2000}"
FQBN="${FQBN:-}"   # auto-detected from arduino-cli board list when empty
HOST="$(hostname)"
log() { echo "[$HOST] $*"; }
die() { echo "[$HOST] FAIL: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root (sudo)"
[[ -f "$SKETCH/$(basename "$SKETCH").ino" ]] || die "no $(basename "$SKETCH").ino in $SKETCH"

# --- arduino-cli ------------------------------------------------------------
CLI="${CLI:-$(command -v arduino-cli || true)}"
[[ -z "$CLI" && -x /usr/local/bin/arduino-cli ]] && CLI=/usr/local/bin/arduino-cli
if [[ -z "$CLI" ]]; then
    log "arduino-cli not found - installing"
    curl -fsSL --max-time 60 https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh \
        | BINDIR=/usr/local/bin sh >/dev/null \
        || die "could not download arduino-cli (no internet? join the lab Wi-Fi and retry)"
    CLI=/usr/local/bin/arduino-cli
fi
PORT="${PORT:-$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1 || true)}"
[[ -n "$PORT" ]] || die "no Arduino serial port found (USB cable?)"
if [[ -z "$FQBN" ]]; then
    FQBN="$("$CLI" board list 2>/dev/null | awk -v p="$PORT" '$1==p{for(i=1;i<=NF;i++) if($i ~ /^[a-z0-9_]+:[a-z0-9_]+:[a-z0-9_]+$/){print $i; exit}}')"
    FQBN="${FQBN:-arduino:renesas_uno:minima}"
fi
CORE="${FQBN%:*}"
log "board on $PORT: $FQBN"
if ! "$CLI" core list 2>/dev/null | awk '{print $1}' | grep -qxF "$CORE"; then
    log "installing core $CORE"
    "$CLI" core update-index >/dev/null && "$CLI" core install "$CORE" >/dev/null \
        || die "could not install core $CORE (no internet? join the lab Wi-Fi and retry)"
fi

# --- build first, so a compile error never leaves the agent stopped ---------
log "compiling $SKETCH"
"$CLI" compile --fqbn "$FQBN" "$SKETCH" >/tmp/flash_trigger_build.log 2>&1 \
    || { tail -20 /tmp/flash_trigger_build.log; die "compile failed"; }

restart_agent() { systemctl start picamera-capture 2>/dev/null || true; }
trap restart_agent EXIT
log "stopping picamera-capture"
systemctl stop picamera-capture
sleep 2

log "uploading to $PORT"
"$CLI" upload --fqbn "$FQBN" -p "$PORT" "$SKETCH" >/tmp/flash_trigger_upload.log 2>&1 \
    || { tail -20 /tmp/flash_trigger_upload.log; die "upload failed"; }
sleep 4
PORT="$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1 || true)"
[[ -n "$PORT" ]] || die "board did not come back after upload"

# --- settings (saved to EEPROM by the firmware) -----------------------------
log "writing settings E=$EXPOSURE_US T=$PERIOD_MS F=$FPS B=$BURST_MS"
python3 - "$PORT" "$EXPOSURE_US" "$PERIOD_MS" "$FPS" "$BURST_MS" <<'PY' || die "settings not confirmed by R"
import sys, time, serial
port, e, t, f, b = sys.argv[1:]
s = serial.Serial(port, 115200, timeout=1)
time.sleep(2.5)                      # an R3 resets on open; harmless on R4
s.reset_input_buffer()
for cmd in (f"E{e}", f"T{t}", f"F{f}", f"B{b}", "R"):
    s.write((cmd + "\n").encode())
    time.sleep(0.4)
time.sleep(0.5)
out = s.read(4096).decode(errors="replace")
print(out.strip())
s.close()
want = {"EXP": e, "PERIOD": t}
bad = [k for k, v in want.items() if f"{k} {v}" not in out]
sys.exit(1 if bad else 0)
PY

log "OK - firmware uploaded and settings stored"
