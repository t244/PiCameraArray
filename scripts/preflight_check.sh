#!/usr/bin/env bash
#
# preflight_check.sh - report the readiness of one Pi as a single line of
# key=value pairs. Read-only; changes nothing.
#
# Emitted keys:
#   host     hostname
#   svc      picamera-capture service state
#   ssd      yes | NO   - is the SSD actually mounted (not just the directory)
#   free     free space on the SSD
#   fstab    number of LABEL=HIKSEMI entries in /etc/fstab (expect 1)
#   wifi     comma-separated "profile/priority" for every wifi profile
#   mode     capture | preview
#   count    frames captured this session (0 = no trigger arriving)
#   bursts   bursts written this session
#   arduino  True | False - serial link detected by the agent
#   sess     session directory in use
#   temp     CPU temperature
#
set -uo pipefail

MOUNT="${MOUNT:-/media/pi/HIKSEMI}"

# Several of the commands below print a value AND exit non-zero (systemctl
# is-active on a stopped unit, grep -c with no match). Take the first line and
# fall back only when it is genuinely empty, so no stray newline sneaks into
# the single-line output contract.
first() { head -n 1 | tr -d '\r\n'; }

out="host=$(hostname | first)"

svc="$(systemctl is-active picamera-capture 2>/dev/null | first)"
out+=";svc=${svc:-unknown}"

if mountpoint -q "$MOUNT"; then
    out+=";ssd=yes"
    out+=";free=$(df -h --output=avail "$MOUNT" 2>/dev/null | tail -1 | tr -d ' \r\n')"
else
    out+=";ssd=NO;free=-"
fi

fstab="$(grep -c '^LABEL=HIKSEMI[[:space:]]' /etc/fstab 2>/dev/null | first)"
out+=";fstab=${fstab:-0}"

wifi="$(nmcli -t -f NAME,TYPE connection show 2>/dev/null \
        | awk -F: '$NF=="802-11-wireless"{NF--; print}' OFS=: \
        | while IFS= read -r n; do
              [[ -n "$n" ]] || continue
              p="$(nmcli -g connection.autoconnect-priority connection show "$n" 2>/dev/null || echo '?')"
              printf '%s/%s,' "$n" "$p"
          done)"
out+=";wifi=${wifi%,}"

status="$(curl -s -m 5 localhost:8000/status 2>/dev/null || true)"
if [[ -n "$status" ]]; then
    agent="$(printf '%s' "$status" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print("mode=?;count=-1;bursts=-1;arduino=?;sess=?;temp=?")
    sys.exit()
t = d.get("cpu_temp")
print("mode=%s;count=%s;bursts=%s;arduino=%s;sess=%s;temp=%s" % (
    d.get("mode", "?"),
    d.get("capture_count", -1),
    d.get("burst_count", -1),
    d.get("has_arduino", "?"),
    d.get("session_dir", "?"),
    ("%.1f" % t) if isinstance(t, (int, float)) else "?",
))' 2>/dev/null)"
    out+=";${agent:-mode=?;count=-1;bursts=-1;arduino=?;sess=?;temp=?}"
else
    out+=";mode=DOWN;count=-1;bursts=-1;arduino=?;sess=?;temp=?"
fi

# One line, always - the caller parses it as key=value pairs.
printf '%s\n' "$(printf '%s' "$out" | tr -d '\r\n')"
