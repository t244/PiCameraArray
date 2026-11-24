#!/bin/bash
# setup_ready_line.sh - hold the shared READY line LOW from early boot.
#
# Each Pi drives GPIO21 (header pin 40; ground on pin 34) onto the READY line read by the
# trigger Arduino (see capture/exposure_trigger/exposure_trigger.ino).
# The line must be LOW until camera_agent.py is recording; without this the
# pin floats as an input for the first seconds after power-on and the
# Arduino could briefly see "all ready". Idempotent; reboot to apply.
#
#   sudo bash setup_ready_line.sh
set -euo pipefail
CFG=/boot/firmware/config.txt
[ -f "$CFG" ] || CFG=/boot/config.txt
LINE="gpio=21=op,dl"
OLD_LINES=("gpio=26=op,dl")   # earlier READY pin (37); removed if present
for old in "${OLD_LINES[@]}"; do
    if grep -qxF "$old" "$CFG"; then
        sed -i "\|^${old}\$|d" "$CFG"
        echo "removed old '$old' from $CFG"
    fi
done
if grep -qxF "$LINE" "$CFG"; then
    echo "already set in $CFG"
else
    printf '\n# Shared READY line to the trigger Arduino: LOW until recording\n%s\n' "$LINE" >> "$CFG"
    echo "added '$LINE' to $CFG - reboot to apply"
fi
command -v pinctrl >/dev/null || command -v raspi-gpio >/dev/null \
    || echo "WARNING: neither pinctrl nor raspi-gpio found; camera_agent.py cannot drive the line"
