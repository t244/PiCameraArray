#!/bin/bash
# setup_ready_line.sh - hold the shared READY line LOW from early boot.
#
# Each Pi drives GPIO26 (header pin 37) onto the READY line read by the
# trigger Arduino (see capture/exposure_trigger/exposure_trigger.ino).
# The line must be LOW until camera_agent.py is recording; without this the
# pin floats as an input for the first seconds after power-on and the
# Arduino could briefly see "all ready". Idempotent; reboot to apply.
#
#   sudo bash setup_ready_line.sh
set -euo pipefail
CFG=/boot/firmware/config.txt
[ -f "$CFG" ] || CFG=/boot/config.txt
LINE="gpio=26=op,dl"
if grep -qxF "$LINE" "$CFG"; then
    echo "already set in $CFG"
else
    printf '\n# Shared READY line to the trigger Arduino: LOW until recording\n%s\n' "$LINE" >> "$CFG"
    echo "added '$LINE' to $CFG - reboot to apply"
fi
command -v pinctrl >/dev/null || command -v raspi-gpio >/dev/null \
    || echo "WARNING: neither pinctrl nor raspi-gpio found; camera_agent.py cannot drive the line"
