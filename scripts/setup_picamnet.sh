#!/usr/bin/env bash
#
# setup_picamnet.sh - register the field pocket-router SSIDs on a Pi and give
# them priority over the lab WiFi.
#
# The pocket router (GL.iNet) broadcasts a separate SSID per band, typically
# "PiCamNet" on 2.4 GHz and "PiCamNet5G" on 5 GHz. Both bridge to the same
# LAN, so it does not matter which band a given device lands on. We register
# a profile for each and let NetworkManager pick whichever is in range.
#
# SAFE TO RUN OVER SSH FROM THE LAB NETWORK. Profiles are created but are
# deliberately NOT activated. NetworkManager only evaluates
# autoconnect-priority when it has to choose a connection (boot, or after the
# current one drops) and never roams away from a healthy link, so the live
# lab connection is untouched.
#
# Usage on the Pi:
#     sudo bash setup_picamnet.sh
#
# Environment overrides:
#     SSIDS      space-separated SSIDs, highest priority first
#                (default: "PiCamNet5G PiCamNet")
#     PSK        WPA2 passphrase, shared by all of them (default: PiCamNet)
#     PRIORITY   priority of the first SSID; each later one gets one less
#                (default: 100; higher wins)
#     STATIC_IP  e.g. 192.168.8.112/24    (default: empty -> DHCP)
#     GATEWAY    used only with STATIC_IP (default: 192.168.8.1)
#
set -euo pipefail

SSIDS="${SSIDS:-PiCamNet5G PiCamNet}"
PSK="${PSK:-PiCamNet}"
PRIORITY="${PRIORITY:-100}"
STATIC_IP="${STATIC_IP:-}"
GATEWAY="${GATEWAY:-192.168.8.1}"

HOST="$(hostname)"
log() { echo "[$HOST] $*"; }
die() { echo "[$HOST] ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root (use sudo)"

command -v nmcli >/dev/null 2>&1 \
    || die "nmcli not found - this script needs NetworkManager (Raspberry Pi OS Bookworm+)"

# WPA2-PSK requires 8..63 characters.
[[ ${#PSK} -ge 8 && ${#PSK} -le 63 ]] \
    || die "passphrase must be 8..63 characters (got ${#PSK})"

read -r -a SSID_LIST <<< "$SSIDS"
[[ ${#SSID_LIST[@]} -gt 0 ]] || die "SSIDS is empty"

WIFI_DEV="$(nmcli -t -f DEVICE,TYPE device | awk -F: '$2=="wifi"{print $1; exit}')"
[[ -n "$WIFI_DEV" ]] || die "no wifi device found"

ACTIVE_BEFORE="$(nmcli -t -f NAME,DEVICE connection show --active | tr '\n' ' ')"
log "wifi device: $WIFI_DEV"
log "active before: ${ACTIVE_BEFORE:-none}"

# Helper: list connection names of type wifi (handles names containing ':')
wifi_profiles() {
    nmcli -t -f NAME,TYPE connection show \
        | awk -F: '$NF=="802-11-wireless"{NF--; print}' OFS=:
}

# ---------------------------------------------------------------------------
# 1. Create or update one profile per SSID, descending priority
# ---------------------------------------------------------------------------
prio="$PRIORITY"
for ssid in "${SSID_LIST[@]}"; do
    if nmcli -t -f NAME connection show | grep -qxF "$ssid"; then
        log "profile '$ssid' exists - updating in place"
    else
        log "creating profile '$ssid'"
        nmcli connection add type wifi con-name "$ssid" \
            ifname "$WIFI_DEV" ssid "$ssid" >/dev/null
    fi

    nmcli connection modify "$ssid" \
        connection.interface-name "$WIFI_DEV" \
        802-11-wireless.ssid "$ssid" \
        802-11-wireless.mode infrastructure \
        802-11-wireless-security.key-mgmt wpa-psk \
        802-11-wireless-security.psk "$PSK" \
        connection.autoconnect yes \
        connection.autoconnect-retries 0 \
        connection.autoconnect-priority "$prio"

    if [[ -n "$STATIC_IP" ]]; then
        nmcli connection modify "$ssid" \
            ipv4.method manual ipv4.addresses "$STATIC_IP" ipv4.gateway "$GATEWAY"
    else
        nmcli connection modify "$ssid" ipv4.method auto
    fi

    # The pocket router has no uplink; a missing DNS server must not stall boot.
    nmcli connection modify "$ssid" ipv4.may-fail yes ipv6.method ignore

    log "configured '$ssid' priority=$prio"
    prio=$((prio - 1))
done

LOWEST="$prio"   # anything at or above this belongs to us

# ---------------------------------------------------------------------------
# 2. Make sure no other WiFi profile can outrank ours. Relative order of the
#    remaining profiles is preserved, so the lab network stays usable as a
#    fallback when the array comes back.
# ---------------------------------------------------------------------------
is_ours() {
    local n="$1"
    for s in "${SSID_LIST[@]}"; do [[ "$n" == "$s" ]] && return 0; done
    return 1
}

while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    is_ours "$name" && continue
    cur="$(nmcli -g connection.autoconnect-priority connection show "$name" 2>/dev/null || true)"
    [[ "$cur" =~ ^-?[0-9]+$ ]] || cur=0
    if [[ "$cur" -gt "$LOWEST" ]]; then
        nmcli connection modify "$name" connection.autoconnect-priority 0
        log "demoted '$name' (priority $cur -> 0)"
    else
        log "kept    '$name' (priority $cur) as fallback"
    fi
done < <(wifi_profiles)

# ---------------------------------------------------------------------------
# 3. mDNS - the laptop reaches the array as e00.local .. e15.local
# ---------------------------------------------------------------------------
if systemctl list-unit-files 2>/dev/null | grep -q '^avahi-daemon'; then
    systemctl enable --now avahi-daemon >/dev/null 2>&1 || true
    log "avahi-daemon: $(systemctl is-active avahi-daemon 2>/dev/null || echo unknown)"
else
    log "WARNING: avahi-daemon not installed; ${HOST}.local will not resolve."
    log "         Install before leaving:  sudo apt install -y avahi-daemon"
fi

# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------
log "--- wifi profiles (autoconnect priority, higher wins) ---"
while IFS= read -r n; do
    [[ -n "$n" ]] || continue
    p="$(nmcli -g connection.autoconnect-priority connection show "$n" 2>/dev/null || echo '?')"
    a="$(nmcli -g connection.autoconnect connection show "$n" 2>/dev/null || echo '?')"
    printf '[%s]     %-28s priority=%-5s autoconnect=%s\n' "$HOST" "$n" "$p" "$a"
done < <(wifi_profiles)

log "--- SSIDs currently visible ---"
nmcli -t -f SSID,CHAN,SIGNAL device wifi list --rescan yes 2>/dev/null \
    | awk -F: 'NF>=3 && $1!=""{printf "  %-24s ch=%-4s signal=%s\n", $1, $2, $3}' \
    | sort -u | sed "s/^/[$HOST]   /" || log "  (scan failed)"

for ssid in "${SSID_LIST[@]}"; do
    if nmcli -t -f SSID device wifi list 2>/dev/null | grep -qxF "$ssid"; then
        log "OK: '$ssid' is in range"
    else
        log "WARNING: '$ssid' NOT visible from this Pi"
    fi
done

ACTIVE_AFTER="$(nmcli -t -f NAME,DEVICE connection show --active | tr '\n' ' ')"
log "active after:  ${ACTIVE_AFTER:-none}"

if [[ "$ACTIVE_BEFORE" != "$ACTIVE_AFTER" ]]; then
    log "NOTE: the active connection changed during this run."
else
    log "OK - current connection untouched. NetworkManager will pick the"
    log "     highest-priority SSID that is in range at the next boot, or as"
    log "     soon as the current link drops."
fi
