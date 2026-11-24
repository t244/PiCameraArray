<#
.SYNOPSIS
    Apply the READY-line settings and reinstall the capture service on every Pi.

.DESCRIPTION
    On each Pi (repo already pulled, e.g. with deploy_pull.ps1):
      1. sudo bash scripts/setup_ready_line.sh
         -> adds "gpio=21=op,dl" to config.txt so GPIO21 (READY, pin 40) is LOW from
            early boot. Takes effect after a reboot.
      2. copies config/picamera-capture.service to /etc/systemd/system and
         reloads systemd (the unit pulls READY LOW when the agent stops).
    Then reports, per Pi, whether config.txt has the line and the installed
    unit matches the repo copy.

    -Reboot reboots each Pi afterwards. Without it, remember to reboot before
    the READY line is relied on.
    -Verify only reports; nothing is changed.

    Works on Windows PowerShell 5.1 and PowerShell 7+. Needs plink (PuTTY).

.EXAMPLE
    .\deploy_ready_line.ps1 -Only 0      # one Pi first
    .\deploy_ready_line.ps1 -Reboot      # all 16, then reboot
    .\deploy_ready_line.ps1 -Verify      # check only
#>
[CmdletBinding()]
param(
    [int]    $First      = 0,
    [int]    $Last       = 15,
    [int[]]  $Only,
    [string] $HostFormat = "e{0:D2}",
    [string] $User       = "pi",
    [string] $Password   = "pi",
    [string] $RepoDir    = "/home/pi/PiCameraArray",
    [switch] $Reboot,
    [switch] $Verify
)

if (-not (Get-Command plink -ErrorAction SilentlyContinue)) {
    throw "plink not found on PATH. Install the PuTTY tools first."
}

# Single-quoted here-strings: PowerShell expands nothing inside; __REPO__ is
# substituted below. Lines are joined with '; ' for plink.
$apply = @'
cd '__REPO__' || { echo 'FAIL: repo not found'; exit 2; }
sudo sed -i 's/\r//g' scripts/setup_ready_line.sh
sudo bash scripts/setup_ready_line.sh || { echo 'FAIL: setup_ready_line.sh'; exit 3; }
sudo cp config/picamera-capture.service /etc/systemd/system/picamera-capture.service || { echo 'FAIL: copy unit'; exit 4; }
sudo systemctl daemon-reload
'@

$check = @'
cd '__REPO__' 2>/dev/null
CFG=/boot/firmware/config.txt; [ -f "$CFG" ] || CFG=/boot/config.txt
if grep -qxF 'gpio=21=op,dl' "$CFG"; then echo 'config=ok'; else echo 'config=MISSING'; fi
if cmp -s config/picamera-capture.service /etc/systemd/system/picamera-capture.service; then echo 'unit=ok'; else echo 'unit=DIFFERS'; fi
echo "head=$(git log -1 --format=%h 2>/dev/null) svc=$(systemctl is-active picamera-capture 2>/dev/null)"
'@

function Join-Remote([string]$text) {
    (($text -replace '__REPO__', $RepoDir) -replace "`r", "") -replace "`n", "; "
}

$cmd = if ($Verify) { Join-Remote $check } else { Join-Remote ($apply + "`n" + $check) }
if ($Reboot -and -not $Verify) { $cmd += "; echo 'rebooting'; sudo systemctl reboot" }

# Not "if ($Only)": an array holding the single value 0 counts as false, and
# "-Only 0" would then mean all hosts.
$indices = if ($PSBoundParameters.ContainsKey('Only')) { $Only } else { $First..$Last }
$mode = if ($Verify) { 'VERIFY' } elseif ($Reboot) { 'APPLY + REBOOT' } else { 'APPLY' }
Write-Host "=== $mode : $($indices.Count) hosts ===" -ForegroundColor Cyan

$results = @()
foreach ($i in $indices) {
    $h = $HostFormat -f $i
    Write-Host "[$h] ..." -ForegroundColor DarkGray
    $out = (plink -pw $Password -batch "$User@$h" $cmd 2>&1 | Out-String).TrimEnd()
    # A reboot drops the session, so judge by the report lines, not the exit code.
    $ok = ($out -match 'config=ok') -and ($out -match 'unit=ok') -and ($out -notmatch 'FAIL')
    $results += [pscustomobject]@{ Host = $h; Ok = $ok; Output = $out }
}

Write-Host ""
foreach ($r in $results) {
    $c = if ($r.Ok) { 'Green' } else { 'Red' }
    Write-Host "=== $($r.Host) === $(if ($r.Ok) {'OK'} else {'CHECK'})" -ForegroundColor $c
    Write-Host $r.Output
}
$bad = @($results | Where-Object { -not $_.Ok })
Write-Host ""
if ($bad.Count -eq 0) {
    Write-Host "All $($results.Count) Pis OK." -ForegroundColor Green
    if (-not $Reboot -and -not $Verify) {
        Write-Host "Reboot the Pis for gpio=21=op,dl to take effect (or rerun with -Reboot)." -ForegroundColor Yellow
    }
} else {
    Write-Host ("Check: {0}" -f (($bad | ForEach-Object Host) -join ', ')) -ForegroundColor Red
}
