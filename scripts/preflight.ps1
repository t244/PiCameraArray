<#
.SYNOPSIS
    Pre-departure readiness check for the whole camera array.

.DESCRIPTION
    Queries every Pi and prints one table row each, flagging anything that
    would ruin a field session:

      SSD      the capture SSD is really mounted (not just the directory)
      FSTAB    the LABEL=HIKSEMI entry exists, so the mount survives reboot
      SVC      picamera-capture is running
      TRIG     frames are arriving - catches broken trigger wiring
      SESS     the session directory is on the SSD, not the SD card fallback
      WIFI     both PiCamNet profiles are registered and outrank the lab AP
      ARDUINO  the master Pi has found the trigger board

    Read-only. Changes nothing.

.EXAMPLE
    .\preflight.ps1                  # check e00..e15
    .\preflight.ps1 -Suffix .local   # when the array is on the pocket router
    .\preflight.ps1 -Detail          # also print the raw values per host
#>
[CmdletBinding()]
param(
    [int]    $First      = 0,
    [int]    $Last       = 15,
    [int[]]  $Only,
    [string] $HostFormat = "e{0:D2}",
    [string] $Suffix     = "",
    [string] $User       = "pi",
    [string] $Password   = "pi",
    [string[]] $Ssids    = @("PiCamNet5G", "PiCamNet"),
    [string] $Mount      = "/media/pi/HIKSEMI",
    [switch] $Detail,
    [switch] $Parallel,
    [int]    $Throttle   = 16
)

$ErrorActionPreference = 'Stop'

foreach ($exe in @('plink', 'pscp')) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        throw "$exe not found on PATH. Install the PuTTY tools first."
    }
}

$localScript = Join-Path $PSScriptRoot 'preflight_check.sh'
if (-not (Test-Path $localScript)) {
    throw "preflight_check.sh not found next to this script ($localScript)"
}

if ($Parallel -and $PSVersionTable.PSVersion.Major -lt 7) {
    Write-Warning "-Parallel needs PowerShell 7+; falling back to sequential."
    $Parallel = $false
}

$indices = if ($Only) { $Only } else { $First..$Last }
$targets = @($indices | ForEach-Object {
    [pscustomobject]@{ Index = $_; Hostname = ("$HostFormat$Suffix" -f $_) }
})

Write-Host "=== PREFLIGHT : $($targets.Count) hosts ===" -ForegroundColor Cyan

$cfg = @{
    User = $User; Password = $Password; Mount = $Mount; LocalScript = $localScript
}

$workerBody = @'
param($Target, $Cfg)

# Native-command stderr becomes a NativeCommandError record in Windows
# PowerShell; under ErrorActionPreference 'Stop' that is terminating and would
# discard the real output. Judge success by $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'

$h = $Target.Hostname
try {
    $copy = (pscp -pw $Cfg.Password -batch $Cfg.LocalScript "$($Cfg.User)@${h}:/tmp/preflight_check.sh" 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "pscp: $($copy.Trim())" }
    $line = (plink -pw $Cfg.Password -batch "$($Cfg.User)@$h" `
        "sudo sed -i 's/\r//g' /tmp/preflight_check.sh; MOUNT='$($Cfg.Mount)' bash /tmp/preflight_check.sh" 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw $line.Trim() }
    [pscustomobject]@{ Hostname = $h; Line = ($line -replace "`r", "").Trim(); Error = $null }
}
catch {
    [pscustomobject]@{ Hostname = $h; Line = $null; Error = $_.Exception.Message }
}
'@

if ($Parallel) {
    $raw = @($targets | ForEach-Object -Parallel {
        & ([scriptblock]::Create($using:workerBody)) $_ $using:cfg
    } -ThrottleLimit $Throttle)
}
else {
    $worker = [scriptblock]::Create($workerBody)
    $raw = @(foreach ($t in $targets) {
        Write-Host "  querying $($t.Hostname) ..." -ForegroundColor DarkGray
        & $worker $t $cfg
    })
}

# --- parse -----------------------------------------------------------------

$rows = foreach ($r in $raw) {
    if (-not $r.Line) {
        [pscustomobject]@{
            Host = $r.Hostname; SSD = '?'; FSTAB = '?'; SVC = '?'; TRIG = '?'
            SESS = '?'; WIFI = '?'; ARDUINO = '?'; Free = '-'; Temp = '-'
            Problems = @("UNREACHABLE: $($r.Error)")
        }
        continue
    }

    $kv = @{}
    foreach ($pair in ($r.Line -split ';')) {
        $i = $pair.IndexOf('=')
        if ($i -gt 0) { $kv[$pair.Substring(0, $i)] = $pair.Substring($i + 1) }
    }

    $problems = @()

    $ssdOk = $kv['ssd'] -eq 'yes'
    if (-not $ssdOk) { $problems += "SSD not mounted" }

    $fstabOk = [int]($kv['fstab'] -as [int]) -ge 1
    if (-not $fstabOk) { $problems += "no fstab entry - mount will not survive reboot" }

    $svcOk = $kv['svc'] -eq 'active'
    if (-not $svcOk) { $problems += "service $($kv['svc'])" }

    $count = [int]($kv['count'] -as [int])
    $trigOk = $count -gt 0
    if (-not $trigOk) { $problems += "no frames captured - check trigger wiring" }

    $sessOk = $kv['sess'] -like "$Mount*"
    if (-not $sessOk) { $problems += "session dir not on SSD: $($kv['sess'])" }

    $wifiOk = $true
    foreach ($s in $Ssids) {
        if ($kv['wifi'] -notmatch [regex]::Escape("$s/")) { $wifiOk = $false; $problems += "wifi profile missing: $s" }
    }

    [pscustomobject]@{
        Host    = $kv['host']
        SSD     = if ($ssdOk)   { 'ok' } else { 'FAIL' }
        FSTAB   = if ($fstabOk) { 'ok' } else { 'FAIL' }
        SVC     = if ($svcOk)   { 'ok' } else { 'FAIL' }
        TRIG    = if ($trigOk)  { "ok($count)" } else { 'FAIL(0)' }
        SESS    = if ($sessOk)  { 'ok' } else { 'FAIL' }
        WIFI    = if ($wifiOk)  { 'ok' } else { 'FAIL' }
        ARDUINO = $kv['arduino']
        Free    = $kv['free']
        Temp    = $kv['temp']
        Problems = $problems
    }
}

# --- report ----------------------------------------------------------------

Write-Host ""
$rows | Sort-Object Host |
    Format-Table Host, SSD, FSTAB, SVC, TRIG, SESS, WIFI, ARDUINO, Free, Temp -AutoSize |
    Out-String -Width 200 | Write-Host

if ($Detail) {
    foreach ($r in $raw) { Write-Host "$($r.Hostname): $($r.Line)$($r.Error)" -ForegroundColor DarkGray }
    Write-Host ""
}

$bad = @($rows | Where-Object { $_.Problems.Count -gt 0 })
if ($bad.Count -eq 0) {
    Write-Host "All $($rows.Count) hosts ready." -ForegroundColor Green
}
else {
    Write-Host "$($bad.Count) of $($rows.Count) hosts need attention:" -ForegroundColor Red
    foreach ($b in ($bad | Sort-Object Host)) {
        Write-Host "  $($b.Host):" -ForegroundColor Red
        foreach ($p in $b.Problems) { Write-Host "      - $p" -ForegroundColor Red }
    }
}

# The Arduino is only expected on the master Pi, so report it separately
# rather than failing every other host.
$withArduino = @($rows | Where-Object { $_.ARDUINO -eq 'True' })
Write-Host ""
if ($withArduino.Count -eq 1) {
    Write-Host "Arduino detected on $($withArduino.Host) (master)." -ForegroundColor Green
}
elseif ($withArduino.Count -eq 0) {
    Write-Host "No Pi has detected the Arduino. Trigger settings cannot be changed." -ForegroundColor Yellow
    Write-Host "The agent scans for the serial port only at startup - restart it after plugging in:" -ForegroundColor Yellow
    Write-Host "    plink -pw $Password -batch $User@$($targets[0].Hostname) `"sudo systemctl restart picamera-capture`"" -ForegroundColor Yellow
}
else {
    Write-Host "More than one Pi reports an Arduino: $(($withArduino | ForEach-Object { $_.Host }) -join ', ')" -ForegroundColor Yellow
}
