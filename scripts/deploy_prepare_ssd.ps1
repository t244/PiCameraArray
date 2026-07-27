<#
.SYNOPSIS
    Format replacement SSDs across the camera array and mount them at
    /media/pi/HIKSEMI via /etc/fstab.

.DESCRIPTION
    Copies prepare_ssd.sh to each Pi and runs it.

    INSPECT IS THE DEFAULT. Without -Format the script only reports what it
    sees and changes nothing. Read that output before formatting anything.

    Works on Windows PowerShell 5.1 and PowerShell 7+. Sequential by default;
    -Parallel needs PowerShell 7.

.PARAMETER Format
    Actually partition and format. DESTROYS ALL DATA on the target disk.
    Refused automatically for the boot disk.

.PARAMETER Device
    Force a specific disk, e.g. /dev/sda. Only needed when a Pi has more than
    one candidate disk attached and the script cannot choose.

.EXAMPLE
    .\deploy_prepare_ssd.ps1 -Only 15              # inspect one Pi
    .\deploy_prepare_ssd.ps1 -Only 15 -Format      # format that one Pi
    .\deploy_prepare_ssd.ps1                       # inspect all 16
    .\deploy_prepare_ssd.ps1 -Format               # format all 16
#>
[CmdletBinding()]
param(
    [int]      $First      = 0,
    [int]      $Last       = 15,
    [int[]]    $Only,
    [string]   $HostFormat = "e{0:D2}",
    [string]   $User       = "pi",
    [string]   $Password   = "pi",
    [string]   $Label      = "HIKSEMI",
    [string]   $Mountpoint = "",
    [string]   $Device     = "",
    [int]      $MinSizeGb  = 50,
    # Eager inode-table init was observed to fail on this hardware; the
    # mke2fs default (lazy) is used unless you ask for eager explicitly.
    [switch]   $EagerInit,
    [switch]   $Format,
    # Repair the fstab entry and mount an already-formatted SSD. Keeps all
    # data; no confirmation needed because nothing is destroyed.
    [switch]   $FixMount,
    [switch]   $Parallel,
    [int]      $Throttle   = 16
)

$ErrorActionPreference = 'Stop'

foreach ($exe in @('plink', 'pscp')) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        throw "$exe not found on PATH. Install the PuTTY tools first."
    }
}

$localScript = Join-Path $PSScriptRoot 'prepare_ssd.sh'
if (-not (Test-Path $localScript)) {
    throw "prepare_ssd.sh not found next to this script ($localScript)"
}

if ($Parallel -and $PSVersionTable.PSVersion.Major -lt 7) {
    Write-Warning ("-Parallel needs PowerShell 7+ (running {0}); falling back to sequential." `
        -f $PSVersionTable.PSVersion)
    $Parallel = $false
}

$indices = if ($Only) { $Only } else { $First..$Last }
$targets = @($indices | ForEach-Object {
    [pscustomobject]@{ Index = $_; Hostname = ($HostFormat -f $_) }
})

if ($Format -and $FixMount) { throw "-Format and -FixMount are mutually exclusive." }

$mode = if ($Format) { 'FORMAT' } elseif ($FixMount) { 'FIX-MOUNT' } else { 'INSPECT' }
Write-Host "=== $mode : $($targets.Count) hosts ($($targets[0].Hostname)..$($targets[-1].Hostname)) ===" -ForegroundColor Cyan

if ($Format) {
    Write-Host ""
    Write-Host "  THIS WILL DESTROY ALL DATA on the target disk of each Pi listed above." -ForegroundColor Red
    Write-Host "  Detach any old SSD first - two disks sharing LABEL=$Label is refused." -ForegroundColor Red
    Write-Host ""
    $answer = Read-Host "  Type the word FORMAT to continue"
    if ($answer -cne 'FORMAT') {
        Write-Host "aborted." -ForegroundColor Yellow
        exit 1
    }
}

$cfg = @{
    User        = $User
    Password    = $Password
    Label       = $Label
    Mountpoint  = $Mountpoint
    Device      = $Device
    MinSizeGb   = $MinSizeGb
    EagerInit   = [bool]$EagerInit
    FixMount    = [bool]$FixMount
    Confirm     = [bool]$Format
    LocalScript = $localScript
}

# Worker kept as source text: ForEach-Object -Parallel cannot marshal a live
# scriptblock across runspaces, but each runspace can recompile a string.
$workerBody = @'
param($Target, $Cfg)

# Windows PowerShell turns a native command's stderr into NativeCommandError
# records. With ErrorActionPreference 'Stop' that becomes a *terminating*
# error, so a harmless banner like "mke2fs 1.47.2" aborts the whole call and
# the real output is lost. Success here is decided by $LASTEXITCODE alone.
$ErrorActionPreference = 'Continue'

$h    = $Target.Hostname
$pw   = $Cfg.Password
$user = $Cfg.User
$ok   = $true
$out  = ""

try {
    $copy = (pscp -pw $pw -batch $Cfg.LocalScript "$user@${h}:/tmp/prepare_ssd.sh" 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "pscp failed: $($copy.Trim())" }

    $envs = "LABEL='$($Cfg.Label)' MIN_SIZE_GB=$($Cfg.MinSizeGb)"
    if ($Cfg.Mountpoint) { $envs += " MOUNTPOINT='$($Cfg.Mountpoint)'" }
    if ($Cfg.Device)     { $envs += " DEVICE='$($Cfg.Device)'" }
    if ($Cfg.EagerInit)  { $envs += " LAZY_INIT=0" }
    if ($Cfg.FixMount)   { $envs += " FSTAB_ONLY=1" }
    if ($Cfg.Confirm)    { $envs += " CONFIRM=YES" }

    # 'sudo env VAR=... cmd', not 'sudo VAR=... cmd': the latter needs SETENV
    # in sudoers, which Raspberry Pi OS does not grant.
    $run = "sudo sed -i 's/\r//g' /tmp/prepare_ssd.sh; " +
           "sudo env $envs bash /tmp/prepare_ssd.sh"
    $out = (plink -pw $pw -batch "$user@$h" $run 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { $ok = $false }
}
catch {
    $ok  = $false
    $out = $_.Exception.Message
}

# mke2fs draws progress with CR and backspace, which erases earlier lines in
# the console and hides the real error. Normalise before returning.
$clean = ("$out") -replace "`r`n", "`n" -replace "`r", "`n" -replace "[\x08]+", ""

[pscustomobject]@{ Hostname = $h; Ok = $ok; Output = $clean.TrimEnd() }
'@

if ($Parallel) {
    $results = @($targets | ForEach-Object -Parallel {
        $sb = [scriptblock]::Create($using:workerBody)
        & $sb $_ $using:cfg
    } -ThrottleLimit $Throttle)
}
else {
    $worker  = [scriptblock]::Create($workerBody)
    $n       = 0
    $results = @(foreach ($t in $targets) {
        $n++
        Write-Host ("[{0,2}/{1}] {2} ..." -f $n, $targets.Count, $t.Hostname) -ForegroundColor DarkGray
        & $worker $t $cfg
    })
}

Write-Host ""
foreach ($r in ($results | Sort-Object Hostname)) {
    $colour = if ($r.Ok) { 'Green' } else { 'Red' }
    $label  = if ($r.Ok) { 'OK' } else { 'FAILED' }
    Write-Host "=== $($r.Hostname) === $label" -ForegroundColor $colour
    if ($r.Output) { Write-Host $r.Output }
}

$failed = @($results | Where-Object { -not $_.Ok })
$total  = @($results).Count
Write-Host ""
Write-Host "--- $($total - $failed.Count)/$total succeeded ---" -ForegroundColor $(if ($failed.Count) { 'Red' } else { 'Green' })

if ($failed.Count) {
    Write-Host "failed: $(($failed | ForEach-Object { $_.Hostname }) -join ', ')" -ForegroundColor Red
    exit 1
}

if (-not $Format -and -not $FixMount) {
    Write-Host ""
    Write-Host "Inspect only - nothing was changed." -ForegroundColor Cyan
    Write-Host "Check the target disk on each Pi above, then re-run with -Format." -ForegroundColor Cyan
}
else {
    Write-Host ""
    Write-Host "Verify after a reboot that the mount survives:" -ForegroundColor Cyan
    Write-Host "  plink -pw $Password -batch $User@$($targets[0].Hostname) `"findmnt /media/pi/$Label; curl -s localhost:8000/status`"" -ForegroundColor Cyan
}
