<#
.SYNOPSIS
    Empty the capture data directories across the camera array.

.DESCRIPTION
    Copies clear_capture_data.sh to each Pi and runs it.

    DRY RUN IS THE DEFAULT. Without -Delete it only reports how much is
    stored where. Read that before deleting anything.

    Two directories can hold captures:
      /media/pi/HIKSEMI/data        the SSD, where captures belong
      /home/pi/PiCameraArray/data   the SD card fallback, which fills in
                                    under an hour and should stay empty

    Clearing the SSD is skipped on any Pi where the SSD is not mounted, so a
    stray directory on the root filesystem is never mistaken for the disk.

.PARAMETER Delete
    Actually delete. Prompts for confirmation first.

.PARAMETER Target
    ssd (default), sd, or both.

.EXAMPLE
    .\deploy_clear_data.ps1                        # report what is stored
    .\deploy_clear_data.ps1 -Delete                # clear the SSDs
    .\deploy_clear_data.ps1 -Target both -Delete   # clear SSD and SD fallback
    .\deploy_clear_data.ps1 -Only 3 -Delete        # one Pi
#>
[CmdletBinding()]
param(
    [int]      $First      = 0,
    [int]      $Last       = 15,
    [int[]]    $Only,
    [string]   $HostFormat = "e{0:D2}",
    [string]   $Suffix     = "",
    [string]   $User       = "pi",
    [string]   $Password   = "pi",
    [ValidateSet('ssd', 'sd', 'both')]
    [string]   $Target     = 'ssd',
    [string]   $Mount      = "/media/pi/HIKSEMI",
    [switch]   $Delete,
    [switch]   $Parallel,
    [int]      $Throttle   = 16
)

$ErrorActionPreference = 'Stop'

foreach ($exe in @('plink', 'pscp')) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        throw "$exe not found on PATH. Install the PuTTY tools first."
    }
}

$localScript = Join-Path $PSScriptRoot 'clear_capture_data.sh'
if (-not (Test-Path $localScript)) {
    throw "clear_capture_data.sh not found next to this script ($localScript)"
}

if ($Parallel -and $PSVersionTable.PSVersion.Major -lt 7) {
    Write-Warning "-Parallel needs PowerShell 7+; falling back to sequential."
    $Parallel = $false
}

$indices = if ($Only) { $Only } else { $First..$Last }
$targets = @($indices | ForEach-Object {
    [pscustomobject]@{ Index = $_; Hostname = ("$HostFormat$Suffix" -f $_) }
})

$mode = if ($Delete) { 'DELETE' } else { 'DRY RUN' }
Write-Host "=== $mode : $($targets.Count) hosts, target=$Target ===" -ForegroundColor Cyan

if ($Delete) {
    Write-Host ""
    Write-Host "  This deletes every capture under:" -ForegroundColor Red
    if ($Target -ne 'sd')  { Write-Host "    $Mount/data" -ForegroundColor Red }
    if ($Target -ne 'ssd') { Write-Host "    /home/pi/PiCameraArray/data" -ForegroundColor Red }
    Write-Host "  on $($targets.Count) hosts. Make sure anything you need is already copied off." -ForegroundColor Red
    Write-Host ""
    $answer = Read-Host "  Type the word DELETE to continue"
    if ($answer -cne 'DELETE') {
        Write-Host "aborted." -ForegroundColor Yellow
        exit 1
    }
}

$cfg = @{
    User = $User; Password = $Password; Target = $Target
    Mount = $Mount; Confirm = [bool]$Delete; LocalScript = $localScript
}

# Worker kept as source text: ForEach-Object -Parallel cannot marshal a live
# scriptblock across runspaces, but each runspace can recompile a string.
$workerBody = @'
param($Target, $Cfg)

# Native-command stderr becomes a NativeCommandError record in Windows
# PowerShell; under ErrorActionPreference 'Stop' that is terminating and would
# discard the real output. Judge success by $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'

$h    = $Target.Hostname
$pw   = $Cfg.Password
$user = $Cfg.User
$ok   = $true
$out  = ""

try {
    $copy = (pscp -pw $pw -batch $Cfg.LocalScript "$user@${h}:/tmp/clear_capture_data.sh" 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "pscp failed: $($copy.Trim())" }

    $envs = "TARGET='$($Cfg.Target)' MOUNT='$($Cfg.Mount)'"
    if ($Cfg.Confirm) { $envs += " CONFIRM=YES" }

    # 'sudo env VAR=... cmd', not 'sudo VAR=... cmd': the latter needs SETENV
    # in sudoers, which Raspberry Pi OS does not grant.
    $run = "sudo sed -i 's/\r//g' /tmp/clear_capture_data.sh; " +
           "sudo env $envs bash /tmp/clear_capture_data.sh"
    $out = (plink -pw $pw -batch "$user@$h" $run 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { $ok = $false }
}
catch {
    $ok  = $false
    $out = $_.Exception.Message
}

$clean = ("$out") -replace "`r`n", "`n" -replace "`r", "`n" -replace "[\x08]+", ""
[pscustomobject]@{ Hostname = $h; Ok = $ok; Output = $clean.TrimEnd() }
'@

if ($Parallel) {
    $results = @($targets | ForEach-Object -Parallel {
        & ([scriptblock]::Create($using:workerBody)) $_ $using:cfg
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

if (-not $Delete) {
    Write-Host ""
    Write-Host "Dry run - nothing was deleted. Re-run with -Delete to clear." -ForegroundColor Cyan
}
