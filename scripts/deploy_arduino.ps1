<#
.SYNOPSIS
    Flash exposure_trigger.ino to the trigger-box Uno through e00 and store
    the capture settings in its EEPROM.

.DESCRIPTION
    Copies capture/exposure_trigger/exposure_trigger.ino and
    scripts/flash_trigger.sh from this PC to the Pi (no git pull needed),
    then runs flash_trigger.sh there: compile, stop picamera-capture, upload,
    send E/T/F/B, confirm with R, start picamera-capture again.

    The first run installs arduino-cli and the AVR core on the Pi, which needs
    internet: put that Pi on the lab Wi-Fi for the first run.

    Works on Windows PowerShell 5.1 and PowerShell 7+. Needs plink/pscp.

.EXAMPLE
    .\deploy_arduino.ps1
    .\deploy_arduino.ps1 -ExposureUs 20000
    .\deploy_arduino.ps1 -HostName e00.local
#>
[CmdletBinding()]
param(
    [string] $HostName   = "e00",
    [string] $User       = "pi",
    [string] $Password   = "pi",
    [int]    $ExposureUs = 30000,
    [int]    $PeriodMs   = 120000,
    [int]    $Fps        = 30,
    [int]    $BurstMs    = 2000
)

foreach ($exe in @('plink', 'pscp')) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        throw "$exe not found on PATH. Install the PuTTY tools first."
    }
}
$repo   = Split-Path $PSScriptRoot -Parent
$ino    = Join-Path $repo 'capture\exposure_trigger\exposure_trigger.ino'
$script = Join-Path $PSScriptRoot 'flash_trigger.sh'
foreach ($f in @($ino, $script)) { if (-not (Test-Path $f)) { throw "not found: $f" } }

$target = "$User@$HostName"
Write-Host "=== flash trigger box via $HostName : E=$ExposureUs us T=$PeriodMs ms F=$Fps B=$BurstMs ms ===" -ForegroundColor Cyan

$ErrorActionPreference = 'Continue'
plink -pw $Password -batch $target "rm -rf /tmp/exposure_trigger; mkdir -p /tmp/exposure_trigger" | Out-Null
pscp -pw $Password -batch -q $ino "${target}:/tmp/exposure_trigger/exposure_trigger.ino"
if ($LASTEXITCODE -ne 0) { throw "pscp of the sketch failed" }
pscp -pw $Password -batch -q $script "${target}:/tmp/flash_trigger.sh"
if ($LASTEXITCODE -ne 0) { throw "pscp of flash_trigger.sh failed" }

$run = "sed -i 's/\r//g' /tmp/flash_trigger.sh; " +
       "sudo env EXPOSURE_US=$ExposureUs PERIOD_MS=$PeriodMs FPS=$Fps BURST_MS=$BurstMs " +
       "bash /tmp/flash_trigger.sh /tmp/exposure_trigger"
plink -pw $Password -batch $target $run 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK" -ForegroundColor Green
} else {
    Write-Host "FAILED (see above). picamera-capture was restarted on $HostName." -ForegroundColor Red
}
