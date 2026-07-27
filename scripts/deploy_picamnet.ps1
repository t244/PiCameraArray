<#
.SYNOPSIS
    Push the PiCamNet WiFi profile to the whole camera array.

.DESCRIPTION
    Copies setup_picamnet.sh to each Pi and runs it. The script registers the
    pocket-router SSID with a higher autoconnect priority than the lab WiFi
    but does NOT activate it, so running this from the lab is safe - no Pi
    drops off the network mid-deploy.

    Run this with the pocket router POWERED OFF, while the array is still on
    the lab network.

    Works on both Windows PowerShell 5.1 and PowerShell 7+. Hosts are
    processed sequentially by default; add -Parallel on PowerShell 7+.

.PARAMETER Verify
    Don't change anything; just report the current WiFi profiles and priority
    on each Pi.

.PARAMETER Remove
    Roll back: delete the PiCamNet profile from each Pi.

.PARAMETER Static
    Use static addresses instead of DHCP: e00 -> 192.168.8.112 ... e15 -> .127.
    Defaults assume the GL.iNet pocket router (LAN 192.168.8.0/24). Check the
    router's actual subnet before using this.

.EXAMPLE
    .\deploy_picamnet.ps1                    # deploy to e00..e15 (DHCP)
    .\deploy_picamnet.ps1 -Only 15           # stage on one Pi first
    .\deploy_picamnet.ps1 -Verify            # check what is configured
    .\deploy_picamnet.ps1 -Remove            # undo
#>
[CmdletBinding()]
param(
    [int]      $First      = 0,
    [int]      $Last       = 15,
    [int[]]    $Only,
    [string]   $HostFormat = "e{0:D2}",
    [string]   $User       = "pi",
    [string]   $Password   = "pi",
    # The GL.iNet router splits bands into separate SSIDs. Highest priority
    # first; every one of them is registered on each Pi.
    [string[]] $Ssids      = @("PiCamNet5G", "PiCamNet"),
    [string]   $Psk        = "PiCamNet",
    [int]      $Priority   = 100,
    [switch]   $Static,
    # Defaults match the field pocket router (GL.iNet, LAN 192.168.8.0/24).
    [string]   $StaticBase = "192.168.8",
    [int]      $StaticFrom = 112,
    [string]   $Gateway    = "192.168.8.1",
    [switch]   $Verify,
    [switch]   $Remove,
    [switch]   $Parallel,
    [int]      $Throttle   = 16
)

$ErrorActionPreference = 'Stop'

# --- preflight -------------------------------------------------------------

foreach ($exe in @('plink', 'pscp')) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        throw "$exe not found on PATH. Install the PuTTY tools first."
    }
}

$localScript = Join-Path $PSScriptRoot 'setup_picamnet.sh'
if (-not $Verify -and -not $Remove -and -not (Test-Path $localScript)) {
    throw "setup_picamnet.sh not found next to this script ($localScript)"
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

$mode = if ($Verify) { 'VERIFY' } elseif ($Remove) { 'REMOVE' } else { 'DEPLOY' }
$addressing = if ($Static) { "static $StaticBase.$StaticFrom+" } else { 'DHCP' }

Write-Host "=== $mode : $($targets.Count) hosts ($($targets[0].Hostname)..$($targets[-1].Hostname)) ===" -ForegroundColor Cyan
if ($mode -eq 'DEPLOY') {
    Write-Host "    SSIDs=$($Ssids -join ', ')  priority=$Priority  addressing=$addressing" -ForegroundColor Cyan
    Write-Host "    Make sure the pocket router is POWERED OFF before continuing." -ForegroundColor Yellow
}

# Config bundle handed to every worker invocation (plain data only, so it
# crosses runspace boundaries cleanly when running in parallel).
$cfg = @{
    User        = $User
    Password    = $Password
    Ssids       = $Ssids
    Psk         = $Psk
    Priority    = $Priority
    Static      = [bool]$Static
    StaticBase  = $StaticBase
    StaticFrom  = $StaticFrom
    Gateway     = $Gateway
    Verify      = [bool]$Verify
    Remove      = [bool]$Remove
    LocalScript = $localScript
}

# --- worker ----------------------------------------------------------------
# Kept as source text rather than a scriptblock object: ForEach-Object
# -Parallel cannot marshal a live scriptblock across runspaces, but it can
# marshal a string that each runspace recompiles.

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
    if ($Cfg.Verify) {
        $cmd = "nmcli -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show | " +
               "grep -E 'NAME|wireless'; echo '-- active --'; " +
               "nmcli -t -f NAME,DEVICE connection show --active"
        $out = (plink -pw $pw -batch "$user@$h" $cmd 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) { $ok = $false }
    }
    elseif ($Cfg.Remove) {
        $dels = ($Cfg.Ssids | ForEach-Object {
            "sudo nmcli connection delete '$_' 2>&1 || echo 'not present: $_'"
        }) -join '; '
        $out = (plink -pw $pw -batch "$user@$h" $dels 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) { $ok = $false }
    }
    else {
        $copy = (pscp -pw $pw -batch $Cfg.LocalScript "$user@${h}:/tmp/setup_picamnet.sh" 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) { throw "pscp failed: $($copy.Trim())" }

        $envs = "SSIDS='$($Cfg.Ssids -join ' ')' PSK='$($Cfg.Psk)' PRIORITY=$($Cfg.Priority)"
        if ($Cfg.Static) {
            $ip = "$($Cfg.StaticBase).$($Cfg.StaticFrom + $Target.Index)/24"
            $envs += " STATIC_IP='$ip' GATEWAY='$($Cfg.Gateway)'"
        }
        # Normalise line endings in case the file was saved with CRLF.
        # GNU sed understands \r; the expression contains no '$', so
        # PowerShell interpolation leaves it untouched.
        # 'sudo env VAR=... cmd' rather than 'sudo VAR=... cmd': the latter is
        # rejected unless sudoers grants SETENV, which Raspberry Pi OS does not.
        $run = "sudo sed -i 's/\r//g' /tmp/setup_picamnet.sh; " +
               "sudo env $envs bash /tmp/setup_picamnet.sh"
        $out = (plink -pw $pw -batch "$user@$h" $run 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) { $ok = $false }
    }
}
catch {
    $ok  = $false
    $out = $_.Exception.Message
}

[pscustomobject]@{ Hostname = $h; Ok = $ok; Output = ("$out").TrimEnd() }
'@

# --- run -------------------------------------------------------------------

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

# --- report ----------------------------------------------------------------

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

if ($mode -eq 'DEPLOY') {
    $canary = $targets[-1].Hostname
    Write-Host ""
    Write-Host "Next steps" -ForegroundColor Cyan
    Write-Host "  1. Power the pocket router on." -ForegroundColor Cyan
    Write-Host "  2. Reboot ONE Pi as a canary:" -ForegroundColor Cyan
    Write-Host "         plink -pw $Password -batch $User@$canary `"sudo reboot`"" -ForegroundColor Cyan
    Write-Host "  3. Move the laptop onto $($Ssids -join ' or '), then check it came back:" -ForegroundColor Cyan
    Write-Host "         curl http://$canary.local:8000/status" -ForegroundColor Cyan
    Write-Host "  4. Only if that works, reboot the rest." -ForegroundColor Cyan
}
