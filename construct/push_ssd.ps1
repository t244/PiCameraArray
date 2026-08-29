<#
.SYNOPSIS
    Push a Pi capture SSD from this Windows machine to Thunder over Tailscale.

.DESCRIPTION
    The SSDs are ext4, which Windows cannot read on its own, so this script
    mounts the disk inside WSL2 (read-only) and runs construct/ingest_ssd.sh
    there. That script decides which collection on Thunder the session belongs
    to - by FILE COUNT, the same rule get_data.sh uses - creates the directory,
    rsyncs the files over SSH, and verifies the result. Re-running it copies
    only what is missing, so an interrupted transfer just needs the same
    command again.

    Nothing is ever written to the SSD: it is mounted read-only.

    Requirements on this machine:
      - WSL2 with a distro installed        (wsl --install -d Ubuntu)
      - rsync inside that distro           (wsl -u root apt install -y rsync)
      - an Administrator PowerShell        (bare-disk mounting needs it)
      - Tailscale up, and SSH to Thunder working from either WSL or Windows

.PARAMETER Drive
    Physical drive number of the SSD (see -ListDisks). Auto-detected when the
    machine has exactly one removable/USB disk of about 233 GB.

.PARAMETER Target
    user@host on the far side. Default takatani@thunder (Tailscale MagicDNS);
    use 100.109.26.105 if MagicDNS is not resolving.

.PARAMETER Session
    Collection folder on Thunder to merge into, e.g. 20260728_073439. Without
    it the matching collection is worked out from the file count.

.PARAMETER List
    Show what is on the disk and what would happen, then stop.

.PARAMETER ExtraArgs
    Anything else is passed straight through to ingest_ssd.sh, e.g.
    --all-sessions, --source-session, --host, --checksum, --bwlimit 50000.

.EXAMPLE
    .\push_ssd.ps1 -ListDisks
    .\push_ssd.ps1 -List                       # inspect, copy nothing
    .\push_ssd.ps1 -Session 20260728_073439    # push the session of that run
    .\push_ssd.ps1 -Drive 2 -ExtraArgs '--bwlimit','50000'
#>
[CmdletBinding()]
param(
    [int]      $Drive = -1,
    [int]      $Partition = 1,
    [string]   $Target = "takatani@thunder",
    [string]   $Session = "",
    [string]   $DataRoot = "/data/CameraArray",
    [string]   $Distro = "",
    [switch]   $ListDisks,
    [switch]   $ForceUnmount,
    [switch]   $List,
    [switch]   $DryRun,
    [switch]   $Yes,
    [int]      $ProgressInterval = 15,
    [string[]] $ExtraArgs = @()
)

# NOT "Stop": with Stop, one line on a native command's stderr - ssh saying
# "Permanently added 'thunder' to the list of known hosts", wsl printing a
# notice - becomes a NativeCommandError and kills the run. Every external
# call below is checked by its exit code instead, which is what actually
# tells us whether it worked.
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"   # wsl.exe otherwise answers in UTF-16 and mangles parsing
function Info($m) { Write-Host "[push_ssd] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[push_ssd] $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[push_ssd] $m" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- disk listing
function Get-CandidateDisks {
    Get-Disk | Where-Object { -not $_.IsBoot -and -not $_.IsSystem } |
        Select-Object Number, FriendlyName, BusType,
                      @{n = 'SizeGB'; e = { [math]::Round($_.Size / 1GB) } }, OperationalStatus
}

if ($ForceUnmount) {
    if ($Drive -lt 0) { Die "-ForceUnmount needs -Drive <Number>" }
    wsl.exe --unmount "\\.\PHYSICALDRIVE$Drive"
    exit $LASTEXITCODE
}

if ($ListDisks) {
    Info "disks that are neither boot nor system (the Pi SSD is ~233 GB, USB):"
    Get-CandidateDisks | Format-Table -AutoSize
    Info "then:  .\push_ssd.ps1 -Drive <Number> -List"
    exit 0
}

# ------------------------------------------------------------ prerequisites
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Die "WSL is not installed. The SSD is ext4 and Windows cannot read it without WSL.`n  Install with:  wsl --install -d Ubuntu   (then reboot)"
}
# a distro name is a bare token; anything else on that stream is a notice
$wslDistros = (wsl.exe --list --quiet 2>&1) -replace "`0", "" |
              ForEach-Object { "$_".Trim() } |
              Where-Object { $_ -match '^[A-Za-z0-9._-]+$' }
if (-not $wslDistros) { Die "WSL has no distro installed. Run:  wsl --install -d Ubuntu" }
if (-not $Distro) { $Distro = ($wslDistros | Select-Object -First 1).Trim() }
Info "WSL distro: $Distro"

$admin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Die "mounting a bare disk needs an elevated shell. Re-open PowerShell with 'Run as Administrator'."
}

# rsync must exist inside the distro; it is what makes the transfer resumable
wsl.exe -d $Distro -e bash -lc "command -v rsync" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Die "rsync is missing inside WSL. Install it with:`n  wsl -d $Distro -u root apt-get update && wsl -d $Distro -u root apt-get install -y rsync"
}

# --------------------------------------------------------------- pick the disk
if ($Drive -lt 0) {
    $cands = @(Get-CandidateDisks | Where-Object { $_.SizeGB -gt 180 -and $_.SizeGB -lt 300 })
    if ($cands.Count -eq 1) {
        $Drive = $cands[0].Number
        Info "auto-detected disk $Drive : $($cands[0].FriendlyName) ($($cands[0].SizeGB) GB, $($cands[0].BusType))"
    } else {
        Info "cannot pick a disk on my own; candidates:"
        Get-CandidateDisks | Format-Table -AutoSize
        Die "name it with -Drive <Number>"
    }
}

$devicePath = "\\.\PHYSICALDRIVE$Drive"
$mountPoint = "/mnt/wsl/PHYSICALDRIVE${Drive}p${Partition}"

# ---------------------------------------------------- where the engine lives
# ingest_ssd.sh sits next to this file; hand WSL the translated path.
$engineWin = Join-Path $PSScriptRoot "ingest_ssd.sh"
if (-not (Test-Path $engineWin)) { Die "ingest_ssd.sh not found next to this script ($PSScriptRoot)" }
$engineWsl = (wsl.exe -d $Distro -e wslpath -a "$engineWin" 2>&1 | Select-Object -First 1)
if (-not $engineWsl) { Die "could not translate $engineWin into a WSL path" }
$engineWsl = $engineWsl.Trim()

# A checkout on the Windows filesystem may carry CRLF line endings, which bash
# refuses. Copy the engine into the distro with the carriage returns stripped.
$engineRun = "/tmp/picam_ingest_ssd.sh"
wsl.exe -d $Distro -e bash -lc "tr -d '\r' < '$engineWsl' > $engineRun && chmod +x $engineRun" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Die "could not stage ingest_ssd.sh inside WSL" }

# --------------------------------------------- how WSL should reach Thunder
# WSL2 sits behind a NAT and does not always see the host's Tailscale network.
# When it cannot, borrow the Windows ssh.exe as rsync's transport: the traffic
# then goes out through the host, where Tailscale certainly works.
$sshCmd = "ssh"
Info "checking SSH from WSL to $Target ..."
wsl.exe -d $Distro -e bash -lc "ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new $Target true" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Warn "WSL cannot reach $Target directly (Tailscale is usually not routed into WSL2)."
    $winSsh = "/mnt/c/Windows/System32/OpenSSH/ssh.exe"
    wsl.exe -d $Distro -e bash -lc "$winSsh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new $Target true" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Die @"
Neither WSL's ssh nor the Windows ssh.exe could reach $Target.
Check, on this machine:
  tailscale status                     # is Thunder listed and online?
  ssh $Target                          # does a plain SSH session work?
If SSH asks for a password every time, either copy a key over
  (ssh-keygen; type `$env:USERPROFILE\.ssh\id_ed25519.pub | ssh $Target 'cat >> .ssh/authorized_keys')
or rely on Tailscale SSH, which Thunder already has enabled.
"@
    }
    $sshCmd = $winSsh
    Info "using the Windows ssh.exe as the transport ($winSsh)"
} else {
    Info "WSL reaches $Target directly"
}

# ------------------------------------------------------------- mount and run
$mounted = $false
try {
    Info "mounting $devicePath partition $Partition read-only into WSL ..."
    wsl.exe --mount $devicePath --partition $Partition --type ext4 --options "ro" 2>&1 | ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -ne 0) {
        Die "wsl --mount failed. Is the disk still in use by Windows, or already mounted?`n  Try:  wsl --unmount $devicePath"
    }
    $mounted = $true

    $engineArgs = @("--target", $Target, "--ssh-cmd", $sshCmd, "--data-root", $DataRoot,
                    "--progress-interval", "$ProgressInterval")
    if ($Session)  { $engineArgs += @("--session", $Session) }
    if ($List)     { $engineArgs += "--list" }
    if ($DryRun)   { $engineArgs += "--dry-run" }
    if ($Yes)      { $engineArgs += "-y" }
    $engineArgs += $ExtraArgs
    $engineArgs += $mountPoint

    Info ("running: bash $engineRun " + ($engineArgs -join " "))
    Write-Host ""
    wsl.exe -d $Distro -e bash $engineRun @engineArgs
    $rc = $LASTEXITCODE
    Write-Host ""
    if ($rc -eq 0) { Info "done" } else { Warn "ingest_ssd.sh exited with $rc (see the messages above)" }
    exit $rc
}
finally {
    if ($mounted) {
        Info "unmounting $devicePath"
        wsl.exe --unmount $devicePath 2>&1 | ForEach-Object { Write-Host "  $_" }
    }
}
