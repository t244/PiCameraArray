<#
.SYNOPSIS
    Push a Pi capture SSD from this Windows machine to Thunder over Tailscale.

.DESCRIPTION
    The SSDs are ext4, which Windows cannot read on its own, so the disk is
    made visible inside WSL2 and construct/ingest_ssd.sh runs there. That
    script decides which collection on Thunder the session belongs to - by
    FILE COUNT, the same rule get_data.sh uses - creates the directory,
    rsyncs the files over SSH and verifies the result. Re-running copies only
    what is missing, so an interrupted transfer just needs the same command
    again. Nothing is ever written to the SSD; it is mounted read-only.

    Three ways to get the disk into WSL, tried in this order:

      1. -MountPath        you mounted it yourself; nothing is mounted here
      2. wsl --mount       the built-in way. NOT available on ARM64 Windows
                           before build 27653, which is why 3 exists
      3. usbipd-win        the whole USB device is handed to WSL, which then
                           mounts the partition normally. Works on ARM64.
                           Install once:  winget install dorssel.usbipd-win

    Requirements: WSL2 with a distro, rsync inside it, an Administrator
    PowerShell, Tailscale up, and SSH to Thunder working from WSL or Windows.

.PARAMETER Drive
    Physical drive number for the wsl --mount path (see -ListDisks).

.PARAMETER BusId
    USB bus id for the usbipd path, e.g. 2-1 (see -ListDisks). Detected
    automatically when exactly one mass-storage device is attached.

.PARAMETER MountPath
    A path inside WSL where the SSD is already mounted, e.g. /mnt/ssd.
    Skips every mounting step.

.PARAMETER Target
    user@host on the far side. Default takatani@thunder (Tailscale MagicDNS);
    use 100.109.26.105 if MagicDNS is not resolving.

.PARAMETER Session
    Collection folder on Thunder to merge into, e.g. 20260728_073439. Without
    it the matching collection is worked out from the file count.

.PARAMETER ExtraArgs
    Passed straight through to ingest_ssd.sh: --all-sessions,
    --source-session, --host, --checksum, --bwlimit 50000, ...

.EXAMPLE
    .\push_ssd.ps1 -ListDisks
    .\push_ssd.ps1 -List                       # inspect, copy nothing
    .\push_ssd.ps1 -Session 20260728_073439 -Yes
    .\push_ssd.ps1 -MountPath /mnt/ssd -List   # disk already mounted in WSL
#>
[CmdletBinding()]
param(
    [int]      $Drive = -1,
    [int]      $Partition = 1,
    [string]   $BusId = "",
    [string]   $MountPath = "",
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
# "Permanently added 'thunder' to the list of known hosts", usbipd printing a
# notice - becomes a NativeCommandError and kills the run. Every external call
# below is judged by its exit code instead, which is what actually tells us
# whether it worked.
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"   # wsl.exe otherwise answers in UTF-16 and mangles parsing

function Info($m) { Write-Host "[push_ssd] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[push_ssd] $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[push_ssd] $m" -ForegroundColor Red; exit 1 }

function Have($exe) { [bool](Get-Command $exe -ErrorAction SilentlyContinue) }

# Run a command inside the distro and return its stdout, stderr dropped.
function WslOut([string]$cmd, [switch]$AsRoot) {
    if ($AsRoot) { $out = wsl.exe -d $Distro -u root -e bash -lc $cmd 2>&1 }
    else         { $out = wsl.exe -d $Distro       -e bash -lc $cmd 2>&1 }
    $script:WslExit = $LASTEXITCODE
    return ($out | ForEach-Object { "$_" -replace "`r", "" })
}

# ---------------------------------------------------------------- disk listing
function Get-CandidateDisks {
    Get-Disk | Where-Object { -not $_.IsBoot -and -not $_.IsSystem } |
        Select-Object Number, FriendlyName, BusType,
                      @{n = 'SizeGB'; e = { [math]::Round($_.Size / 1GB) } }, OperationalStatus
}

function Get-UsbDevices {
    if (-not (Have usbipd)) { return @() }
    $json = (usbipd state 2>&1) -join "`n"
    try {
        $devs = (ConvertFrom-Json $json).Devices
        if ($devs) { return @($devs) }
    } catch { }
    # older usbipd, or a build whose state JSON we do not recognise: read the
    # table it prints instead. Lines look like
    #   2-1    0480:a202  USB Attached SCSI (UAS) Mass Storage Device   Not shared
    $rows = @()
    foreach ($line in (usbipd list 2>&1)) {
        if ("$line" -match '^\s*(\d+-\d+)\s+([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\s+(.+?)\s{2,}(\S.*)$') {
            $rows += [pscustomobject]@{
                BusId = $Matches[1]; VidPid = $Matches[2]
                Description = $Matches[3].Trim(); StateText = $Matches[4].Trim()
                ClientIPAddress = $null; PersistedGuid = $null
            }
        }
    }
    return $rows
}

function Show-UsbDevices($devs) {
    if (-not $devs) {
        Warn "usbipd reports no USB devices at all. Is the service running?  (usbipd list)"
        return
    }
    $devs | Select-Object BusId, Description,
           @{n = 'State'; e = {
                 if ($_.ClientIPAddress) { "attached to WSL" }
                 elseif ($_.PersistedGuid) { "shared" }
                 elseif ($_.StateText) { $_.StateText }
                 else { "-" } }} |
        Format-Table -AutoSize | Out-String -Width 200 | Write-Host
}

if ($ListDisks) {
    Info "disks that are neither boot nor system (the Pi SSD is ~233 GB, USB):"
    Get-CandidateDisks | Format-Table -AutoSize
    $usb = Get-UsbDevices
    if ($usb) {
        Info "USB devices usbipd can hand to WSL (use -BusId):"
        Show-UsbDevices $usb
    } elseif (-not (Have usbipd)) {
        Info "usbipd-win is not installed; on ARM64 Windows it is the way in:"
        Info "  winget install --exact dorssel.usbipd-win"
    }
    exit 0
}

# ------------------------------------------------------------ prerequisites
if (-not (Have wsl.exe)) {
    Die "WSL is not installed. The SSD is ext4 and Windows cannot read it without WSL.`n  Install with:  wsl --install -d Ubuntu   (then reboot)"
}
# a distro name is a bare token; anything else on that stream is a notice
$wslDistros = (wsl.exe --list --quiet 2>&1) -replace "`0", "" |
              ForEach-Object { "$_".Trim() } |
              Where-Object { $_ -match '^[A-Za-z0-9._-]+$' }
if (-not $wslDistros) { Die "WSL has no distro installed. Run:  wsl --install -d Ubuntu" }
if (-not $Distro) { $Distro = ($wslDistros | Select-Object -First 1) }
Info "WSL distro: $Distro"

$admin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin -and -not $MountPath) {
    Die "attaching or mounting a disk needs an elevated shell. Re-open PowerShell with 'Run as Administrator'."
}

if ($ForceUnmount) {
    if ($Drive -ge 0) { wsl.exe --unmount "\\.\PHYSICALDRIVE$Drive" }
    if ($BusId)       { usbipd detach --busid $BusId }
    WslOut "umount /mnt/picam_ssd 2>/dev/null; true" -AsRoot | Out-Null
    Info "cleaned up"
    exit 0
}

WslOut "command -v rsync" | Out-Null
if ($WslExit -ne 0) {
    Die "rsync is missing inside WSL. Install it with:`n  wsl -d $Distro -u root apt-get update; wsl -d $Distro -u root apt-get install -y rsync"
}

# ---------------------------------------------------- where the engine lives
$engineWin = Join-Path $PSScriptRoot "ingest_ssd.sh"
if (-not (Test-Path $engineWin)) { Die "ingest_ssd.sh not found next to this script ($PSScriptRoot)" }
$engineWsl = (wsl.exe -d $Distro -e wslpath -a "$engineWin" 2>&1 | Select-Object -First 1)
if (-not $engineWsl) { Die "could not translate $engineWin into a WSL path" }
$engineWsl = "$engineWsl".Trim()

# A checkout on the Windows filesystem carries CRLF line endings for some
# files; bash refuses those. Stage the engine inside the distro without them.
$engineRun = "/tmp/picam_ingest_ssd.sh"
WslOut "tr -d '\r' < '$engineWsl' > $engineRun && chmod +x $engineRun" | Out-Null
if ($WslExit -ne 0) { Die "could not stage ingest_ssd.sh inside WSL" }

# --------------------------------------------- how WSL should reach Thunder
# WSL2 sits behind a NAT and does not always see the host's Tailscale network.
# When it cannot, borrow the Windows ssh.exe as rsync's transport: that
# traffic leaves through the host, where Tailscale certainly works.
$sshProbe = "ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new $Target true"
$sshCmd = "ssh"
Info "checking SSH from WSL to $Target ..."
WslOut $sshProbe | Out-Null
if ($WslExit -ne 0) {
    Warn "WSL cannot reach $Target directly (Tailscale is often not routed into WSL2)."
    $winSsh = "/mnt/c/Windows/System32/OpenSSH/ssh.exe"
    WslOut "$winSsh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new $Target true" | Out-Null
    if ($WslExit -ne 0) {
        Die @"
Neither WSL's ssh nor the Windows ssh.exe could reach $Target.
Check, on this machine:
  tailscale status         # is Thunder listed and online?
  ssh $Target              # does a plain SSH session work?
Thunder has Tailscale SSH enabled, so no key should be needed from a
machine on the same tailnet.
"@
    }
    $sshCmd = $winSsh
    Info "using the Windows ssh.exe as the transport"
} else {
    Info "WSL reaches $Target directly"
}

# ------------------------------------------------- get the disk into WSL -----
$cleanupWslMount = ""
$cleanupBusId    = ""
$cleanupUmount   = ""

function Mount-ViaWsl {
    if ($Drive -lt 0) {
        $cands = @(Get-CandidateDisks | Where-Object { $_.SizeGB -gt 180 -and $_.SizeGB -lt 300 })
        if ($cands.Count -eq 1) {
            $script:Drive = $cands[0].Number
            Info "auto-detected disk $($script:Drive) : $($cands[0].FriendlyName) ($($cands[0].SizeGB) GB, $($cands[0].BusType))"
        } else { return "" }
    }
    $dev = "\\.\PHYSICALDRIVE$Drive"
    Info "trying wsl --mount $dev partition $Partition (read-only) ..."
    $out = (wsl.exe --mount $dev --partition $Partition --type ext4 --options "ro" 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        if ($out -match "WSL_E_WSL_MOUNT_NOT_SUPPORTED") {
            Warn "wsl --mount is not available on this machine (ARM64 needs Windows build 27653+)."
        } else {
            Warn ("wsl --mount failed: " + ($out -replace "`n", " "))
        }
        return ""
    }
    $script:cleanupWslMount = $dev
    return "/mnt/wsl/PHYSICALDRIVE${Drive}p${Partition}"
}

function Mount-ViaUsbIp {
    if (-not (Have usbipd)) {
        Warn "usbipd-win is not installed. Install it once with:"
        Warn "  winget install --exact dorssel.usbipd-win     (then re-open this shell)"
        return ""
    }
    $devs = Get-UsbDevices
    if ($BusId) {
        $dev = $devs | Where-Object { $_.BusId -eq $BusId } | Select-Object -First 1
        if (-not $dev) {
            Warn "no USB device with bus id $BusId. usbipd sees:"
            Show-UsbDevices $devs
            return ""
        }
    } else {
        # what a mass-storage bridge tends to call itself, plus the drive's own
        # name as Windows reports it, since some enclosures announce that
        # Windows names devices in the display language, so the Japanese terms
        # are here too - as escapes, because a .ps1 without a BOM is read in
        # the ANSI code page by PowerShell 5.1 and literal kanji would rot.
        $wanted = 'Mass Storage|UAS|SCSI|SSD|Disk|Storage|External|Portable' +
                  '|\u5927\u5BB9\u91CF|\u8A18\u61B6\u88C5\u7F6E' +
                  '|\u30C7\u30A3\u30B9\u30AF|\u30B9\u30C8\u30EC\u30FC\u30B8'
        $model = ""
        if ($Drive -ge 0) {
            $d = Get-Disk -Number $Drive -ErrorAction SilentlyContinue
            if ($d) { $model = ($d.FriendlyName -split '\s+')[0] }
        }
        $cand = @($devs | Where-Object {
            $_.Description -match $wanted -or ($model -and $_.Description -match [regex]::Escape($model)) })
        if ($cand.Count -ne 1) {
            # one already attached to WSL is unambiguous enough
            $attached = @($devs | Where-Object { $_.ClientIPAddress })
            if ($attached.Count -eq 1) { $cand = $attached }
        }
        if ($cand.Count -ne 1) {
            Warn "cannot tell which of these is the SSD; re-run with -BusId <id>:"
            Show-UsbDevices $devs
            Warn "example:  .\construct\push_ssd.ps1 -BusId $(if ($devs) { $devs[0].BusId } else { '2-1' }) -List"
            return ""
        }
        $dev = $cand[0]
        $script:BusId = $dev.BusId
    }
    Info "handing USB device $($script:BusId) ($($dev.Description)) to WSL ..."
    usbipd bind --busid $script:BusId 2>&1 | Out-Null      # no-op when already shared
    usbipd attach --wsl --busid $script:BusId 2>&1 | ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -ne 0) {
        Warn "usbipd attach failed. Is the device in use by Windows? Try unplugging and replugging it."
        return ""
    }
    $script:cleanupBusId = $script:BusId
    Start-Sleep -Seconds 3

    # find the ext4 partition that just appeared
    $part = WslOut "lsblk -rno NAME,FSTYPE,LABEL,SIZE | grep -w ext4 | sort -k4 -h | tail -1" -AsRoot
    $part = ($part | Where-Object { $_ -ne "" } | Select-Object -First 1)
    if (-not $part) {
        Warn "the device is attached but WSL sees no ext4 partition on it."
        Warn "check with:  wsl -u root -e lsblk -f"
        Warn "(a WSL kernel without USB storage support would explain it: wsl --update, then retry)"
        return ""
    }
    $name = ($part -split '\s+')[0]
    Info "found /dev/$name ($part)"
    WslOut "mkdir -p /mnt/picam_ssd && mount -o ro /dev/$name /mnt/picam_ssd" -AsRoot | Out-Null
    if ($WslExit -ne 0) { Warn "mounting /dev/$name inside WSL failed"; return "" }
    $script:cleanupUmount = "/mnt/picam_ssd"
    return "/mnt/picam_ssd"
}

try {
    $srcPath = ""
    if ($MountPath) {
        Info "using the path you gave: $MountPath (nothing will be mounted or unmounted)"
        $srcPath = $MountPath
    } elseif ($BusId) {
        $srcPath = Mount-ViaUsbIp
    } else {
        $srcPath = Mount-ViaWsl
        if (-not $srcPath) {
            Info "falling back to usbipd-win passthrough"
            $srcPath = Mount-ViaUsbIp
        }
    }
    if (-not $srcPath) {
        Die @"
Could not make the SSD readable from WSL.

On ARM64 Windows, 'wsl --mount' needs build 27653 or newer, so the usual
route is usbipd-win:

  winget install --exact dorssel.usbipd-win     # once, then re-open PowerShell
  .\construct\push_ssd.ps1 -ListDisks           # note the SSD's bus id
  .\construct\push_ssd.ps1 -BusId 2-1 -List

If that is not an option, mount the disk in WSL by any means you like and
point this script at it:

  .\construct\push_ssd.ps1 -MountPath /mnt/ssd -List
"@
    }

    $engineArgs = @("--target", $Target, "--ssh-cmd", $sshCmd, "--data-root", $DataRoot,
                    "--progress-interval", "$ProgressInterval")
    if ($Session)  { $engineArgs += @("--session", $Session) }
    if ($List)     { $engineArgs += "--list" }
    if ($DryRun)   { $engineArgs += "--dry-run" }
    if ($Yes)      { $engineArgs += "-y" }
    $engineArgs += $ExtraArgs
    $engineArgs += $srcPath

    Info ("running: bash $engineRun " + ($engineArgs -join " "))
    Write-Host ""
    wsl.exe -d $Distro -e bash $engineRun @engineArgs
    $rc = $LASTEXITCODE
    Write-Host ""
    if ($rc -eq 0) { Info "done" } else { Warn "ingest_ssd.sh exited with $rc (see above)" }
    exit $rc
}
finally {
    if ($cleanupUmount)   { Info "unmounting $cleanupUmount inside WSL"; WslOut "umount '$cleanupUmount'" -AsRoot | Out-Null }
    if ($cleanupBusId)    { Info "detaching USB device $cleanupBusId";   usbipd detach --busid $cleanupBusId 2>&1 | Out-Null }
    if ($cleanupWslMount) { Info "unmounting $cleanupWslMount";          wsl.exe --unmount $cleanupWslMount 2>&1 | Out-Null }
}
