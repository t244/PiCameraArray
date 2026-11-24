<#
.SYNOPSIS
    Check that every burst on every Pi holds the expected number of frames.

.DESCRIPTION
    Runs scripts/count_frames.py on each Pi (newest session on its SSD) and
    tabulates the result: bursts in the session, how many held exactly
    -Expect frames, and which ones came up short. Read-only on the Pis.

    Use it after a trial run to confirm that 30 ms exposure at 30 fps
    (33.3 ms period) does not drop frames: every Pi should show short=0 and
    the same total. The first burst of a session is the trigger box's
    warm-up train and is counted separately, not as a failure.

    Needs count_frames.py on the Pis (deploy_pull.ps1 first).
    Works on Windows PowerShell 5.1 and PowerShell 7+. Needs plink (PuTTY).

.EXAMPLE
    .\check_frames.ps1                # e00..e15, all bursts of the newest session
    .\check_frames.ps1 -Last 5        # only the 5 most recent bursts per Pi
    .\check_frames.ps1 -Only 0,5 -Detail
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
    [int]    $Expect     = 60,
    [int]    $LastBursts = 0,
    [switch] $Detail       # also print every non-ok burst line
)

if (-not (Get-Command plink -ErrorAction SilentlyContinue)) {
    throw "plink not found on PATH. Install the PuTTY tools first."
}

$indices = if ($Only) { $Only } else { $First..$Last }
$cmd = "python3 '$RepoDir/scripts/count_frames.py' --no-gaps --expect $Expect"
if ($LastBursts -gt 0) { $cmd += " --last $LastBursts" }
if ($Detail) { $cmd = $cmd -replace ' --no-gaps', '' }

function ConvertFrom-KvLine([string] $line) {
    $h = @{}
    foreach ($pair in ($line -split ';')) {
        $kv = $pair -split '=', 2
        if ($kv.Count -eq 2) { $h[$kv[0].Trim()] = $kv[1].Trim() }
    }
    return $h
}

$rows = @()
$details = @()
foreach ($i in $indices) {
    $h = $HostFormat -f $i
    Write-Host "[$h] ..." -ForegroundColor DarkGray
    $out = @(plink -pw $Password -batch "$User@$h" $cmd 2>&1 | ForEach-Object { "$_" })
    $summary = $out | Where-Object { $_ -like 'host host=*' } | Select-Object -First 1
    if ($summary) {
        $kv = ConvertFrom-KvLine ($summary.Substring(5))
        $rows += [pscustomobject]@{
            Host   = $h
            Total  = [int]$kv['total']
            Ok     = [int]$kv['ok']
            Warmup = [int]$kv['warmup']
            Short  = [int]$kv['short']
            Long   = [int]$kv['long']
            Bad    = [int]$kv['bad']
            Short_bursts = $kv['short_list']
        }
        $details += $out | Where-Object { $_ -like 'burst *' -and $_ -notmatch 'status=(ok|warmup)' } |
            ForEach-Object { "$h  $($_.Substring(6))" }
    } else {
        $msg = ($out -join ' ').Trim()
        if (-not $msg) { $msg = "no output (unreachable?)" }
        $rows += [pscustomobject]@{
            Host = $h; Total = -1; Ok = 0; Warmup = 0; Short = 0; Long = 0; Bad = 0
            Short_bursts = $msg
        }
    }
}

Write-Host ""
$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

if ($Detail -and $details) {
    Write-Host "Bursts that are not ok (maxgap_ms is a hint only):" -ForegroundColor Yellow
    $details | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
}

$reached = @($rows | Where-Object { $_.Total -ge 0 })
$totals  = @($reached | ForEach-Object { $_.Total } | Sort-Object -Unique)
$failed  = @($rows | Where-Object { $_.Total -lt 0 -or $_.Short -gt 0 -or $_.Long -gt 0 -or $_.Bad -gt 0 })

if ($totals.Count -gt 1) {
    Write-Host ("Burst totals differ across Pis ({0}): a Pi with a lower total missed whole bursts, or the Pis are in different sessions." -f ($totals -join ', ')) -ForegroundColor Yellow
}
if ($failed.Count -eq 0 -and $totals.Count -le 1) {
    Write-Host ("All {0} Pis: every burst has {1} frames." -f $rows.Count, $Expect) -ForegroundColor Green
} else {
    Write-Host ("CHECK: {0}" -f (($failed | ForEach-Object { $_.Host }) -join ', ')) -ForegroundColor Red
}
