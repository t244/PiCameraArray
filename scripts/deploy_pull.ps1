<#
.SYNOPSIS
    Run 'git pull' on every Pi in the camera array and report each Pi's commit.

.DESCRIPTION
    Pulls the branch each Pi already has checked out (fast-forward only), so a
    Pi with local edits or a diverged branch is reported instead of being
    merged. Run from the lab network; the Pis need internet access to GitHub.

    Works on Windows PowerShell 5.1 and PowerShell 7+. Needs plink (PuTTY).

.EXAMPLE
    .\deploy_pull.ps1                 # e00..e15
    .\deploy_pull.ps1 -Only 0,5       # selected Pis
    .\deploy_pull.ps1 -Branch thunder # switch to this branch first, then pull
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
    [string] $Branch     = ""
)

if (-not (Get-Command plink -ErrorAction SilentlyContinue)) {
    throw "plink not found on PATH. Install the PuTTY tools first."
}

$indices = if ($Only) { $Only } else { $First..$Last }
# Remote shell command. Single-quoted here-string: PowerShell expands nothing
# inside it; __REPO__ and __CHECKOUT__ are substituted below.
$remote = @'
cd '__REPO__' || exit 2
__CHECKOUT__
git pull --ff-only 2>&1 | tail -2
echo "branch=$(git rev-parse --abbrev-ref HEAD) head=$(git log -1 --format=%h)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo 'WARNING: local changes present'
'@
$checkout = if ($Branch) { "git fetch -q origin && git checkout -q '$Branch' || exit 3" } else { ":" }
$cmd = (($remote -replace '__REPO__', $RepoDir) -replace '__CHECKOUT__', $checkout) -replace "`r", "" -replace "`n", "; "

$results = @()
foreach ($i in $indices) {
    $h = $HostFormat -f $i
    Write-Host "[$h] ..." -ForegroundColor DarkGray
    $out = (plink -pw $Password -batch "$User@$h" $cmd 2>&1 | Out-String).TrimEnd()
    $ok  = ($LASTEXITCODE -eq 0) -and ($out -notmatch 'fatal|error|WARNING')
    $results += [pscustomobject]@{ Host = $h; Ok = $ok; Output = $out }
}

Write-Host ""
foreach ($r in $results) {
    $c = if ($r.Ok) { 'Green' } else { 'Red' }
    Write-Host "=== $($r.Host) === $(if ($r.Ok) {'OK'} else {'CHECK'})" -ForegroundColor $c
    Write-Host $r.Output
}
$heads = $results | ForEach-Object { if ($_.Output -match 'head=(\S+)') { $Matches[1] } } | Sort-Object -Unique
Write-Host ""
Write-Host ("Distinct HEADs across Pis: {0}" -f ($heads -join ', ')) -ForegroundColor Cyan
