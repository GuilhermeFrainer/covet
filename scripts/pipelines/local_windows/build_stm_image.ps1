# Builds the STM image locally and uploads it to PCAD.
#
# The SLURM worker (scripts/experiments/slurm_stm_job.sh) loads
# ~/docker_images/cast_stm-lite-v<Version>.tar.gz on each node it runs on.
# After a new version, update STM_IMAGE in src/experiment_queue.py.
#
# Usage: .\scripts\pipelines\local_windows\build_stm_image.ps1 -Version 0.2.0
param (
    [Parameter(Mandatory=$true)]
    [string]$Version
)

$ErrorActionPreference = "Stop"

# Docker Image Variables
$ImageName = "cast"
$TagVersion = "${ImageName}:stm-lite-v$Version"

# File System Variables
$DownloadsDir = Join-Path $env:USERPROFILE "Downloads"
$TarName = "${ImageName}_stm-lite-v${Version}.tar"
$TarPath = Join-Path $DownloadsDir $TarName

# Rsync and Remote Variables
$RemoteUser = "gdsfrainer"
$RemoteHost = "gppd-hpc.inf.ufrgs.br"
$RemotePath = "~/docker_images"
$SshKeyPath = "~/.ssh/pcad_ufrgs" # Path inside WSL as per original command

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$QueueModule = Join-Path $RepoRoot "src\experiment_queue.py"
if (-not (Select-String -Path $QueueModule -Pattern ([regex]::Escape("`"$TagVersion`"")) -Quiet)) {
    Write-Warning "STM_IMAGE in src/experiment_queue.py is not $TagVersion; queued jobs will not use this image."
}

# The image copies nothing from the repository, so it is built without a context.
Write-Host "`n>>> Building Docker image: $TagVersion" -ForegroundColor Cyan
Get-Content (Join-Path $RepoRoot "Dockerfile.stm") -Raw | docker build --platform linux/amd64 -t $TagVersion -
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

Write-Host "`n>>> Saving image to $TarPath" -ForegroundColor Cyan
if (!(Test-Path $DownloadsDir)) {
    New-Item -ItemType Directory -Path $DownloadsDir | Out-Null
}
docker save $TagVersion -o $TarPath
if ($LASTEXITCODE -ne 0) { throw "docker save failed" }

Write-Host "`n>>> Converting path for WSL using wslpath" -ForegroundColor Cyan
# Replace backslashes with forward slashes to prevent stripping when passing to WSL
$NormalizedPath = $TarPath.Replace('\', '/')
$WslTarPath = wsl wslpath -u $NormalizedPath

if ([string]::IsNullOrWhiteSpace($WslTarPath)) {
    Write-Error "Failed to convert Windows path to WSL path. wslpath returned no output."
    exit 1
}
$WslTarPath = $WslTarPath.Trim()

Write-Host "`n>>> Compressing image" -ForegroundColor Cyan
wsl gzip -f $WslTarPath
if ($LASTEXITCODE -ne 0) { throw "gzip failed" }

Write-Host "`n>>> Transferring image to remote server via wsl rsync" -ForegroundColor Cyan
$RemoteDest = "${RemoteUser}@${RemoteHost}:$RemotePath"
$SshCommand = "ssh -i $SshKeyPath"

# Run rsync through WSL
wsl rsync -avP -e "$SshCommand" "$WslTarPath.gz" $RemoteDest
if ($LASTEXITCODE -ne 0) { throw "rsync failed" }

Write-Host "`nProcess completed successfully!" -ForegroundColor Green
