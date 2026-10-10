# Local batch of 2026-10-10: the Fed and Yelp runs the cluster could not
# finish (see docs/paper_submission_checklist.md).
#
# - Fed UMAP + Spectral runs 1-5: stuck on the cluster for 23 h on run 1
#   (k = 10); every other run of this model takes about 7 min.
# - Fed STM K = 10, 20, 40, 50 and Yelp STM K = 10, 40, 50: killed on the
#   cluster (exit 137), probably by node memory pressure.
#
# Every run is started on its own and has a time limit; a run that exceeds it
# is killed and marked TIMEOUT, and the batch moves on. Each run prints a
# banner, and a status table (OK / FAILED / TIMEOUT / NOT RUN) is printed
# after every run and at the end, even after Ctrl+C. To resume, comment out
# the finished lines in $Runs below.
#
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/pipelines/local_windows/fill_gaps_2026-10-10.ps1
# The status table is also written to logs/fill_gaps_<timestamp>.log.

$ErrorActionPreference = "Continue"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $RepoRoot

# One line per run. Comment out a line to skip that run.
# Runner "opt" = run_optimizer.py, "stm" = run_stm.py (local R).
# UMAP + Spectral run N: k = 10,10,10,20,20 for N = 1..5 (seeds vary fastest).
# STM run N: K = 10, 20, 30, 40, 50 for N = 1..5.
$Runs = @(
    @{ Runner = "opt"; Exp = "fed/fed_standard_umap_spectral"; Run = 1; Label = "Fed UMAP+Spectral k=10 seed 36201624 (hung on cluster)"; Minutes = 30 }
    @{ Runner = "opt"; Exp = "fed/fed_standard_umap_spectral"; Run = 2; Label = "Fed UMAP+Spectral k=10 seed 62613654"; Minutes = 30 }
    @{ Runner = "opt"; Exp = "fed/fed_standard_umap_spectral"; Run = 3; Label = "Fed UMAP+Spectral k=10 seed 57116123"; Minutes = 30 }
    @{ Runner = "opt"; Exp = "fed/fed_standard_umap_spectral"; Run = 4; Label = "Fed UMAP+Spectral k=20 seed 36201624"; Minutes = 30 }
    @{ Runner = "opt"; Exp = "fed/fed_standard_umap_spectral"; Run = 5; Label = "Fed UMAP+Spectral k=20 seed 62613654"; Minutes = 30 }
    @{ Runner = "stm"; Exp = "fed/fed_standard_stm"; Run = 1; Label = "Fed STM K=10"; Minutes = 120 }
    @{ Runner = "stm"; Exp = "fed/fed_standard_stm"; Run = 2; Label = "Fed STM K=20"; Minutes = 120 }
    @{ Runner = "stm"; Exp = "fed/fed_standard_stm"; Run = 4; Label = "Fed STM K=40"; Minutes = 120 }
    @{ Runner = "stm"; Exp = "fed/fed_standard_stm"; Run = 5; Label = "Fed STM K=50"; Minutes = 120 }
    @{ Runner = "stm"; Exp = "yelp/yelp_standard_stm"; Run = 1; Label = "Yelp STM K=10"; Minutes = 120 }
    @{ Runner = "stm"; Exp = "yelp/yelp_standard_stm"; Run = 4; Label = "Yelp STM K=40"; Minutes = 120 }
    @{ Runner = "stm"; Exp = "yelp/yelp_standard_stm"; Run = 5; Label = "Yelp STM K=50"; Minutes = 120 }
)

$Scripts = @{
    opt = "scripts/experiments/run_optimizer.py"
    stm = "scripts/experiments/run_stm.py"
}

New-Item -ItemType Directory -Force "logs" | Out-Null
$Log = "logs/fill_gaps_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$Line = "=" * 78
foreach ($r in $Runs) {
    $r.Status = "NOT RUN"
    $r.Minutes_Taken = ""
}

function Show-Status {
    param([string]$Title)
    $text = @("", $Line, $Title, $Line)
    $text += "{0,-8} {1,-56} {2}" -f "STATUS", "RUN", "MINUTES"
    foreach ($r in $Runs) {
        $text += "{0,-8} {1,-56} {2}" -f $r.Status, "$($r.Label)", $r.Minutes_Taken
    }
    $text += $Line
    Set-Content -Path $Log -Value ($text -join "`n")
    foreach ($t in $text) {
        $color = if ($t -match "^OK ") { "Green" } elseif ($t -match "^(FAILED|TIMEOUT)") { "Red" } else { "Cyan" }
        Write-Host $t -ForegroundColor $color
    }
}

Write-Host $Line -ForegroundColor Cyan
Write-Host "Local gap-filling batch: $($Runs.Count) runs. Status log: $Log" -ForegroundColor Cyan
Write-Host $Line -ForegroundColor Cyan

try {
    for ($i = 0; $i -lt $Runs.Count; $i++) {
        $r = $Runs[$i]
        $script = $Scripts[$r.Runner]
        $cmd = "uv run python $script --exp $($r.Exp) --model $($r.Run)"

        Write-Host ""
        Write-Host $Line -ForegroundColor Yellow
        Write-Host ("[{0}/{1}]  {2}   NOW RUNNING: {3}" -f ($i + 1), $Runs.Count, (Get-Date -Format "HH:mm:ss"), $r.Label) -ForegroundColor Yellow
        Write-Host "  CMD     : $cmd" -ForegroundColor Yellow
        Write-Host "  LIMIT   : $($r.Minutes) min" -ForegroundColor Yellow
        Write-Host "  IF YOU STOP NOW: comment out the lines above '$($r.Label)' in `$Runs" -ForegroundColor Yellow
        Write-Host $Line -ForegroundColor Yellow

        $r.Status = "RUNNING"
        $start = Get-Date
        $proc = Start-Process -FilePath "uv" -NoNewWindow -PassThru `
            -ArgumentList @("run", "python", $script, "--exp", $r.Exp, "--model", "$($r.Run)")
        $finished = $proc.WaitForExit($r.Minutes * 60 * 1000)
        if (-not $finished) {
            # Kill the whole tree: uv, Python and (for STM) Rscript.
            & taskkill /PID $proc.Id /T /F | Out-Null
            $r.Status = "TIMEOUT"
        } else {
            $proc.WaitForExit()
            $r.Status = if ($proc.ExitCode -eq 0) { "OK" } else { "FAILED" }
        }
        $r.Minutes_Taken = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
        Show-Status "STATUS after run $($i + 1) of $($Runs.Count)  ($(Get-Date -Format 'HH:mm:ss'))"
    }
}
finally {
    foreach ($r in $Runs) {
        if ($r.Status -eq "RUNNING") { $r.Status = "STOPPED" }
    }
    Show-Status "FINAL STATUS  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))  -- comment out the OK lines to resume"
    Write-Host "Next: uv run python scripts/analysis/merge_results.py" -ForegroundColor Cyan
}
